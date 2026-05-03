import os
import uuid
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from typing_extensions import TypedDict

from app.agents.planner import analyze_question
from app.retrieval import search_news
from database import SessionLocal
from models import StoryMetadata
from schemas import (
    ChatResponse,
    ContextAssessment,
    NewsSource,
    QueryAnalysis,
    QueryRewrite,
    SynthesizedAnswer,
    TraceStep,
)


MAX_RETRIEVAL_ATTEMPTS = 2
MIN_CONTEXT_DOCS = 2
MAX_ANSWER_SOURCES = 6
MAX_CONTEXT_CHARS_PER_SOURCE = 2500

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "missing")

REWRITE_SYSTEM_PROMPT = """
You are a query rewriting assistant for a news chatbot.

The previous search did not retrieve enough context.

Create a better search query for retrieving relevant news stories.

Rules:
- Keep the query concise.
- Preserve important names, organizations, topics, dates, and places.
- Expand abbreviations when helpful.
- Do not answer the question.
- Only produce the structured rewritten query.
"""

ANSWER_SYSTEM_PROMPT = """
You are a careful news assistant.

Answer the user's question using only the provided news sources.

Rules:
- Give a direct answer when the sources support it.
- Cite sources using [Source 1], [Source 2], etc.
- If the retrieved sources do not answer the question, say that clearly.
- Do not invent facts.
- Do not use outside knowledge.
- Keep the answer concise but useful.
- Only produce the structured answer.
"""

CONTEXT_SYSTEM_PROMPT = """
You are a context quality judge for a news RAG chatbot.

Decide whether the retrieved news sources are sufficient to give a useful grounded answer.

Rules:
- Return context_enough true when the sources directly support a useful answer, even if the answer is not exhaustive.
- For summaries, briefings, and timelines, true means the sources contain enough relevant events or facts to summarize what is available.
- If sources are irrelevant, generic, or only weakly related, return false.
- If false, suggest a better concise search query.
- Do not answer the user's question.
- Only produce the structured assessment.
"""


class ChatState(TypedDict):
    question: str
    analysis: QueryAnalysis | None
    current_query: str | None
    documents: list[Any]
    attempts: int
    context_enough: bool
    suggested_query: str | None
    steps: list[TraceStep]
    response: ChatResponse | None


def require_analysis(state: ChatState) -> QueryAnalysis:
    analysis = state["analysis"]
    if analysis is None:
        raise ValueError("Missing query analysis in chat state.")
    return analysis


def plan_query(state: ChatState):
    analysis = analyze_question(state["question"])
    return {
        "analysis": analysis,
        "current_query": analysis.search_query,
        "attempts": 0,
        "context_enough": False,
        "suggested_query": None,
        "steps": [
            TraceStep(
                name="Parsed query",
                detail=f"Intent: {analysis.intent}; search query: {analysis.search_query}",
            )
        ],
    }


def route_after_planning(state: ChatState) -> Literal["ask_clarification", "retrieve"]:
    analysis = require_analysis(state)
    if analysis.clarification_needed:
        return "ask_clarification"
    return "retrieve"


def ask_clarification(state: ChatState):
    analysis = require_analysis(state)

    return {
        "response": ChatResponse(
            type="clarification_needed",
            message=analysis.clarification_question
            or "Could you add a little more context?",
            analysis=analysis,
            sources=[],
            steps=state.get("steps", [])
            + [TraceStep(name="Asked for clarification", detail=analysis.clarification_question)],
        )
    }


def retrieve(state: ChatState):
    analysis = require_analysis(state)
    query = state["current_query"] or analysis.search_query
    documents = search_news(query, k=analysis.k)

    return {
        "documents": documents,
        "attempts": state["attempts"] + 1,
        "steps": state.get("steps", [])
        + [
            TraceStep(
                name="Retrieved sources",
                detail=(
                    f"Attempt {state['attempts'] + 1}: hybrid search returned "
                    f"{len(documents)} candidate chunk(s)."
                ),
            )
        ],
    }


def check_context(state: ChatState):
    documents = state["documents"]
    analysis = require_analysis(state)

    if not documents:
        return {
            "context_enough": False,
            "steps": state.get("steps", [])
            + [TraceStep(name="Checked context", detail="No documents found.")],
        }

    _, context_block = build_sources_and_context(documents)

    try:
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": CONTEXT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['question']}\n"
                        f"Planned search query: {analysis.search_query}\n"
                        f"Detected entities: {analysis.entities}\n\n"
                        f"Retrieved sources:\n{context_block}"
                    ),
                },
            ],
            text_format=ContextAssessment,
        )
        assessment = response.output_parsed
        return {
            "context_enough": assessment.context_enough,
            "suggested_query": assessment.suggested_query,
            "steps": state.get("steps", [])
            + [
                TraceStep(
                    name="Judged context",
                    detail=assessment.reason,
                )
            ],
        }
    except Exception as exc:
        context_enough = len(documents) >= MIN_CONTEXT_DOCS
        return {
            "context_enough": context_enough,
            "suggested_query": None,
            "steps": state.get("steps", [])
            + [
                TraceStep(
                    name="Checked context",
                    detail=(
                        f"LLM context judge failed: {exc}. "
                        f"Fallback context enough: {context_enough}."
                    ),
                )
            ],
        }


def route_after_context(
    state: ChatState,
) -> Literal["answer", "rewrite_query", "limited_answer"]:
    if state["context_enough"]:
        return "answer"

    if state["attempts"] < MAX_RETRIEVAL_ATTEMPTS:
        return "rewrite_query"

    if state["documents"]:
        return "answer"

    return "limited_answer"


def story_id_to_uuid(story_id: str | None):
    if not story_id:
        return None

    try:
        return uuid.UUID(story_id)
    except ValueError:
        return None


def build_sources_and_context(documents: list[Any]) -> tuple[list[NewsSource], str]:
    seen_story_ids = set()
    sources = []
    context_block = ""
    db = SessionLocal()

    try:
        for document in documents:
            metadata = document.metadata
            story_id = metadata.get("story_id")
            dedupe_key = story_id or document.page_content[:160]

            if dedupe_key in seen_story_ids:
                continue

            seen_story_ids.add(dedupe_key)
            story_uuid = story_id_to_uuid(story_id)
            story = db.get(StoryMetadata, story_uuid) if story_uuid else None

            headline = story.headline if story else metadata.get("headline", "Untitled")
            summary = story.summary if story else None

            sources.append(
                NewsSource(
                    headline=headline,
                    summary=summary,
                    match_snippet=document.page_content,
                    story_id=story_id,
                )
            )

            story_context = story.full_content if story else document.page_content
            context_block += (
                f"\n[Source {len(sources)}]\n"
                f"Headline: {headline}\n"
                f"Summary: {summary or 'Not available'}\n"
                f"Context: {story_context[:MAX_CONTEXT_CHARS_PER_SOURCE]}\n"
            )

            if len(sources) >= MAX_ANSWER_SOURCES:
                break

        return sources, context_block

    finally:
        db.close()


def build_sources(documents: list[Any]) -> list[NewsSource]:
    sources, _ = build_sources_and_context(documents)
    return sources


def build_fallback_sources(documents: list[Any]) -> list[NewsSource]:
    sources = []
    for document in documents[:MAX_ANSWER_SOURCES]:
        metadata = document.metadata
        sources.append(
            NewsSource(
                headline=metadata.get("headline", "Untitled"),
                summary=None,
                match_snippet=document.page_content,
                story_id=metadata.get("story_id"),
            )
        )
    return sources


def answer(state: ChatState):
    sources, context_block = build_sources_and_context(state["documents"])
    steps = state.get("steps", []) + [
        TraceStep(
            name="Hydrated sources",
            detail="Fetched full story context from Postgres where available.",
        )
    ]

    try:
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": ANSWER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['question']}\n\n"
                        f"Sources:\n{context_block}"
                    ),
                },
            ],
            text_format=SynthesizedAnswer,
        )
        message = response.output_parsed.answer
        steps.append(
            TraceStep(
                name="Generated answer",
                detail=f"Used {len(sources)} unique source(s).",
            )
        )

    except Exception as exc:
        print(f"OpenAI answer generation failed: {exc}")
        citation_list = ", ".join(
            f"[Source {index}]"
            for index in range(1, len(sources) + 1)
        )
        message = (
            "I found matching news sources, but the answer generator failed. "
            f"Please inspect these sources directly: {citation_list}."
        )
        steps.append(
            TraceStep(
                name="Generated fallback",
                detail=f"Answer LLM failed: {exc}",
            )
        )

    return {
        "response": ChatResponse(
            type="answer",
            message=message,
            analysis=state["analysis"],
            sources=sources,
            steps=steps,
        )
    }


def rewrite_query(state: ChatState):
    analysis = require_analysis(state)

    if state.get("suggested_query"):
        return {
            "current_query": state["suggested_query"],
            "context_enough": False,
            "steps": state.get("steps", [])
            + [
                TraceStep(
                    name="Rewrote query",
                    detail=f"Used context judge suggestion: {state['suggested_query']}",
                )
            ],
        }

    try:
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": REWRITE_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Original question: {state['question']}\n"
                        f"Previous search query: {state['current_query']}\n"
                        f"Detected entities: {analysis.entities}"
                    ),
                },
            ],
            text_format=QueryRewrite,
        )
        rewritten_query = response.output_parsed.rewritten_query

    except Exception as exc:
        print(f"OpenAI query rewrite failed: {exc}")
        rewritten_query = " ".join([state["question"], *analysis.entities]).strip()

    return {
        "current_query": rewritten_query,
        "context_enough": False,
        "steps": state.get("steps", [])
        + [
            TraceStep(
                name="Rewrote query",
                detail=f"New query: {rewritten_query}",
            )
        ],
    }


def limited_answer(state: ChatState):
    analysis = require_analysis(state)
    sources = build_sources(state["documents"])

    return {
        "response": ChatResponse(
            type="limited_answer",
            message=(
                "I could not find enough matching news context to answer "
                "confidently. Try adding a person, organization, topic, or "
                "time period. I am showing the closest sources I found below."
            ),
            analysis=analysis,
            sources=sources,
            steps=state.get("steps", [])
            + [TraceStep(name="Stopped search", detail="Reached retrieval attempt limit.")],
        )
    }


builder = StateGraph(ChatState)

builder.add_node("plan_query", plan_query)
builder.add_node("ask_clarification", ask_clarification)
builder.add_node("retrieve", retrieve)
builder.add_node("check_context", check_context)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("limited_answer", limited_answer)
builder.add_node("answer", answer)

builder.add_edge(START, "plan_query")
builder.add_conditional_edges(
    "plan_query",
    route_after_planning,
    {
        "ask_clarification": "ask_clarification",
        "retrieve": "retrieve",
    },
)
builder.add_edge("ask_clarification", END)
builder.add_edge("retrieve", "check_context")
builder.add_conditional_edges(
    "check_context",
    route_after_context,
    {
        "answer": "answer",
        "rewrite_query": "rewrite_query",
        "limited_answer": "limited_answer",
    },
)
builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("answer", END)
builder.add_edge("limited_answer", END)

chat_graph = builder.compile()
