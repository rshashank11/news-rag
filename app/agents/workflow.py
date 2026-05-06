import os  # Reads environment variables such as OPENAI_API_KEY.
import uuid  # Converts story_id strings back into UUID objects for Postgres lookup.
from typing import Any, Literal  # Any is flexible typing; Literal restricts route names to fixed strings.
from zoneinfo import ZoneInfo  # Provides timezone handling; we use IST for publish dates.

from langgraph.graph import END, START, StateGraph  # LangGraph building blocks for the chatbot workflow.
from openai import OpenAI  # OpenAI client for planner/judge/rewrite/answer calls.
from typing_extensions import TypedDict  # Typed dictionary used to describe LangGraph state.

from app.agents.planner import analyze_question  # Query planner function.
from app.retrieval import search_news  # Hybrid retrieval function.
from database import SessionLocal  # Creates Postgres sessions for full-story hydration.
from models import StoryMetadata  # SQLAlchemy model for the stories table.
from schemas import (  # Pydantic schemas used for structured inputs/outputs.
    ChatResponse,  # Final API response shape.
    ContextAssessment,  # Structured context-judge result.
    NewsSource,  # Source card shape.
    QueryAnalysis,  # Planner result shape.
    QueryRewrite,  # Query rewrite result shape.
    SynthesizedAnswer,  # Answer-generator result shape.
    TraceStep,  # UI trace/process step shape.
)


MAX_RETRIEVAL_ATTEMPTS = 2  # Maximum number of searches before we stop and give a limited answer.
MAX_ANSWER_SOURCES = 6  # Maximum number of unique full stories passed to the answer LLM.
MAX_CONTEXT_CHARS_PER_SOURCE = 2500  # Max characters from each full story sent to the LLM to control token usage.
IST = ZoneInfo("Asia/Kolkata")  # Timezone used for showing article dates as India dates.

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "missing")  # OpenAI client; "missing" avoids import-time crashes.

REWRITE_SYSTEM_PROMPT = """  # Instructions for rewriting weak search queries.
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

ANSWER_SYSTEM_PROMPT = """  # Instructions for generating grounded final answers.
You are a careful news assistant.

Answer the user's question using only the provided news sources.

Rules:
- Give a direct answer when the sources support it.
- Cite sources using [Source 1], [Source 2], etc.
- Every factual paragraph or bullet must contain at least one source citation.
- Do not cite a source unless that source directly supports the sentence.
- If the retrieved sources do not answer the question, say that clearly.
- Do not invent facts.
- Do not use outside knowledge.
- Keep the answer concise but useful.
- Only produce the structured answer.
"""

CONTEXT_SYSTEM_PROMPT = """  # Instructions for judging whether retrieved sources are good enough.
You are a context quality judge for a news RAG chatbot.

Decide whether the retrieved news sources are sufficient to give a useful grounded answer.

Rules:
- Return context_enough true only when the sources directly support the specific question.
- If the question asks about a named person, case, organization, court, statute, date, or event, the sources must discuss that same subject.
- Similar legal topics are not enough. For example, an Article 21 story is not enough for a different Article 21 highway-safety case.
- For summaries, briefings, and timelines, true means the sources contain enough relevant events or facts to summarize what is available.
- If sources are irrelevant, generic, or only weakly related, return false.
- If false, suggest a better concise search query.
- Do not answer the user's question.
- Only produce the structured assessment.
"""


class ChatState(TypedDict):  # Shared state object passed between LangGraph nodes.
    question: str  # Original user question.
    analysis: QueryAnalysis | None  # Planner output; None before plan_query runs.
    current_query: str | None  # Search query currently being used.
    documents: list[Any]  # Retrieved LangChain Document objects.
    attempts: int  # Number of retrieval attempts already made.
    context_enough: bool  # Whether sources are good enough to answer.
    suggested_query: str | None  # Better query suggested by context judge.
    steps: list[TraceStep]  # Human-readable process trace for UI.
    response: ChatResponse | None  # Final response once workflow finishes.


def require_analysis(state: ChatState) -> QueryAnalysis:  # Helper that safely fetches planner output.
    analysis = state["analysis"]  # Reads analysis from LangGraph state.
    if analysis is None:  # If this happens, the graph order is wrong or a node failed.
        raise ValueError("Missing query analysis in chat state.")  # Fail loudly because later nodes need analysis.
    return analysis  # Return non-None planner result.


def plan_query(state: ChatState):  # First workflow node: understand the user's question.
    analysis = analyze_question(state["question"])  # Calls planner LLM to create structured search plan.
    return {  # LangGraph merges this dictionary into the shared state.
        "analysis": analysis,  # Save planner result for later nodes.
        "current_query": analysis.search_query,  # Initial query used by retrieval.
        "attempts": 0,  # Reset retrieval count for this request.
        "context_enough": False,  # Nothing has been judged yet.
        "suggested_query": None,  # No rewrite suggestion yet.
        "steps": [  # Start visible trace steps for the UI.
            TraceStep(  # One process step.
                name="Parsed query",  # Short label shown in UI.
                detail=f"Intent: {analysis.intent}; search query: {analysis.search_query}",  # Human-readable planner result.
            )
        ],
    }


def route_after_planning(state: ChatState) -> Literal["ask_clarification", "retrieve"]:  # Chooses next node after planning.
    analysis = require_analysis(state)  # Gets planner output safely.
    if analysis.clarification_needed:  # If question is too vague...
        return "ask_clarification"  # ...ask user for more details instead of searching badly.
    return "retrieve"  # Otherwise continue to retrieval.


def ask_clarification(state: ChatState):  # Terminal node when the question is too vague.
    analysis = require_analysis(state)  # Gets planner output.

    return {  # Writes final response into graph state.
        "response": ChatResponse(  # Structured API response.
            type="clarification_needed",  # Tells frontend this is not a final answer.
            message=analysis.clarification_question  # Use planner's follow-up question if available.
            or "Could you add a little more context?",  # Fallback follow-up question.
            analysis=analysis,  # Include planner analysis for debugging/UI.
            sources=[],  # No sources because no search happened.
            steps=state.get("steps", [])  # Existing trace steps.
            + [TraceStep(name="Asked for clarification", detail=analysis.clarification_question)],  # Add final trace step.
        )
    }


def retrieve(state: ChatState):  # Node that searches OpenSearch for relevant chunks.
    analysis = require_analysis(state)  # Gets planner output.
    query = state["current_query"] or analysis.search_query  # Uses rewritten query if available, otherwise planner query.
    documents = search_news(  # Runs hybrid retrieval.
        query,  # Search query.
        k=analysis.k,  # Number of final documents requested.
        from_date=analysis.from_date,  # Optional start date filter.
        to_date=analysis.to_date,  # Optional end date filter.
    )

    date_detail = ""  # Extra text for the trace panel.
    if analysis.from_date or analysis.to_date:  # Only mention date filter if one exists.
        date_detail = f" Date filter: {analysis.from_date or 'any'} to {analysis.to_date or 'any'}."  # Human-readable date range.

    return {  # Updates graph state.
        "documents": documents,  # Save retrieved documents for context checking.
        "attempts": state["attempts"] + 1,  # Increment retrieval attempt count.
        "steps": state.get("steps", [])  # Existing trace steps.
        + [  # Add retrieval trace step.
            TraceStep(  # Trace object shown in UI.
                name="Retrieved sources",  # Step label.
                detail=(  # Step detail.
                    f"Attempt {state['attempts'] + 1}: hybrid search returned "
                    f"{len(documents)} candidate chunk(s).{date_detail}"
                ),
            )
        ],
    }


def check_context(state: ChatState):  # Node that decides whether retrieved documents are good enough.
    documents = state["documents"]  # Retrieved chunks from previous node.
    analysis = require_analysis(state)  # Planner output used for entities/date/query.

    if not documents:  # If retrieval returned nothing...
        return {  # Update state and do not call answer generation.
            "context_enough": False,  # No documents means no grounded answer.
            "steps": state.get("steps", [])  # Existing trace.
            + [TraceStep(name="Checked context", detail="No documents found.")],  # Explain the failure.
        }

    _, context_block = build_sources_and_context(documents)  # Hydrates sources and builds text block for judge LLM.

    try:  # LLM judge can fail, so we catch errors and fail closed.
        response = client.responses.parse(  # Asks LLM for structured context quality judgment.
            model="gpt-4o-mini",  # Cheap model used as judge.
            input=[  # Messages passed to judge.
                {  # System instruction.
                    "role": "system",  # High-priority role.
                    "content": CONTEXT_SYSTEM_PROMPT,  # Guardrail rules.
                },
                {  # User message contains the actual question and retrieved context.
                    "role": "user",  # Normal prompt content.
                    "content": (  # Multi-line prompt with all relevant details.
                        f"Question: {state['question']}\n"
                        f"Planned search query: {analysis.search_query}\n"
                        f"Detected entities: {analysis.entities}\n\n"
                        f"Retrieved sources:\n{context_block}"
                    ),
                },
            ],
            text_format=ContextAssessment,  # Force output into context_enough/reason/suggested_query.
        )
        assessment = response.output_parsed  # Parsed Pydantic object from LLM.
        return {  # Update graph state with judge result.
            "context_enough": assessment.context_enough,  # True means answer node may run.
            "suggested_query": assessment.suggested_query,  # Optional better query for retry.
            "steps": state.get("steps", [])  # Existing trace.
            + [  # Add context judgment trace.
                TraceStep(  # One UI process step.
                    name="Judged context",  # Step title.
                    detail=assessment.reason,  # Explain why sources were accepted/rejected.
                )
            ],
        }
    except Exception as exc:  # If context judge fails, safer to refuse than hallucinate.
        return {  # Fail closed.
            "context_enough": False,  # Do not answer without judge approval.
            "suggested_query": None,  # No query suggestion available.
            "steps": state.get("steps", [])  # Existing trace.
            + [  # Add failure trace.
                TraceStep(  # UI trace step.
                    name="Checked context",  # Step label.
                    detail=(  # Detail explains fail-closed behavior.
                        f"LLM context judge failed: {exc}. "
                        "Fallback marked context as insufficient."
                    ),
                )
            ],
        }


def route_after_context(
    state: ChatState,  # Current LangGraph state after context check.
) -> Literal["answer", "rewrite_query", "limited_answer"]:  # Allowed route names.
    if state["context_enough"]:  # If guardrail approves the sources...
        return "answer"  # ...generate final answer.

    if state["attempts"] < MAX_RETRIEVAL_ATTEMPTS:  # If we still have retry budget...
        return "rewrite_query"  # ...try a better search query.

    return "limited_answer"  # Otherwise stop and safely refuse/limit the answer.


def story_id_to_uuid(story_id: str | None):  # Converts string story ID into uuid.UUID for database lookup.
    if not story_id:  # Missing story ID cannot be converted.
        return None  # Tell caller there is no valid UUID.

    try:  # UUID conversion can fail if the string is malformed.
        return uuid.UUID(story_id)  # Return a real UUID object.
    except ValueError:  # Bad UUID string.
        return None  # Fail gently instead of crashing retrieval.


def datetime_to_ist_date(value):  # Converts datetime into YYYY-MM-DD date string in IST.
    if value is None:  # No datetime means no date to show.
        return None  # Keep source date empty.

    return value.astimezone(IST).date().isoformat()  # Convert timezone, take date part, format for UI/filtering.


def build_sources_and_context(documents: list[Any]) -> tuple[list[NewsSource], str]:  # Turns retrieved chunks into UI sources + LLM context.
    seen_story_ids = set()  # Tracks stories already included so repeated chunks do not create duplicate source cards.
    sources = []  # List of NewsSource objects returned to frontend.
    context_block = ""  # Plain text source block sent to judge/answer LLM.
    db = SessionLocal()  # Opens Postgres session to fetch full story text.

    try:  # Ensure DB session closes even if source building fails.
        for document in documents:  # Loop through retrieved chunks in ranked order.
            metadata = document.metadata  # Metadata attached during ingestion/retrieval.
            story_id = metadata.get("story_id")  # Original story UUID as string.
            dedupe_key = story_id or document.page_content[:160]  # Prefer story ID; fallback to text prefix.

            if dedupe_key in seen_story_ids:  # If this story was already included...
                continue  # ...skip duplicate chunk/source.

            seen_story_ids.add(dedupe_key)  # Mark story/chunk as already included.
            story_uuid = story_id_to_uuid(story_id)  # Convert string ID to UUID for DB query.
            story = db.get(StoryMetadata, story_uuid) if story_uuid else None  # Fetch full story from Postgres when possible.

            headline = story.headline if story else metadata.get("headline", "Untitled")  # Prefer DB headline, fallback to metadata.
            summary = story.summary if story else None  # Summary comes from DB when story exists.
            story_context = story.full_content if story else document.page_content  # Hydrate full story; fallback to retrieved chunk.
            published_at = (  # Display date for source card and LLM context.
                datetime_to_ist_date(story.published_at)  # Convert DB datetime to IST date string.
                if story and story.published_at  # Use DB date when available.
                else metadata.get("published_at")  # Fallback to OpenSearch metadata date.
            )

            sources.append(  # Add a source card for the frontend.
                NewsSource(  # Pydantic model for one source.
                    headline=headline,  # Article title.
                    summary=summary,  # Article summary if available.
                    published_at=published_at,  # Publish date.
                    match_snippet=story_context,  # Evidence text shown in source card.
                    story_id=story_id,  # Original story ID for traceability.
                )
            )

            context_block += (  # Append this source to the text block for the LLM.
                f"\n[Source {len(sources)}]\n"  # Source number used for citations.
                f"Headline: {headline}\n"  # Give LLM title context.
                f"Published: {published_at or 'Unknown'}\n"  # Give LLM date context.
                f"Summary: {summary or 'Not available'}\n"  # Give LLM summary context if available.
                f"Context: {story_context[:MAX_CONTEXT_CHARS_PER_SOURCE]}\n"  # Limit story text to control token usage.
            )

            if len(sources) >= MAX_ANSWER_SOURCES:  # Stop after enough unique sources.
                break  # Prevent too much context/token cost.

        return sources, context_block  # Return both UI source objects and LLM source text.

    finally:  # Always runs after try block.
        db.close()  # Closes DB session so connections are not leaked.


def build_sources(documents: list[Any]) -> list[NewsSource]:  # Convenience helper when only UI sources are needed.
    sources, _ = build_sources_and_context(documents)  # Ignore the context text with "_".
    return sources  # Return source list.


def build_fallback_sources(documents: list[Any]) -> list[NewsSource]:  # Builds sources without Postgres hydration.
    sources = []  # Empty source list.
    for document in documents[:MAX_ANSWER_SOURCES]:  # Use only top documents to avoid noisy UI.
        metadata = document.metadata  # Read metadata from retrieved chunk.
        sources.append(  # Add one fallback source card.
            NewsSource(  # Source model.
                headline=metadata.get("headline", "Untitled"),  # Headline from metadata.
                summary=None,  # No DB summary in fallback mode.
                published_at=metadata.get("published_at"),  # Date from OpenSearch metadata.
                match_snippet=document.page_content,  # Show retrieved chunk as evidence.
                story_id=metadata.get("story_id"),  # Story ID if available.
            )
        )
    return sources  # Return fallback source list.


def answer(state: ChatState):  # Node that generates the final answer.
    sources, context_block = build_sources_and_context(state["documents"])  # Hydrate full sources and build prompt context.
    steps = state.get("steps", []) + [  # Start with previous trace and add hydration step.
        TraceStep(  # UI trace step.
            name="Hydrated sources",  # Step title.
            detail="Fetched full story context from Postgres where available.",  # Explains why answer has fuller context than chunks.
        )
    ]

    try:  # Answer generation can fail due to API/network/schema issues.
        response = client.responses.parse(  # Ask OpenAI for a structured answer.
            model="gpt-4o-mini",  # Cheap model used for final answer.
            input=[  # Messages for the model.
                {  # System instruction.
                    "role": "system",  # High-priority instructions.
                    "content": ANSWER_SYSTEM_PROMPT,  # Grounding/citation rules.
                },
                {  # User message includes question plus trusted sources.
                    "role": "user",  # User role because this is the task payload.
                    "content": (  # Prompt text.
                        f"Question: {state['question']}\n\n"
                        f"Sources:\n{context_block}"
                    ),
                },
            ],
            text_format=SynthesizedAnswer,  # Force model output into {"answer": "..."}.
        )
        message = response.output_parsed.answer  # Extract final answer text.
        steps.append(  # Add answer-generation trace step.
            TraceStep(  # UI trace step.
                name="Generated answer",  # Step title.
                detail=f"Used {len(sources)} unique source(s).",  # Shows source count.
            )
        )

    except Exception as exc:  # If answer LLM fails, return a useful fallback instead of crashing.
        print(f"OpenAI answer generation failed: {exc}")  # Logs the failure.
        citation_list = ", ".join(  # Builds text like "[Source 1], [Source 2]".
            f"[Source {index}]"  # One citation marker.
            for index in range(1, len(sources) + 1)  # One marker per source.
        )
        message = (  # Fallback message shown to user.
            "I found matching news sources, but the answer generator failed. "
            f"Please inspect these sources directly: {citation_list}."
        )
        steps.append(  # Add fallback trace.
            TraceStep(  # UI trace step.
                name="Generated fallback",  # Step title.
                detail=f"Answer LLM failed: {exc}",  # Error detail for debugging.
            )
        )

    return {  # Write final response into graph state.
        "response": ChatResponse(  # API response model.
            type="answer",  # Marks this as a normal answer.
            message=message,  # Final answer text.
            analysis=state["analysis"],  # Include planner output for UI/debugging.
            sources=sources,  # Source cards used by answer.
            steps=steps,  # Full process trace.
        )
    }


def rewrite_query(state: ChatState):  # Node that improves the search query after weak context.
    analysis = require_analysis(state)  # Planner output.

    if state.get("suggested_query"):  # If context judge already gave a better query...
        return {  # Use it directly without another LLM call.
            "current_query": state["suggested_query"],  # Replace search query.
            "context_enough": False,  # Reset context flag before retrying retrieval.
            "steps": state.get("steps", [])  # Existing trace.
            + [  # Add rewrite trace.
                TraceStep(  # UI trace step.
                    name="Rewrote query",  # Step title.
                    detail=f"Used context judge suggestion: {state['suggested_query']}",  # Shows new query.
                )
            ],
        }

    try:  # If no suggestion exists, ask LLM to rewrite the query.
        response = client.responses.parse(  # Structured OpenAI call.
            model="gpt-4o-mini",  # Cheap model used for query rewrite.
            input=[  # Messages to rewriting model.
                {  # System instruction.
                    "role": "system",  # High-priority instruction.
                    "content": REWRITE_SYSTEM_PROMPT,  # Rewrite rules.
                },
                {  # User payload with previous failed query.
                    "role": "user",  # User role for task content.
                    "content": (  # Context for rewriting.
                        f"Original question: {state['question']}\n"
                        f"Previous search query: {state['current_query']}\n"
                        f"Detected entities: {analysis.entities}"
                    ),
                },
            ],
            text_format=QueryRewrite,  # Force output to contain rewritten_query.
        )
        rewritten_query = response.output_parsed.rewritten_query  # Extract rewritten query.

    except Exception as exc:  # If rewrite LLM fails, use a simple fallback query.
        print(f"OpenAI query rewrite failed: {exc}")  # Log rewrite failure.
        rewritten_query = " ".join([state["question"], *analysis.entities]).strip()  # Combine question + entities.

    return {  # Update graph state before retrying retrieval.
        "current_query": rewritten_query,  # New query for next retrieval attempt.
        "context_enough": False,  # Reset context flag.
        "steps": state.get("steps", [])  # Existing trace.
        + [  # Add rewrite trace.
            TraceStep(  # UI trace step.
                name="Rewrote query",  # Step title.
                detail=f"New query: {rewritten_query}",  # Shows rewritten query.
            )
        ],
    }


def limited_answer(state: ChatState):  # Safe terminal node when sources are missing/weak.
    analysis = require_analysis(state)  # Planner output.

    return {  # Writes final limited response into graph state.
        "response": ChatResponse(  # API response model.
            type="limited_answer",  # Tells frontend this is a refusal/limited response.
            message=(  # User-safe explanation.
                "I could not find enough matching news context to answer "
                "confidently in the indexed Bar & Bench stories. Try adding "
                "a person, organization, case name, topic, or time period."
            ),
            analysis=analysis,  # Include planner output.
            sources=[],  # No sources shown because context was not trustworthy.
            steps=state.get("steps", [])  # Existing trace.
            + [TraceStep(name="Stopped search", detail="Reached retrieval attempt limit.")],  # Final trace step.
        )
    }


builder = StateGraph(ChatState)  # Creates a LangGraph workflow whose state follows ChatState.

builder.add_node("plan_query", plan_query)  # Node 1: analyze the user question.
builder.add_node("ask_clarification", ask_clarification)  # Node: ask follow-up if question is vague.
builder.add_node("retrieve", retrieve)  # Node: run hybrid search.
builder.add_node("check_context", check_context)  # Node: judge whether retrieved sources are good enough.
builder.add_node("rewrite_query", rewrite_query)  # Node: improve query after weak retrieval.
builder.add_node("limited_answer", limited_answer)  # Node: safe response when context is insufficient.
builder.add_node("answer", answer)  # Node: generate grounded final answer.

builder.add_edge(START, "plan_query")  # Workflow always starts by planning the query.
builder.add_conditional_edges(  # Conditional edge means a function chooses the next node.
    "plan_query",  # Current node.
    route_after_planning,  # Function that returns "ask_clarification" or "retrieve".
    {  # Mapping from route string to actual node name.
        "ask_clarification": "ask_clarification",  # Vague question route.
        "retrieve": "retrieve",  # Clear question route.
    },
)
builder.add_edge("ask_clarification", END)  # Clarification response ends the workflow.
builder.add_edge("retrieve", "check_context")  # After retrieval, always check context before answering.
builder.add_conditional_edges(  # Choose what happens after context judge.
    "check_context",  # Current node.
    route_after_context,  # Function that returns answer/rewrite/limited_answer.
    {  # Route map.
        "answer": "answer",  # Sources are good enough.
        "rewrite_query": "rewrite_query",  # Sources weak but retry budget remains.
        "limited_answer": "limited_answer",  # Sources weak and retry budget exhausted.
    },
)
builder.add_edge("rewrite_query", "retrieve")  # After rewriting, search again.
builder.add_edge("answer", END)  # Final answer ends workflow.
builder.add_edge("limited_answer", END)  # Limited answer ends workflow.

chat_graph = builder.compile()  # Compiles the graph into a runnable object used by main.py.
