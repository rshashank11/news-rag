import re
import uuid
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agents.planner import analyze_question
from app.agents.prompts import (
    ANSWER_SYSTEM_PROMPT,
    CONTEXT_JUDGE_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
)
from app.openai_client import get_chat_model, make_chat_client
from app.retrieval import retrieve_chunks
from database import SessionLocal
from models import StoryMetaData
from schemas import (
    ChatResponse,
    ChatMessage,
    ContextAssessment,
    NewsSource,
    ProcessNote,
    QueryAnalysis,
    QueryRewrite,
    RetrievedChunk,
    SynthesizedAnswer,
    TraceStep,
)


MAX_RETRIEVAL_ATTEMPTS = 2
MAX_ANSWER_SOURCES = 6
MAX_CONTEXT_CHARS_PER_SOURCE = 6000
MAX_STORY_EXCERPT_CHARS = 5000
MIN_TIMELINE_DATED_SOURCES = 2
MIN_TIMELINE_RELEVANCE_SCORE = 5
MIN_PARTIAL_BRIEFING_RELEVANCE_SCORE = 4
MIN_NEGATIVE_LIST_RELEVANCE_SCORE = 3
CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]")

client = make_chat_client()


class ChatState(TypedDict):
    question: str
    history: list[ChatMessage]
    analysis: QueryAnalysis | None
    current_query: str | None
    chunks: list[RetrievedChunk]
    sources: list[NewsSource]
    attempts: int
    context_enough: bool
    suggested_query: str | None
    steps: list[TraceStep]
    response: ChatResponse | None


def require_analysis(state: ChatState) -> QueryAnalysis:
    analysis = state["analysis"]
    if analysis is None:
        raise ValueError("Missing query analysis in workflow state.")
    return analysis


def format_story_date(story: StoryMetaData) -> str | None:
    if story.published_at is None:
        return None

    return story.published_at.date().isoformat()


def fetch_stories_by_id(story_ids: list[str]) -> dict[str, StoryMetaData]:
    parsed_story_ids = []

    for story_id in story_ids:
        try:
            parsed_story_ids.append(uuid.UUID(story_id))
        except (TypeError, ValueError):
            continue

    if not parsed_story_ids:
        return {}

    db = SessionLocal()

    try:
        stories = (
            db.query(StoryMetaData)
            .filter(StoryMetaData.id.in_(parsed_story_ids))
            .all()
        )
        return {
            str(story.id): story
            for story in stories
        }
    finally:
        db.close()


def build_source_context(chunk: RetrievedChunk, story: StoryMetaData | None) -> str:
    if story is None:
        return chunk.chunk_text

    return "\n\n".join(
        [
            f"Matched paragraph: {chunk.chunk_text}",
            f"Full story excerpt: {story.full_content[:MAX_STORY_EXCERPT_CHARS]}",
        ]
    )


def build_sources_from_chunks(chunks: list[RetrievedChunk]) -> list[NewsSource]:
    sources = []
    seen_story_ids = set()
    story_ids = [
        chunk.story_id
        for chunk in chunks
        if chunk.story_id
    ]
    stories_by_id = fetch_stories_by_id(story_ids)

    for chunk in chunks:
        dedupe_key = chunk.story_id or chunk.id

        if dedupe_key in seen_story_ids:
            continue

        seen_story_ids.add(dedupe_key)
        story = stories_by_id.get(chunk.story_id or "")

        sources.append(
            NewsSource(
                source_number=len(sources) + 1,
                headline=story.headline if story else chunk.headline,
                published_at=format_story_date(story) if story else chunk.published_at,
                match_snippet=build_source_context(chunk, story),
            )
        )

        if len(sources) >= MAX_ANSWER_SOURCES:
            break

    return sources


def order_sources_for_intent(
    sources: list[NewsSource],
    analysis: QueryAnalysis,
) -> list[NewsSource]:
    if analysis.intent != "timeline":
        return sources

    ordered_sources = sorted(
        sources,
        key=lambda source: source.published_at or "9999-99-99",
    )

    return [
        source.model_copy(update={"source_number": index + 1})
        for index, source in enumerate(ordered_sources)
    ]


def build_context_block(sources: list[NewsSource]) -> str:
    context_parts = []

    for source in sources:
        context_parts.append(
            "\n".join(
                [
                    f"[Source {source.source_number}]",
                    f"Headline: {source.headline}",
                    f"Published: {source.published_at or 'Unknown'}",
                    f"Context: {source.match_snippet[:MAX_CONTEXT_CHARS_PER_SOURCE]}",
                ]
            )
        )

    return "\n\n".join(context_parts)


def extract_cited_source_numbers(answer_text: str) -> set[int]:
    return {
        int(match)
        for match in CITATION_PATTERN.findall(answer_text)
    }


def cited_sources_for_answer(
    answer: SynthesizedAnswer,
    sources: list[NewsSource],
) -> list[NewsSource]:
    cited_numbers = (
        set(answer.cited_source_numbers)
        | extract_cited_source_numbers(answer.answer)
    )

    if not cited_numbers:
        return sources

    cited_sources = [
        source
        for source in sources
        if source.source_number in cited_numbers
    ]

    return cited_sources or sources


def answer_has_valid_citations(answer: SynthesizedAnswer, sources: list[NewsSource]) -> bool:
    if answer.unable_to_answer:
        return True

    available_numbers = {
        source.source_number
        for source in sources
    }
    cited_numbers = set(answer.cited_source_numbers)
    inline_numbers = extract_cited_source_numbers(answer.answer)

    if not cited_numbers:
        return False

    if not cited_numbers.issubset(available_numbers):
        return False

    if not inline_numbers:
        return False

    return inline_numbers.issubset(available_numbers)


def question_requests_exhaustive_coverage(question: str) -> bool:
    normalized_question = question.lower()
    exhaustive_terms = [
        "all ",
        "every ",
        "each ",
        "this month",
        "which grounds were most commonly cited",
    ]

    return any(term in normalized_question for term in exhaustive_terms)


def question_allows_negative_list_answer(question: str) -> bool:
    normalized_question = question.lower()
    negative_list_terms = [
        "did any",
        "were any",
        "are any",
        "which of these",
        "of these",
        "list only those",
    ]

    return any(term in normalized_question for term in negative_list_terms)


def requires_employment_arbitration_scope(state: ChatState, analysis: QueryAnalysis) -> bool:
    text = " ".join(
        [
            state["question"],
            analysis.search_query,
            state.get("current_query") or "",
        ]
    ).lower()

    has_arbitration = "arbitration" in text or "arbitrator" in text
    has_employment = "employment" in text or "employee" in text

    return has_arbitration and has_employment


def source_supports_employment_arbitration(source: NewsSource) -> bool:
    text = f"{source.headline} {source.match_snippet}".lower()
    has_arbitration = "arbitration" in text or "arbitrator" in text
    has_unilateral = "unilateral" in text or "unilaterally" in text
    has_employment = (
        "employment contract" in text
        or "employment agreement" in text
    )

    return has_arbitration and has_unilateral and has_employment


def answer_coverage_note(state: ChatState, analysis: QueryAnalysis) -> str:
    notes = []
    normalized_question = state["question"].lower()

    if analysis.from_date or analysis.to_date:
        notes.append(
            "Resolved date window: "
            f"{analysis.from_date or 'earliest indexed date'} to "
            f"{analysis.to_date or 'latest indexed date'}."
        )

    partial_briefing = any(
        step.detail and "Accepted as a partial briefing" in step.detail
        for step in state.get("steps", [])
    )

    if partial_briefing or (
        analysis.intent == "briefing"
        and question_requests_exhaustive_coverage(state["question"])
    ):
        notes.append(
            "Coverage instruction: answer only from the retrieved indexed stories. "
            "If the sources do not prove exhaustive coverage, start by saying the "
            "answer is based on the indexed stories found and cannot confirm every "
            "order in the period."
        )

    if question_allows_negative_list_answer(state["question"]):
        notes.append(
            "List instruction: if the retrieved sources cover the candidate stories "
            "but do not explicitly identify any item matching the requested criterion, "
            "say that no matching item was found in the retrieved indexed stories. "
            "Still cite the source or sources reviewed."
        )

    if requires_employment_arbitration_scope(state, analysis):
        notes.append(
            "Scope instruction: the answer must stay within employment-contract "
            "arbitration sources. General unilateral arbitrator appointment cases "
            "outside employment contracts may be mentioned only as non-answer "
            "background, not as matching cases."
        )

    if "resolution professional" in normalized_question:
        notes.append(
            "Role instruction: list only people explicitly described as a resolution "
            "professional, RP, interim resolution professional, or IRP. If the "
            "sources only list advocates or party representatives, say that no "
            "resolution professional names were explicitly identified and cite the "
            "sources reviewed."
        )

    return "\n".join(notes) or "No additional coverage constraints."


def describe_topic(analysis: QueryAnalysis) -> str:
    if analysis.entities:
        return ", ".join(analysis.entities[:4])

    return analysis.search_query


def describe_intent(analysis: QueryAnalysis) -> str:
    if analysis.intent == "timeline":
        return "build a timeline"

    if analysis.intent == "briefing":
        return "prepare a briefing"

    if analysis.intent == "clarify":
        return "clarify the question"

    return "answer the question"


def extract_score_from_steps(steps: list[TraceStep]) -> str | None:
    for step in steps:
        if step.name != "Judged context" or not step.detail:
            continue

        match = re.search(r"Score\s+(\d+/10)", step.detail)
        if match:
            return match.group(1)

    return None


def extract_relevance_scores_from_steps(steps: list[TraceStep]) -> list[int]:
    scores = []

    for step in steps:
        if step.name != "Judged context" or not step.detail:
            continue

        match = re.search(r"Score\s+(\d+)/10", step.detail)
        if match:
            scores.append(int(match.group(1)))

    return scores


def build_process_notes(
    analysis: QueryAnalysis,
    sources: list[NewsSource],
    steps: list[TraceStep],
    response_type: str,
    reason: str | None = None,
) -> list[ProcessNote]:
    topic = describe_topic(analysis)
    notes = [
        ProcessNote(
            title="Planning the answer",
            detail=(
                f"I treated this as a request to {describe_intent(analysis)} "
                f"using Bar & Bench stories about {topic}."
            ),
        )
    ]

    if analysis.from_date or analysis.to_date:
        notes.append(
            ProcessNote(
                title="Applying the time window",
                detail=(
                    "I limited the search to stories published from "
                    f"{analysis.from_date or 'the earliest indexed story'} to "
                    f"{analysis.to_date or 'the latest indexed date'}."
                ),
            )
        )

    if sources:
        notes.append(
            ProcessNote(
                title="Reviewing news stories",
                detail=(
                    f"I found {len(sources)} usable news "
                    f"{'story' if len(sources) == 1 else 'stories'} "
                    "from the page data and kept them available below."
                ),
                source_numbers=[
                    source.source_number
                    for source in sources[:MAX_ANSWER_SOURCES]
                ],
            )
        )
    else:
        notes.append(
            ProcessNote(
                title="Reviewing news stories",
                detail=(
                    "I could not find enough directly relevant news stories "
                    "to support a confident answer."
                ),
            )
        )

    if any(step.name == "Rewrote query" for step in steps):
        notes.append(
            ProcessNote(
                title="Refining the search",
                detail=(
                    "The first pass was not strong enough, so I tried a more "
                    "focused version of the search before deciding whether to answer."
                ),
            )
        )

    score = extract_score_from_steps(steps)
    if score:
        notes.append(
            ProcessNote(
                title="Checking source support",
                detail=(
                    f"The retrieved stories looked {score} strong for this request, "
                    "so I used them as the basis for the response."
                ),
            )
        )

    if response_type == "answer":
        notes.append(
            ProcessNote(
                title="Preparing the response",
                detail=(
                    "I wrote the answer from the selected stories and kept the "
                    "citations tied to those story sources."
                ),
                source_numbers=[
                    source.source_number
                    for source in sources[:MAX_ANSWER_SOURCES]
                ],
            )
        )
    elif response_type == "out_of_scope":
        notes.append(
            ProcessNote(
                title="Stopping safely",
                detail=reason or "The request was outside the legal-news story data.",
            )
        )
    elif response_type == "clarification_needed":
        notes.append(
            ProcessNote(
                title="Asking for a sharper question",
                detail=(
                    "The request was too broad to ground safely, so I asked for "
                    "a case, court, person, organization, topic, or time period."
                ),
            )
        )
    else:
        notes.append(
            ProcessNote(
                title="Stopping safely",
                detail=(
                    reason
                    or "The indexed news stories did not provide enough support for a confident answer."
                ),
            )
        )

    return notes


def plan_query(state: ChatState):
    analysis = analyze_question(
        question=state["question"],
        history=state.get("history", []),
    )

    return {
        "analysis": analysis,
        "current_query": analysis.search_query,
        "chunks": [],
        "sources": [],
        "attempts": 0,
        "context_enough": False,
        "suggested_query": None,
        "steps": [
            TraceStep(
                name="Parsed query",
                detail=(
                    f"Intent: {analysis.intent}; "
                    f"search query: {analysis.search_query}"
                ),
            )
        ],
    }


def route_after_planning(
    state: ChatState,
) -> Literal["out_of_scope", "ask_clarification", "retrieve"]:
    analysis = require_analysis(state)

    if analysis.intent == "out_of_scope":
        return "out_of_scope"

    if analysis.clarification_needed:
        return "ask_clarification"

    return "retrieve"


def out_of_scope(state: ChatState):
    analysis = require_analysis(state)
    message = analysis.refusal_reason or (
        "This assistant can only answer questions about indexed legal-news stories."
    )
    steps = state.get("steps", []) + [TraceStep(name="Stopped request", detail=message)]

    return {
        "response": ChatResponse(
            type="out_of_scope",
            message=message,
            sources=[],
            process_notes=build_process_notes(
                analysis=analysis,
                sources=[],
                steps=steps,
                response_type="out_of_scope",
                reason=message,
            ),
        )
    }


def ask_clarification(state: ChatState):
    analysis = require_analysis(state)
    message = analysis.clarification_question or (
        "Could you add a person, case, court, organization, topic, or time period?"
    )
    steps = state.get("steps", []) + [TraceStep(name="Asked clarification", detail=message)]

    return {
        "response": ChatResponse(
            type="clarification_needed",
            message=message,
            sources=[],
            process_notes=build_process_notes(
                analysis=analysis,
                sources=[],
                steps=steps,
                response_type="clarification_needed",
                reason=message,
            ),
        )
    }


def retrieve(state: ChatState):
    analysis = require_analysis(state)
    query = state["current_query"] or analysis.search_query

    chunks = retrieve_chunks(
        query=query,
        top_k=analysis.k,
        from_date=analysis.from_date,
        to_date=analysis.to_date,
    )
    sources = order_sources_for_intent(
        build_sources_from_chunks(chunks),
        analysis,
    )

    date_detail = ""
    if analysis.from_date or analysis.to_date:
        date_detail = f" Date filter: {analysis.from_date or 'any'} to {analysis.to_date or 'any'}."

    return {
        "chunks": chunks,
        "sources": sources,
        "attempts": state["attempts"] + 1,
        "steps": state.get("steps", [])
        + [
            TraceStep(
                name="Retrieved sources",
                detail=(
                    f"Attempt {state['attempts'] + 1}: hybrid search returned "
                    f"{len(chunks)} chunk(s) and {len(sources)} source(s).{date_detail}"
                ),
            )
        ],
    }


def check_context(state: ChatState):
    sources = state["sources"]
    analysis = require_analysis(state)

    if not sources:
        return {
            "context_enough": False,
            "suggested_query": None,
            "steps": state.get("steps", [])
            + [TraceStep(name="Judged context", detail="No sources were retrieved.")],
        }

    context_block = build_context_block(sources)

    try:
        response = client.responses.parse(
            model=get_chat_model(),
            input=[
                {
                    "role": "system",
                    "content": CONTEXT_JUDGE_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['question']}\n"
                        f"Planned search query: {analysis.search_query}\n"
                        f"Current search query: {state['current_query']}\n"
                        f"Detected entities: {analysis.entities}\n\n"
                        f"Retrieved sources:\n{context_block}"
                    ),
                },
            ],
            text_format=ContextAssessment,
        )
        assessment = response.output_parsed
        dated_source_count = sum(
            1
            for source in sources
            if source.published_at
        )
        missing_required_scope = (
            requires_employment_arbitration_scope(state, analysis)
            and not any(source_supports_employment_arbitration(source) for source in sources)
        )
        timeline_supported = (
            analysis.intent == "timeline"
            and not missing_required_scope
            and assessment.relevance_score >= MIN_TIMELINE_RELEVANCE_SCORE
            and dated_source_count >= MIN_TIMELINE_DATED_SOURCES
        )
        partial_briefing_supported = (
            analysis.intent == "briefing"
            and not missing_required_scope
            and state["attempts"] >= MAX_RETRIEVAL_ATTEMPTS
            and max(
                [
                    assessment.relevance_score,
                    *extract_relevance_scores_from_steps(state.get("steps", [])),
                ]
            )
            >= MIN_PARTIAL_BRIEFING_RELEVANCE_SCORE
        )
        negative_list_supported = (
            analysis.intent == "answer"
            and not missing_required_scope
            and state["attempts"] >= MAX_RETRIEVAL_ATTEMPTS
            and question_allows_negative_list_answer(state["question"])
            and max(
                [
                    assessment.relevance_score,
                    *extract_relevance_scores_from_steps(state.get("steps", [])),
                ]
            )
            >= MIN_NEGATIVE_LIST_RELEVANCE_SCORE
        )
        context_enough = (
            (assessment.context_enough and not missing_required_scope)
            or timeline_supported
            or partial_briefing_supported
            or negative_list_supported
        )
        scope_note = ""
        if missing_required_scope:
            scope_note = (
                " Rejected because the retrieved sources do not satisfy the required "
                "employment-contract arbitration setting."
            )
        timeline_note = ""
        if timeline_supported and not assessment.context_enough:
            timeline_note = (
                " Accepted because timeline requests can be answered from "
                "multiple directly relevant dated stories."
            )
        partial_briefing_note = ""
        if partial_briefing_supported and not assessment.context_enough:
            partial_briefing_note = (
                " Accepted as a partial briefing because the sources are directly "
                "relevant, but the answer must avoid claiming exhaustive coverage."
            )
        negative_list_note = ""
        if negative_list_supported and not assessment.context_enough:
            negative_list_note = (
                " Accepted for a negative list answer because the sources cover the "
                "candidate stories but do not show the requested criterion."
            )
        return {
            "context_enough": context_enough,
            "suggested_query": assessment.suggested_query,
            "steps": state.get("steps", [])
            + [
                TraceStep(
                    name="Judged context",
                    detail=(
                        f"Score {assessment.relevance_score}/10. "
                        f"{assessment.reason}{scope_note}{timeline_note}"
                        f"{partial_briefing_note}{negative_list_note}"
                    ),
                )
            ],
        }

    except Exception as exc:
        return {
            "context_enough": False,
            "suggested_query": None,
            "steps": state.get("steps", [])
            + [
                TraceStep(
                    name="Judged context",
                    detail=(
                        f"Context judge failed: {exc}. "
                        "Marked context as insufficient."
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

    return "limited_answer"


def rewrite_query(state: ChatState):
    analysis = require_analysis(state)

    if state.get("suggested_query"):
        rewritten_query = state["suggested_query"]
        reason = "Used context judge suggestion."
    else:
        try:
            response = client.responses.parse(
                model=get_chat_model(),
                input=[
                    {
                        "role": "system",
                        "content": QUERY_REWRITE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Original question: {state['question']}\n"
                            f"Previous search query: {state['current_query']}\n"
                            f"Detected entities: {analysis.entities}\n"
                        ),
                    },
                ],
                text_format=QueryRewrite,
            )
            rewrite = response.output_parsed
            rewritten_query = rewrite.rewritten_query
            reason = rewrite.reason or "LLM rewrote the query."

        except Exception as exc:
            rewritten_query = " ".join([state["question"], *analysis.entities]).strip()
            reason = f"Rewrite failed: {exc}. Used question plus detected entities."

    return {
        "current_query": rewritten_query,
        "context_enough": False,
        "suggested_query": None,
        "steps": state.get("steps", [])
        + [TraceStep(name="Rewrote query", detail=f"{reason} Query: {rewritten_query}")],
    }


def answer(state: ChatState):
    analysis = require_analysis(state)
    sources = state["sources"]
    context_block = build_context_block(sources)
    coverage_note = answer_coverage_note(state, analysis)

    try:
        response = client.responses.parse(
            model=get_chat_model(),
            input=[
                {
                    "role": "system",
                    "content": ANSWER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['question']}\n"
                        f"{coverage_note}\n\n"
                        f"Approved sources:\n{context_block}"
                    ),
                },
            ],
            text_format=SynthesizedAnswer,
        )
        synthesized_answer = response.output_parsed

        if not answer_has_valid_citations(synthesized_answer, sources):
            return limited_answer_with_reason(
                state,
                "The generated answer did not pass citation validation.",
            )
        cited_sources = cited_sources_for_answer(synthesized_answer, sources)
        steps = state.get("steps", []) + [
            TraceStep(
                name="Generated answer",
                detail=(
                    f"Used {len(cited_sources)} source(s). "
                    f"Confidence: {synthesized_answer.confidence}."
                ),
            )
        ]

        return {
            "response": ChatResponse(
                type="answer",
                message=synthesized_answer.answer,
                sources=cited_sources,
                process_notes=build_process_notes(
                    analysis=analysis,
                    sources=cited_sources,
                    steps=steps,
                    response_type="answer",
                ),
            )
        }

    except Exception as exc:
        return limited_answer_with_reason(
            state,
            f"Answer generation failed: {exc}",
        )


def limited_answer_with_reason(state: ChatState, reason: str):
    analysis = require_analysis(state)
    steps = state.get("steps", []) + [TraceStep(name="Stopped safely", detail=reason)]

    return {
        "response": ChatResponse(
            type="limited_answer",
            message=(
                "I could not find enough directly supported indexed news context "
                "to answer confidently. Try adding a case name, court, person, "
                "organization, topic, or time period."
            ),
            sources=[],
            process_notes=build_process_notes(
                analysis=analysis,
                sources=state.get("sources", []),
                steps=steps,
                response_type="limited_answer",
                reason=reason,
            ),
        )
    }


def limited_answer(state: ChatState):
    return limited_answer_with_reason(
        state,
        "Reached retrieval attempt limit without sufficient context.",
    )


builder = StateGraph(ChatState)

builder.add_node("plan_query", plan_query)
builder.add_node("out_of_scope", out_of_scope)
builder.add_node("ask_clarification", ask_clarification)
builder.add_node("retrieve", retrieve)
builder.add_node("check_context", check_context)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("answer", answer)
builder.add_node("limited_answer", limited_answer)

builder.add_edge(START, "plan_query")
builder.add_conditional_edges(
    "plan_query",
    route_after_planning,
    {
        "out_of_scope": "out_of_scope",
        "ask_clarification": "ask_clarification",
        "retrieve": "retrieve",
    },
)
builder.add_edge("out_of_scope", END)
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
