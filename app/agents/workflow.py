import re
import uuid
from datetime import datetime
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agents.planner import analyze_question
from app.agents.prompts import (
    ANSWER_SYSTEM_PROMPT,
    CONTEXT_JUDGE_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
)
from app.config import settings
from app.openai_client import (
    get_answer_model,
    get_context_judge_model,
    get_query_rewrite_model,
    make_sync_chat_client,
)
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


MAX_RETRIEVAL_ATTEMPTS = settings.max_retrieval_attempts
MAX_ANSWER_SOURCES = settings.max_answer_sources
MAX_CONTEXT_CHARS_PER_SOURCE = settings.max_context_chars_per_source
MAX_STORY_EXCERPT_CHARS = settings.max_story_excerpt_chars
SAKAL_CHUNK_OVERLAP_WORDS = 50
MIN_TIMELINE_DATED_SOURCES = settings.min_timeline_dated_sources
MIN_TIMELINE_RELEVANCE_SCORE = settings.min_timeline_relevance_score
MIN_PARTIAL_BRIEFING_RELEVANCE_SCORE = settings.min_partial_briefing_relevance_score
MIN_NEGATIVE_LIST_RELEVANCE_SCORE = settings.min_negative_list_relevance_score
CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]")

client = make_sync_chat_client()


def display_source_name(source: str | None) -> str:
    if source == "barandbench":
        return "Bar & Bench"

    if source == "sakal":
        return "Sakal"

    return "selected"


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    suffix = " ... [truncated]"
    return f"{text[:max_chars - len(suffix)].rstrip()}{suffix}"


def process_note_detail(text: str) -> str:
    return truncate_text(text, 700)


def trace_detail(text: str) -> str:
    return truncate_text(text, 1000)


class ChatState(TypedDict):
    question: str
    source: str
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


def story_id_to_uuid(story_id: str | None) -> uuid.UUID | None:
    if not story_id:
        return None

    try:
        return uuid.UUID(str(story_id))
    except (TypeError, ValueError):
        return None


def datetime_to_iso_date(value) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        return value[:10] if value else None

    if isinstance(value, datetime):
        return value.date().isoformat()

    if hasattr(value, "date"):
        return value.date().isoformat()

    return None


def list_to_line(label: str, values: list[str] | None) -> str | None:
    if not values:
        return None

    cleaned_values = [
        str(value).strip()
        for value in values
        if str(value).strip()
    ]

    if not cleaned_values:
        return None

    return f"{label}: {', '.join(cleaned_values)}"


def merge_consecutive_chunk_texts(chunks: list[RetrievedChunk]) -> str:
    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: chunk.chunk_index if chunk.chunk_index is not None else 0,
    )

    merged_words = []
    previous_chunk_index = None

    for chunk in ordered_chunks:
        chunk_words = chunk.chunk_text.split()

        if (
            previous_chunk_index is not None
            and chunk.chunk_index == previous_chunk_index + 1
        ):
            chunk_words = chunk_words[SAKAL_CHUNK_OVERLAP_WORDS:]

        merged_words.extend(chunk_words)
        previous_chunk_index = chunk.chunk_index

    return " ".join(merged_words)


def build_combined_source_context(story_chunks: list[RetrievedChunk]) -> str:
    best_chunk = story_chunks[0]
    metadata_lines = []

    topics_line = list_to_line("Topics", best_chunk.topics)
    categories_line = list_to_line("Categories", best_chunk.categories)

    if topics_line:
        metadata_lines.append(topics_line)

    if categories_line:
        metadata_lines.append(categories_line)

    context_parts = []

    if metadata_lines:
        context_parts.append("\n".join(metadata_lines))

    context_parts.append(
        f"{merge_consecutive_chunk_texts(story_chunks)}"
    )

    return "\n\n".join(context_parts)


def build_full_article_context_from_story(
    story: StoryMetaData,
    fallback_chunk: RetrievedChunk,
) -> str:
    metadata_lines = []

    topics_line = list_to_line("Topics", story.topics)
    categories_line = list_to_line("Categories", story.categories)

    if topics_line:
        metadata_lines.append(topics_line)

    if categories_line:
        metadata_lines.append(categories_line)

    if story.summary:
        metadata_lines.append(f"Summary: {story.summary}")

    context_parts = []

    if metadata_lines:
        context_parts.append("\n".join(metadata_lines))

    if story.full_content:
        context_parts.append(f"Full article context: {story.full_content}")

    context = "\n\n".join(context_parts).strip()

    if not context:
        return build_combined_source_context([fallback_chunk])

    return truncate_text(context, MAX_STORY_EXCERPT_CHARS)


def top_unique_story_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen_story_ids = set()
    unique_story_chunks = []

    for chunk in chunks:
        dedupe_key = chunk.story_id or chunk.id

        if dedupe_key in seen_story_ids:
            continue

        seen_story_ids.add(dedupe_key)
        unique_story_chunks.append(chunk)

    return unique_story_chunks


def source_selection_detail(chunks: list[RetrievedChunk]) -> str:
    unique_story_chunks = top_unique_story_chunks(chunks)

    if not unique_story_chunks:
        return ""

    debug_chunks = unique_story_chunks[:settings.rerank_debug_story_count]
    top_story_text = "; ".join(
        (
            f"{chunk.headline}"
            f" (rerank={chunk.rerank_score:.2f})"
            if chunk.rerank_score is not None
            else chunk.headline
        )
        for chunk in debug_chunks
    )

    detail = f" Top reranked stories: {top_story_text}."

    if len(unique_story_chunks) > MAX_ANSWER_SOURCES:
        dropped_chunk = unique_story_chunks[MAX_ANSWER_SOURCES]
        dropped_score = (
            f" rerank={dropped_chunk.rerank_score:.2f}"
            if dropped_chunk.rerank_score is not None
            else ""
        )
        detail += (
            " First story after the source cap: "
            f"{dropped_chunk.headline}{dropped_score}."
        )

    return detail[:450]


def group_chunks_by_story(chunks: list[RetrievedChunk]) -> dict[str, list[RetrievedChunk]]:
    grouped_chunks: dict[str, list[RetrievedChunk]] = {}

    for chunk in chunks:
        grouped_chunks.setdefault(chunk.story_id or chunk.id, []).append(chunk)

    return grouped_chunks


def build_chunk_based_sources_from_chunks(chunks: list[RetrievedChunk]) -> list[NewsSource]:
    sources = []
    chunks_by_story = group_chunks_by_story(chunks)

    for chunk in top_unique_story_chunks(chunks):
        story_chunks = chunks_by_story.get(chunk.story_id or chunk.id, [chunk])

        sources.append(
            NewsSource(
                source_number=len(sources) + 1,
                article_id=chunk.story_id or chunk.id,
                headline=chunk.headline,
                published_at=chunk.published_at,
                match_snippet=truncate_text(
                    build_combined_source_context(story_chunks),
                    MAX_STORY_EXCERPT_CHARS,
                ),
            )
        )

        if len(sources) >= MAX_ANSWER_SOURCES:
            break

    return sources


def build_barandbench_sources_from_postgres(
    chunks: list[RetrievedChunk],
) -> list[NewsSource]:
    sources = []
    db = SessionLocal()

    try:
        for chunk in top_unique_story_chunks(chunks):
            story_uuid = story_id_to_uuid(chunk.story_id)
            story = db.get(StoryMetaData, story_uuid) if story_uuid else None

            if story:
                headline = story.headline or chunk.headline
                published_at = datetime_to_iso_date(story.published_at) or chunk.published_at
                match_snippet = build_full_article_context_from_story(story, chunk)
            else:
                headline = chunk.headline
                published_at = chunk.published_at
                match_snippet = build_combined_source_context([chunk])

            sources.append(
                NewsSource(
                    source_number=len(sources) + 1,
                    article_id=chunk.story_id or chunk.id,
                    headline=headline,
                    published_at=published_at,
                    match_snippet=truncate_text(match_snippet, MAX_STORY_EXCERPT_CHARS),
                )
            )

            if len(sources) >= MAX_ANSWER_SOURCES:
                break

        return sources

    finally:
        db.close()


def build_sources_from_chunks(
    chunks: list[RetrievedChunk],
    source: str,
) -> list[NewsSource]:
    if source == "barandbench":
        return build_barandbench_sources_from_postgres(chunks)

    return build_chunk_based_sources_from_chunks(chunks)


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


def answer_system_prompt_for_source(source: str | None) -> str:
    prompt = ANSWER_SYSTEM_PROMPT

    if source == "sakal":
        prompt += (
            "\n\nWhen answering questions about Sakal news content, provide the "
            "response in both English and Marathi. Give the user the key answer "
            "in English, then repeat or summarize the same answer in Marathi. "
            "Keep your response grounded in the approved sources."
        )

    return prompt


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


def answer_has_valid_citations(
    answer: SynthesizedAnswer,
    sources: list[NewsSource],
) -> bool:
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


def answer_coverage_note(state: ChatState, analysis: QueryAnalysis) -> str:
    notes = []

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
            "item in the period."
        )

    if question_allows_negative_list_answer(state["question"]):
        notes.append(
            "List instruction: if the retrieved sources cover the candidate stories "
            "but do not explicitly identify any item matching the requested criterion, "
            "say that no matching item was found in the retrieved indexed stories. "
            "Still cite the source or sources reviewed."
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
    source: str,
    reason: str | None = None,
) -> list[ProcessNote]:
    topic = describe_topic(analysis)
    source_name = display_source_name(source)

    notes = [
        ProcessNote(
            title="Planning the answer",
            detail=(
                f"I treated this as a request to {describe_intent(analysis)} "
                f"using {source_name} news stories about {topic}."
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
        if source == "barandbench":
            source_detail = (
                f"I found {len(sources)} usable Bar & Bench "
                f"{'story' if len(sources) == 1 else 'stories'} and hydrated "
                "the selected story context from Postgres where possible."
            )
        else:
            source_detail = (
                f"I found {len(sources)} usable Sakal "
                f"{'story' if len(sources) == 1 else 'stories'} from matched chunks "
                "and merged chunks belonging to the same article where available."
            )

        notes.append(
            ProcessNote(
                title="Reviewing news stories",
                detail=source_detail,
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
                detail=process_note_detail(
                    reason or f"The request was outside the {source_name} news story data."
                ),
            )
        )

    elif response_type == "clarification_needed":
        notes.append(
            ProcessNote(
                title="Asking for a sharper question",
                detail=(
                    "The request was too broad to ground safely, so I asked for "
                    "a story, person, organization, location, topic, or time period."
                ),
            )
        )

    else:
        notes.append(
            ProcessNote(
                title="Stopping safely",
                detail=process_note_detail(
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
                detail=trace_detail(
                    f"Intent: {analysis.intent}; "
                    f"history: {'used' if analysis.uses_history else 'not used'}; "
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
        f"This assistant can only answer questions about indexed "
        f"{display_source_name(state['source'])} news stories."
    )

    steps = state.get("steps", []) + [
        TraceStep(name="Stopped request", detail=message)
    ]

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
                source=state["source"],
                reason=message,
            ),
        )
    }


def ask_clarification(state: ChatState):
    analysis = require_analysis(state)
    message = analysis.clarification_question or (
        "Could you add a person, case, court, organization, topic, or time period?"
    )

    steps = state.get("steps", []) + [
        TraceStep(name="Asked clarification", detail=message)
    ]

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
                source=state["source"],
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
        source=state["source"],
    )

    sources = order_sources_for_intent(
        build_sources_from_chunks(chunks, state["source"]),
        analysis,
    )

    date_detail = ""
    if analysis.from_date or analysis.to_date:
        date_detail = (
            f" Date filter: {analysis.from_date or 'any'} "
            f"to {analysis.to_date or 'any'}."
        )

    selection_detail = source_selection_detail(chunks)

    source_mode = (
        "Bar & Bench full-article Postgres hydration"
        if state["source"] == "barandbench"
        else "Sakal same-story chunk merge"
    )

    return {
        "chunks": chunks,
        "sources": sources,
        "attempts": state["attempts"] + 1,
        "steps": state.get("steps", [])
        + [
            TraceStep(
                name="Retrieved sources",
                detail=(
                    f"Attempt {state['attempts'] + 1}: reranked candidate pool up to "
                    f"{settings.rerank_candidate_top_k} Pinecone chunk(s), selected "
                    f"{len(chunks)} chunk(s) and {len(sources)} source(s). "
                    f"Source workflow: {source_mode}."
                    f"{date_detail}{selection_detail}"
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
            model=get_context_judge_model(),
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

        timeline_supported = (
            analysis.intent == "timeline"
            and assessment.relevance_score >= MIN_TIMELINE_RELEVANCE_SCORE
            and dated_source_count >= MIN_TIMELINE_DATED_SOURCES
        )

        partial_briefing_supported = (
            analysis.intent == "briefing"
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
            assessment.context_enough
            or timeline_supported
            or partial_briefing_supported
            or negative_list_supported
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
                    detail=trace_detail(
                        f"Score {assessment.relevance_score}/10. "
                        f"{assessment.reason}{timeline_note}"
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
                    detail=trace_detail(
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
                model=get_query_rewrite_model(),
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
            reason = trace_detail(
                f"Rewrite failed: {exc}. Used question plus detected entities."
            )

    return {
        "current_query": rewritten_query,
        "context_enough": False,
        "suggested_query": None,
        "steps": state.get("steps", [])
        + [
            TraceStep(
                name="Rewrote query",
                detail=trace_detail(f"{reason} Query: {rewritten_query}"),
            )
        ],
    }


def answer(state: ChatState):
    analysis = require_analysis(state)
    sources = state["sources"]
    context_block = build_context_block(sources)
    coverage_note = answer_coverage_note(state, analysis)

    try:
        response = client.responses.parse(
            model=get_answer_model(),
            input=[
                {
                    "role": "system",
                    "content": answer_system_prompt_for_source(state["source"]),
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
                    source=state["source"],
                ),
            )
        }

    except Exception as exc:
        return limited_answer_with_reason(
            state,
            trace_detail(f"Answer generation failed: {exc}"),
        )


def limited_answer_with_reason(state: ChatState, reason: str):
    analysis = require_analysis(state)
    safe_reason = trace_detail(reason)

    steps = state.get("steps", []) + [
        TraceStep(name="Stopped safely", detail=safe_reason)
    ]

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
                source=state["source"],
                reason=safe_reason,
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
