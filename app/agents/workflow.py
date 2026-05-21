import re
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.barandbench_sources import (
    build_barandbench_sources_from_postgres as build_barandbench_sources,
)
from app.agents.planner import analyze_question
from app.agents.prompts import (
    ANSWER_SYSTEM_PROMPT,
    CONTEXT_JUDGE_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    answer_source_prompt,
    query_rewrite_source_prompt,
)
from app.config import settings
from app.news_sources import source_profile
from app.openai_client import (
    get_answer_model,
    get_context_judge_model,
    get_query_rewrite_model,
    make_sync_chat_client,
)
from app.retrieval import retrieve_chunks
from app.source_context import build_combined_source_context, truncate_text
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
MIN_TIMELINE_DATED_SOURCES = settings.min_timeline_dated_sources
MIN_TIMELINE_RELEVANCE_SCORE = settings.min_timeline_relevance_score
MIN_PARTIAL_BRIEFING_RELEVANCE_SCORE = settings.min_partial_briefing_relevance_score
MIN_NEGATIVE_LIST_RELEVANCE_SCORE = settings.min_negative_list_relevance_score
CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]")

client = make_sync_chat_client()


def display_source_name(source: str | None) -> str:
    """
    Return the friendly name shown to users.

    Example:
    "barandbench" becomes "Bar & Bench".
    """
    return source_profile(source).display_name


def process_note_detail(text: str) -> str:
    """
    Keep process-note text short enough for the API response schema.

    Process notes are user-visible status explanations, not full debug logs.
    """
    return truncate_text(text, 700)


def trace_detail(text: str) -> str:
    """
    Keep internal trace details within the schema size limit.

    Trace details help debugging, but they should not become huge prompt dumps.
    """
    return truncate_text(text, 1000)


class ChatState(TypedDict):
    """
    Data passed between LangGraph workflow steps.

    Example:
    plan_query fills "analysis", retrieve fills "chunks" and "sources",
    answer fills "response".
    """
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
    """
    Read the planned query analysis from graph state.

    If analysis is missing, the workflow is in a bad state and should fail early
    instead of producing an ungrounded answer.
    """
    analysis = state["analysis"]

    if analysis is None:
        raise ValueError("Missing query analysis in workflow state.")

    return analysis


def top_unique_story_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Keep only the best chunk from each story.

    Example:
    If one article returned 5 chunks, this keeps the first one for source
    selection so other stories still get a chance.
    """
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
    """
    Build a short debug note about which stories retrieval selected.

    This is useful when checking why the chatbot chose certain sources.
    """
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
    """
    Put chunks from the same article/story together.

    Example:
    "PNE26Y81513-0" and "PNE26Y81513-1" should be grouped under the same
    article before we rebuild context for the answer.
    """
    grouped_chunks: dict[str, list[RetrievedChunk]] = {}

    for chunk in chunks:
        grouped_chunks.setdefault(chunk.story_id or chunk.id, []).append(chunk)

    return grouped_chunks


def build_chunk_based_sources_from_chunks(
    chunks: list[RetrievedChunk],
    source: str,
) -> list[NewsSource]:
    """
    Build answer sources directly from retrieved Pinecone chunks.

    Example:
    Sakal does not hydrate full articles from Postgres here. It uses the matched
    chunk text plus nearby same-article chunks from Pinecone.
    """
    sources = []
    chunks_by_story = group_chunks_by_story(chunks)
    profile = source_profile(source)

    for chunk in top_unique_story_chunks(chunks):
        story_chunks = chunks_by_story.get(chunk.story_id or chunk.id, [chunk])

        sources.append(
            NewsSource(
                source_number=len(sources) + 1,
                article_id=chunk.story_id or chunk.id,
                headline=chunk.headline,
                published_at=chunk.published_at,
                match_snippet=truncate_text(
                    build_combined_source_context(
                        story_chunks,
                        profile.chunk_overlap_words,
                    ),
                    MAX_STORY_EXCERPT_CHARS,
                ),
            )
        )

        if len(sources) >= MAX_ANSWER_SOURCES:
            break

    return sources


def build_barandbench_sources_from_chunks(
    chunks: list[RetrievedChunk],
) -> list[NewsSource]:
    """
    Build Bar & Bench sources using Postgres full-article hydration.

    Pinecone finds the story first. Postgres then provides the fuller story text
    for answer generation.
    """
    return build_barandbench_sources(
        chunks=top_unique_story_chunks(chunks),
        max_sources=MAX_ANSWER_SOURCES,
        max_context_chars=MAX_STORY_EXCERPT_CHARS,
    )


def build_sources_from_chunks(
    chunks: list[RetrievedChunk],
    source: str,
) -> list[NewsSource]:
    """
    Choose how to build answer sources for the selected archive.

    Example:
    Sakal uses chunk-based context.
    Bar & Bench uses Postgres-hydrated story context.
    """
    profile = source_profile(source)

    if profile.hydrate_sources_from_postgres:
        return build_barandbench_sources_from_chunks(chunks)

    return build_chunk_based_sources_from_chunks(chunks, source)


def order_sources_for_intent(
    sources: list[NewsSource],
    analysis: QueryAnalysis,
) -> list[NewsSource]:
    """
    Sort sources oldest-to-newest for timeline questions.

    Example:
    A timeline answer should list April 1 before April 10.
    """
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
    """
    Turn selected sources into the text block given to the model.

    The model only answers from this block, so every source includes its number,
    headline, publish date, and context snippet.
    """
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


DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")


def response_language_for_question(question: str) -> str:
    """
    Decide whether the answer should be English or Marathi.

    Simple rule:
    If the question contains enough Devanagari characters, answer in Marathi.
    Otherwise answer in English.
    """
    devanagari_chars = len(DEVANAGARI_PATTERN.findall(question or ""))

    if devanagari_chars >= 2:
        return "Marathi"

    return "English"


def response_language_instruction(question: str) -> str:
    """
    Build the instruction that locks the answer language.

    Example:
    If the user asked in English but Sakal source text is Marathi, the answer
    should still be in English.
    """
    response_language = response_language_for_question(question)

    if response_language == "Marathi":
        return (
            "Response language instruction: The user asked in Marathi. "
            "Answer in Marathi only. Do not add a separate English translation."
        )

    return (
        "Response language instruction: The user asked in English. "
        "Answer in English only. Do not add a separate Marathi translation."
    )


def answer_system_prompt_for_source(source: str | None, question: str) -> str:
    """
    Combine all answer rules into one system prompt.

    It includes:
    - generic grounding/citation rules,
    - source-specific notes,
    - language instruction based on the user's question.
    """
    prompt = ANSWER_SYSTEM_PROMPT
    source_prompt = answer_source_prompt(source)

    if source_prompt:
        prompt += f"\n\n{source_prompt}"

    prompt += f"\n\n{response_language_instruction(question)}"

    return prompt


def extract_cited_source_numbers(answer_text: str) -> set[int]:
    """
    Find citation numbers written inside the answer text.

    Example:
    "The court granted bail [Source 2]." returns {2}.
    """
    return {
        int(match)
        for match in CITATION_PATTERN.findall(answer_text)
    }


def cited_sources_for_answer(
    answer: SynthesizedAnswer,
    sources: list[NewsSource],
) -> list[NewsSource]:
    """
    Return only sources that the final answer actually cited.

    Example:
    If 6 sources were reviewed but the answer cites Source 1 and Source 3,
    the API response should expose those cited sources first.
    """
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
    """
    Check that the answer cites real sources from the provided context.

    This prevents the model from inventing citations like [Source 99].
    """
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
    """
    Detect questions that ask for "all" or complete coverage.

    Example:
    If the user asks "all stories this month", the answer must say whether the
    retrieved sources may be incomplete.
    """
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
    """
    Detect questions where "none found" can be a valid grounded answer.

    Example:
    "Did any of these cases mention bail?" can be answered as "No matching item
    was found" if the reviewed sources support that.
    """
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
    """
    Build extra instructions for answer generation.

    Example:
    If the retrieval was accepted as only a partial briefing, this note tells the
    answer model not to claim the results are exhaustive.
    """
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
    """
    Pick a short topic label for user-visible process notes.

    Example:
    If entities are ["Pune", "Market Yard"], use those instead of a long query.
    """
    if analysis.entities:
        return ", ".join(analysis.entities[:4])

    return analysis.search_query


def describe_intent(analysis: QueryAnalysis) -> str:
    """
    Convert internal intent into simple wording for process notes.

    Example:
    intent="timeline" becomes "build a timeline".
    """
    if analysis.intent == "timeline":
        return "build a timeline"

    if analysis.intent == "briefing":
        return "prepare a briefing"

    if analysis.intent == "clarify":
        return "clarify the question"

    return "answer the question"


def extract_score_from_steps(steps: list[TraceStep]) -> str | None:
    """
    Pull the context judge score from workflow trace steps.

    Example:
    A trace detail like "Score 8/10..." returns "8/10".
    """
    for step in steps:
        if step.name != "Judged context" or not step.detail:
            continue

        match = re.search(r"Score\s+(\d+/10)", step.detail)
        if match:
            return match.group(1)

    return None


def extract_relevance_scores_from_steps(steps: list[TraceStep]) -> list[int]:
    """
    Collect numeric context judge scores from previous attempts.

    These scores help decide whether a partial briefing or negative-list answer
    is acceptable after retries.
    """
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
    """
    Build the "how I worked" notes returned with the answer.

    Example:
    The user can see that the bot planned a query, searched stories, checked
    source support, and then wrote a cited answer.
    """
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
        profile = source_profile(source)

        if profile.hydrate_sources_from_postgres:
            source_detail = (
                f"I found {len(sources)} usable {profile.display_name} "
                f"{'story' if len(sources) == 1 else 'stories'} and hydrated "
                "the selected story context from Postgres where possible."
            )
        else:
            source_detail = (
                f"I found {len(sources)} usable {profile.display_name} "
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
    """
    First workflow step: understand the user's question.

    Example:
    "What happened next in this case?" may need chat history.
    "Find Pune traffic stories" can be searched directly.
    """
    analysis = analyze_question(
        question=state["question"],
        history=state.get("history", []),
        source=state["source"],
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
    """
    Decide where the workflow goes after planning.

    Possible next steps:
    - stop safely,
    - ask for clarification,
    - retrieve sources.
    """
    analysis = require_analysis(state)

    if analysis.intent == "out_of_scope":
        return "out_of_scope"

    if analysis.clarification_needed:
        return "ask_clarification"

    return "retrieve"


def out_of_scope(state: ChatState):
    """
    Stop when the request is outside the chatbot's job.

    Example:
    The chatbot can summarize news coverage, but it should not provide legal
    advice or reveal hidden system prompts.
    """
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
    """
    Ask the user for more detail when retrieval would be too vague.

    Example:
    "Tell me about that issue" needs a specific story, topic, or prior context.
    """
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
    """
    Search the selected news archive and prepare candidate sources.

    This step:
    - runs hybrid retrieval,
    - reranks chunks,
    - turns chunks into answer-ready sources.
    """
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

    source_mode = source_profile(state["source"]).retrieval_workflow_detail

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
    """
    Check whether the retrieved sources actually answer the question.

    Example:
    A source about "Pune traffic" is not enough if the user asked about a very
    specific Hinjewadi helmet campaign.
    """
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
    """
    Decide what to do after source-quality checking.

    If sources are strong, answer.
    If sources are weak but retries remain, rewrite the query.
    If retries are exhausted, stop with a limited answer.
    """
    if state["context_enough"]:
        return "answer"

    if state["attempts"] < MAX_RETRIEVAL_ATTEMPTS:
        return "rewrite_query"

    return "limited_answer"


def rewrite_query(state: ChatState):
    """
    Create a better search query after weak retrieval results.

    Example:
    If "teacher recruitment" was too broad, the rewrite may add "Pavitra portal"
    or another detected entity.
    """
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
                        "role": "system",
                        "content": query_rewrite_source_prompt(state["source"]),
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
    """
    Generate the final answer using only approved sources.

    The answer must cite source numbers like [Source 1]. If citation validation
    fails, the workflow stops instead of returning an unsupported answer.
    """
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
                    "content": answer_system_prompt_for_source(
                        state["source"],
                        state["question"],
                    ),
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
    """
    Return a safe response when the system cannot answer confidently.

    Example:
    If retrieval fails or the answer has invalid citations, the user gets a
    limited-answer message instead of a guessed answer.
    """
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
    """
    Stop after all retrieval attempts are used.

    This keeps the chatbot grounded: no enough source support means no confident
    answer.
    """
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
