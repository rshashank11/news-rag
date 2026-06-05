import openai
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from app.agents.prompts import PLANNER_SYSTEM_PROMPT, planner_source_prompt
from app.config import settings
from app.openai_client import get_planner_model, make_sync_chat_client
from schemas import ChatMessage, QueryAnalysis, clean_text


client = make_sync_chat_client()

APP_TIMEZONE = ZoneInfo("Asia/Kolkata")
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
PLANNER_FILLER_TOKENS = {
    "about",
    "and",
    "article",
    "articles",
    "bench",
    "bar",
    "coverage",
    "find",
    "from",
    "give",
    "in",
    "involving",
    "me",
    "news",
    "of",
    "on",
    "report",
    "reports",
    "sakal",
    "show",
    "stories",
    "the",
    "what",
    "which",
}


def clarification_analysis(question: str, message: str) -> QueryAnalysis:
    """
    Build a planner result when the question is too vague to search.

    Example:
    If the user asks "tell me about this", but there is no previous context,
    the chatbot should ask for a person, topic, case, or time period.
    """
    return QueryAnalysis(
        intent="clarify",
        search_query=clean_text(question)[:300] or "clarification needed",
        entities=[],
        k=settings.default_query_k,
        clarification_needed=True,
        clarification_question=message,
        from_date=None,
        to_date=None,
        refusal_reason=None,
    )


def out_of_scope_analysis(question: str, message: str) -> QueryAnalysis:
    """
    Build a planner result when the request should not be answered.

    Example:
    A request for legal advice or hidden system prompts is outside the news
    archive search boundary.
    """
    return QueryAnalysis(
        intent="out_of_scope",
        search_query=clean_text(question)[:300] or "out of scope",
        entities=[],
        k=settings.default_query_k,
        clarification_needed=False,
        clarification_question=None,
        from_date=None,
        to_date=None,
        refusal_reason=message,
    )


def truncate_for_planner(text: str, max_chars: int) -> str:
    """
    Shorten old chat messages before sending them to the planner.

    The planner only needs enough history to understand follow-ups like
    "what happened next?".
    """
    cleaned = clean_text(text)

    if len(cleaned) <= max_chars:
        return cleaned

    suffix = " ... [truncated]"
    if max_chars <= len(suffix):
        return cleaned[:max_chars]

    return f"{cleaned[:max_chars - len(suffix)].rstrip()}{suffix}"


def format_history_for_planner(history: list[ChatMessage]) -> str:
    """
    Format recent chat history for the planner.

    Important:
    History is only for resolving references like "that case".
    It is not treated as factual evidence for the answer.
    """
    if not history:
        return "No recent conversation."

    lines: list[str] = []
    total_chars = 0

    for message in reversed(history[-settings.max_history_messages:]):
        prefix = f"{message.role}: "
        remaining_chars = settings.max_history_chars - total_chars - len(prefix)

        if remaining_chars <= 0:
            break

        max_content_chars = min(settings.max_history_message_chars, remaining_chars)
        content = truncate_for_planner(message.content, max_content_chars)
        line = f"{prefix}{content}"
        lines.append(line)
        total_chars += len(line)

    return "\n".join(reversed(lines)) or "No recent conversation."


def truncate_query(value: str, max_chars: int = 300) -> str:
    """
    Keep a planned search query under the schema length limit.

    Example:
    If the model returns a very long query, this trims it without cutting the
    final word in half when possible.
    """
    cleaned = clean_text(value)

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[:max_chars].rsplit(" ", 1)[0].strip() or cleaned[:max_chars]


def normalize_analysis_search_query(
    question: str,
    analysis: QueryAnalysis,
) -> QueryAnalysis:
    """
    Repair weak planner search queries before retrieval.

    Example:
    If the user asks "Which vegetables became costlier in Pune Market Yard?"
    but the planner only returns "vegetables", we use the fuller question because
    "vegetables" alone is too broad.
    """
    if analysis.intent in {"clarify", "out_of_scope"}:
        return analysis

    question_clean = clean_text(question)
    search_query_clean = clean_text(analysis.search_query)

    question_tokens = set(TOKEN_PATTERN.findall(question_clean.lower()))
    search_tokens = set(TOKEN_PATTERN.findall(search_query_clean.lower()))

    normalized_query = search_query_clean

    if len(search_tokens) < 2 and len(question_tokens) >= 3:
        normalized_query = question_clean
    elif search_tokens and search_tokens.issubset(question_tokens):
        meaningful_missing_tokens = (
            question_tokens
            - search_tokens
            - PLANNER_FILLER_TOKENS
        )

        if meaningful_missing_tokens:
            normalized_query = question_clean

    return analysis.model_copy(
        update={
            "search_query": truncate_query(normalized_query),
        }
    )


def analyze_question(
    question: str,
    history: list[ChatMessage] | None = None,
    source: str | None = None,
) -> QueryAnalysis:
    """
    Turn the user's message into a structured search plan.

    The planner decides:
    - what to search for,
    - whether history is needed,
    - whether this is an answer, briefing, timeline, clarification, or refusal.
    """
    cleaned_question = clean_text(question)
    history = history or []
    selected_source = settings.news_source_config(source)["source"]

    # Only bail out early for very short inputs when there is no prior context to
    # resolve the meaning from. With history, a single word like "yes", "crime",
    # or "opinion" can be a meaningful continuation and must go through the LLM.
    if (not cleaned_question or len(cleaned_question.split()) < 2) and not history:
        return clarification_analysis(
            cleaned_question,
            "Could you add a person, case, court, organization, topic, or time period?",
        )

    try:
        today = datetime.now(APP_TIMEZONE).date()
        current_date = today.isoformat()
        response = client.responses.parse(
            model=get_planner_model(),
            timeout=60,
            temperature=0,
            input=[
                {
                    "role": "system",
                    "content": PLANNER_SYSTEM_PROMPT,
                },
                {
                    "role": "system",
                    "content": (
                        "Runtime date context: "
                        f"today is {current_date} in Asia/Kolkata. "
                        "Resolve relative date phrases using this date."
                    ),
                },
                {
                    "role": "system",
                    "content": (
                        f"Selected news source: {selected_source}. "
                        "Build search_query for this source's indexed archive."
                    ),
                },
                {
                    "role": "system",
                    "content": planner_source_prompt(selected_source),
                },
                {
                    "role": "system",
                    "content": (
                        "Recent conversation is provided only to resolve follow-up "
                        "references like 'that case', 'the court', 'him', or 'the FIR'. "
                        "Do not treat conversation history as evidence. Evidence must "
                        "come from retrieved archive sources later."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Recent conversation:\n{format_history_for_planner(history)}\n\n"
                        f"Current question: {cleaned_question}"
                    ),
                },
            ],
            text_format=QueryAnalysis,
        )
        analysis = response.output_parsed

        if analysis.clarification_needed:
            return analysis.model_copy(update={"intent": "clarify"})

        return normalize_analysis_search_query(cleaned_question, analysis)

    except openai.BadRequestError as exc:
        return out_of_scope_analysis(
            cleaned_question,
            (
                "I cannot help with requests to bypass instructions, ignore "
                "source grounding, or override the chatbot's guardrails."
            ),
        )
    except openai.APIError as exc:
        return clarification_analysis(
            cleaned_question,
            (
                "I am having trouble reaching the AI services right now. Please try again in a moment."
            ),
        )
    except Exception as exc:
        return clarification_analysis(
            cleaned_question,
            (
                "I could not plan that search safely. Please mention the case, court, "
                "person, organization, topic, or date range you want me to search."
            ),
        )
