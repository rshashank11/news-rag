import openai
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from app.agents.prompts import PLANNER_SYSTEM_PROMPT
from app.config import settings
from app.openai_client import get_planner_model, make_sync_chat_client
from schemas import ChatMessage, QueryAnalysis, clean_text


client = make_sync_chat_client()

APP_TIMEZONE = ZoneInfo("Asia/Kolkata")
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

PRICE_INTENT_TERMS = {
    "price",
    "prices",
    "rate",
    "rates",
    "costlier",
    "cheaper",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "stable",
}

def clarification_analysis(question: str, message: str) -> QueryAnalysis:
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
    cleaned = clean_text(text)

    if len(cleaned) <= max_chars:
        return cleaned

    suffix = " ... [truncated]"
    if max_chars <= len(suffix):
        return cleaned[:max_chars]

    return f"{cleaned[:max_chars - len(suffix)].rstrip()}{suffix}"


def format_history_for_planner(history: list[ChatMessage]) -> str:
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
    cleaned = clean_text(value)

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[:max_chars].rsplit(" ", 1)[0].strip() or cleaned[:max_chars]


def normalize_analysis_search_query(
    question: str,
    analysis: QueryAnalysis,
) -> QueryAnalysis:
    if analysis.intent in {"clarify", "out_of_scope"}:
        return analysis

    question_clean = clean_text(question)
    search_query_clean = clean_text(analysis.search_query)

    question_tokens = set(TOKEN_PATTERN.findall(question_clean.lower()))
    search_tokens = set(TOKEN_PATTERN.findall(search_query_clean.lower()))

    required_terms: list[str] = []

    for term in sorted(PRICE_INTENT_TERMS):
        if term in question_tokens and term not in search_tokens:
            required_terms.append(term)

    if (
        {"market", "yard"}.issubset(question_tokens)
        and ({"vegetable", "vegetables"} & question_tokens)
        and not ({"vegetable", "vegetables"} & search_tokens)
    ):
        required_terms.append("vegetables")

    normalized_query = search_query_clean

    if len(search_tokens) < 2 and len(question_tokens) >= 3:
        normalized_query = question_clean

    if required_terms:
        normalized_query = clean_text(" ".join([normalized_query, *required_terms]))

    return analysis.model_copy(
        update={
            "search_query": truncate_query(normalized_query),
        }
    )


def analyze_question(
    question: str,
    history: list[ChatMessage] | None = None,
) -> QueryAnalysis:
    cleaned_question = clean_text(question)
    history = history or []

    if not cleaned_question or len(cleaned_question.split()) < 2:
        return clarification_analysis(
            cleaned_question,
            "Could you add a person, case, court, organization, topic, or time period?",
        )

    try:
        today = datetime.now(APP_TIMEZONE).date()
        current_date = today.isoformat()
        response = client.responses.parse(
            model=get_planner_model(),
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
                        "Recent conversation is provided only to resolve follow-up "
                        "references like 'that case', 'the court', 'him', or 'the FIR'. "
                        "Do not treat conversation history as evidence. Evidence must "
                        "come from retrieved archive sources later.\n\n"
                        f"Recent conversation:\n{format_history_for_planner(history)}"
                    ),
                },
                {
                    "role": "user",
                    "content": cleaned_question,
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
