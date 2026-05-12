import calendar
import openai
from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.agents.prompts import PLANNER_SYSTEM_PROMPT
from app.config import settings
from app.openai_client import get_planner_model, make_sync_chat_client
from schemas import ChatMessage, QueryAnalysis, clean_text


client = make_sync_chat_client()

APP_TIMEZONE = ZoneInfo("Asia/Kolkata")
FOLLOW_UP_REFERENCE_PHRASES = [
    "this",
    "that",
    "these",
    "those",
    "those judgments",
    "the court",
    "the bench",
    "of these",
    "of those",
    "above",
    "same issue",
    "contrary view",
    "both sides",
    "distinguish between",
    "repeat names",
]


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


def format_history_for_planner(history: list[ChatMessage]) -> str:
    if not history:
        return "No recent conversation."

    lines = []
    total_chars = 0

    for message in history[-settings.max_history_messages:]:
        line = f"{message.role}: {clean_text(message.content)}"
        total_chars += len(line)

        if total_chars > settings.max_history_chars:
            break

        lines.append(line)

    return "\n".join(lines) or "No recent conversation."


def latest_user_question(history: list[ChatMessage]) -> str | None:
    for message in reversed(history):
        if message.role == "user":
            return clean_text(message.content)

    return None


def is_follow_up_question(question: str) -> bool:
    normalized_question = question.lower()
    return any(phrase in normalized_question for phrase in FOLLOW_UP_REFERENCE_PHRASES)


def month_date_window_for_text(text: str, today: date) -> tuple[str | None, str | None]:
    normalized_text = text.lower()

    if "this month" not in normalized_text:
        return None, None

    last_day = calendar.monthrange(today.year, today.month)[1]
    return (
        today.replace(day=1).isoformat(),
        today.replace(day=last_day).isoformat(),
    )


def recover_follow_up_analysis(
    question: str,
    history: list[ChatMessage],
    today: date,
) -> QueryAnalysis | None:
    parent_question = latest_user_question(history)

    if not parent_question:
        return None

    if not is_follow_up_question(question):
        return None

    from_date, to_date = month_date_window_for_text(parent_question, today)
    search_query = clean_text(f"{parent_question} {question}")[:300]

    return QueryAnalysis(
        intent="answer",
        search_query=search_query,
        entities=[],
        k=settings.follow_up_query_k,
        clarification_needed=False,
        clarification_question=None,
        from_date=from_date,
        to_date=to_date,
        refusal_reason=None,
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
            recovered_analysis = recover_follow_up_analysis(
                cleaned_question,
                history,
                today,
            )

            if recovered_analysis is not None:
                return recovered_analysis

            return analysis.model_copy(update={"intent": "clarify"})

        return analysis

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
