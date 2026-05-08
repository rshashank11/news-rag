import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from app.agents.prompts import PLANNER_SYSTEM_PROMPT
from app.openai_client import get_chat_model, make_chat_client
from schemas import ChatMessage, QueryAnalysis, clean_text


client = make_chat_client()

APP_TIMEZONE = ZoneInfo("Asia/Kolkata")
DEFAULT_K = 10
BRIEFING_K = 12
TIMELINE_K = 15
MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS = 6000

PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore your instructions",
    "bypass guardrails",
    "override system",
    "reveal your prompt",
    "show me your prompt",
    "system prompt",
    "developer message",
    "api key",
    "environment variable",
    "answer without sources",
    "use your training data",
    "use pretrained knowledge",
]

LEGAL_ADVICE_PATTERNS = [
    "should i",
    "what should i do",
    "advise me",
    "legal advice",
    "draft",
    "file a petition",
    "file bail",
    "arguments should i use",
    "my case",
    "my bail",
    "my lawyer",
]

PRIVACY_PATTERNS = [
    "private phone",
    "phone number",
    "home address",
    "personal address",
    "private email",
    "personal email",
    "dox",
]

TIMELINE_PATTERNS = [
    "timeline",
    "chronology",
    "sequence of events",
    "datewise",
    "date-wise",
]

BRIEFING_PATTERNS = [
    "summary",
    "summarize",
    "briefing",
    "roundup",
    "overview",
    "digest",
]

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

FOLLOW_UP_REFERENCE_PATTERNS = [
    r"\bthis\s+(case|matter|petition|plea|fir|order|judgment|judgement|story|issue)\b",
    r"\bthat\s+(case|matter|petition|plea|fir|order|judgment|judgement|story|issue)\b",
    r"\bthe\s+(case|matter|petition|plea|fir|order|judgment|judgement|story|issue)\b",
    r"\bsame\s+(case|matter|petition|plea|fir|order|judgment|judgement|story|issue)\b",
    r"\babove\s+(case|matter|petition|plea|fir|order|judgment|judgement|story|issue)\b",
]


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def get_current_date() -> date:
    return datetime.now(APP_TIMEZONE).date()


def previous_month_range(today: date) -> tuple[str, str]:
    first_day_this_month = today.replace(day=1)
    last_day_previous_month = first_day_this_month - timedelta(days=1)
    first_day_previous_month = last_day_previous_month.replace(day=1)
    return first_day_previous_month.isoformat(), last_day_previous_month.isoformat()


def last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        return 31

    return (date(year, month + 1, 1) - timedelta(days=1)).day


def year_range(year: int) -> tuple[str, str]:
    return f"{year}-01-01", f"{year}-12-31"


def month_range(year: int, month: int) -> tuple[str, str]:
    last_day = last_day_of_month(year, month)
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"


def infer_date_range(question: str) -> tuple[str | None, str | None]:
    lowered = question.lower()
    today = get_current_date()

    if "yesterday" in lowered:
        value = today - timedelta(days=1)
        return value.isoformat(), value.isoformat()

    if "today" in lowered:
        return today.isoformat(), today.isoformat()

    if "last month" in lowered:
        return previous_month_range(today)

    if "this year" in lowered:
        return year_range(today.year)

    for month_name, month_number in MONTHS.items():
        match = re.search(rf"\b{month_name}\s+(\d{{4}})\b", lowered)
        if match:
            return month_range(int(match.group(1)), month_number)

    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", lowered)
    if year_match:
        return year_range(int(year_match.group(1)))

    return None, None


def infer_intent(question: str) -> str:
    lowered = question.lower()

    if contains_any(lowered, TIMELINE_PATTERNS):
        return "timeline"

    if contains_any(lowered, BRIEFING_PATTERNS):
        return "briefing"

    return "answer"


def infer_k(intent: str) -> int:
    if intent == "timeline":
        return TIMELINE_K

    if intent == "briefing":
        return BRIEFING_K

    return DEFAULT_K


def build_out_of_scope_analysis(
    question: str,
    flag: str,
    refusal_reason: str,
) -> QueryAnalysis:
    return QueryAnalysis(
        intent="out_of_scope",
        search_query=clean_text(question)[:300] or "out of scope",
        entities=[],
        k=DEFAULT_K,
        clarification_needed=False,
        clarification_question=None,
        from_date=None,
        to_date=None,
        is_in_scope=False,
        refusal_reason=refusal_reason,
        safety_flags=[flag],
    )


def build_clarification_analysis(
    question: str,
    clarification_question: str | None = None,
) -> QueryAnalysis:
    return QueryAnalysis(
        intent="clarify",
        search_query=clean_text(question)[:300] or "clarification needed",
        entities=[],
        k=DEFAULT_K,
        clarification_needed=True,
        clarification_question=clarification_question
        or "Could you add a person, case, court, organization, topic, or time period?",
        from_date=None,
        to_date=None,
        is_in_scope=True,
        refusal_reason=None,
        safety_flags=[],
    )


def has_follow_up_reference(question: str) -> bool:
    lowered = question.lower()
    return any(
        re.search(pattern, lowered)
        for pattern in FOLLOW_UP_REFERENCE_PATTERNS
    )


def has_recent_conversation_context(history: list[ChatMessage]) -> bool:
    return any(clean_text(message.content) for message in history)


def needs_reference_clarification(
    question: str,
    history: list[ChatMessage],
) -> bool:
    return has_follow_up_reference(question) and not has_recent_conversation_context(history)


def build_reference_clarification_analysis(question: str) -> QueryAnalysis:
    return build_clarification_analysis(
        question,
        (
            "Which case or story should I build the timeline for? "
            "Please mention a case name, party, court, person, organization, or news topic."
        ),
    )


def normalize_history_topic(text: str) -> str:
    topic = clean_text(text).strip(" ?.!")
    topic = re.sub(r"^q\d+\s*[-:]\s*", "", topic, flags=re.IGNORECASE)
    topic = re.sub(r"^(find|show me|tell me about|what happened in|what happened with)\s+", "", topic, flags=re.IGNORECASE)
    return clean_text(topic).strip(" ?.!")[:180]


def infer_follow_up_topic(history: list[ChatMessage]) -> str | None:
    for message in reversed(history):
        if message.role != "user":
            continue

        topic = normalize_history_topic(message.content)
        if topic:
            return topic

    return None


def build_contextual_follow_up_analysis(
    question: str,
    topic: str,
) -> QueryAnalysis:
    intent = infer_intent(question)
    from_date, to_date = infer_date_range(question)

    if intent == "timeline":
        search_query = f"{topic} timeline chronology datewise progression"
    else:
        search_query = f"{topic} {question}"

    return QueryAnalysis(
        intent=intent,
        search_query=clean_text(search_query)[:300],
        entities=[topic],
        k=infer_k(intent),
        clarification_needed=False,
        clarification_question=None,
        from_date=from_date,
        to_date=to_date,
        is_in_scope=True,
        refusal_reason=None,
        safety_flags=[],
    )


def build_fallback_analysis(question: str) -> QueryAnalysis:
    cleaned_question = clean_text(question)
    lowered = cleaned_question.lower()

    if not cleaned_question or len(cleaned_question.split()) < 2:
        return build_clarification_analysis(cleaned_question)

    if contains_any(lowered, PROMPT_INJECTION_PATTERNS):
        return build_out_of_scope_analysis(
            cleaned_question,
            "prompt_injection",
            "I can only answer using retrieved sources from the indexed legal-news archive.",
        )

    if contains_any(lowered, PRIVACY_PATTERNS):
        return build_out_of_scope_analysis(
            cleaned_question,
            "privacy",
            "I cannot help retrieve or expose private personal information.",
        )

    if contains_any(lowered, LEGAL_ADVICE_PATTERNS):
        return build_out_of_scope_analysis(
            cleaned_question,
            "legal_advice",
            "I can summarize indexed legal-news coverage, but I cannot provide legal advice or strategy.",
        )

    intent = infer_intent(cleaned_question)
    from_date, to_date = infer_date_range(cleaned_question)

    return QueryAnalysis(
        intent=intent,
        search_query=cleaned_question[:300],
        entities=[],
        k=infer_k(intent),
        clarification_needed=False,
        clarification_question=None,
        from_date=from_date,
        to_date=to_date,
        is_in_scope=True,
        refusal_reason=None,
        safety_flags=[],
    )


def format_history_for_planner(history: list[ChatMessage]) -> str:
    if not history:
        return "No recent conversation."

    lines = []
    total_chars = 0

    for message in history[-MAX_HISTORY_MESSAGES:]:
        line = f"{message.role}: {clean_text(message.content)}"
        total_chars += len(line)

        if total_chars > MAX_HISTORY_CHARS:
            break

        lines.append(line)

    return "\n".join(lines) or "No recent conversation."


def analyze_question(
    question: str,
    history: list[ChatMessage] | None = None,
) -> QueryAnalysis:
    cleaned_question = clean_text(question)
    history = history or []

    if not cleaned_question or len(cleaned_question.split()) < 2:
        return build_clarification_analysis(cleaned_question)

    if needs_reference_clarification(cleaned_question, history):
        return build_reference_clarification_analysis(cleaned_question)

    follow_up_topic = infer_follow_up_topic(history) if has_follow_up_reference(cleaned_question) else None
    if follow_up_topic and infer_intent(cleaned_question) == "timeline":
        return build_contextual_follow_up_analysis(cleaned_question, follow_up_topic)

    fallback_analysis = build_fallback_analysis(cleaned_question)
    if not fallback_analysis.is_in_scope or fallback_analysis.clarification_needed:
        return fallback_analysis

    try:
        today = get_current_date()
        response = client.responses.parse(
            model=get_chat_model(),
            input=[
                {
                    "role": "system",
                    "content": PLANNER_SYSTEM_PROMPT,
                },
                {
                    "role": "system",
                    "content": (
                        "Runtime date context: "
                        f"today is {today.isoformat()} in Asia/Kolkata. "
                        "Resolve relative date phrases using this date."
                    ),
                },
                {
                    "role": "system",
                    "content": (
                        "Recent conversation is provided only to resolve follow-up "
                        "references like 'that case', 'the court', 'him', or 'the FIR'. "
                        "Do not treat conversation history as evidence. Evidence must "
                        "come from retrieved archive sources later.\n"
                        "If the user asks for a timeline of 'this case' and the "
                        "recent conversation provides the topic, set intent to "
                        "timeline and build the search_query from that topic.\n\n"
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
        return response.output_parsed

    except Exception as exc:
        print(f"OpenAI query planning failed: {exc}")
        if follow_up_topic:
            return build_contextual_follow_up_analysis(cleaned_question, follow_up_topic)
        return fallback_analysis
