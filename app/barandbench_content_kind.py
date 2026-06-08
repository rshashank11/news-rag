import logging
from functools import lru_cache
from typing import Literal

from pydantic import Field

from app.openai_client import get_content_kind_model, make_sync_chat_client
from schemas import StrictBaseModel, clean_text

logger = logging.getLogger(__name__)

BARANDBENCH_CONTENT_KIND_NEWS_ARTICLE = "news_article"
BARANDBENCH_CONTENT_KIND_JOB_POSTING = "job_posting"
BARANDBENCH_CONTENT_KIND_OTHER = "other"
BARANDBENCH_ALLOWED_CONTENT_KINDS = {
    BARANDBENCH_CONTENT_KIND_NEWS_ARTICLE,
    BARANDBENCH_CONTENT_KIND_JOB_POSTING,
    BARANDBENCH_CONTENT_KIND_OTHER,
}
MAX_CLASSIFIER_BODY_CHARS = 5000

CONTENT_KIND_SYSTEM_PROMPT = """
You classify Bar & Bench archive items for a news RAG system.

Return one content_kind:
- news_article: normal legal journalism or reported coverage. This includes
  court cases, hearings, judgments, investigations, appointments, law firm
  developments, and news reports about hiring, recruitment disputes, or vacancy
  controversies.
- job_posting: the main purpose is to advertise an opening, internship,
  clerkship, role, or application opportunity, or to tell people how to apply.
- other: anything else that is neither normal reporting nor a job posting.

Important:
- Reporting about a recruitment issue is news_article, not job_posting.
- A hiring controversy, exam dispute, recruitment scam, or appointment-related
  court case is news_article.
- A call for applications, internship notice, law clerk opening, vacancy post,
  or article whose main value is application details is job_posting.
"""

client = make_sync_chat_client()


class BarAndBenchContentKindAssessment(StrictBaseModel):
    content_kind: Literal[
        "news_article",
        "job_posting",
        "other",
    ]
    reason: str = Field(min_length=1, max_length=400)


def normalize_content_kind(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = clean_text(value).lower()
    if normalized in BARANDBENCH_ALLOWED_CONTENT_KINDS:
        return normalized

    return None


def truncate_classifier_body(text: str, max_chars: int = MAX_CLASSIFIER_BODY_CHARS) -> str:
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0].strip() or cleaned[:max_chars]


@lru_cache(maxsize=4096)
def _classify_cached(
    headline: str,
    summary: str,
    categories_text: str,
    body_text: str,
) -> str | None:
    try:
        response = client.responses.parse(
            model=get_content_kind_model(),
            timeout=30,
            temperature=0,
            input=[
                {
                    "role": "system",
                    "content": CONTENT_KIND_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Headline: {headline or 'Unknown'}\n"
                        f"Summary: {summary or 'None'}\n"
                        f"Categories: {categories_text or 'None'}\n"
                        f"Body:\n{body_text or 'None'}"
                    ),
                },
            ],
            text_format=BarAndBenchContentKindAssessment,
        )
    except Exception as exc:
        logger.warning("Bar & Bench content kind classification failed: %s", exc)
        return None

    return normalize_content_kind(response.output_parsed.content_kind)


def classify_barandbench_story_content_kind(
    headline: str,
    summary: str,
    categories: list[str],
    full_content: str,
) -> str | None:
    return _classify_cached(
        clean_text(headline or "Untitled"),
        clean_text(summary or ""),
        ", ".join(clean_text(category) for category in categories if clean_text(category)),
        truncate_classifier_body(full_content),
    )


def classify_barandbench_chunk_content_kind(
    headline: str,
    categories: list[str],
    chunk_text: str,
) -> str | None:
    return _classify_cached(
        clean_text(headline or "Untitled"),
        "",
        ", ".join(clean_text(category) for category in categories if clean_text(category)),
        truncate_classifier_body(chunk_text),
    )
