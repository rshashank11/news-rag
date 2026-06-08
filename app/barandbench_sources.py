import logging
import uuid
from datetime import datetime

from sqlalchemy.exc import SQLAlchemyError

from app.source_context import (
    build_combined_source_context,
    list_to_line,
    truncate_text,
)
from database import SessionLocal
from models import StoryMetaData
from schemas import NewsSource, RetrievedChunk

logger = logging.getLogger(__name__)


def story_id_to_uuid(story_id: str | None) -> uuid.UUID | None:
    """
    Convert a Bar & Bench story ID string into a UUID for Postgres lookup.

    Example:
    Pinecone metadata stores story_id as text.
    Postgres stores StoryMetaData.id as UUID.
    This function bridges those two formats.
    """
    if not story_id:
        return None

    try:
        return uuid.UUID(str(story_id))
    except (TypeError, ValueError):
        return None


def datetime_to_iso_date(value) -> str | None:
    """
    Convert a database date value into a simple YYYY-MM-DD string.

    The API response schema expects dates as strings, not Python datetime
    objects.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return value[:10] if value else None

    if isinstance(value, datetime):
        return value.date().isoformat()

    if hasattr(value, "date"):
        return value.date().isoformat()

    return None


def build_full_article_context_from_story(
    story: StoryMetaData,
    fallback_chunk: RetrievedChunk,
    max_chars: int,
) -> str:
    """
    Build answer context from the full Bar & Bench article in Postgres.

    Pinecone gives us the matching chunk first.
    Postgres gives us the fuller article body after we know which story matched.

    If the Postgres row is incomplete, we fall back to the Pinecone chunk so the
    user can still get an answer from retrieved evidence.
    """
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

    return truncate_text(context, max_chars)


def build_barandbench_sources_from_postgres(
    chunks: list[RetrievedChunk],
    max_sources: int,
    max_context_chars: int,
) -> list[NewsSource]:
    """
    Turn Bar & Bench Pinecone matches into full answer sources.

    Example flow:
    1. Pinecone finds story IDs that match the query.
    2. This function fetches those stories from Postgres.
    3. The answer model receives fuller article context instead of only one
       small paragraph chunk.
    """
    sources = []
    seen_story_ids = set()
    db = None
    postgres_available = True

    try:
        db = SessionLocal()
    except Exception as exc:
        postgres_available = False
        logger.warning(
            "Bar & Bench Postgres session could not be created; "
            "falling back to Pinecone chunk context. Error: %s",
            exc,
        )

    try:
        for chunk in chunks:
            dedupe_key = chunk.story_id or chunk.id

            if dedupe_key in seen_story_ids:
                continue

            seen_story_ids.add(dedupe_key)

            story = None
            story_uuid = story_id_to_uuid(chunk.story_id)

            if postgres_available and db is not None and story_uuid:
                try:
                    story = db.get(StoryMetaData, story_uuid)
                except SQLAlchemyError as exc:
                    postgres_available = False
                    logger.warning(
                        "Bar & Bench Postgres lookup failed for story_id=%s; "
                        "falling back to Pinecone chunk context for this and "
                        "remaining sources. Error: %s",
                        chunk.story_id,
                        exc,
                    )

            if story:
                headline = story.headline or chunk.headline
                published_at = (
                    datetime_to_iso_date(story.published_at)
                    or chunk.published_at
                )
                match_snippet = build_full_article_context_from_story(
                    story,
                    chunk,
                    max_context_chars,
                )
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
                    match_snippet=truncate_text(match_snippet, max_context_chars),
                )
            )

            if len(sources) >= max_sources:
                break

        return sources

    finally:
        if db is not None:
            db.close()
