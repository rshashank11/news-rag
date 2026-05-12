import re
from datetime import date

from app.config import settings
from app.embeddings import embed_text
from app.sparse import encode_sparse_query
from app.vectorstore import hybrid_query
from schemas import RetrievedChunk, clean_text


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "about",
    "after",
    "against",
    "also",
    "and",
    "any",
    "are",
    "before",
    "between",
    "but",
    "can",
    "case",
    "cases",
    "did",
    "does",
    "for",
    "from",
    "give",
    "has",
    "have",
    "into",
    "last",
    "latest",
    "list",
    "month",
    "need",
    "orders",
    "recent",
    "said",
    "same",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "under",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def clamp_top_k(top_k: int | None) -> int:
    requested_top_k = settings.retrieval_top_k if top_k is None else top_k
    return min(max(requested_top_k, settings.min_retrieval_top_k), settings.max_retrieval_top_k)


def iso_date_to_yyyymmdd(value: str | None) -> int | None:
    if value is None:
        return None

    return int(value.replace("-", ""))


def build_pinecone_date_filter(
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict | None:
    date_filter = {}

    from_date_number = iso_date_to_yyyymmdd(from_date)
    to_date_number = iso_date_to_yyyymmdd(to_date)

    if from_date_number is not None:
        date_filter["$gte"] = from_date_number

    if to_date_number is not None:
        date_filter["$lte"] = to_date_number

    if not date_filter:
        return None

    return {"published_at_yyyymmdd": date_filter}


def get_match_value(match, name: str, default=None):
    if isinstance(match, dict):
        return match.get(name, default)

    return getattr(match, name, default)


def get_match_metadata(match) -> dict:
    metadata = get_match_value(match, "metadata", {}) or {}

    if hasattr(metadata, "to_dict"):
        return metadata.to_dict()

    return dict(metadata)


def get_match_score(match) -> float | None:
    score = get_match_value(match, "score")

    if score is None:
        return None

    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def list_metadata_values(value) -> list[str]:
    if not value:
        return []

    if isinstance(value, str):
        return [clean_text(value)] if clean_text(value) else []

    if isinstance(value, list):
        values = []

        for item in value:
            if not isinstance(item, str):
                continue

            cleaned_item = clean_text(item)
            if cleaned_item:
                values.append(cleaned_item)

        return values

    return []


def document_matches_date_filter(
    chunk: RetrievedChunk,
    from_date: str | None = None,
    to_date: str | None = None,
) -> bool:
    published_at = getattr(chunk, "published_at", None)

    if not published_at:
        return not (from_date or to_date)

    if from_date and published_at < from_date:
        return False

    if to_date and published_at > to_date:
        return False

    return True


def format_match(match) -> RetrievedChunk | None:
    metadata = get_match_metadata(match)
    chunk_text = metadata.get("chunk_text")

    if not chunk_text:
        return None

    return RetrievedChunk(
        id=str(get_match_value(match, "id", "")),
        story_id=metadata.get("story_id"),
        chunk_index=metadata.get("chunk_index"),
        headline=metadata.get("headline") or "Untitled",
        published_at=metadata.get("published_at"),
        topics=list_metadata_values(metadata.get("topics")),
        categories=list_metadata_values(metadata.get("categories")),
        retrieval_score=get_match_score(match),
        chunk_text=chunk_text,
    )


def dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen_ids = set()
    unique_chunks = []

    for chunk in chunks:
        if chunk.id in seen_ids:
            continue

        seen_ids.add(chunk.id)
        unique_chunks.append(chunk)

    return unique_chunks


def token_set(text: str) -> set[str]:
    min_length = settings.rerank_min_token_length

    return {
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) >= min_length and token not in STOPWORDS
    }


def overlap_ratio(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0

    text_tokens = token_set(text)

    if not text_tokens:
        return 0.0

    return len(query_tokens & text_tokens) / len(query_tokens)


def parse_chunk_date(chunk: RetrievedChunk) -> date | None:
    if not chunk.published_at:
        return None

    try:
        return date.fromisoformat(chunk.published_at)
    except ValueError:
        return None


def normalize_float(value: float | None, minimum: float, maximum: float) -> float:
    if value is None:
        return 0.0

    if maximum <= minimum:
        return 1.0

    return (value - minimum) / (maximum - minimum)


def normalize_chunk_date(
    chunk_date: date | None,
    earliest_date: date | None,
    latest_date: date | None,
) -> float:
    if chunk_date is None or earliest_date is None or latest_date is None:
        return 0.0

    total_days = (latest_date - earliest_date).days

    if total_days <= 0:
        return 0.0

    return (chunk_date - earliest_date).days / total_days


def rerank_chunk(
    chunk: RetrievedChunk,
    query_tokens: set[str],
    normalized_vector_score: float,
    normalized_date_score: float,
) -> RetrievedChunk:
    metadata_text = " ".join([*chunk.topics, *chunk.categories])
    rerank_score = (
        normalized_vector_score * settings.rerank_vector_score_weight
        + overlap_ratio(query_tokens, chunk.headline) * settings.rerank_headline_overlap_weight
        + overlap_ratio(query_tokens, metadata_text) * settings.rerank_metadata_overlap_weight
        + overlap_ratio(query_tokens, chunk.chunk_text) * settings.rerank_chunk_overlap_weight
        + normalized_date_score * settings.rerank_date_score_weight
    )

    return chunk.model_copy(update={"rerank_score": round(rerank_score, 6)})


def story_key(chunk: RetrievedChunk) -> str:
    return chunk.story_id or chunk.id


def rerank_chunks_by_story(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return []

    query_tokens = token_set(query)
    retrieval_scores = [
        chunk.retrieval_score
        for chunk in chunks
        if chunk.retrieval_score is not None
    ]
    minimum_score = min(retrieval_scores) if retrieval_scores else 0.0
    maximum_score = max(retrieval_scores) if retrieval_scores else 0.0
    chunk_dates = [
        chunk_date
        for chunk in chunks
        if (chunk_date := parse_chunk_date(chunk)) is not None
    ]
    earliest_date = min(chunk_dates) if chunk_dates else None
    latest_date = max(chunk_dates) if chunk_dates else None

    scored_chunks = [
        rerank_chunk(
            chunk=chunk,
            query_tokens=query_tokens,
            normalized_vector_score=normalize_float(
                chunk.retrieval_score,
                minimum_score,
                maximum_score,
            ),
            normalized_date_score=normalize_chunk_date(
                parse_chunk_date(chunk),
                earliest_date,
                latest_date,
            ),
        )
        for chunk in chunks
    ]
    grouped_chunks: dict[str, list[RetrievedChunk]] = {}

    for chunk in scored_chunks:
        grouped_chunks.setdefault(story_key(chunk), []).append(chunk)

    story_groups = []

    for group_key, story_chunks in grouped_chunks.items():
        ordered_story_chunks = sorted(
            story_chunks,
            key=lambda chunk: (
                chunk.rerank_score or 0.0,
                chunk.retrieval_score or 0.0,
                -(chunk.chunk_index or 0),
            ),
            reverse=True,
        )
        evidence_count = min(len(ordered_story_chunks), settings.rerank_story_evidence_cap)
        story_score = (
            (ordered_story_chunks[0].rerank_score or 0.0)
            + evidence_count * settings.rerank_story_evidence_weight
        )
        story_groups.append((story_score, group_key, ordered_story_chunks))

    ordered_groups = sorted(
        story_groups,
        key=lambda item: (
            item[0],
            item[2][0].retrieval_score or 0.0,
            item[2][0].published_at or "",
        ),
        reverse=True,
    )
    max_group_length = max(len(item[2]) for item in ordered_groups)
    ordered_chunks = []

    for chunk_offset in range(max_group_length):
        for _, _, story_chunks in ordered_groups:
            if chunk_offset < len(story_chunks):
                ordered_chunks.append(story_chunks[chunk_offset])

    return ordered_chunks


def retrieve_chunks(
    query: str,
    top_k: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[RetrievedChunk]:
    cleaned_query = clean_text(query)

    if not cleaned_query:
        raise ValueError("Search query cannot be empty.")

    search_top_k = clamp_top_k(top_k)
    candidate_top_k = clamp_top_k(max(search_top_k, settings.rerank_candidate_top_k))
    dense_vector = embed_text(cleaned_query)
    sparse_vector = encode_sparse_query(cleaned_query)
    date_filter = build_pinecone_date_filter(from_date, to_date)
    metadata_filter = date_filter

    response = hybrid_query(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=candidate_top_k,
        metadata_filter=metadata_filter,
    )

    chunks = []

    for match in get_match_value(response, "matches", []) or []:
        chunk = format_match(match)
        if chunk is None:
            continue

        if document_matches_date_filter(chunk, from_date, to_date):
            chunks.append(chunk)

    chunks = rerank_chunks_by_story(cleaned_query, dedupe_chunks(chunks))

    if chunks or metadata_filter is None:
        return chunks[:search_top_k]

    # Older Pinecone records may only have the old metadata fields.
    # Fall back to a wider unfiltered search, then apply the same date check in Python.
    fallback_response = hybrid_query(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=max(candidate_top_k, settings.date_fallback_top_k),
    )
    fallback_chunks = []

    for match in get_match_value(fallback_response, "matches", []) or []:
        chunk = format_match(match)
        if chunk is None:
            continue

        if document_matches_date_filter(chunk, from_date, to_date):
            fallback_chunks.append(chunk)

    return rerank_chunks_by_story(cleaned_query, dedupe_chunks(fallback_chunks))[:search_top_k]
