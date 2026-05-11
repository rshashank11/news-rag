from app.config import settings
from app.embeddings import embed_text
from app.sparse import encode_sparse_query
from app.vectorstore import hybrid_query
from schemas import RetrievedChunk, clean_text


MIN_TOP_K = 1
MAX_TOP_K = 80
DATE_FALLBACK_TOP_K = 400


def clamp_top_k(top_k: int | None) -> int:
    requested_top_k = settings.retrieval_top_k if top_k is None else top_k
    return min(max(requested_top_k, MIN_TOP_K), MAX_TOP_K)


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
        headline=metadata.get("headline") or "Untitled",
        published_at=metadata.get("published_at"),
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
    dense_vector = embed_text(cleaned_query)
    sparse_vector = encode_sparse_query(cleaned_query)
    date_filter = build_pinecone_date_filter(from_date, to_date)
    metadata_filter = date_filter

    response = hybrid_query(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=search_top_k,
        metadata_filter=metadata_filter,
    )

    chunks = []

    for match in get_match_value(response, "matches", []) or []:
        chunk = format_match(match)
        if chunk is None:
            continue

        if document_matches_date_filter(chunk, from_date, to_date):
            chunks.append(chunk)

    chunks = dedupe_chunks(chunks)

    if chunks or metadata_filter is None:
        return chunks[:search_top_k]

    # Older Pinecone records may only have the old metadata fields.
    # Fall back to a wider unfiltered search, then apply the same date check in Python.
    fallback_response = hybrid_query(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=max(search_top_k, DATE_FALLBACK_TOP_K),
    )
    fallback_chunks = []

    for match in get_match_value(fallback_response, "matches", []) or []:
        chunk = format_match(match)
        if chunk is None:
            continue

        if document_matches_date_filter(chunk, from_date, to_date):
            fallback_chunks.append(chunk)

    return dedupe_chunks(fallback_chunks)[:search_top_k]
