from app.config import settings
from app.embeddings import embed_text
from app.sparse import encode_sparse_query
from app.vectorstore import hybrid_query
from schemas import RetrievedChunk, clean_text


MIN_TOP_K = 1
MAX_TOP_K = 80


def clamp_top_k(top_k: int | None) -> int:
    requested_top_k = settings.retrieval_top_k if top_k is None else top_k
    return min(max(requested_top_k, MIN_TOP_K), MAX_TOP_K)


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
        score=get_match_value(match, "score"),
        story_id=metadata.get("story_id"),
        chunk_index=metadata.get("chunk_index"),
        headline=metadata.get("headline") or "Untitled",
        published_at=metadata.get("published_at"),
        chunk_text=chunk_text,
        search_type="hybrid",
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
    response = hybrid_query(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=search_top_k,
    )

    chunks = []

    for match in get_match_value(response, "matches", []) or []:
        chunk = format_match(match)
        if chunk is None:
            continue

        if document_matches_date_filter(chunk, from_date, to_date):
            chunks.append(chunk)

    return dedupe_chunks(chunks)
