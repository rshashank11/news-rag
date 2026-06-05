import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config import settings
from app.embeddings import embed_text
from app.legal_extraction import CANONICAL_COURT_MAP, KNOWN_COURT_NAMES, extract_courts
from app.news_sources import source_profile
from app.sparse import encode_sparse_query
from app.vectorstore import hybrid_query
from schemas import RetrievedChunk, clean_text


def runtime_source_profile(source: str | None):
    """
    Get the metadata rules for the selected source.

    Example:
    Sakal stores dates in "date_published".
    Bar & Bench stores dates in "published_at".
    This profile tells retrieval which field to read.
    """
    source_name = settings.news_source_config(source)["source"]
    return source_profile(source_name)


def clamp_top_k(top_k: int | None) -> int:
    """
    Keep requested result count inside safe limits.

    Example:
    If a user asks for 500 results but max_retrieval_top_k is 80,
    this function caps it at 80.
    """
    requested_top_k = settings.retrieval_top_k if top_k is None else top_k
    return min(
        max(requested_top_k, settings.min_retrieval_top_k),
        settings.max_retrieval_top_k,
    )


def iso_date_to_yyyymmdd(value: str | None) -> int | None:
    """
    Convert a date string into the number format stored in Pinecone.

    Example:
    "2026-04-01" becomes 20260401.
    """
    if value is None:
        return None

    try:
        return int(value.replace("-", ""))
    except (ValueError, AttributeError):
        return None


def build_pinecone_date_filter(
    from_date: str | None = None,
    to_date: str | None = None,
    source: str | None = None,
) -> dict | None:
    """
    Build the date filter sent to Pinecone.

    Example:
    from_date="2026-04-01" creates a "$gte" filter.
    to_date="2026-04-30" creates a "$lte" filter.
    """
    date_filter = {}

    from_date_number = iso_date_to_yyyymmdd(from_date)
    to_date_number = iso_date_to_yyyymmdd(to_date)

    if from_date_number is not None:
        date_filter["$gte"] = from_date_number

    if to_date_number is not None:
        date_filter["$lte"] = to_date_number

    if not date_filter:
        return None

    date_field = runtime_source_profile(source).date_filter_field

    return {date_field: date_filter}


def build_pinecone_court_filter(
    entities: list[str],
    source: str | None = None,
) -> dict | None:
    """
    Build an optional Pinecone court filter from extracted entities.

    Only applied for Bar & Bench when one or more entities map cleanly to a
    known court name. Court-based filtering is high-precision because court
    names are exact and unambiguous.

    Example:
    entities=["Supreme Court", "Delhi High Court"] →
        {"court": {"$in": ["Supreme Court", "Delhi High Court"]}}
    """
    if not entities:
        return None

    source_name = settings.news_source_config(source).get("source")
    if source_name != "barandbench":
        return None

    matched_courts = [
        CANONICAL_COURT_MAP[entity.lower()]
        for entity in entities
        if entity.lower() in KNOWN_COURT_NAMES
    ]

    if not matched_courts:
        return None

    return {"court": {"$in": matched_courts}}


def combine_pinecone_filters(*filters: dict | None) -> dict | None:
    """
    Combine multiple Pinecone metadata filters with $and.

    Skips None filters and returns a single filter or $and clause.
    """
    active = [f for f in filters if f is not None]

    if not active:
        return None

    if len(active) == 1:
        return active[0]

    return {"$and": active}


def get_match_value(match, name: str, default=None):
    """
    Read a value from a Pinecone match.

    Pinecone responses may behave like objects or dictionaries depending on the
    SDK path used. This function handles both shapes.
    """
    if isinstance(match, dict):
        return match.get(name, default)

    return getattr(match, name, default)


def get_match_metadata(match) -> dict:
    """
    Convert Pinecone metadata into a normal Python dictionary.

    Example:
    Some SDK objects expose metadata.to_dict().
    Our app code wants a plain dict either way.
    """
    metadata = get_match_value(match, "metadata", {}) or {}

    if hasattr(metadata, "to_dict"):
        return metadata.to_dict()

    return dict(metadata)


def get_match_score(match) -> float | None:
    """
    Read the Pinecone similarity score as a float.

    The score becomes retrieval_score on RetrievedChunk and can be used as a
    fallback ranking signal.
    """
    score = get_match_value(match, "score")

    if score is None:
        return None

    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def list_metadata_values(value) -> list[str]:
    """
    Convert metadata into a clean list of strings.

    Example:
    "Pune" becomes ["Pune"].
    [" Pune ", "", "Traffic"] becomes ["Pune", "Traffic"].
    """
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
    """
    Check whether one retrieved chunk is inside the requested date window.

    Pinecone normally handles this filter. We also check locally as a safety net,
    especially during index rebuilds or fallback searches.
    """
    published_at = getattr(chunk, "published_at", None)

    if not published_at:
        return not (from_date or to_date)

    if from_date and published_at < from_date:
        return False

    if to_date and published_at > to_date:
        return False

    return True


def metadata_categories_for_source(
    metadata: dict,
    source: str | None,
) -> list[str]:
    """
    Read category fields using the selected source's metadata rules.

    Example:
    Bar & Bench uses "categories".
    Sakal combines "edition", "source", and "location".
    """
    profile = runtime_source_profile(source)
    values = [
        metadata.get(field_name)
        for field_name in profile.category_metadata_fields
    ]

    if len(values) == 1:
        return list_metadata_values(values[0])

    return list_metadata_values(values)


def format_match(match, source: str | None = None) -> RetrievedChunk | None:
    """
    Convert one Pinecone result into the app's RetrievedChunk object.

    Pinecone gives us raw metadata.
    The workflow expects a consistent object with story_id, headline, date,
    topics, categories, score, and chunk text.
    """
    metadata = get_match_metadata(match)
    chunk_text = metadata.get("chunk_text")
    profile = runtime_source_profile(source)

    if not chunk_text:
        return None

    return RetrievedChunk(
        id=str(get_match_value(match, "id", "")),
        story_id=metadata.get(profile.id_metadata_field),
        chunk_index=metadata.get("chunk_index"),
        headline=metadata.get("headline") or "Untitled",
        published_at=metadata.get(profile.published_at_metadata_field),
        topics=list_metadata_values(metadata.get(profile.topics_metadata_field)),
        categories=metadata_categories_for_source(metadata, source),
        retrieval_score=get_match_score(match),
        chunk_text=chunk_text,
    )


def dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Remove duplicate chunks while keeping the first/highest-ranked copy.

    Example:
    If Pinecone returns the same chunk twice, the answer model should see it once.
    """
    seen_ids = set()
    unique_chunks = []

    for chunk in chunks:
        if chunk.id in seen_ids:
            continue

        seen_ids.add(chunk.id)
        unique_chunks.append(chunk)

    return unique_chunks


def story_key(chunk: RetrievedChunk) -> str:
    """
    Return the ID used to group chunks from the same story.

    Example:
    Three chunks from one Sakal article should share the same story_key.
    """
    return chunk.story_id or chunk.id


def chunk_to_jina_document(chunk: RetrievedChunk) -> str:
    """
    Format a retrieved chunk for the optional Jina reranker.

    The reranker sees headline, date, metadata, and text together so it can judge
    relevance more accurately than using body text alone.
    """
    metadata_text = " ".join([*chunk.topics, *chunk.categories])

    return "\n".join(
        [
            f"Headline: {chunk.headline or 'Untitled'}",
            f"Published: {chunk.published_at or 'Unknown'}",
            f"Metadata: {metadata_text}",
            f"Text: {chunk.chunk_text}",
        ]
    )


def call_jina_reranker(
    query: str,
    chunks: list[RetrievedChunk],
) -> list[tuple[RetrievedChunk, float]]:
    """
    Ask Jina to rerank retrieved chunks by relevance to the query.

    Example:
    Pinecone may return 30 candidate chunks.
    Jina can reorder them so the most answer-worthy chunks rise to the top.
    """
    if not chunks:
        return []

    if not settings.jina_api_key:
        raise RuntimeError(
            "JINA_API_KEY is missing. Add it to your .env or Hugging Face secrets."
        )

    documents = [
        chunk_to_jina_document(chunk)
        for chunk in chunks
    ]

    top_n = min(settings.jina_rerank_top_n, len(documents))

    payload = {
        "model": settings.jina_rerank_model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }

    headers = {
        "Authorization": f"Bearer {settings.jina_api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        settings.jina_rerank_url,
        headers=headers,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()

    reranked_chunks: list[tuple[RetrievedChunk, float]] = []

    for item in data.get("results", []):
        chunk_index = item.get("index")
        relevance_score = item.get("relevance_score", 0.0)

        if chunk_index is None:
            continue

        if chunk_index < 0 or chunk_index >= len(chunks):
            continue

        reranked_chunks.append(
            (
                chunks[chunk_index],
                float(relevance_score),
            )
        )

    return reranked_chunks


def build_retrieval_score_ranked_chunks(
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    Use Pinecone's original score when no external reranker is active.

    Example:
    If RERANK_MODE="none", we still attach rerank_score so downstream code can
    use one common ranking field.
    """
    return [
        chunk.model_copy(
            update={
                "rerank_score": round(chunk.retrieval_score or 0.0, 6),
            }
        )
        for chunk in chunks
    ]


def rerank_chunks_by_story(
    query: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    Rank chunks while avoiding too many chunks from one story.

    Example:
    If one article has 6 matching chunks and another article has 1 strong chunk,
    we do not want the first article to crowd out everything else. Grouping by
    story helps the answer model see a better spread of sources.
    """
    if not chunks:
        return []

    rerank_mode = clean_text(settings.rerank_mode or "none").lower()

    if rerank_mode != "jina":
        scored_chunks = build_retrieval_score_ranked_chunks(chunks)
    else:
        try:
            jina_ranked_chunks = call_jina_reranker(
                query=query,
                chunks=chunks,
            )

            scored_chunks = [
                chunk.model_copy(
                    update={
                        "rerank_score": round(jina_score, 6),
                    }
                )
                for chunk, jina_score in jina_ranked_chunks
            ]

        except (requests.RequestException, RuntimeError, ValueError) as exc:
            print(
                "Warning: Jina reranker unavailable; "
                f"falling back to retrieval scores ({exc})."
            )
            scored_chunks = build_retrieval_score_ranked_chunks(chunks)

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

        evidence_count = min(
            len(ordered_story_chunks),
            settings.rerank_story_evidence_cap,
        )

        story_score = (
            (ordered_story_chunks[0].rerank_score or 0.0)
            + evidence_count * settings.rerank_story_evidence_weight
        )

        story_groups.append(
            (
                story_score,
                group_key,
                ordered_story_chunks,
            )
        )

    ordered_groups = sorted(
        story_groups,
        key=lambda item: (
            item[0],
            item[2][0].rerank_score or 0.0,
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
    source: str | None = None,
    entities: list[str] | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the best matching news chunks for one planned search query.

    Full flow:
    1. embed the query for meaning-based search,
    2. encode the query for keyword/BM25 search,
    3. send both to Pinecone as a hybrid query,
    4. format and rerank the returned chunks.
    """
    cleaned_query = clean_text(query)

    if not cleaned_query:
        raise ValueError("Search query cannot be empty.")

    search_top_k = clamp_top_k(top_k)
    candidate_top_k = clamp_top_k(
        max(search_top_k, settings.rerank_candidate_top_k)
    )

    source_config = settings.news_source_config(source)
    source_alpha = source_config.get("hybrid_alpha")

    dense_vector = embed_text(cleaned_query)
    sparse_vector = encode_sparse_query(cleaned_query, source=source)
    date_filter = build_pinecone_date_filter(
        from_date,
        to_date,
        source=source,
    )
    court_filter = build_pinecone_court_filter(entities or [], source=source)
    metadata_filter = combine_pinecone_filters(date_filter, court_filter)

    response = hybrid_query(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=candidate_top_k,
        alpha=source_alpha,
        metadata_filter=metadata_filter,
        source=source,
    )

    chunks = []

    for match in get_match_value(response, "matches", []) or []:
        chunk = format_match(match, source=source)

        if chunk is None:
            continue

        if document_matches_date_filter(chunk, from_date, to_date):
            chunks.append(chunk)

    chunks = rerank_chunks_by_story(
        cleaned_query,
        dedupe_chunks(chunks),
    )

    if chunks or metadata_filter is None:
        return chunks[:search_top_k]

    # If both a court filter and a date filter were active but returned nothing,
    # try dropping the court filter first while keeping the date filter. Court
    # metadata is not always populated consistently, so some articles about a
    # specific court may be missing the court field entirely.
    if court_filter is not None and date_filter is not None:
        date_only_response = hybrid_query(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=candidate_top_k,
            alpha=source_alpha,
            metadata_filter=date_filter,
            source=source,
        )

        date_only_chunks = []

        for match in get_match_value(date_only_response, "matches", []) or []:
            chunk = format_match(match, source=source)

            if chunk is None:
                continue

            if document_matches_date_filter(chunk, from_date, to_date):
                date_only_chunks.append(chunk)

        date_only_chunks = rerank_chunks_by_story(
            cleaned_query,
            dedupe_chunks(date_only_chunks),
        )

        if date_only_chunks:
            return date_only_chunks[:search_top_k]

    # If Pinecone returns no filtered matches, run one wider search and apply the
    # same date check in Python. This helps while a namespace is still being rebuilt.
    fallback_response = hybrid_query(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=max(candidate_top_k, settings.date_fallback_top_k),
        alpha=source_alpha,
        source=source,
    )

    fallback_chunks = []

    for match in get_match_value(fallback_response, "matches", []) or []:
        chunk = format_match(match, source=source)

        if chunk is None:
            continue

        if document_matches_date_filter(chunk, from_date, to_date):
            fallback_chunks.append(chunk)

    return rerank_chunks_by_story(
        cleaned_query,
        dedupe_chunks(fallback_chunks),
    )[:search_top_k]


def retrieve_chunks_fan_out(
    query: str,
    query_variants: list[str],
    top_k: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    source: str | None = None,
    entities: list[str] | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve chunks using the primary query plus alternate query variants.

    Each variant is searched independently at a reduced candidate pool, then
    all results are merged, deduplicated by chunk ID (keeping the best score),
    and re-ranked together. This improves recall when the primary query misses
    articles using different legal terminology for the same concept.

    Example:
    Primary: "ED PMLA bail Supreme Court"
    Variant 1: "Enforcement Directorate money laundering bail order"
    Variant 2: "Prevention of Money Laundering Act arrest bail judgment"
    → merged pool covers both abbreviations and full-form phrasing.
    """
    search_top_k = clamp_top_k(top_k)
    main_candidate_top_k = clamp_top_k(
        max(search_top_k, settings.rerank_candidate_top_k)
    )
    variant_candidate_top_k = max(
        int(main_candidate_top_k * settings.multi_query_variant_top_k_fraction),
        settings.min_retrieval_top_k,
    )

    # Build list of (query_text, candidate_k) pairs: primary first, then variants
    query_jobs: list[tuple[str, int]] = [(clean_text(query), main_candidate_top_k)]
    for variant in query_variants:
        variant_clean = clean_text(variant)
        if variant_clean:
            query_jobs.append((variant_clean, variant_candidate_top_k))

    def _run_query(job: tuple[str, int]) -> list[RetrievedChunk]:
        q, k = job
        return retrieve_chunks(
            query=q,
            top_k=k,
            from_date=from_date,
            to_date=to_date,
            source=source,
            entities=entities,
        )

    all_chunks: list[RetrievedChunk] = []
    with ThreadPoolExecutor(max_workers=len(query_jobs)) as executor:
        futures = {executor.submit(_run_query, job): job for job in query_jobs}
        for future in as_completed(futures):
            try:
                all_chunks.extend(future.result())
            except Exception as exc:
                print(f"Warning: fan-out query failed: {exc}")

    # Dedupe by chunk ID, keeping the copy with the best retrieval_score
    best_by_id: dict[str, RetrievedChunk] = {}
    for chunk in all_chunks:
        existing = best_by_id.get(chunk.id)
        if existing is None:
            best_by_id[chunk.id] = chunk
        elif (chunk.retrieval_score or 0.0) > (existing.retrieval_score or 0.0):
            best_by_id[chunk.id] = chunk

    merged_chunks = list(best_by_id.values())

    # Re-rank the merged pool using the primary query as the ranking signal
    reranked = rerank_chunks_by_story(
        clean_text(query),
        merged_chunks,
    )

    return reranked[:search_top_k]
