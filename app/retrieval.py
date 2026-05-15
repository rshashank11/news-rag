import re
import requests

from app.config import settings
from app.embeddings import embed_text
from app.sparse import encode_sparse_query
from app.vectorstore import hybrid_query
from schemas import RetrievedChunk, clean_text


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
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

SAKAL_ENGLISH_QUERY_TERMS = {
    "additives": ["भेसळ", "भेसळयुक्त", "घटक"],
    "admission": ["प्रवेश"],
    "admissions": ["प्रवेश"],
    "agriculture": ["शेती", "कृषी"],
    "bribe": ["लाच", "लाचखोरी"],
    "bribery": ["लाच", "लाचखोरी"],
    "bus": ["बस", "एसटी"],
    "cleanliness": ["स्वच्छता"],
    "college": ["महाविद्यालय"],
    "colleges": ["महाविद्यालय"],
    "compensation": ["नुकसानभरपाई", "भरपाई", "मदत"],
    "consumer": ["ग्राहक"],
    "costlier": ["महाग", "भावात वाढ", "दर वाढ"],
    "crop": ["पीक", "पिके"],
    "crops": ["पीक", "पिके"],
    "damage": ["नुकसान"],
    "damselfly": ["डॅम्सेलफ्लाय", "किटक"],
    "damselflies": ["डॅम्सेलफ्लाय", "किटक"],
    "date": ["तारीख"],
    "development": ["विकास"],
    "diversity": ["विविधता"],
    "dragonfly": ["ड्रॅगनफ्लाय", "किटक"],
    "dragonflies": ["ड्रॅगनफ्लाय", "किटक"],
    "education": ["शिक्षण"],
    "fee": ["शुल्क"],
    "farmers": ["शेतकरी", "शेतकऱ्यांना"],
    "fake": ["बनावट", "निकृष्ट"],
    "free": ["विनामूल्य"],
    "ghats": ["घाट", "पश्चिम घाट"],
    "hygiene": ["स्वच्छता", "अस्वच्छता"],
    "illness": ["आजार", "लक्षणे"],
    "insects": ["किटक"],
    "jobs": ["नोकरी", "भरती"],
    "kalewadi": ["काळेवाडी"],
    "mango": ["मँगो", "आंबा"],
    "market": ["मार्केट", "बाजार", "मार्केटयार्ड"],
    "money": ["पैसे", "लाच"],
    "online": ["ऑनलाइन"],
    "parents": ["पालक"],
    "peas": ["मटार"],
    "pimpri": ["पिंपरी"],
    "portal": ["पोर्टल"],
    "price": ["भाव", "दर"],
    "prices": ["भाव", "दर"],
    "pulp": ["पल्प", "गर"],
    "rain": ["पाऊस", "अवकाळी पाऊस"],
    "rainfall": ["पाऊस", "पावसामुळे"],
    "report": ["अहवाल", "बातमी"],
    "research": ["संशोधन"],
    "rte": ["आरटीई"],
    "school": ["शाळा"],
    "schools": ["शाळा"],
    "st": ["एसटी"],
    "stand": ["स्थानक", "बसस्थानक"],
    "stands": ["स्थानक", "बसस्थानक"],
    "study": ["अभ्यास", "संशोधन"],
    "swargate": ["स्वारगेट"],
    "symptoms": ["लक्षणे"],
    "tomato": ["टोमॅटो"],
    "tomatoes": ["टोमॅटो"],
    "transparent": ["पारदर्शक"],
    "truckloads": ["ट्रक", "आवक"],
    "vegetable": ["भाजी", "भाज्या"],
    "vegetables": ["भाजी", "भाज्या"],
    "wakdewadi": ["वाकडेवाडी"],
    "watermelon": ["कलिंगड"],
    "western": ["पश्चिम"],
    "workshop": ["कार्यशाळा"],
}

SAKAL_ENGLISH_PHRASE_TERMS = {
    "market yard": ["मार्केटयार्ड"],
    "pune market yard": ["पुणे", "मार्केटयार्ड"],
    "rte admission": ["आरटीई", "प्रवेश"],
    "rte admissions": ["आरटीई", "प्रवेश"],
    "pimpri chinchwad": ["पिंपरी", "चिंचवड"],
    "western ghats": ["पश्चिम घाट"],
    "mango pulp": ["मँगो पल्प", "आंबा गर"],
    "cleanliness fee": ["स्वच्छता शुल्क"],
    "bus stand": ["बसस्थानक"],
    "bus stands": ["बसस्थानक"],
}


def clamp_top_k(top_k: int | None) -> int:
    requested_top_k = settings.retrieval_top_k if top_k is None else top_k
    return min(
        max(requested_top_k, settings.min_retrieval_top_k),
        settings.max_retrieval_top_k,
    )


def iso_date_to_yyyymmdd(value: str | None) -> int | None:
    if value is None:
        return None

    return int(value.replace("-", ""))


def build_pinecone_date_filter(
    from_date: str | None = None,
    to_date: str | None = None,
    source: str | None = None,
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

    source_name = settings.news_source_config(source)["source"]
    date_field = (
        "published_at_yyyymmdd"
        if source_name == "barandbench"
        else "date_published_yyyymmdd"
    )

    return {date_field: date_filter}


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


def is_probably_english_query(query: str) -> bool:
    return len(DEVANAGARI_PATTERN.findall(query or "")) < 2


def expand_sakal_english_query(query: str, source: str | None = None) -> str:
    source_name = settings.news_source_config(source)["source"]

    if source_name != "sakal" or not is_probably_english_query(query):
        return query

    lowered_query = query.lower()
    expanded_terms: list[str] = []
    seen = set()

    def add_terms(terms: list[str]) -> None:
        for term in terms:
            key = term.lower()
            if key in seen:
                continue

            seen.add(key)
            expanded_terms.append(term)

    for phrase, terms in SAKAL_ENGLISH_PHRASE_TERMS.items():
        if phrase in lowered_query:
            add_terms(terms)

    for token in TOKEN_PATTERN.findall(lowered_query):
        if token in STOPWORDS:
            continue

        terms = SAKAL_ENGLISH_QUERY_TERMS.get(token)
        if terms:
            add_terms(terms)

    if not expanded_terms:
        return query

    return clean_text(" ".join([query, *expanded_terms]))


def metadata_categories_for_source(
    metadata: dict,
    source: str | None,
) -> list[str]:
    source_name = settings.news_source_config(source)["source"]

    if source_name == "barandbench":
        return list_metadata_values(metadata.get("categories"))

    return list_metadata_values(
        [
            metadata.get("edition"),
            metadata.get("source"),
            metadata.get("location"),
        ]
    )


def format_match(match, source: str | None = None) -> RetrievedChunk | None:
    metadata = get_match_metadata(match)
    chunk_text = metadata.get("chunk_text")
    source_name = settings.news_source_config(source)["source"]

    if not chunk_text:
        return None

    if source_name == "barandbench":
        story_id = metadata.get("story_id")
        published_at = metadata.get("published_at")
        topics = list_metadata_values(metadata.get("topics"))
    else:
        story_id = metadata.get("article_id")
        published_at = metadata.get("date_published")
        topics = list_metadata_values(metadata.get("keywords"))

    return RetrievedChunk(
        id=str(get_match_value(match, "id", "")),
        story_id=story_id,
        chunk_index=metadata.get("chunk_index"),
        headline=metadata.get("headline") or "Untitled",
        published_at=published_at,
        topics=topics,
        categories=metadata_categories_for_source(metadata, source),
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


# -------------------------------------------------------------------
# OLD CUSTOM RERANKER - COMMENTED OUT
# -------------------------------------------------------------------
# We are no longer actively using this heuristic reranker.
#
# It used:
# - Pinecone vector/hybrid score
# - headline keyword overlap
# - metadata keyword overlap
# - chunk text keyword overlap
# - date recency
# - same-story evidence bonus
#
# We are replacing that active reranking logic with Jina reranker below.
#
# from datetime import date
#
#
# def token_set(text: str) -> set[str]:
#     min_length = settings.rerank_min_token_length
#
#     return {
#         token
#         for token in TOKEN_PATTERN.findall(text.lower())
#         if len(token) >= min_length and token not in STOPWORDS
#     }
#
#
# def overlap_ratio(query_tokens: set[str], text: str) -> float:
#     if not query_tokens:
#         return 0.0
#
#     text_tokens = token_set(text)
#
#     if not text_tokens:
#         return 0.0
#
#     return len(query_tokens & text_tokens) / len(query_tokens)
#
#
# def parse_chunk_date(chunk: RetrievedChunk) -> date | None:
#     if not chunk.published_at:
#         return None
#
#     try:
#         return date.fromisoformat(chunk.published_at)
#     except ValueError:
#         return None
#
#
# def normalize_float(value: float | None, minimum: float, maximum: float) -> float:
#     if value is None:
#         return 0.0
#
#     if maximum <= minimum:
#         return 1.0
#
#     return (value - minimum) / (maximum - minimum)
#
#
# def normalize_chunk_date(
#     chunk_date: date | None,
#     earliest_date: date | None,
#     latest_date: date | None,
# ) -> float:
#     if chunk_date is None or earliest_date is None or latest_date is None:
#         return 0.0
#
#     total_days = (latest_date - earliest_date).days
#
#     if total_days <= 0:
#         return 0.0
#
#     return (chunk_date - earliest_date).days / total_days
#
#
# def rerank_chunk(
#     chunk: RetrievedChunk,
#     query_tokens: set[str],
#     normalized_vector_score: float,
#     normalized_date_score: float,
# ) -> RetrievedChunk:
#     metadata_text = " ".join([*chunk.topics, *chunk.categories])
#
#     rerank_score = (
#         normalized_vector_score * settings.rerank_vector_score_weight
#         + overlap_ratio(query_tokens, chunk.headline) * settings.rerank_headline_overlap_weight
#         + overlap_ratio(query_tokens, metadata_text) * settings.rerank_metadata_overlap_weight
#         + overlap_ratio(query_tokens, chunk.chunk_text) * settings.rerank_chunk_overlap_weight
#         + normalized_date_score * settings.rerank_date_score_weight
#     )
#
#     return chunk.model_copy(update={"rerank_score": round(rerank_score, 6)})


# -------------------------------------------------------------------
# JINA RERANKER
# -------------------------------------------------------------------


def story_key(chunk: RetrievedChunk) -> str:
    return chunk.story_id or chunk.id


def chunk_to_jina_document(chunk: RetrievedChunk) -> str:
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
) -> list[RetrievedChunk]:
    cleaned_query = clean_text(query)
    retrieval_query = expand_sakal_english_query(cleaned_query, source=source)

    if not cleaned_query:
        raise ValueError("Search query cannot be empty.")

    search_top_k = clamp_top_k(top_k)
    candidate_top_k = clamp_top_k(
        max(search_top_k, settings.rerank_candidate_top_k)
    )

    dense_vector = embed_text(retrieval_query)
    sparse_vector = encode_sparse_query(retrieval_query, source=source)
    date_filter = build_pinecone_date_filter(
        from_date,
        to_date,
        source=source,
    )
    metadata_filter = date_filter

    response = hybrid_query(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=candidate_top_k,
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
        retrieval_query,
        dedupe_chunks(chunks),
    )

    if chunks or metadata_filter is None:
        return chunks[:search_top_k]

    # If Pinecone returns no filtered matches, run one wider search and apply the
    # same date check in Python. This helps while a namespace is still being rebuilt.
    fallback_response = hybrid_query(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=max(candidate_top_k, settings.date_fallback_top_k),
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
        retrieval_query,
        dedupe_chunks(fallback_chunks),
    )[:search_top_k]
