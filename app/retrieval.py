import os

import requests
from langchain_core.documents import Document
from opensearchpy import OpenSearch

from app.config import DEFAULT_SEARCH_K, OPENSEARCH_INDEX_NAME, OPENSEARCH_URL
from app.vectorstore import get_vectorstore


docsearch = get_vectorstore()

os_client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

JINA_API_KEY = os.environ.get("JINA_API_KEY")


def semantic_search(query: str, k: int = DEFAULT_SEARCH_K) -> list[Document]:
    try:
        documents = docsearch.similarity_search(query, k=k)
        for rank, document in enumerate(documents, start=1):
            document.metadata["search_type"] = "semantic"
            document.metadata["semantic_rank"] = rank

        return documents
    except Exception as exc:
        print(f"Semantic search failed: {exc}")
        return []


def keyword_search(query: str, k: int = DEFAULT_SEARCH_K) -> list[Document]:
    try:
        response = os_client.search(
            index=OPENSEARCH_INDEX_NAME,
            body={
                "query": {
                    "match": {
                        "text": query,
                    }
                },
                "size": k,
            },
        )
    except Exception as exc:
        print(f"Keyword search failed: {exc}")
        return []

    documents = []

    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        metadata = source.get("metadata") or {}
        metadata["search_type"] = "keyword"
        metadata["keyword_score"] = hit.get("_score")

        documents.append(
            Document(
                page_content=source.get("text", ""),
                metadata=metadata,
            )
        )

    return documents


def dedupe_documents(documents: list[Document]) -> list[Document]:
    seen_keys = set()
    unique_documents = []

    for document in documents:
        story_id = document.metadata.get("story_id")
        chunk_index = document.metadata.get("chunk_index")
        dedupe_key = (
            (story_id, chunk_index)
            if story_id is not None
            else document.page_content[:160]
        )

        if dedupe_key in seen_keys:
            continue

        seen_keys.add(dedupe_key)
        unique_documents.append(document)

    return unique_documents


def document_key(document: Document) -> tuple[str | None, int | None] | str:
    story_id = document.metadata.get("story_id")
    chunk_index = document.metadata.get("chunk_index")

    if story_id is not None and chunk_index is not None:
        return story_id, chunk_index

    return document.page_content[:160]


def combine_hybrid_results(
    keyword_documents: list[Document],
    semantic_documents: list[Document],
) -> list[Document]:
    documents_by_key = {}
    scores_by_key = {}

    for result_list in (keyword_documents, semantic_documents):
        for rank, document in enumerate(result_list, start=1):
            key = document_key(document)

            if key not in documents_by_key:
                documents_by_key[key] = document
                scores_by_key[key] = 0.0

            scores_by_key[key] += 1 / (60 + rank)

    combined_documents = list(documents_by_key.values())
    combined_documents.sort(
        key=lambda document: scores_by_key[document_key(document)],
        reverse=True,
    )

    for document in combined_documents:
        document.metadata["hybrid_score"] = scores_by_key[document_key(document)]

    return combined_documents


def rerank_documents(query: str, documents: list[Document], top_n: int) -> list[Document]:
    if not documents:
        return []

    if not JINA_API_KEY:
        return documents[:top_n]

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "jina-reranker-v3",
        "query": query,
        "documents": [document.page_content for document in documents],
        "top_n": min(top_n, len(documents)),
    }

    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers=headers,
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        rankings = response.json().get("results", [])

        reranked_documents = []
        for item in rankings:
            document = documents[item["index"]]
            document.metadata["rerank_score"] = item.get("relevance_score")
            reranked_documents.append(document)

        return reranked_documents

    except Exception as exc:
        print(f"Rerank failed: {exc}")
        return documents[:top_n]


def search_news(query: str, k: int = DEFAULT_SEARCH_K) -> list[Document]:
    candidate_k = max(k * 4, 12)

    keyword_documents = keyword_search(query, k=candidate_k)
    semantic_documents = semantic_search(query, k=candidate_k)
    candidates = combine_hybrid_results(keyword_documents, semantic_documents)
    candidates = dedupe_documents(candidates)

    return rerank_documents(query, candidates, top_n=k)
