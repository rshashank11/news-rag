import os  # Reads optional API keys and settings from environment variables.

import requests  # Used to call the optional Jina reranker API.
from langchain_core.documents import Document  # Standard LangChain object for retrieved text + metadata.
from opensearchpy import OpenSearch  # Low-level OpenSearch client used for keyword search.

from app.config import DEFAULT_SEARCH_K, OPENSEARCH_INDEX_NAME, OPENSEARCH_URL  # Shared retrieval configuration.
from app.vectorstore import get_vectorstore  # Creates LangChain OpenSearch vector store for semantic search.


docsearch = get_vectorstore()  # Vector-search object; used for semantic similarity search.

os_client = OpenSearch(  # Plain OpenSearch client; used for keyword/BM25-style search.
    hosts=[OPENSEARCH_URL],  # OpenSearch host URL.
    use_ssl=True,  # Uses HTTPS for hosted OpenSearch.
    verify_certs=True,  # Verifies SSL certificates.
    ssl_assert_hostname=False,  # Keeps compatibility with managed OpenSearch certs.
    ssl_show_warn=False,  # Hides noisy SSL warnings.
)

JINA_API_KEY = os.environ.get("JINA_API_KEY")  # Optional reranker key; if missing, reranking is skipped.


def document_matches_date_filter(  # Checks whether a retrieved document is inside the requested date range.
    document: Document,  # One retrieved chunk.
    from_date: str | None = None,  # Optional start date in YYYY-MM-DD format.
    to_date: str | None = None,  # Optional end date in YYYY-MM-DD format.
) -> bool:  # Returns True if the document passes the date filter.
    published_at = document.metadata.get("published_at")  # Reads publish date stored during ingestion.

    if not published_at:  # If the chunk has no date, we cannot prove it belongs in the range.
        return False  # Reject undated documents when a date filter is being applied.

    if from_date and published_at < from_date:  # String comparison works because dates are YYYY-MM-DD.
        return False  # Reject documents older than the requested start date.

    if to_date and published_at > to_date:  # Checks upper date bound.
        return False  # Reject documents newer than the requested end date.

    return True  # Document is inside the requested date range.


def semantic_search(  # Finds chunks by meaning, not just exact words.
    query: str,  # Search text produced by the planner.
    k: int = DEFAULT_SEARCH_K,  # Number of documents requested.
    from_date: str | None = None,  # Optional start date filter.
    to_date: str | None = None,  # Optional end date filter.
) -> list[Document]:  # Returns LangChain Document objects.
    try:  # Retrieval should fail gracefully, not crash the whole chatbot.
        search_k = max(k * 3, 30) if from_date or to_date else k  # Over-fetch when filtering dates because vector store filtering is manual here.
        documents = docsearch.similarity_search(query, k=search_k)  # Runs vector similarity search in OpenSearch.
        for rank, document in enumerate(documents, start=1):  # Adds rank metadata for debugging and hybrid scoring.
            document.metadata["search_type"] = "semantic"  # Marks this result as semantic/vector search.
            document.metadata["semantic_rank"] = rank  # Stores semantic rank for traceability.

        if from_date or to_date:  # Only apply manual date filtering when the user asked for dates.
            documents = [  # Keeps only documents that pass the date guardrail.
                document  # Current retrieved document.
                for document in documents  # Loops over vector results.
                if document_matches_date_filter(document, from_date, to_date)  # True means date is acceptable.
            ]

        documents = documents[:k]  # Trims back to requested count after filtering.

        return documents  # Sends semantic results back to hybrid retrieval.
    except Exception as exc:  # Catches OpenSearch/embedding errors.
        print(f"Semantic search failed: {exc}")  # Logs the failure for debugging.
        return []  # Returns empty list so keyword search can still work.


def build_date_range_filter(  # Builds an OpenSearch filter for keyword search dates.
    from_date: str | None = None,  # Optional start date.
    to_date: str | None = None,  # Optional end date.
) -> dict | None:  # Returns an OpenSearch filter dictionary or None.
    date_range = {}  # Empty range that we fill only if dates are provided.

    if from_date:  # Checks if a start date exists.
        date_range["gte"] = from_date  # gte means "greater than or equal to" this date.

    if to_date:  # Checks if an end date exists.
        date_range["lte"] = to_date  # lte means "less than or equal to" this date.

    if not date_range:  # If no dates were provided, there is no filter to build.
        return None  # Keyword search will run without a date filter.

    return {  # OpenSearch query syntax for a range filter.
        "range": {  # Range means compare a field against min/max values.
            "metadata.published_at": date_range,  # Field where ingestion stored the publish date.
        }
    }


def keyword_search(  # Finds chunks using exact-ish word matching in OpenSearch.
    query: str,  # Search text.
    k: int = DEFAULT_SEARCH_K,  # Number of keyword results to request.
    from_date: str | None = None,  # Optional start date.
    to_date: str | None = None,  # Optional end date.
) -> list[Document]:  # Returns matching chunks as LangChain Documents.
    filters = []  # OpenSearch filters that do not affect text relevance score.
    date_filter = build_date_range_filter(from_date, to_date)  # Builds date filter if needed.

    if date_filter:  # If a date filter exists...
        filters.append(date_filter)  # ...include it in the OpenSearch bool query.

    try:  # OpenSearch can fail due to connection/index problems, so catch errors.
        response = os_client.search(  # Sends a search request to OpenSearch.
            index=OPENSEARCH_INDEX_NAME,  # Index containing our text chunks.
            body={  # OpenSearch query body.
                "query": {  # Main query object.
                    "bool": {  # Combines text matching and filters.
                        "must": [  # Conditions that must match and affect score.
                            {  # One text-matching condition.
                                "match": {  # Match query does keyword-style text relevance.
                                    "text": query,  # Searches the stored chunk text field.
                                }
                            }
                        ],
                        "filter": filters,  # Date filters go here if present.
                    }
                },
                "size": k,  # Maximum number of hits to return.
            },
        )
    except Exception as exc:  # Handles OpenSearch errors gracefully.
        print(f"Keyword search failed: {exc}")  # Logs the failure.
        return []  # Lets semantic search still contribute.

    documents = []  # Will hold converted LangChain Document objects.

    for hit in response.get("hits", {}).get("hits", []):  # Loops over OpenSearch hits safely.
        source = hit.get("_source", {})  # Actual stored document content from OpenSearch.
        metadata = source.get("metadata") or {}  # Metadata stored with the chunk during ingestion.
        metadata["search_type"] = "keyword"  # Marks this result as keyword search.
        metadata["keyword_score"] = hit.get("_score")  # Stores OpenSearch relevance score.

        documents.append(  # Converts OpenSearch hit into LangChain Document.
            Document(  # Standard text+metadata wrapper.
                page_content=source.get("text", ""),  # Chunk text used as retrieved context.
                metadata=metadata,  # Story id, headline, publish date, rank info, etc.
            )
        )

    return documents  # Returns keyword results.


def dedupe_documents(documents: list[Document]) -> list[Document]:  # Removes duplicate chunks from combined search results.
    seen_keys = set()  # Tracks which story/chunk combinations we have already kept.
    unique_documents = []  # Final list with duplicates removed.

    for document in documents:  # Checks each retrieved document in order.
        story_id = document.metadata.get("story_id")  # Original story UUID from metadata.
        chunk_index = document.metadata.get("chunk_index")  # Chunk number inside that story.
        dedupe_key = (  # Unique identity for this result.
            (story_id, chunk_index)  # Best key when story ID exists.
            if story_id is not None  # Use story/chunk ID when available.
            else document.page_content[:160]  # Fallback key when metadata is missing.
        )

        if dedupe_key in seen_keys:  # If we already kept this chunk...
            continue  # ...skip it so sources are not repeated.

        seen_keys.add(dedupe_key)  # Remember this chunk as already seen.
        unique_documents.append(document)  # Keep this document.

    return unique_documents  # Returns de-duplicated results in original order.


def document_key(document: Document) -> tuple[str | None, int | None] | str:  # Creates a stable key for scoring/merging.
    story_id = document.metadata.get("story_id")  # Story UUID from ingestion.
    chunk_index = document.metadata.get("chunk_index")  # Chunk number from ingestion.

    if story_id is not None and chunk_index is not None:  # Best case: we know exactly which chunk this is.
        return story_id, chunk_index  # Tuple uniquely identifies the chunk.

    return document.page_content[:160]  # Fallback key based on text if metadata is incomplete.


def combine_hybrid_results(  # Combines keyword and semantic results into one ranking.
    keyword_documents: list[Document],  # Results from keyword search.
    semantic_documents: list[Document],  # Results from vector/semantic search.
) -> list[Document]:  # Returns one ranked result list.
    documents_by_key = {}  # Maps each unique document key to one Document object.
    scores_by_key = {}  # Maps each unique document key to its combined hybrid score.

    for result_list in (keyword_documents, semantic_documents):  # Processes both retrieval lists.
        for rank, document in enumerate(result_list, start=1):  # Rank starts at 1 because first result is strongest.
            key = document_key(document)  # Finds the stable identity of this chunk.

            if key not in documents_by_key:  # If this chunk has not appeared before...
                documents_by_key[key] = document  # Store the document once.
                scores_by_key[key] = 0.0  # Start its combined score at zero.

            scores_by_key[key] += 1 / (60 + rank)  # Reciprocal Rank Fusion: higher-ranked docs get more points.

    combined_documents = list(documents_by_key.values())  # Converts dictionary values into a list.
    combined_documents.sort(  # Sorts final documents by hybrid score.
        key=lambda document: scores_by_key[document_key(document)],  # Looks up each document's combined score.
        reverse=True,  # Highest score should come first.
    )

    for document in combined_documents:  # Adds the score to metadata for debugging/UI traces.
        document.metadata["hybrid_score"] = scores_by_key[document_key(document)]  # Saves final hybrid score.

    return combined_documents  # Returns merged ranking.


def rerank_documents(query: str, documents: list[Document], top_n: int) -> list[Document]:  # Optional final relevance sort.
    if not documents:  # No documents means there is nothing to rerank.
        return []  # Return empty list immediately.

    if not JINA_API_KEY:  # If no Jina key is configured...
        return documents[:top_n]  # ...skip reranking and keep the hybrid order.

    headers = {  # HTTP headers required by Jina API.
        "Authorization": f"Bearer {JINA_API_KEY}",  # Authenticates the request.
        "Content-Type": "application/json",  # Says we are sending JSON.
    }
    payload = {  # JSON body sent to the reranker.
        "model": "jina-reranker-v3",  # Reranker model name.
        "query": query,  # User/search query to compare documents against.
        "documents": [document.page_content for document in documents],  # Candidate texts to rerank.
        "top_n": min(top_n, len(documents)),  # Ask only for as many as we need.
    }

    try:  # External APIs can fail, so reranking must be optional.
        response = requests.post(  # Sends rerank request to Jina.
            "https://api.jina.ai/v1/rerank",  # Jina rerank endpoint.
            headers=headers,  # Auth and content type.
            json=payload,  # Request body.
            timeout=20,  # Prevents the app from hanging too long.
        )
        response.raise_for_status()  # Raises an error for non-2xx HTTP responses.
        rankings = response.json().get("results", [])  # Reads ranked result metadata from Jina response.

        reranked_documents = []  # Final list ordered by Jina.
        for item in rankings:  # Each item points back to one original document index.
            document = documents[item["index"]]  # Gets the original Document object.
            document.metadata["rerank_score"] = item.get("relevance_score")  # Stores Jina relevance score.
            reranked_documents.append(document)  # Adds document in reranked order.

        return reranked_documents  # Returns Jina-ranked documents.

    except Exception as exc:  # If reranker fails, retrieval should still work.
        print(f"Rerank failed: {exc}")  # Logs the reranker issue.
        return documents[:top_n]  # Falls back to hybrid ranking.


def search_news(  # Main retrieval function used by the LangGraph workflow.
    query: str,  # Search query from the planner or rewrite step.
    k: int = DEFAULT_SEARCH_K,  # Final number of documents wanted.
    from_date: str | None = None,  # Optional start date filter.
    to_date: str | None = None,  # Optional end date filter.
) -> list[Document]:  # Returns final retrieved documents.
    candidate_k = max(k * 4, 12)  # Fetch more candidates first so reranking/dedupe has room to work.

    keyword_documents = keyword_search(  # Exact-word retrieval branch.
        query,  # Same search query.
        k=candidate_k,  # Over-fetch candidates.
        from_date=from_date,  # Pass date filter through.
        to_date=to_date,  # Pass date filter through.
    )
    semantic_documents = semantic_search(  # Meaning-based retrieval branch.
        query,  # Same search query.
        k=candidate_k,  # Over-fetch candidates.
        from_date=from_date,  # Pass date filter through.
        to_date=to_date,  # Pass date filter through.
    )
    candidates = combine_hybrid_results(keyword_documents, semantic_documents)  # Merge keyword + semantic rankings.
    candidates = dedupe_documents(candidates)  # Remove duplicate chunks before final rerank.

    return rerank_documents(query, candidates, top_n=k)  # Return the final top-k retrieved sources.
