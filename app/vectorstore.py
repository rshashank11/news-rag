from functools import lru_cache
from typing import Any

from pinecone.grpc import PineconeGRPC as Pinecone

from app.config import settings


def scale_dense_vector(vector: list[float], weight: float) -> list[float]:
    return [
        value * weight
        for value in vector
    ]


def scale_sparse_vector(sparse_vector: dict, weight: float) -> dict:
    return {
        "indices": sparse_vector["indices"],
        "values": [
            value * weight
            for value in sparse_vector["values"]
        ],
    }


def get_hybrid_weights(alpha: float | None = None) -> tuple[float, float]:
    dense_weight = settings.hybrid_alpha if alpha is None else alpha

    if not 0 <= dense_weight <= 1:
        raise ValueError("Hybrid alpha must be between 0 and 1.")

    sparse_weight = 1 - dense_weight
    return dense_weight, sparse_weight


@lru_cache(maxsize=4)
def get_index(source: str | None = None):
    source_config = settings.news_source_config(source)
    pc = Pinecone(api_key=settings.pinecone_api_key)

    if source_config.get("pinecone_index_host"):
        return pc.Index(host=source_config["pinecone_index_host"])

    return pc.Index(source_config["pinecone_index_name"])


def hybrid_query(
    dense_vector: list[float],
    sparse_vector: dict,
    top_k: int | None = None,
    alpha: float | None = None,
    metadata_filter: dict | None = None,
    source: str | None = None,
) -> Any:
    source_config = settings.news_source_config(source)
    dense_weight, sparse_weight = get_hybrid_weights(alpha)
    query_kwargs = {
        "vector": scale_dense_vector(dense_vector, dense_weight),
        "sparse_vector": scale_sparse_vector(sparse_vector, sparse_weight),
        "top_k": top_k or settings.retrieval_top_k,
        "namespace": source_config["pinecone_namespace"],
        "include_metadata": True,
    }

    if metadata_filter is not None:
        query_kwargs["filter"] = metadata_filter

    return get_index(source_config["source"]).query(**query_kwargs)
