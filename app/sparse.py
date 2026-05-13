from pathlib import Path
from functools import lru_cache

from pinecone_text.sparse import BM25Encoder

from app.config import settings


@lru_cache(maxsize=4)
def _load_bm25_encoder(source: str | None = None) -> BM25Encoder:
    source_config = settings.news_source_config(source)
    encoder_path = Path(source_config["bm25_encoder_path"])

    if not encoder_path.exists():
        raise RuntimeError(
            f"BM25 encoder file not found at {encoder_path}. "
            "Run ingestion first so the source-specific BM25 file is created."
        )

    return BM25Encoder().load(str(encoder_path))


def to_pinecone_sparse_values(sparse_vector: dict) -> dict:
    return {
        "indices": [int(index) for index in sparse_vector["indices"]],
        "values": [float(value) for value in sparse_vector["values"]],
    }


def encode_sparse_query(query: str, source: str | None = None) -> dict:
    bm25_encoder = _load_bm25_encoder(source)
    sparse_vector = bm25_encoder.encode_queries(query)
    return to_pinecone_sparse_values(sparse_vector)
