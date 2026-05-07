from pathlib import Path

from pinecone_text.sparse import BM25Encoder

from app.config import settings


def _load_bm25_encoder() -> BM25Encoder:
    encoder_path = Path(settings.bm25_encoder_path)

    if not encoder_path.exists():
        raise RuntimeError(
            f"BM25 encoder file not found at {encoder_path}. "
            "Run ingestion first so bm25_values.json is created."
        )

    return BM25Encoder().load(str(encoder_path))


bm25_encoder = _load_bm25_encoder()


def to_pinecone_sparse_values(sparse_vector: dict) -> dict:
    return {
        "indices": [int(index) for index in sparse_vector["indices"]],
        "values": [float(value) for value in sparse_vector["values"]],
    }


def encode_sparse_query(query: str) -> dict:
    sparse_vector = bm25_encoder.encode_queries(query)
    return to_pinecone_sparse_values(sparse_vector)
