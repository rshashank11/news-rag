from pathlib import Path
from functools import lru_cache

from pinecone_text.sparse import BM25Encoder

from app.config import settings
from app.legal_tokenizer import apply_legal_tokenizer


def _load_bm25_encoder(source: str | None = None) -> BM25Encoder:
    """
    Load the BM25 keyword-search encoder for one source.

    BM25 needs a saved vocabulary from ingestion time.

    Example:
    Sakal and Bar & Bench have different words, so each source gets its own
    BM25 file instead of sharing one global encoder.
    """
    source_name = settings.news_source_config(source)["source"]
    return _load_bm25_encoder_cached(source_name)


@lru_cache(maxsize=4)
def _load_bm25_encoder_cached(source_name: str) -> BM25Encoder:
    source_config = settings.news_source_config(source_name)
    encoder_path = Path(source_config["bm25_encoder_path"])

    if not encoder_path.exists():
        raise RuntimeError(
            f"BM25 encoder file not found at {encoder_path}. "
            "Run ingestion first so the source-specific BM25 file is created."
        )

    # Bar & Bench encoder was fit with legal compound terms pre-processed.
    # We must load with the same configuration flags so vocabulary hashes match.
    return BM25Encoder(
        lower_case=True,
        remove_punctuation=True,
        remove_stopwords=True,
        stem=True,
        language="english",
    ).load(str(encoder_path))


def to_pinecone_sparse_values(sparse_vector: dict) -> dict:
    """
    Convert BM25 output into the shape Pinecone expects.

    Pinecone wants:
    - indices = word/token IDs
    - values = importance score for each word/token
    """
    return {
        "indices": [int(index) for index in sparse_vector["indices"]],
        "values": [float(value) for value in sparse_vector["values"]],
    }


def encode_sparse_query(query: str, source: str | None = None) -> dict:
    """
    Convert the search query into a BM25 sparse vector.

    BM25 helps exact words matter.

    Example:
    A query like "PMLA bail" should strongly match chunks containing
    the exact words "PMLA" and "bail".

    For Bar & Bench, legal compound terms are pre-processed so that
    "Supreme Court" becomes "supreme_court" before tokenization, matching
    the vocabulary built during ingestion.
    """
    source_config = settings.news_source_config(source)
    if source_config.get("source") == "barandbench":
        query = apply_legal_tokenizer(query)
    bm25_encoder = _load_bm25_encoder(source)
    sparse_vector = bm25_encoder.encode_queries(query)
    return to_pinecone_sparse_values(sparse_vector)
