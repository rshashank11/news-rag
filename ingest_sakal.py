import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from app.embeddings import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    clean_text_part,
    embed_texts,
)
from app.sparse import to_pinecone_sparse_values


SAKAL_JSON_PATH = Path("sakal.json")
SAKAL_INDEX_NAME = "sakal"
SAKAL_NAMESPACE = "sakal_v1"
SAKAL_INDEX_DIMENSION = 1536
SAKAL_INDEX_METRIC = "dotproduct"
SAKAL_INDEX_VECTOR_TYPE = "dense"
SAKAL_CHUNK_SIZE_WORDS = 500
SAKAL_CHUNK_OVERLAP_WORDS = 50
SAKAL_BM25_ENCODER_PATH = Path(
    os.environ.get("SAKAL_BM25_ENCODER_PATH", "bm25_sakal_values.json")
)


def load_articles(json_path: Path, limit: int | None = None) -> list[dict]:
    """
    Load Sakal articles from the converted JSON file.

    Example:
    convert_sakal_xml_to_json.py creates sakal.json.
    This function reads sakal.json so ingestion can turn articles into vectors.

    limit is useful for testing ingestion on the first few articles only.
    """
    with json_path.open("r", encoding="utf-8") as file:
        articles = json.load(file)

    if limit is not None:
        return articles[:limit]

    return articles


def load_or_fit_sakal_bm25_encoder(
    embedding_texts: list[str],
    encoder_path: Path,
    rebuild: bool = False,
) -> BM25Encoder:
    """
    Load or create the Sakal BM25 keyword-search encoder.

    BM25 learns which words exist in the Sakal chunks.

    Example:
    If we change how chunks are built, we should use --rebuild-bm25 so the saved
    keyword vocabulary matches the new chunk text.
    """
    if encoder_path.exists() and not rebuild:
        return BM25Encoder().load(str(encoder_path))

    encoder = BM25Encoder().default()
    encoder.fit(embedding_texts)
    encoder.dump(str(encoder_path))
    return encoder


def build_sakal_chunk_embedding_text(
    article: dict,
    chunk_text: str,
) -> str:
    """
    Build the exact text that gets embedded and stored for one Sakal chunk.

    We include labels like "Headline", "Keywords", and "Body chunk" so the
    embedding model gets clearer context.

    Example:
    A body chunk may not say "Pune Market Yard", but the headline might.
    Repeating the headline helps that chunk still match Pune Market Yard queries.
    """
    headline = clean_text_part(article.get("headline"))
    keywords = article.get("keywords") or []
    keyword_text = ", ".join(clean_text_part(keyword) for keyword in keywords)
    chunk_text = clean_text_part(chunk_text)

    labeled_parts = [
        ("Headline", headline),
        ("Keywords", keyword_text),
        ("Body chunk", chunk_text),
    ]

    parts = [
        f"{label}: {text}"
        for label, text in labeled_parts
        if text
    ]

    return "\n\n".join(parts)


def split_sakal_body_into_chunks(
    text: str,
    chunk_size_words: int = SAKAL_CHUNK_SIZE_WORDS,
    overlap_words: int = SAKAL_CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Split a Sakal article body into smaller word chunks.

    Why chunk?
    A full article can be too large or too broad for one embedding.

    Why overlap?
    If important context is split between two chunks, overlap keeps a little
    repeated text so neither chunk loses the meaning.

    Example:
    chunk_size_words=500 and overlap_words=50 means:
    - chunk 1: words 1-500
    - chunk 2: words 451-950
    """
    words = clean_text_part(text).split()

    if not words:
        return []

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than 0.")

    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")

    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words.")

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - overlap_words

    return chunks


def sakal_article_body_chunks(article: dict) -> list[str]:
    """
    Return all body chunks for one Sakal article.

    Example:
    A short article may return 1 chunk.
    A long article may return several overlapping chunks.
    """
    body = clean_text_part(article.get("body"))
    return split_sakal_body_into_chunks(body)


def sakal_article_chunk_records(
    articles: list[dict],
) -> list[tuple[dict, str, int, str]]:
    """
    Convert many articles into a flat list of chunk records.

    Example output item:
    (article_dict, "PNE26Y81513", 0, "first chunk text")

    We need this shape because each article can become multiple Pinecone vectors.
    """
    records = []

    for article in articles:
        article_id = clean_text_part(article.get("id"))

        if not article_id:
            continue

        for chunk_index, chunk_text in enumerate(sakal_article_body_chunks(article)):
            records.append((article, article_id, chunk_index, chunk_text))

    return records


def sakal_article_metadata(
    article: dict,
    chunk_index: int | None = None,
    chunk_text: str | None = None,
) -> dict:
    """
    Build the metadata stored next to each Sakal vector in Pinecone.

    Metadata is not the vector itself. It is searchable/display information.

    Example:
    - article_id helps group chunks from the same article.
    - date_published_yyyymmdd helps Pinecone filter by date.
    - headline is shown back to the user as source information.
    """
    raw = article.get("raw") or {}
    metadata = {
        "article_id": article.get("id"),
        "headline": article.get("headline"),
        "date_published": article.get("date_published"),
        "date_published_yyyymmdd": article.get("date_published_yyyymmdd"),
        "date_created": article.get("date_created"),
        "date_created_yyyymmdd": article.get("date_created_yyyymmdd"),
        "location": article.get("location"),
        "edition": article.get("edition"),
        "source": article.get("source"),
        "language": article.get("language"),
        "keywords": article.get("keywords") or [],
        "xml_id": raw.get("xml_id"),
        "chunk_index": chunk_index,
        "chunk_text": clean_text_part(chunk_text),
    }

    return {
        key: value
        for key, value in metadata.items()
        if value not in (None, "", [])
    }


def create_sakal_index_if_needed(
    pc: Pinecone,
    index_name: str,
    cloud: str,
    region: str,
) -> None:
    """
    Create the Sakal Pinecone index if it is missing.

    Example:
    The first ingestion run may need to create the index.
    Later runs should reuse the existing index.
    """
    if pc.has_index(index_name):
        return

    pc.create_index(
        name=index_name,
        dimension=SAKAL_INDEX_DIMENSION,
        metric=SAKAL_INDEX_METRIC,
        vector_type=SAKAL_INDEX_VECTOR_TYPE,
        spec=ServerlessSpec(
            cloud=cloud,
            region=region,
        ),
    )


def build_sakal_vectors(
    article_chunk_batch: list[tuple[dict, str, int, str]],
    bm25_encoder: BM25Encoder,
) -> list[dict]:
    """
    Convert Sakal chunks into Pinecone upload objects.

    Each Pinecone vector contains:
    - id: unique chunk ID
    - values: dense/semantic embedding
    - sparse_values: BM25 keyword vector
    - metadata: headline, dates, article ID, chunk text

    This lets retrieval use both meaning and exact Marathi keywords.
    """
    embedding_texts = [
        build_sakal_chunk_embedding_text(article, chunk_text)
        for article, _, _, chunk_text in article_chunk_batch
    ]

    dense_embeddings = embed_texts(embedding_texts)

    sparse_vectors = [
        to_pinecone_sparse_values(bm25_encoder.encode_documents(embedding_text))
        for embedding_text in embedding_texts
    ]

    vectors = []

    for (
        article,
        article_id,
        chunk_index,
        chunk_text,
    ), dense_embedding, sparse_vector in zip(
        article_chunk_batch,
        dense_embeddings,
        sparse_vectors,
    ):
        vector = {
            "id": f"{article_id}-{chunk_index}",
            "values": dense_embedding,
            "sparse_values": sparse_vector,
            "metadata": sakal_article_metadata(
                article,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
            ),
        }
        vectors.append(vector)

    return vectors


def batched(items: list, batch_size: int):
    """
    Yield a large list in smaller batches.

    Example:
    If there are 1,000 chunks and batch_size=100,
    this yields 10 batches of 100 chunks.
    """
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def ingest_sakal(
    json_path: Path,
    index_name: str,
    namespace: str,
    batch_size: int,
    limit: int | None,
    dry_run: bool,
    rebuild_bm25: bool,
    start_batch: int,
) -> None:
    """
    Run the full Sakal ingestion pipeline.

    What happens:
    1. load converted JSON articles,
    2. split each article into chunks,
    3. turn each chunk into dense + sparse vectors,
    4. upload those vectors to Pinecone.

    Example:
    One Sakal article with a long body may become 3 Pinecone vectors.
    """
    load_dotenv()

    api_key = os.environ.get("PINECONE_API_KEY")
    cloud = os.environ.get("PINECONE_CLOUD", "aws")
    region = os.environ.get("PINECONE_REGION", "us-east-1")

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing.")

    if start_batch < 1:
        raise ValueError("--start-batch must be 1 or greater.")

    articles = load_articles(json_path, limit=limit)

    article_chunks = sakal_article_chunk_records(articles)

    embedding_texts = [
        build_sakal_chunk_embedding_text(article, chunk_text)
        for article, _, _, chunk_text in article_chunks
    ]

    print(f"Loaded {len(articles)} articles.")
    print(f"Prepared {len(article_chunks)} article chunks.")
    print(f"Index: {index_name}")
    print(f"Dimension: {SAKAL_INDEX_DIMENSION}")
    print(f"Metric: {SAKAL_INDEX_METRIC}")
    print(f"Vector type: {SAKAL_INDEX_VECTOR_TYPE}")
    print(f"Namespace: {namespace}")

    if dry_run:
        for article, article_id, chunk_index, chunk_text in article_chunks[:3]:
            preview = build_sakal_chunk_embedding_text(article, chunk_text)
            print(f"DRY RUN {article_id}-{chunk_index}: {preview[:300]}")
        return

    bm25_encoder = load_or_fit_sakal_bm25_encoder(
        embedding_texts=embedding_texts,
        encoder_path=SAKAL_BM25_ENCODER_PATH,
        rebuild=rebuild_bm25,
    )

    pc = Pinecone(api_key=api_key)
    create_sakal_index_if_needed(
        pc=pc,
        index_name=index_name,
        cloud=cloud,
        region=region,
    )
    index = pc.Index(index_name)

    uploaded = min((start_batch - 1) * batch_size, len(article_chunks))

    for batch_number, article_chunk_batch in enumerate(
        batched(article_chunks, batch_size),
        start=1,
    ):
        if batch_number < start_batch:
            continue

        vectors = build_sakal_vectors(
            article_chunk_batch=article_chunk_batch,
            bm25_encoder=bm25_encoder,
        )
        index.upsert(vectors=vectors, namespace=namespace)
        uploaded += len(vectors)
        print(f"Uploaded batch {batch_number}: {uploaded}/{len(article_chunks)}")

    print(f"Uploaded {uploaded} vectors to Pinecone index '{index_name}'.")


def parse_args():
    """
    Parse command-line options for Sakal ingestion.

    Example:
    python ingest_sakal.py --limit 10 --dry-run
    previews the first 10 articles without writing to Pinecone.
    """
    parser = argparse.ArgumentParser(
        description="Embed Sakal XML-converted JSON and store it in Pinecone."
    )
    parser.add_argument("--json-file", type=Path, default=SAKAL_JSON_PATH)
    parser.add_argument("--index-name", default=SAKAL_INDEX_NAME)
    parser.add_argument(
        "--namespace",
        default=os.environ.get("SAKAL_PINECONE_NAMESPACE", SAKAL_NAMESPACE),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-bm25", action="store_true")
    parser.add_argument("--start-batch", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_sakal(
        json_path=args.json_file,
        index_name=args.index_name,
        namespace=args.namespace,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
        rebuild_bm25=args.rebuild_bm25,
        start_batch=args.start_batch,
    )
