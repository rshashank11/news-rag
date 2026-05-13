import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from app.embeddings import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    article_chunk_records,
    article_metadata,
    build_article_chunk_embedding_text,
    embed_texts,
)


SAKAL_JSON_PATH = Path("sakal.json")
SAKAL_INDEX_NAME = "sakal"
SAKAL_INDEX_DIMENSION = 1536
SAKAL_INDEX_METRIC = "dotproduct"
SAKAL_INDEX_VECTOR_TYPE = "dense"
SAKAL_BM25_ENCODER_PATH = Path("bm25_sakal_values.json")


def load_articles(json_path: Path, limit: int | None = None) -> list[dict]:
    # Read the converted XML data from sakal.json.
    with json_path.open("r", encoding="utf-8") as file:
        articles = json.load(file)

    if limit is not None:
        return articles[:limit]

    return articles


def to_pinecone_sparse_values(sparse_vector: dict) -> dict:
    # Pinecone expects sparse indices as ints and sparse values as floats.
    return {
        "indices": [int(index) for index in sparse_vector["indices"]],
        "values": [float(value) for value in sparse_vector["values"]],
    }


def load_or_fit_sakal_bm25_encoder(
    embedding_texts: list[str],
    encoder_path: Path,
    rebuild: bool = False,
) -> BM25Encoder:
    # BM25 must be fitted on the same Sakal chunk texts that we upload.
    # If the saved encoder exists, reuse it unless --rebuild-bm25 is passed.
    if encoder_path.exists() and not rebuild:
        return BM25Encoder().load(str(encoder_path))

    encoder = BM25Encoder().default()
    encoder.fit(embedding_texts)
    encoder.dump(str(encoder_path))
    return encoder


def create_sakal_index_if_needed(
    pc: Pinecone,
    index_name: str,
    cloud: str,
    region: str,
) -> None:
    # Create the Pinecone index only if it does not already exist.
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
    # This is the exact text used for both dense embeddings and sparse BM25.
    embedding_texts = [
        build_article_chunk_embedding_text(article, chunk_text)
        for article, _, _, chunk_text in article_chunk_batch
    ]

    # Dense vectors capture semantic meaning.
    dense_embeddings = embed_texts(embedding_texts)

    # Sparse vectors capture exact word/keyword matching.
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
        # Each chunk becomes one Pinecone vector.
        vector = {
            "id": f"{article_id}-{chunk_index}",
            "values": dense_embedding,
            "sparse_values": sparse_vector,
            "metadata": article_metadata(
                article,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
            ),
        }
        vectors.append(vector)

    return vectors


def batched(items: list, batch_size: int):
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
    # Load API keys from .env.
    load_dotenv()

    api_key = os.environ.get("PINECONE_API_KEY")
    cloud = os.environ.get("PINECONE_CLOUD", "aws")
    region = os.environ.get("PINECONE_REGION", "us-east-1")

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing.")

    if start_batch < 1:
        raise ValueError("--start-batch must be 1 or greater.")

    # Step 1: read sakal.json.
    articles = load_articles(json_path, limit=limit)

    # Step 2: split articles into 500-word chunks with 50-word overlap.
    article_chunks = article_chunk_records(articles)

    # Step 3: build the searchable text for every chunk.
    embedding_texts = [
        build_article_chunk_embedding_text(article, chunk_text)
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
            preview = build_article_chunk_embedding_text(article, chunk_text)
            print(f"DRY RUN {article_id}-{chunk_index}: {preview[:300]}")
        return

    # Step 4: fit or load BM25 for sparse search.
    bm25_encoder = load_or_fit_sakal_bm25_encoder(
        embedding_texts=embedding_texts,
        encoder_path=SAKAL_BM25_ENCODER_PATH,
        rebuild=rebuild_bm25,
    )

    # Step 5: connect to Pinecone and create the sakal index if needed.
    pc = Pinecone(api_key=api_key)
    create_sakal_index_if_needed(
        pc=pc,
        index_name=index_name,
        cloud=cloud,
        region=region,
    )
    index = pc.Index(index_name)

    uploaded = min((start_batch - 1) * batch_size, len(article_chunks))

    # Step 6: embed chunks, create sparse values, and upload vectors to Pinecone.
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
    parser = argparse.ArgumentParser(
        description="Embed Sakal XML-converted JSON and store it in Pinecone."
    )
    parser.add_argument("--json-file", type=Path, default=SAKAL_JSON_PATH)
    parser.add_argument("--index-name", default=SAKAL_INDEX_NAME)
    parser.add_argument("--namespace", default=os.environ.get("PINECONE_NAMESPACE", "default"))
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
