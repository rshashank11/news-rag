import json
import os
import re
import uuid
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone_text.sparse import BM25Encoder

from app.openai_client import get_embedding_model, make_embedding_client
from database import SessionLocal, engine, Base
from models import StoryMetaData

load_dotenv()

client = make_embedding_client()
IST = ZoneInfo("Asia/Kolkata")

Base.metadata.create_all(bind=engine)

BM25_ENCODER_PATH = Path(os.environ.get("BM25_ENCODER_PATH", "bm25_values.json"))
INGEST_CHECKPOINT_PATH = Path(os.environ.get("INGEST_CHECKPOINT_PATH", "ingest_checkpoint.json"))


def clean_html_text(tag) -> str:
    text = tag.get_text("", strip=False)
    return re.sub(r"\s+", " ", text).strip()


def extract_paragraphs(data: dict) -> list[str]:
    paragraphs = []

    for card in data.get("cards", []):
        for element in card.get("story-elements", []):
            if element.get("type") != "text":
                continue

            raw_html = element.get("text", "")
            soup = BeautifulSoup(raw_html, "html.parser")

            p_tags = soup.find_all("p")

            if p_tags:
                for p in p_tags:
                    text = clean_html_text(p)
                    if text:
                        paragraphs.append(text)

    return paragraphs


def build_full_content(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs)


def timestamp_ms_to_datetime(value):
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=IST)


def timestamp_ms_to_date_string(value) -> str | None:
    published_at = timestamp_ms_to_datetime(value)
    if published_at is None:
        return None
    return published_at.date().isoformat()


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=get_embedding_model(),
        input=texts
    )

    sorted_items = sorted(response.data, key=lambda item: item.index)

    return [
        item.embedding
        for item in sorted_items
    ]


def load_checkpoint() -> dict | None:
    if os.environ.get("INGEST_IGNORE_CHECKPOINT", "").lower() == "true":
        return None

    if not INGEST_CHECKPOINT_PATH.exists():
        return None

    with INGEST_CHECKPOINT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def checkpoint_from_env() -> dict | None:
    start_file = os.environ.get("INGEST_START_FILE")
    start_line = os.environ.get("INGEST_START_LINE")

    if not start_file and not start_line:
        return None

    if not start_file or not start_line:
        raise RuntimeError("Both INGEST_START_FILE and INGEST_START_LINE must be set to resume manually.")

    return {
        "file": Path(start_file).name,
        "line": int(start_line) - 1,
    }


def get_resume_checkpoint() -> dict | None:
    return checkpoint_from_env() or load_checkpoint()


def should_skip_for_checkpoint(path: Path, line_number: int, checkpoint: dict | None) -> bool:
    if not checkpoint:
        return False

    checkpoint_file = checkpoint.get("file")
    checkpoint_line = int(checkpoint.get("line", 0))

    if not checkpoint_file:
        return False

    if path.name < checkpoint_file:
        return True

    if path.name == checkpoint_file and line_number <= checkpoint_line:
        return True

    return False


def save_checkpoint(path: str, line_number: int):
    checkpoint = {
        "file": Path(path).name,
        "path": path,
        "line": line_number,
        "updated_at": datetime.now(tz=IST).isoformat(),
    }

    INGEST_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INGEST_CHECKPOINT_PATH.open("w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=2)


def iter_story_data(file_paths: list[str], checkpoint: dict | None = None):
    for file_path in file_paths:
        path = Path(file_path)

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if should_skip_for_checkpoint(path, line_number, checkpoint):
                    continue

                if not line.strip():
                    continue

                yield path, line_number, json.loads(line)


def build_bm25_encoder(file_paths: list[str]) -> BM25Encoder:
    if BM25_ENCODER_PATH.exists():
        print(f"Loading BM25 encoder from {BM25_ENCODER_PATH}")
        return BM25Encoder().load(str(BM25_ENCODER_PATH))

    print("Fitting BM25 encoder on paragraph chunks...")
    corpus = []

    for _, _, data in iter_story_data(file_paths):
        corpus.extend(extract_paragraphs(data))

    if not corpus:
        raise RuntimeError("Cannot fit BM25 encoder because no paragraphs were found.")

    bm25_encoder = BM25Encoder().default()
    # which words exist
    # how common each word is
    # how important rare words should be
    bm25_encoder.fit(corpus)

    BM25_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    bm25_encoder.dump(str(BM25_ENCODER_PATH))
    print(f"Saved BM25 encoder to {BM25_ENCODER_PATH}")

    return bm25_encoder


def to_pinecone_sparse_values(sparse_vector: dict) -> dict:
    """{
    "indices": [123, 456],
    "values": [0.8, 1.2]
    }
    
    token 123 has weight 0.8
    token 456 has weight 1.2

    """
    return {
        "indices": [int(index) for index in sparse_vector["indices"]],
        "values": [float(value) for value in sparse_vector["values"]],
    }


def has_sparse_values(sparse_vector: dict) -> bool:
    return bool(sparse_vector["indices"]) and bool(sparse_vector["values"])


def flush_stories(db, pending_stories):
    if not pending_stories:
        return

    db.add_all(pending_stories)
    db.commit()
    pending_stories.clear()


def flush_chunks(index, namespace, pending_chunks, bm25_encoder):
    if not pending_chunks:
        return

    last_flushed_position = None

    while pending_chunks:
        batch = pending_chunks[:100]
        del pending_chunks[:100]

        chunk_texts = [
            chunk["chunk_text"]
            for chunk in batch
        ]

        embeddings = embed_texts(chunk_texts)
        sparse_vectors = [
            to_pinecone_sparse_values(bm25_encoder.encode_documents(chunk_text))
            for chunk_text in chunk_texts
        ]

        vectors = []

        # zip lets us walk through three lists side by side
        for chunk, embedding, sparse_vector in zip(batch, embeddings, sparse_vectors):
            vector = {
                "id": chunk["chunk_id"],
                "values": embedding,
                "metadata": {
                    "story_id": chunk["story_id"],
                    "chunk_index": chunk["chunk_index"],
                    "headline": chunk["headline"],
                    "published_at": chunk["published_at"],
                    "chunk_text": chunk["chunk_text"],
                },
            }

            if has_sparse_values(sparse_vector):
                vector["sparse_values"] = sparse_vector

            vectors.append(vector)

        index.upsert(vectors=vectors, namespace=namespace)
        last_chunk = batch[-1]
        last_flushed_position = (last_chunk["source_path"], last_chunk["source_line"])
        save_checkpoint(*last_flushed_position)

    return last_flushed_position


def ingest_files(file_paths: list[str]):
    db = SessionLocal()

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(host=os.environ.get("PINECONE_INDEX_HOST"))

    namespace = os.environ.get("PINECONE_NAMESPACE")
    bm25_encoder = build_bm25_encoder(file_paths)
    checkpoint = get_resume_checkpoint()
    pending_stories = []
    pending_chunks = []

    if checkpoint:
        print(f"Resuming after {checkpoint.get('file')}:{checkpoint.get('line')}")

    try:
        for path, line_number, data in iter_story_data(file_paths, checkpoint):
            source_id = data.get("id")

            if not source_id:
                continue

            try:
                story_id = uuid.UUID(source_id)
            except ValueError:
                continue

            paragraphs = extract_paragraphs(data)

            if not paragraphs:
                continue

            full_content = build_full_content(paragraphs)
            published_at = timestamp_ms_to_datetime(data.get("published-at"))
            published_at_date = timestamp_ms_to_date_string(data.get("published-at"))

            existing_story = db.get(StoryMetaData, story_id)

            if not existing_story:
                story = StoryMetaData(
                    id=story_id,
                    headline=data.get("headline", "Untitled"),
                    summary=data.get("seo", {}).get("meta-description", ""),
                    published_at=published_at,
                    topics=[
                        tag.get("name")
                        for tag in data.get("tags", [])
                        if tag.get("name")
                    ],
                    categories=[
                        section.get("name")
                        for section in data.get("sections", [])
                        if section.get("name")
                    ],
                    full_content=full_content,
                )

                pending_stories.append(story)

            for chunk_index, paragraph in enumerate(paragraphs):
                pending_chunks.append(
                    {
                        "chunk_id": f"{story_id}-{chunk_index}",
                        "story_id": str(story_id),
                        "chunk_index": chunk_index,
                        "headline": data.get("headline", "Untitled"),
                        "published_at": published_at_date,
                        "chunk_text": paragraph,
                        "source_path": str(path),
                        "source_line": line_number,
                    }
                )

            if len(pending_stories) >= 500:
                flush_stories(db, pending_stories)

            if len(pending_chunks) >= 100:
                flush_stories(db, pending_stories)
                flush_chunks(index, namespace, pending_chunks, bm25_encoder)

            print(f"Queued {path.name}:{line_number}")

        flush_stories(db, pending_stories)
        flush_chunks(index, namespace, pending_chunks, bm25_encoder)

    finally:
        db.close()


if __name__ == "__main__":
    ingest_files(
        [
            "data/stories-barandbench-1.txt",
            "data/stories-barandbench-2.txt",
            "data/stories-barandbench-3.txt",
            "data/stories-barandbench-4.txt",
            "data/stories-barandbench-5.txt",
            "data/stories-barandbench-6.txt",
            "data/stories-barandbench-7.txt",
        ]
    )
