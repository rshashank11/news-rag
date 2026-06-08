import argparse
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

from app.barandbench_content_kind import classify_barandbench_story_content_kind
from app.embeddings import embed_texts
from app.legal_extraction import extract_courts, extract_statutes, extract_case_type
from app.legal_tokenizer import apply_legal_tokenizer
from app.sparse import to_pinecone_sparse_values
from database import SessionLocal, engine, Base
from models import StoryMetaData

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")

Base.metadata.create_all(bind=engine)

DEFAULT_DATA_FILES = [
    "data/stories-barandbench-1.txt",
    "data/stories-barandbench-2.txt",
    "data/stories-barandbench-3.txt",
    "data/stories-barandbench-4.txt",
    "data/stories-barandbench-5.txt",
    "data/stories-barandbench-6.txt",
    "data/stories-barandbench-7.txt",
]
BM25_ENCODER_PATH = Path(
    os.environ.get("BARANDBENCH_BM25_ENCODER_PATH")
    or os.environ.get("BM25_ENCODER_PATH", "bm25_barandbench_values.json")
)
INGEST_CHECKPOINT_PATH = Path(
    os.environ.get(
        "BARANDBENCH_INGEST_CHECKPOINT_PATH",
        "barandbench_ingest_checkpoint.json",
    )
)


def clean_html_text(tag) -> str:
    """
    Extract readable text from one HTML tag.

    Example:
    "<p>Hello <b>world</b></p>" becomes "Hello world".
    """
    text = tag.get_text("", strip=False)
    return re.sub(r"\s+", " ", text).strip()


def extract_paragraphs(data: dict) -> list[str]:
    """
    Pull paragraph chunks from one Bar & Bench story record.

    Bar & Bench stories arrive as nested cards/elements with HTML inside. This
    function extracts the article paragraphs that we embed and store.
    """
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
    """
    Join paragraph chunks into the full article body.

    Pinecone stores individual paragraph chunks.
    Postgres stores the fuller article text for answer generation.
    """
    return "\n\n".join(paragraphs)


def timestamp_ms_to_datetime(value):
    """
    Convert a Quintype timestamp into a Python datetime.

    Quintype timestamps are milliseconds since Unix epoch.
    """
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=IST)


def timestamp_ms_to_date_string(value) -> str | None:
    """
    Convert a Quintype timestamp into a YYYY-MM-DD date string.

    Example:
    1775000000000 becomes something like "2026-04-01".
    """
    published_at = timestamp_ms_to_datetime(value)
    if published_at is None:
        return None
    return published_at.date().isoformat()


def date_string_to_yyyymmdd(value: str | None) -> int | None:
    """
    Convert a date string into Pinecone's numeric date format.

    Example:
    "2026-04-01" becomes 20260401.
    """
    if value is None:
        return None

    return int(value.replace("-", ""))


def normalized_metadata_values(values: list[str]) -> list[str]:
    """
    Lowercase and dedupe topic/category values.

    Example:
    [" Supreme Court ", "supreme court"] becomes ["supreme court"].
    """
    normalized_values = []
    seen = set()

    for value in values:
        normalized_value = re.sub(r"\s+", " ", value).strip().lower()

        if normalized_value and normalized_value not in seen:
            normalized_values.append(normalized_value)
            seen.add(normalized_value)

    return normalized_values


def extract_tag_names(data: dict) -> list[str]:
    """
    Extract topic/tag names from one story record.

    Example:
    Tags may include courts, statutes, legal topics, or people.
    """
    return [
        tag.get("name")
        for tag in data.get("tags", [])
        if tag.get("name")
    ]


def extract_category_names(data: dict) -> list[str]:
    """
    Extract section/category names from one story record.

    Example:
    Categories may include "Litigation", "Corporate", or similar sections.
    """
    return [
        section.get("name")
        for section in data.get("sections", [])
        if section.get("name")
    ]


def load_checkpoint() -> dict | None:
    """
    Load the last saved ingestion position.

    Example:
    If ingestion stopped at stories-barandbench-3.txt line 2000, the next run can
    continue after that line instead of starting over.
    """
    ignore_checkpoint = (
        os.environ.get("BARANDBENCH_INGEST_IGNORE_CHECKPOINT")
        or os.environ.get("INGEST_IGNORE_CHECKPOINT", "")
    )

    if ignore_checkpoint.lower() == "true":
        return None

    if not INGEST_CHECKPOINT_PATH.exists():
        return None

    with INGEST_CHECKPOINT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def checkpoint_from_env() -> dict | None:
    """
    Build a manual resume point from environment variables.

    Example:
    BARANDBENCH_INGEST_START_FILE and BARANDBENCH_INGEST_START_LINE let an
    engineer restart from a known dump position.
    """
    start_file = (
        os.environ.get("BARANDBENCH_INGEST_START_FILE")
        or os.environ.get("INGEST_START_FILE")
    )
    start_line = (
        os.environ.get("BARANDBENCH_INGEST_START_LINE")
        or os.environ.get("INGEST_START_LINE")
    )

    if not start_file and not start_line:
        return None

    if not start_file or not start_line:
        raise RuntimeError(
            "Both BARANDBENCH_INGEST_START_FILE and "
            "BARANDBENCH_INGEST_START_LINE must be set to resume manually."
        )

    return {
        "file": Path(start_file).name,
        "line": int(start_line) - 1,
    }


def get_resume_checkpoint() -> dict | None:
    """
    Choose the checkpoint used for this ingestion run.

    Manual environment settings win over the saved checkpoint file.
    """
    return checkpoint_from_env() or load_checkpoint()


def should_skip_for_checkpoint(path: Path, line_number: int, checkpoint: dict | None) -> bool:
    """
    Decide whether a dump line is before the resume point.

    This prevents re-uploading chunks that were already flushed successfully.
    """
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
    """
    Save the latest safely uploaded dump position.

    The checkpoint is written after Pinecone upsert so retries do not skip data.
    """
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
    """
    Read story dump lines as JSON records.

    Each non-empty line is one story. Checkpoint logic skips lines already
    processed by a previous run.
    """
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
    """
    Load or create the Bar & Bench BM25 keyword-search encoder.

    BM25 learns the Bar & Bench paragraph vocabulary, so exact legal terms like
    "PMLA", "NCLT", or "bail" can match strongly.

    Legal compound terms (e.g. "Supreme Court", "Enforcement Directorate") are
    pre-processed into single underscore-joined tokens before fitting, so they
    are treated as one BM25 token rather than two separate words.
    """
    if BM25_ENCODER_PATH.exists():
        print(f"Loading BM25 encoder from {BM25_ENCODER_PATH}")
        return BM25Encoder().load(str(BM25_ENCODER_PATH))

    print("Fitting BM25 encoder on paragraph chunks...")
    corpus = []

    for _, _, data in iter_story_data(file_paths):
        for paragraph in extract_paragraphs(data):
            corpus.append(apply_legal_tokenizer(paragraph))

    if not corpus:
        raise RuntimeError("Cannot fit BM25 encoder because no paragraphs were found.")

    bm25_encoder = BM25Encoder().default()
    bm25_encoder.fit(corpus)

    BM25_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    bm25_encoder.dump(str(BM25_ENCODER_PATH))
    print(f"Saved BM25 encoder to {BM25_ENCODER_PATH}")

    return bm25_encoder


def has_sparse_values(sparse_vector: dict) -> bool:
    """
    Check whether BM25 found any keyword signal for a chunk.

    Empty sparse vectors are skipped because Pinecone expects meaningful indices
    and values.
    """
    return bool(sparse_vector["indices"]) and bool(sparse_vector["values"])


def flush_stories(db, pending_stories):
    """
    Save pending full-story rows to Postgres.

    Postgres is used later to hydrate full Bar & Bench article context after
    Pinecone finds matching story IDs.
    """
    if not pending_stories:
        return

    db.add_all(pending_stories)
    db.commit()
    pending_stories.clear()


def flush_chunks(index, namespace, pending_chunks, bm25_encoder):
    """
    Embed pending paragraph chunks and upload them to Pinecone.

    Each paragraph chunk gets:
    - dense embedding for meaning,
    - sparse BM25 vector for keywords,
    - metadata for dates, headline, topics, and categories.
    """
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
            to_pinecone_sparse_values(
                bm25_encoder.encode_documents(apply_legal_tokenizer(chunk_text))
            )
            for chunk_text in chunk_texts
        ]

        vectors = []

        for chunk, embedding, sparse_vector in zip(batch, embeddings, sparse_vectors):
            vector = {
                "id": chunk["chunk_id"],
                "values": embedding,
                "metadata": {
                    "story_id": chunk["story_id"],
                    "chunk_index": chunk["chunk_index"],
                    "headline": chunk["headline"],
                    "published_at": chunk["published_at"],
                    "published_at_yyyymmdd": chunk["published_at_yyyymmdd"],
                    "topics": chunk["topics"],
                    "topics_normalized": chunk["topics_normalized"],
                    "categories": chunk["categories"],
                    "categories_normalized": chunk["categories_normalized"],
                    "court": chunk["court"],
                    "statutes": chunk["statutes"],
                    "case_type": chunk["case_type"],
                    "content_kind": chunk["content_kind"],
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
    """
    Run the full Bar & Bench ingestion pipeline.

    What happens:
    1. read story dump files,
    2. save full story text to Postgres,
    3. embed paragraph chunks,
    4. upload paragraph vectors to Pinecone.
    """
    db = SessionLocal()

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_host = (
        os.environ.get("BARANDBENCH_PINECONE_INDEX_HOST")
        or os.environ.get("PINECONE_INDEX_HOST")
    )

    if not index_host:
        raise RuntimeError(
            "BARANDBENCH_PINECONE_INDEX_HOST or PINECONE_INDEX_HOST is missing."
        )

    index = pc.Index(host=index_host)

    namespace = (
        os.environ.get("BARANDBENCH_PINECONE_NAMESPACE")
        or os.environ.get("PINECONE_NAMESPACE")
        or "barandbench"
    )
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
            published_at_yyyymmdd = date_string_to_yyyymmdd(published_at_date)
            topics = extract_tag_names(data)
            categories = extract_category_names(data)
            topics_normalized = normalized_metadata_values(topics)
            categories_normalized = normalized_metadata_values(categories)

            # Extract legal metadata from headline + full article text
            legal_text = (data.get("headline", "") + " " + full_content).strip()
            story_courts = extract_courts(legal_text)
            story_statutes = extract_statutes(legal_text)
            story_case_type = extract_case_type(legal_text)
            story_content_kind = classify_barandbench_story_content_kind(
                headline=data.get("headline", "Untitled"),
                summary=data.get("seo", {}).get("meta-description", ""),
                categories=categories,
                full_content=full_content,
            )

            existing_story = db.get(StoryMetaData, story_id)

            if not existing_story:
                story = StoryMetaData(
                    id=story_id,
                    headline=data.get("headline", "Untitled"),
                    summary=data.get("seo", {}).get("meta-description", ""),
                    published_at=published_at,
                    topics=topics,
                    categories=categories,
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
                        "published_at_yyyymmdd": published_at_yyyymmdd,
                        "topics": topics,
                        "topics_normalized": topics_normalized,
                        "categories": categories,
                        "categories_normalized": categories_normalized,
                        "court": story_courts,
                        "statutes": story_statutes,
                        "case_type": story_case_type,
                        "content_kind": story_content_kind,
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


def parse_args():
    """
    Parse command-line arguments for Bar & Bench ingestion.

    Example:
    python ingest_barandbench.py data/stories-barandbench-1.txt
    ingests only that file.

    python ingest_barandbench.py --bm25-only
    re-fits the BM25 encoder without touching Pinecone or Postgres.
    """
    parser = argparse.ArgumentParser(
        description="Ingest Bar & Bench story dumps into Postgres and Pinecone."
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_DATA_FILES,
        help="Story dump files to ingest.",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help=(
            "Re-fit the BM25 encoder only. Deletes the existing encoder file and "
            "rebuilds it from the corpus with the legal tokenizer. "
            "Does not touch Pinecone or Postgres."
        ),
    )
    return parser.parse_args()


def rebuild_bm25(file_paths: list[str]):
    """
    Re-fit the BM25 encoder and atomically replace the existing file.

    Writes to a temp file first so the existing encoder is never deleted unless
    the new one is successfully written. This prevents the running app from
    crashing if the rebuild fails partway through.
    """
    import shutil

    print("Fitting BM25 encoder on paragraph corpus...")
    corpus = []
    for _, _, data in iter_story_data(file_paths):
        for paragraph in extract_paragraphs(data):
            corpus.append(apply_legal_tokenizer(paragraph))

    if not corpus:
        raise RuntimeError("Cannot fit BM25 encoder: no paragraphs were found.")

    encoder = BM25Encoder().default()
    encoder.fit(corpus)

    tmp_path = BM25_ENCODER_PATH.with_suffix(".tmp")
    BM25_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    encoder.dump(str(tmp_path))

    shutil.move(str(tmp_path), str(BM25_ENCODER_PATH))
    print(f"BM25 encoder rebuilt and saved to {BM25_ENCODER_PATH}")


if __name__ == "__main__":
    args = parse_args()
    if args.bm25_only:
        rebuild_bm25(args.files)
    else:
        ingest_files(args.files)
