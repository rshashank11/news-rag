"""
Backfill legal metadata (court, statutes, case_type) onto existing
Bar & Bench Pinecone vectors.

This script reads story dump files, extracts court names, statute references,
and case type labels from each article, then patches the corresponding Pinecone
chunk vectors with those new metadata fields using index.update().

No re-embedding or Pinecone re-ingestion is needed — only metadata is updated.

Usage:
    python backfill_barandbench_legal_metadata.py
    python backfill_barandbench_legal_metadata.py --dry-run --limit 20
    python backfill_barandbench_legal_metadata.py --start-at 5001 --workers 32
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

from app.legal_extraction import extract_courts, extract_statutes, extract_case_type


DEFAULT_DATA_FILES = [
    "data/stories-barandbench-1.txt",
    "data/stories-barandbench-2.txt",
    "data/stories-barandbench-3.txt",
    "data/stories-barandbench-4.txt",
    "data/stories-barandbench-5.txt",
    "data/stories-barandbench-6.txt",
    "data/stories-barandbench-7.txt",
]


def clean_html_text(tag) -> str:
    text = tag.get_text("", strip=False)
    return re.sub(r"\s+", " ", text).strip()


def extract_paragraphs(data: dict) -> list[str]:
    """
    Extract paragraphs in the exact order used during ingestion.

    Chunk IDs are {story_id}-{chunk_index} where chunk_index is the 0-based
    paragraph position from this function, so the order must be identical to
    ingest_barandbench.py.
    """
    paragraphs = []

    for card in data.get("cards", []):
        for element in card.get("story-elements", []):
            if element.get("type") != "text":
                continue

            raw_html = element.get("text", "")
            soup = BeautifulSoup(raw_html, "html.parser")

            for p in soup.find_all("p"):
                text = clean_html_text(p)
                if text:
                    paragraphs.append(text)

    return paragraphs


def iter_legal_metadata_updates(file_paths: list[str]):
    """
    Yield one Pinecone update dict per chunk for every story in the dump files.

    Legal metadata is extracted from the story headline plus all paragraph text
    combined, so all chunks from the same story get the same legal metadata.
    """
    for file_path in file_paths:
        path = Path(file_path)

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                data = json.loads(line)
                story_id = data.get("id")

                if not story_id:
                    continue

                paragraphs = extract_paragraphs(data)
                if not paragraphs:
                    continue

                headline = data.get("headline", "")
                full_content = "\n\n".join(paragraphs)
                legal_text = (headline + " " + full_content).strip()

                courts = extract_courts(legal_text)
                statutes = extract_statutes(legal_text)
                case_type = extract_case_type(legal_text)

                for chunk_index in range(len(paragraphs)):
                    yield {
                        "id": f"{story_id}-{chunk_index}",
                        "metadata": {
                            "court": courts,
                            "statutes": statutes,
                            "case_type": case_type,
                        },
                        "source": f"{path.name}:{line_number}",
                    }


def update_metadata(index, namespace: str, update: dict, retries: int):
    """
    Update one Pinecone vector's metadata with retries.
    """
    for attempt in range(1, retries + 1):
        try:
            index.update(
                id=update["id"],
                namespace=namespace,
                set_metadata=update["metadata"],
            )
            return update
        except Exception:
            if attempt == retries:
                raise

            time.sleep(min(2 ** attempt, 10))


def process_update_batch(
    index,
    namespace: str,
    updates: list[dict],
    workers: int,
    retries: int,
    completed: int,
    failed: int,
) -> tuple[int, int]:
    """
    Update many Pinecone vectors in parallel using a thread pool.
    """
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(update_metadata, index, namespace, update, retries)
            for update in updates
        ]

        for future in as_completed(futures):
            try:
                completed_update = future.result()
                completed += 1
            except Exception as exc:
                failed += 1
                print(f"FAILED update: {exc}")
                continue

            if completed % 500 == 0:
                print(f"  Updated {completed} chunks so far...")

            if completed <= 5:
                meta = completed_update["metadata"]
                print(
                    f"  Updated {completed_update['id']} from {completed_update['source']}: "
                    f"court={meta['court']}, statutes={meta['statutes']}, case_type={meta['case_type']}"
                )

    return completed, failed


def run_backfill(
    file_paths: list[str],
    workers: int,
    dry_run: bool,
    limit: int | None,
    batch_size: int,
    start_at: int,
    retries: int,
):
    """
    Run the legal metadata backfill against all Bar & Bench Pinecone vectors.

    Steps:
    1. Iterate all dump files to compute legal metadata per chunk.
    2. Call index.update(set_metadata=...) for each chunk ID.
    3. Reports completed and failed counts.

    For large corpora, use --workers 32 and --batch-size 500 to speed up.
    """
    if dry_run:
        prepared = 0

        for update in iter_legal_metadata_updates(file_paths):
            prepared += 1

            if prepared <= 10:
                meta = update["metadata"]
                print(
                    f"DRY RUN {update['id']} ({update['source']}): "
                    f"court={meta['court']}, statutes={meta['statutes']}, "
                    f"case_type={meta['case_type']}"
                )

            if limit is not None and prepared >= limit:
                break

        print(f"Dry run complete. Would update {prepared} chunk(s).")
        return

    load_dotenv()

    api_key = os.environ.get("PINECONE_API_KEY")
    index_host = (
        os.environ.get("BARANDBENCH_PINECONE_INDEX_HOST")
        or os.environ.get("PINECONE_INDEX_HOST")
    )
    namespace = (
        os.environ.get("BARANDBENCH_PINECONE_NAMESPACE")
        or os.environ.get("PINECONE_NAMESPACE")
        or "barandbench"
    )

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing from environment.")

    if not index_host:
        raise RuntimeError(
            "BARANDBENCH_PINECONE_INDEX_HOST or PINECONE_INDEX_HOST is missing."
        )

    pc = Pinecone(api_key=api_key)
    index = pc.Index(host=index_host)

    prepared = 0
    completed = 0
    failed = 0
    batch: list[dict] = []
    update_number = 0

    for update_number, update in enumerate(
        iter_legal_metadata_updates(file_paths), start=1
    ):
        if update_number < start_at:
            continue

        batch.append(update)
        prepared += 1

        if len(batch) >= batch_size:
            batch_start = update_number - len(batch) + 1
            print(f"Processing batch {batch_start}-{update_number}...")
            completed, failed = process_update_batch(
                index=index,
                namespace=namespace,
                updates=batch,
                workers=workers,
                retries=retries,
                completed=completed,
                failed=failed,
            )
            batch.clear()

        if limit is not None and prepared >= limit:
            break

    if batch:
        batch_start = update_number - len(batch) + 1
        print(f"Processing final batch {batch_start}-{update_number}...")
        completed, failed = process_update_batch(
            index=index,
            namespace=namespace,
            updates=batch,
            workers=workers,
            retries=retries,
            completed=completed,
            failed=failed,
        )

    print(
        f"\nFinished. prepared={prepared}, updated={completed}, failed={failed}."
    )


def parse_args():
    """
    Parse command-line options for the legal metadata backfill.

    Example:
        python backfill_barandbench_legal_metadata.py --dry-run --limit 20
        python backfill_barandbench_legal_metadata.py --workers 32 --batch-size 500
        python backfill_barandbench_legal_metadata.py --start-at 10001
    """
    parser = argparse.ArgumentParser(
        description="Backfill court/statutes/case_type metadata onto Bar & Bench Pinecone vectors."
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_DATA_FILES,
        help="Story dump files to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned updates without writing to Pinecone.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many chunk updates (useful for smoke tests).",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=1,
        help="Skip the first N-1 updates and start from update number N.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel Pinecone update threads.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of updates to queue before processing a batch.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts per failed Pinecone update.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_backfill(
        file_paths=args.files,
        workers=args.workers,
        dry_run=args.dry_run,
        limit=args.limit,
        batch_size=args.batch_size,
        start_at=args.start_at,
        retries=args.retries,
    )
