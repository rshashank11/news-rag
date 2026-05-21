import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone


IST = ZoneInfo("Asia/Kolkata")
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
    """
    Extract readable text from one HTML tag.

    This must match ingest_barandbench.py so chunk indexes stay aligned.
    """
    text = tag.get_text("", strip=False)
    return re.sub(r"\s+", " ", text).strip()


def extract_paragraphs(data: dict) -> list[str]:
    """
    Extract paragraph chunks from a Bar & Bench dump record.

    Example:
    If ingestion created vectors story_id-0, story_id-1, story_id-2, this
    function must produce the same paragraph order for backfill updates.
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


def timestamp_ms_to_date_string(value) -> str | None:
    """
    Convert a Quintype millisecond timestamp into YYYY-MM-DD.
    """
    if value is None:
        return None

    published_at = datetime.fromtimestamp(int(value) / 1000, tz=IST)
    return published_at.date().isoformat()


def date_string_to_yyyymmdd(value: str | None) -> int | None:
    """
    Convert a date string into Pinecone's numeric date field.

    Example:
    "2026-04-01" becomes 20260401.
    """
    if value is None:
        return None

    return int(value.replace("-", ""))


def normalized_metadata_values(values: list[str]) -> list[str]:
    """
    Lowercase and dedupe topic/category metadata.

    Example:
    ["PMLA", " pmla "] becomes ["pmla"].
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
    Extract topic/tag names from one story dump record.
    """
    return [
        tag.get("name")
        for tag in data.get("tags", [])
        if tag.get("name")
    ]


def extract_category_names(data: dict) -> list[str]:
    """
    Extract category/section names from one story dump record.
    """
    return [
        section.get("name")
        for section in data.get("sections", [])
        if section.get("name")
    ]


def iter_metadata_updates(file_paths: list[str]):
    """
    Prepare Pinecone metadata updates from Bar & Bench dump files.

    Important:
    Vector IDs must match ingestion exactly:
    story UUID + "-" + paragraph index.
    """
    for file_path in file_paths:
        path = Path(file_path)

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                data = json.loads(line)
                story_id = data.get("id")
                published_at = timestamp_ms_to_date_string(data.get("published-at"))
                published_at_yyyymmdd = date_string_to_yyyymmdd(published_at)
                topics = extract_tag_names(data)
                categories = extract_category_names(data)

                if not story_id or not published_at:
                    continue

                paragraphs = extract_paragraphs(data)

                for chunk_index, _ in enumerate(paragraphs):
                    yield {
                        "id": f"{story_id}-{chunk_index}",
                        "metadata": {
                            "published_at": published_at,
                            "published_at_yyyymmdd": published_at_yyyymmdd,
                            "topics": topics,
                            "topics_normalized": normalized_metadata_values(topics),
                            "categories": categories,
                            "categories_normalized": normalized_metadata_values(categories),
                        },
                        "source": f"{path.name}:{line_number}",
                    }


def update_metadata(index, namespace: str, update: dict, retries: int):
    """
    Update one Pinecone vector's metadata.

    Retries help with temporary network or Pinecone errors during large backfills.
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

            time.sleep(min(2**attempt, 10))


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
    Update many Pinecone vectors in parallel.

    Example:
    workers=16 means up to 16 metadata updates can run at the same time.
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
                print(f"Updated {completed}/{len(updates)} chunks...")

            if completed <= 5:
                print(f"Updated {completed_update['id']} from {completed_update['source']}")

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
    Run the Bar & Bench Pinecone metadata backfill.

    Use this when old vectors already exist in Pinecone but need newer metadata,
    such as published_at_yyyymmdd for date filtering.
    """
    if dry_run:
        prepared = 0

        for update in iter_metadata_updates(file_paths):
            prepared += 1

            if prepared <= 5:
                print(f"DRY RUN {update['id']} -> {update['metadata']}")

            if limit is not None and prepared >= limit:
                break

        print(f"Prepared {prepared} Pinecone metadata update(s).")
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
        raise RuntimeError("PINECONE_API_KEY is missing.")

    if not index_host:
        raise RuntimeError(
            "BARANDBENCH_PINECONE_INDEX_HOST or PINECONE_INDEX_HOST is missing."
        )

    pc = Pinecone(api_key=api_key)
    index = pc.Index(host=index_host)

    prepared = 0
    completed = 0
    failed = 0
    batch = []

    for update_number, update in enumerate(iter_metadata_updates(file_paths), start=1):
        if update_number < start_at:
            continue

        batch.append(update)
        prepared += 1

        if len(batch) >= batch_size:
            batch_start = update_number - len(batch) + 1
            print(f"Processing prepared updates {batch_start}-{update_number}...")
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
        print(f"Processing prepared updates {batch_start}-{update_number}...")
        completed, failed = process_update_batch(
            index=index,
            namespace=namespace,
            updates=batch,
            workers=workers,
            retries=retries,
            completed=completed,
            failed=failed,
        )

    print(f"Finished. Prepared={prepared}, updated={completed}, failed={failed}.")


def parse_args():
    """
    Parse command-line options for the metadata backfill script.

    Example:
    python backfill_barandbench_pinecone_published_at.py --dry-run --limit 10
    previews the first 10 updates without writing to Pinecone.
    """
    parser = argparse.ArgumentParser(
        description="Backfill published_at metadata onto existing Pinecone chunks."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel Pinecone update workers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first few updates without writing to Pinecone.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process this many chunk updates.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of prepared updates to write before reading more input.",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=1,
        help="Resume from this prepared update number.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Number of times to retry each Pinecone metadata update.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_DATA_FILES,
        help="TXT files to read. Defaults to all data/stories-barandbench-*.txt files.",
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
