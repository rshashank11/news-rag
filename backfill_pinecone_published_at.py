import argparse
import json
import os
import re
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

            for p in soup.find_all("p"):
                text = clean_html_text(p)
                if text:
                    paragraphs.append(text)

    return paragraphs


def timestamp_ms_to_date_string(value) -> str | None:
    if value is None:
        return None

    published_at = datetime.fromtimestamp(int(value) / 1000, tz=IST)
    return published_at.date().isoformat()


def iter_metadata_updates(file_paths: list[str]):
    for file_path in file_paths:
        path = Path(file_path)

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                data = json.loads(line)
                story_id = data.get("id")
                published_at = timestamp_ms_to_date_string(data.get("published-at"))

                if not story_id or not published_at:
                    continue

                paragraphs = extract_paragraphs(data)

                for chunk_index, _ in enumerate(paragraphs):
                    yield {
                        "id": f"{story_id}-{chunk_index}",
                        "metadata": {
                            "published_at": published_at,
                        },
                        "source": f"{path.name}:{line_number}",
                    }


def update_metadata(index, namespace: str, update: dict):
    index.update(
        id=update["id"],
        namespace=namespace,
        set_metadata=update["metadata"],
    )
    return update


def run_backfill(file_paths: list[str], workers: int, dry_run: bool, limit: int | None):
    updates = []

    for update in iter_metadata_updates(file_paths):
        updates.append(update)

        if limit is not None and len(updates) >= limit:
            break

    print(f"Prepared {len(updates)} Pinecone metadata updates.")

    if dry_run:
        for update in updates[:5]:
            print(f"DRY RUN {update['id']} -> {update['metadata']}")
        return

    load_dotenv()

    api_key = os.environ.get("PINECONE_API_KEY")
    index_host = os.environ.get("PINECONE_INDEX_HOST")
    namespace = os.environ.get("PINECONE_NAMESPACE", "default")

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing.")

    if not index_host:
        raise RuntimeError("PINECONE_INDEX_HOST is missing.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(host=index_host)

    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(update_metadata, index, namespace, update)
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

    print(f"Finished. Updated={completed}, failed={failed}.")


def parse_args():
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
    )
