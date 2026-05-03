import json
import os
import uuid

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database import Base, SessionLocal, engine
from models import StoryMetadata


load_dotenv()

INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "news_index")
STORY_COMMIT_BATCH_SIZE = int(os.getenv("STORY_COMMIT_BATCH_SIZE", "25"))
VECTOR_BULK_SIZE = int(os.getenv("VECTOR_BULK_SIZE", "64"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"

Base.metadata.create_all(bind=engine)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={
        "device": os.getenv("EMBEDDING_DEVICE", "cpu"),
    },
    encode_kwargs={
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "batch_size": EMBEDDING_BATCH_SIZE,
    },
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

docsearch = OpenSearchVectorSearch(
    index_name=INDEX_NAME,
    embedding_function=embeddings,
    opensearch_url=os.getenv("OPENSEARCH_URL"),
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    engine="lucene",
)


def clean_html_text(raw_html: str) -> str:
    return BeautifulSoup(raw_html, "html.parser").get_text().strip()


def extract_full_content(data: dict) -> str:
    text_blocks = []

    for card in data.get("cards", []):
        for element in card.get("story-elements", []):
            if element.get("type") == "text":
                clean_text = clean_html_text(element.get("text", ""))

                if clean_text:
                    text_blocks.append(clean_text)

    return "\n\n".join(text_blocks)


def flush_vectors(texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
    if not texts:
        return

    for start in range(0, len(texts), VECTOR_BULK_SIZE):
        end = start + VECTOR_BULK_SIZE

        docsearch.add_texts(
            texts=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
            bulk_size=VECTOR_BULK_SIZE,
        )

    texts.clear()
    metadatas.clear()
    ids.clear()


def ingest_data(file_path: str) -> None:
    db = SessionLocal()

    pending_texts = []
    pending_metadatas = []
    pending_ids = []

    stories_since_commit = 0
    inserted_count = 0
    skipped_count = 0
    failed_count = 0

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                clean_line = line.strip()

                if not clean_line:
                    continue

                try:
                    data = json.loads(clean_line)
                except json.JSONDecodeError as exc:
                    failed_count += 1
                    print(f"Line {line_number}: invalid JSON: {exc}")
                    continue

                source_id = data.get("id")

                if not source_id:
                    skipped_count += 1
                    continue

                try:
                    story_id = uuid.UUID(source_id)
                except ValueError as exc:
                    failed_count += 1
                    print(f"Line {line_number}: invalid story id {source_id}: {exc}")
                    continue

                if db.get(StoryMetadata, story_id):
                    skipped_count += 1
                    continue

                headline = data.get("headline", "Untitled")
                summary = data.get("seo", {}).get("meta-description", "")
                topics = [
                    tag.get("name")
                    for tag in data.get("tags", [])
                    if tag.get("name")
                ]
                categories = [
                    section.get("name")
                    for section in data.get("sections", [])
                    if section.get("name")
                ]
                full_content = extract_full_content(data)

                if not full_content:
                    skipped_count += 1
                    continue

                story = StoryMetadata(
                    id=story_id,
                    headline=headline,
                    summary=summary,
                    topics=topics,
                    categories=categories,
                    full_content=full_content,
                )
                db.add(story)

                chunks = text_splitter.split_text(full_content)

                for chunk_index, chunk in enumerate(chunks):
                    pending_texts.append(chunk)
                    pending_metadatas.append(
                        {
                            "story_id": str(story_id),
                            "headline": headline,
                            "chunk_index": chunk_index,
                        }
                    )
                    pending_ids.append(f"{story_id}-{chunk_index}")

                inserted_count += 1
                stories_since_commit += 1

                if stories_since_commit >= STORY_COMMIT_BATCH_SIZE:
                    flush_vectors(pending_texts, pending_metadatas, pending_ids)
                    db.commit()
                    stories_since_commit = 0
                    print(
                        f"Line {line_number}: inserted={inserted_count}, "
                        f"skipped={skipped_count}, failed={failed_count}"
                    )

        flush_vectors(pending_texts, pending_metadatas, pending_ids)
        db.commit()

        print(
            f"Ingestion complete. inserted={inserted_count}, "
            f"skipped={skipped_count}, failed={failed_count}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    ingest_data("data/stories-barandbench-1.txt")
