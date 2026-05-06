import json  # Parses each line of the TXT file because every line is one JSON story.
import os  # Reads ingestion settings from environment variables.
import uuid  # Converts Bar & Bench story IDs into real UUID objects.
from datetime import datetime  # Converts millisecond timestamps into Python datetime objects.
from zoneinfo import ZoneInfo  # Handles timezone conversion; we use IST for publish dates.

from bs4 import BeautifulSoup  # Removes HTML tags from article body text.
from dotenv import load_dotenv  # Loads local .env values when running locally.
from langchain_community.vectorstores import OpenSearchVectorSearch  # Stores embedded chunks in OpenSearch.
from langchain_huggingface import HuggingFaceEmbeddings  # Converts text chunks into vectors.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits long articles into smaller chunks.

from database import Base, SessionLocal, engine, ensure_database_schema  # DB setup/session/schema helpers.
from models import StoryMetadata  # SQLAlchemy model for storing full story metadata/content.


load_dotenv()  # Makes .env values available through os.getenv.

INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "news_index")  # OpenSearch index used for vector chunks.
STORY_COMMIT_BATCH_SIZE = int(os.getenv("STORY_COMMIT_BATCH_SIZE", "25"))  # How many stories to process before DB commit.
VECTOR_BULK_SIZE = int(os.getenv("VECTOR_BULK_SIZE", "64"))  # How many chunks to send to OpenSearch per bulk request.
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))  # How many chunks MiniLM embeds at once.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # Approximate max characters per chunk.
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # Characters shared between neighboring chunks for continuity.
EMBEDDING_MODEL_NAME = os.getenv(  # Embedding model used during ingestion.
    "EMBEDDING_MODEL_NAME",  # Environment override.
    "sentence-transformers/all-MiniLM-L6-v2",  # Default CPU-friendly free model.
)
NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"  # Convert env string into boolean.
REINDEX_EXISTING = os.getenv("REINDEX_EXISTING", "false").lower() == "true"  # If true, rebuild vectors for stories already in DB.
IST = ZoneInfo("Asia/Kolkata")  # Date timezone used for source metadata.

Base.metadata.create_all(bind=engine)  # Creates DB tables if they do not exist yet.
ensure_database_schema()  # Adds newer columns safely if DB was created earlier.

embeddings = HuggingFaceEmbeddings(  # Loads the embedding model once for ingestion.
    model_name=EMBEDDING_MODEL_NAME,  # Model name from config/env.
    model_kwargs={  # Model loading options.
        "device": os.getenv("EMBEDDING_DEVICE", "cpu"),  # CPU by default for Docker/free hardware.
    },
    encode_kwargs={  # Options used while converting text into vectors.
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,  # Makes vector similarity more stable.
        "batch_size": EMBEDDING_BATCH_SIZE,  # Controls memory/CPU load.
    },
)

text_splitter = RecursiveCharacterTextSplitter(  # Splits full articles into searchable chunks.
    chunk_size=CHUNK_SIZE,  # Maximum chunk size target.
    chunk_overlap=CHUNK_OVERLAP,  # Repeated text between chunks so context is not cut abruptly.
)

docsearch = OpenSearchVectorSearch(  # LangChain vector-store object for writing chunks to OpenSearch.
    index_name=INDEX_NAME,  # Index where chunks are stored.
    embedding_function=embeddings,  # Same embedding model used later for query vectors.
    opensearch_url=os.getenv("OPENSEARCH_URL"),  # OpenSearch server URL.
    use_ssl=True,  # Hosted OpenSearch usually requires HTTPS.
    verify_certs=True,  # Verifies SSL certificate.
    ssl_assert_hostname=False,  # Avoids hostname mismatch issues on some managed services.
    ssl_show_warn=False,  # Keeps logs cleaner.
    engine="lucene",  # Vector engine option used by this setup.
)


def clean_html_text(raw_html: str) -> str:  # Converts article HTML into plain readable text.
    return BeautifulSoup(raw_html, "html.parser").get_text(" ", strip=True)  # Removes tags and joins nested text with spaces.


def extract_full_content(data: dict) -> str:  # Pulls article body text from the raw Bar & Bench JSON.
    text_blocks = []  # Stores cleaned paragraphs/blocks from the story.

    for card in data.get("cards", []):  # Stories are organized into cards.
        for element in card.get("story-elements", []):  # Each card has elements like text, image, file, embed.
            if element.get("type") == "text":  # Only ingest actual text elements.
                clean_text = clean_html_text(element.get("text", ""))  # Clean HTML text into normal text.

                if clean_text:  # Ignore empty text blocks.
                    text_blocks.append(clean_text)  # Save useful text.

    return "\n\n".join(text_blocks)  # Join blocks with blank lines to preserve article readability.


def timestamp_ms_to_datetime(value):  # Converts Bar & Bench millisecond timestamps into timezone-aware datetimes.
    if value is None:  # Missing timestamp.
        return None  # Keep date empty.

    return datetime.fromtimestamp(int(value) / 1000, tz=IST)  # Divide by 1000 because raw timestamp is in milliseconds.


def datetime_to_ist_date(value):  # Converts datetime into simple YYYY-MM-DD string for OpenSearch metadata.
    if value is None:  # Missing datetime.
        return None  # Keep metadata date empty.

    return value.astimezone(IST).date().isoformat()  # Convert to IST, keep only date, format as text.


def flush_vectors(texts: list[str], metadatas: list[dict], ids: list[str]) -> None:  # Sends pending chunks to OpenSearch.
    if not texts:  # If there are no chunks waiting...
        return  # ...there is nothing to write.

    for start in range(0, len(texts), VECTOR_BULK_SIZE):  # Process chunks in smaller batches.
        end = start + VECTOR_BULK_SIZE  # End index for current batch.

        docsearch.add_texts(  # Embeds and stores the current batch in OpenSearch.
            texts=texts[start:end],  # Chunk texts for this batch.
            metadatas=metadatas[start:end],  # Matching metadata for each chunk.
            ids=ids[start:end],  # Stable OpenSearch IDs so chunks can be overwritten/reindexed.
            bulk_size=VECTOR_BULK_SIZE,  # Must be at least as large as this batch size.
        )

    texts.clear()  # Empty pending texts after successful write.
    metadatas.clear()  # Empty pending metadata after successful write.
    ids.clear()  # Empty pending IDs after successful write.


def ingest_data(file_path: str) -> None:  # Main ingestion function that reads TXT JSONL and fills Postgres/OpenSearch.
    db = SessionLocal()  # Opens one Postgres session for this ingestion run.

    pending_texts = []  # Chunk texts waiting to be embedded/stored.
    pending_metadatas = []  # Metadata for each pending chunk.
    pending_ids = []  # Stable OpenSearch IDs for each pending chunk.

    stories_since_commit = 0  # Counts stories processed since last DB commit/vector flush.
    inserted_count = 0  # New stories inserted into Postgres.
    reindexed_count = 0  # Existing stories whose vectors were rebuilt.
    skipped_count = 0  # Stories skipped because already ingested or missing required data.
    failed_count = 0  # Lines skipped because of invalid JSON/IDs.

    try:  # Ensures DB session closes even if ingestion fails.
        with open(file_path, "r", encoding="utf-8") as file:  # Opens the JSONL/TXT file.
            for line_number, line in enumerate(file, start=1):  # Reads one story per line with a human-friendly line number.
                clean_line = line.strip()  # Removes whitespace/newline around JSON.

                if not clean_line:  # Blank line has no story.
                    continue  # Skip blank line.

                try:  # JSON parsing can fail for corrupted lines.
                    data = json.loads(clean_line)  # Turns raw JSON text into a Python dictionary.
                except json.JSONDecodeError as exc:  # Invalid JSON line.
                    failed_count += 1  # Track failure.
                    print(f"Line {line_number}: invalid JSON: {exc}")  # Print useful debug message.
                    continue  # Move to next line.

                source_id = data.get("id")  # Bar & Bench story UUID from raw data.

                if not source_id:  # We need an ID to dedupe stories/chunks.
                    skipped_count += 1  # Track skip.
                    continue  # Skip stories with no ID.

                try:  # Raw ID must be a valid UUID string.
                    story_id = uuid.UUID(source_id)  # Converts string to uuid.UUID for Postgres primary key.
                except ValueError as exc:  # Bad UUID string.
                    failed_count += 1  # Track failure.
                    print(f"Line {line_number}: invalid story id {source_id}: {exc}")  # Print debug info.
                    continue  # Skip this story.

                headline = data.get("headline", "Untitled")  # Article headline, fallback if missing.
                summary = data.get("seo", {}).get("meta-description", "")  # SEO summary/meta description.
                published_at = timestamp_ms_to_datetime(  # Convert raw timestamp into datetime.
                    data.get("published-at")  # Preferred publish timestamp.
                    or data.get("first-published-at")  # Fallback timestamp.
                    or data.get("last-published-at")  # Last fallback timestamp.
                )
                topics = [  # Extract tag names.
                    tag.get("name")  # One tag name.
                    for tag in data.get("tags", [])  # Loop through tags list safely.
                    if tag.get("name")  # Keep only tags with names.
                ]
                categories = [  # Extract article section/category names.
                    section.get("name")  # One section name.
                    for section in data.get("sections", [])  # Loop through sections list safely.
                    if section.get("name")  # Keep only sections with names.
                ]
                full_content = extract_full_content(data)  # Clean full article body from cards/story-elements.

                if not full_content:  # A story with no text cannot be useful for RAG.
                    skipped_count += 1  # Track skip.
                    continue  # Skip empty story.

                existing_story = db.get(StoryMetadata, story_id)  # Checks if this story is already in Postgres.

                if existing_story:  # If story already exists, avoid duplicate DB rows.
                    if published_at and not existing_story.published_at:  # Fill missing publish date if old row did not have it.
                        existing_story.published_at = published_at  # Update DB object; commit happens later.

                    if not REINDEX_EXISTING:  # Normal mode: do not rebuild vectors for existing stories.
                        skipped_count += 1  # Count as skipped.
                        continue  # Move to next story.

                    headline = existing_story.headline or headline  # Prefer existing DB headline.
                    summary = existing_story.summary or summary  # Prefer existing DB summary.
                    published_at = existing_story.published_at or published_at  # Prefer existing DB publish date.
                    full_content = existing_story.full_content or full_content  # Prefer existing DB full content.
                    reindexed_count += 1  # Track that vectors will be rebuilt.

                else:  # New story: insert metadata/full content into Postgres.
                    story = StoryMetadata(  # Create SQLAlchemy row object.
                        id=story_id,  # Primary key.
                        headline=headline,  # Article title.
                        summary=summary,  # Article summary/meta description.
                        published_at=published_at,  # Publish datetime.
                        topics=topics,  # Tags/topics.
                        categories=categories,  # Sections/categories.
                        full_content=full_content,  # Clean full article text.
                    )
                    db.add(story)  # Stage row for insertion.
                    inserted_count += 1  # Track new insert.

                chunks = text_splitter.split_text(full_content)  # Split full story into smaller searchable chunks.

                for chunk_index, chunk in enumerate(chunks):  # Give every chunk a number inside its story.
                    pending_texts.append(chunk)  # Queue chunk text for OpenSearch.
                    pending_metadatas.append(  # Queue metadata for same chunk.
                        {  # Metadata travels with the chunk during retrieval.
                            "story_id": str(story_id),  # Lets workflow fetch full story from Postgres later.
                            "headline": headline,  # Lets UI/retrieval show title without DB lookup.
                            "published_at": datetime_to_ist_date(published_at),  # Date string for filtering.
                            "chunk_index": chunk_index,  # Helps dedupe and trace chunk position.
                        }
                    )
                    pending_ids.append(f"{story_id}-{chunk_index}")  # Stable vector ID for this story chunk.

                stories_since_commit += 1  # One more story processed since last flush.

                if stories_since_commit >= STORY_COMMIT_BATCH_SIZE:  # Periodically flush to avoid huge memory buildup.
                    flush_vectors(pending_texts, pending_metadatas, pending_ids)  # Write queued chunks to OpenSearch.
                    db.commit()  # Commit staged Postgres inserts/updates.
                    stories_since_commit = 0  # Reset batch counter.
                    print(  # Progress log so long ingestion does not look stuck.
                        f"Line {line_number}: inserted={inserted_count}, "
                        f"reindexed={reindexed_count}, "
                        f"skipped={skipped_count}, failed={failed_count}"
                    )

        flush_vectors(pending_texts, pending_metadatas, pending_ids)  # Flush remaining chunks after file ends.
        db.commit()  # Commit remaining DB work.

        print(  # Final ingestion summary.
            f"Ingestion complete. inserted={inserted_count}, "
            f"reindexed={reindexed_count}, "
            f"skipped={skipped_count}, failed={failed_count}"
        )

    finally:  # Always close DB session.
        db.close()  # Prevents database connection leaks.


if __name__ == "__main__":  # Runs only when this file is executed directly, not when imported.
    ingest_data("data/stories-barandbench-1.txt")  # Default ingestion file path inside Docker/project data folder.
