import os  # Reads OpenSearch index settings from environment variables.

from dotenv import load_dotenv  # Loads .env values during local development.
from opensearchpy import OpenSearch  # Client used to delete the OpenSearch index.

from database import SessionLocal  # Creates Postgres sessions.
from models import StoryMetadata  # SQLAlchemy model for the stories table.


load_dotenv()  # Makes .env variables available through os.getenv.

INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "news_index")  # Index to delete/reset in OpenSearch.


def reset_postgres() -> None:  # Deletes all stored full stories from Postgres.
    db = SessionLocal()  # Opens DB session.

    try:  # Ensures session closes even if deletion fails.
        deleted_count = db.query(StoryMetadata).delete()  # Deletes rows from stories table.
        db.commit()  # Saves deletion permanently.
        print(f"Deleted {deleted_count} stories from Postgres.")  # Shows how many rows were removed.

    finally:  # Always runs.
        db.close()  # Closes DB session.


def reset_opensearch() -> None:  # Deletes vector chunks from OpenSearch.
    client = OpenSearch(  # Creates OpenSearch client.
        hosts=[os.getenv("OPENSEARCH_URL")],  # OpenSearch URL.
        use_ssl=True,  # Use HTTPS.
        verify_certs=True,  # Verify SSL certificate.
        ssl_assert_hostname=False,  # Avoid hostname mismatch issues on managed OpenSearch.
        ssl_show_warn=False,  # Hide noisy SSL warnings.
    )

    if client.indices.exists(index=INDEX_NAME):  # Check if index exists before deleting.
        client.indices.delete(index=INDEX_NAME)  # Delete vector index.
        print(f"Deleted OpenSearch index: {INDEX_NAME}")  # Confirm deletion.
    else:  # If index is already gone...
        print(f"OpenSearch index did not exist: {INDEX_NAME}")  # Explain no deletion happened.


if __name__ == "__main__":  # Runs only when this script is executed directly.
    reset_postgres()  # Clear Postgres stories.
    reset_opensearch()  # Clear OpenSearch chunks.
    print("Reset complete.")  # Final confirmation.
