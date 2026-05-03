import os

from dotenv import load_dotenv
from opensearchpy import OpenSearch

from database import SessionLocal
from models import StoryMetadata


load_dotenv()

INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "news_index")


def reset_postgres() -> None:
    db = SessionLocal()

    try:
        deleted_count = db.query(StoryMetadata).delete()
        db.commit()
        print(f"Deleted {deleted_count} stories from Postgres.")

    finally:
        db.close()


def reset_opensearch() -> None:
    client = OpenSearch(
        hosts=[os.getenv("OPENSEARCH_URL")],
        use_ssl=True,
        verify_certs=True,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )

    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
        print(f"Deleted OpenSearch index: {INDEX_NAME}")
    else:
        print(f"OpenSearch index did not exist: {INDEX_NAME}")


if __name__ == "__main__":
    reset_postgres()
    reset_opensearch()
    print("Reset complete.")
