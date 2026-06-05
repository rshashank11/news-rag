import time

from openai import APIError, APIStatusError, RateLimitError

from app.openai_client import get_embedding_model, make_sync_embedding_client


client = make_sync_embedding_client()

DEFAULT_EMBEDDING_BATCH_SIZE = 100
MAX_EMBEDDING_RETRIES = 8
MIN_RATE_LIMIT_WAIT_SECONDS = 2


def clean_text_part(value) -> str:
    """
    Clean one text value before it is embedded or stored.

    This file stays source-neutral:
    - Sakal decides which Sakal fields to embed.
    - Bar & Bench decides which Bar & Bench fields to embed.
    - this function only makes the text safe and consistent.

    Example:
    " hello\\n\\n world " becomes "hello world".
    """
    if value is None:
        return ""

    return " ".join(str(value).split())


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Send many text chunks to the embedding model in one batch.

    Embedding means converting text into numbers that represent meaning.

    Example:
    "court grants bail" and "judge allows release" should get vectors that are
    close to each other because their meanings are related.
    """
    if not texts:
        return []

    for attempt in range(MAX_EMBEDDING_RETRIES):
        try:
            response = client.embeddings.create(
                model=get_embedding_model(),
                input=texts,
            )
            break
        except RateLimitError:
            wait_seconds = max(MIN_RATE_LIMIT_WAIT_SECONDS, 2 ** attempt)
            print(
                "Embedding rate limit hit. "
                f"Retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)
        except APIStatusError as exc:
            if exc.status_code < 500:
                raise
            wait_seconds = max(MIN_RATE_LIMIT_WAIT_SECONDS, 2 ** attempt)
            print(
                f"Embedding API error: {exc}. "
                f"Retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)
    else:
        raise RuntimeError("Embedding failed after multiple retries.")

    sorted_items = sorted(response.data, key=lambda item: item.index)

    return [
        item.embedding
        for item in sorted_items
    ]


def embed_text(text: str) -> list[float]:
    """
    Embed one piece of text.

    This is used during retrieval for the user's search query.

    Example:
    The query "teacher recruitment" becomes one vector that Pinecone can compare
    against stored article chunk vectors.
    """
    embeddings = embed_texts([text])
    return embeddings[0]
