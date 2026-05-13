import time

from openai import APIError, RateLimitError

from app.openai_client import get_embedding_model, make_sync_embedding_client


client = make_sync_embedding_client()

DEFAULT_EMBEDDING_BATCH_SIZE = 100
ARTICLE_CHUNK_SIZE_WORDS = 500
ARTICLE_CHUNK_OVERLAP_WORDS = 50
MAX_EMBEDDING_RETRIES = 8
MIN_RATE_LIMIT_WAIT_SECONDS = 2


def clean_article_part(value) -> str:
    # Convert missing values into empty text so embedding code does not crash.
    if value is None:
        return ""

    # Convert the value to text and collapse extra spaces/newlines.
    return " ".join(str(value).split())


def build_article_embedding_text(article: dict) -> str:
    # Pull the main searchable fields from the converted XML JSON article.
    headline = clean_article_part(article.get("headline"))
    slug = clean_article_part(article.get("slug"))
    summary = clean_article_part(article.get("summary"))
    body = clean_article_part(article.get("body"))

    # Keywords are stored as a list, so join them into one readable text line.
    keywords = article.get("keywords") or []
    keyword_text = ", ".join(clean_article_part(keyword) for keyword in keywords)

    # Labels like "Headline:" help the embedding model understand each section.
    labeled_parts = [
        ("Headline", headline),
        ("Slug", slug),
        ("Summary", summary),
        ("Keywords", keyword_text),
        ("Body", body),
    ]

    # Keep only fields that actually have text.
    parts = [
        f"{label}: {text}"
        for label, text in labeled_parts
        if text
    ]

    # This final text is what gets sent to the embedding model.
    return "\n\n".join(parts)


def split_text_into_word_chunks(
    text: str,
    chunk_size_words: int = ARTICLE_CHUNK_SIZE_WORDS,
    overlap_words: int = ARTICLE_CHUNK_OVERLAP_WORDS,
) -> list[str]:
    # Split the body into words because our chunk sizes are word-based.
    words = clean_article_part(text).split()

    if not words:
        return []

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than 0.")

    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")

    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words.")

    chunks = []
    start = 0

    while start < len(words):
        # Take up to chunk_size_words words for this chunk.
        end = start + chunk_size_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        # Move forward, but keep overlap_words from the previous chunk.
        start = end - overlap_words

    return chunks


def build_article_chunk_embedding_text(
    article: dict,
    chunk_text: str,
) -> str:
    # Each chunk repeats headline/keywords so headline searches still work.
    headline = clean_article_part(article.get("headline"))
    keywords = article.get("keywords") or []
    keyword_text = ", ".join(clean_article_part(keyword) for keyword in keywords)
    chunk_text = clean_article_part(chunk_text)

    labeled_parts = [
        ("Headline", headline),
        ("Keywords", keyword_text),
        ("Body chunk", chunk_text),
    ]

    parts = [
        f"{label}: {text}"
        for label, text in labeled_parts
        if text
    ]

    return "\n\n".join(parts)


def article_body_chunks(article: dict) -> list[str]:
    # Most Sakal articles are short, so they will naturally return one chunk.
    body = clean_article_part(article.get("body"))
    return split_text_into_word_chunks(body)


def article_chunk_records(articles: list[dict]) -> list[tuple[dict, str, int, str]]:
    records = []

    for article in articles:
        article_id = clean_article_part(article.get("id"))

        if not article_id:
            continue

        for chunk_index, chunk_text in enumerate(article_body_chunks(article)):
            records.append((article, article_id, chunk_index, chunk_text))

    return records


def article_metadata(
    article: dict,
    chunk_index: int | None = None,
    chunk_text: str | None = None,
) -> dict:
    # raw contains original XML details like PublicIdentifier and source file.
    raw = article.get("raw") or {}

    # Metadata is stored with the vector for filtering and display.
    # The numeric date is important for fast timeline filters in Pinecone.
    metadata = {
        "article_id": article.get("id"),
        "headline": article.get("headline"),
        "date_published": article.get("date_published"),
        "date_published_yyyymmdd": article.get("date_published_yyyymmdd"),
        "date_created": article.get("date_created"),
        "date_created_yyyymmdd": article.get("date_created_yyyymmdd"),
        "location": article.get("location"),
        "edition": article.get("edition"),
        "source": article.get("source"),
        "language": article.get("language"),
        "keywords": article.get("keywords") or [],
        "xml_id": raw.get("xml_id"),
        "chunk_index": chunk_index,
        "chunk_text": clean_article_part(chunk_text),
    }

    # Pinecone metadata should not include empty values.
    return {
        key: value
        for key, value in metadata.items()
        if value not in (None, "", [])
    }


def chunk_items(items: list, batch_size: int):
    # Split a large list into smaller batches so API calls stay manageable.
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    for attempt in range(MAX_EMBEDDING_RETRIES):
        try:
            response = client.embeddings.create(
                model=get_embedding_model(),
                input=texts,
            )
            break
        except RateLimitError as exc:
            wait_seconds = max(MIN_RATE_LIMIT_WAIT_SECONDS, 2 ** attempt)
            print(
                "Embedding rate limit hit. "
                f"Retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)
        except APIError as exc:
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
    embeddings = embed_texts([text])
    return embeddings[0]


def embed_articles(
    articles: list[dict],
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    # This returns one embedding per article chunk, not one embedding per article.
    embeddings = []
    records = article_chunk_records(articles)

    for record_batch in chunk_items(records, batch_size):
        # Convert each article chunk into one searchable text string.
        texts = [
            build_article_chunk_embedding_text(article, chunk_text)
            for article, _, _, chunk_text in record_batch
        ]

        # Send the batch to the embedding model and collect the vectors.
        embeddings.extend(embed_texts(texts))

    return embeddings


def articles_to_pinecone_vectors(
    articles: list[dict],
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> list[dict]:
    # This returns Pinecone-ready vector dictionaries.
    vectors = []
    article_chunks = article_chunk_records(articles)

    for article_chunk_batch in chunk_items(article_chunks, batch_size):
        # Build the text that will be converted into embeddings.
        texts = [
            build_article_chunk_embedding_text(article, chunk_text)
            for article, _, _, chunk_text in article_chunk_batch
        ]

        # Create one embedding vector for each article chunk.
        embeddings = embed_texts(texts)

        for (article, article_id, chunk_index, chunk_text), embedding in zip(
            article_chunk_batch,
            embeddings,
        ):
            # This shape is what Pinecone upsert expects:
            # id = unique article chunk ID
            # values = embedding numbers
            # metadata = filter/display fields
            vectors.append(
                {
                    "id": f"{article_id}-{chunk_index}",
                    "values": embedding,
                    "metadata": article_metadata(
                        article,
                        chunk_index=chunk_index,
                        chunk_text=chunk_text,
                    ),
                }
            )

    return vectors
