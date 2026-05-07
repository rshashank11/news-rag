from app.openai_client import get_embedding_model, make_embedding_client


client = make_embedding_client()


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    response = client.embeddings.create(
        model=get_embedding_model(),
        input=texts,
    )

    sorted_items = sorted(response.data, key=lambda item: item.index)

    return [
        item.embedding
        for item in sorted_items
    ]


def embed_text(text: str) -> list[float]:
    embeddings = embed_texts([text])
    return embeddings[0]
