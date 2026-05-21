from schemas import RetrievedChunk


def truncate_text(text: str, max_chars: int) -> str:
    """
    Shorten very long source text before sending it to the model.

    Example:
    If an article is 20,000 characters but the prompt budget allows 5,000,
    this keeps the first useful part and adds "... [truncated]" so the model
    knows it did not receive the entire article.
    """
    if len(text) <= max_chars:
        return text

    suffix = " ... [truncated]"
    return f"{text[:max_chars - len(suffix)].rstrip()}{suffix}"


def list_to_line(label: str, values: list[str] | None) -> str | None:
    """
    Turn metadata lists into one readable line for the prompt.

    Example:
    label="Topics", values=["PMLA", "Bail"]
    becomes "Topics: PMLA, Bail".
    """
    if not values:
        return None

    cleaned_values = [
        str(value).strip()
        for value in values
        if str(value).strip()
    ]

    if not cleaned_values:
        return None

    return f"{label}: {', '.join(cleaned_values)}"


def merge_consecutive_chunk_texts(
    chunks: list[RetrievedChunk],
    overlap_words: int = 0,
) -> str:
    """
    Join chunks from the same story back into one readable text.

    Some chunks overlap so we do not lose context at the boundary.

    Example:
    chunk 1 ends with 50 words that chunk 2 starts with.
    When rebuilding the article context, we remove those repeated 50 words.
    """
    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: chunk.chunk_index if chunk.chunk_index is not None else 0,
    )

    merged_words = []
    previous_chunk_index = None

    for chunk in ordered_chunks:
        chunk_words = chunk.chunk_text.split()

        if (
            overlap_words > 0
            and previous_chunk_index is not None
            and chunk.chunk_index == previous_chunk_index + 1
        ):
            chunk_words = chunk_words[overlap_words:]

        merged_words.extend(chunk_words)
        previous_chunk_index = chunk.chunk_index

    return " ".join(merged_words)


def build_combined_source_context(
    story_chunks: list[RetrievedChunk],
    overlap_words: int = 0,
) -> str:
    """
    Build the text block that the answer model sees for one source.

    It includes:
    - useful metadata like topics/categories
    - the matched chunk text
    - nearby same-story chunks when available
    """
    best_chunk = story_chunks[0]
    metadata_lines = []

    topics_line = list_to_line("Topics", best_chunk.topics)
    categories_line = list_to_line("Categories", best_chunk.categories)

    if topics_line:
        metadata_lines.append(topics_line)

    if categories_line:
        metadata_lines.append(categories_line)

    context_parts = []

    if metadata_lines:
        context_parts.append("\n".join(metadata_lines))

    context_parts.append(merge_consecutive_chunk_texts(story_chunks, overlap_words))

    return "\n\n".join(context_parts)
