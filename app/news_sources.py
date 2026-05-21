from dataclasses import dataclass


@dataclass(frozen=True)
class NewsSourceProfile:
    """
    Describe how one news source stores data in Pinecone.

    Different sources use different metadata field names.

    Example:
    - Sakal stores article IDs in "article_id".
    - Bar & Bench stores story IDs in "story_id".

    Retrieval code reads this profile so it does not need hardcoded source checks
    scattered everywhere.
    """
    source: str
    display_name: str
    date_filter_field: str
    id_metadata_field: str
    published_at_metadata_field: str
    topics_metadata_field: str
    category_metadata_fields: tuple[str, ...]
    chunk_overlap_words: int = 0
    hydrate_sources_from_postgres: bool = False
    retrieval_workflow_detail: str = "chunk context"


SOURCE_PROFILES = {
    "barandbench": NewsSourceProfile(
        source="barandbench",
        display_name="Bar & Bench",
        date_filter_field="published_at_yyyymmdd",
        id_metadata_field="story_id",
        published_at_metadata_field="published_at",
        topics_metadata_field="topics",
        category_metadata_fields=("categories",),
        hydrate_sources_from_postgres=True,
        retrieval_workflow_detail="Bar & Bench full-article Postgres hydration",
    ),
    "sakal": NewsSourceProfile(
        source="sakal",
        display_name="Sakal",
        date_filter_field="date_published_yyyymmdd",
        id_metadata_field="article_id",
        published_at_metadata_field="date_published",
        topics_metadata_field="keywords",
        category_metadata_fields=("edition", "source", "location"),
        chunk_overlap_words=50,
        retrieval_workflow_detail="Sakal same-story chunk merge",
    ),
}


def source_profile(source: str | None) -> NewsSourceProfile:
    """
    Return the profile for the selected news source.

    Example:
    source_profile("sakal") tells retrieval to use Sakal date fields,
    Sakal article IDs, and Sakal chunk-overlap rules.
    """
    source_name = (source or "sakal").strip().lower()

    try:
        return SOURCE_PROFILES[source_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported news source: {source}") from exc
