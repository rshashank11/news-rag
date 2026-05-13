from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    postgresql_url: str

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-12-01-preview"
    azure_openai_embedding_api_version: str = ""
    azure_openai_chat_api_version: str = "2025-04-01-preview"
    azure_openai_embedding_deployment: str = ""
    azure_openai_chat_deployment: str = ""

    pinecone_api_key: str
    pinecone_index_host: str
    pinecone_namespace: str = "default"

    bm25_encoder_path: str = "bm25_sakal_values.json"
    default_news_source: str = "sakal"

    barandbench_pinecone_index_host: str = ""
    barandbench_pinecone_namespace: str = "barandbench"
    barandbench_bm25_encoder_path: str = "bm25_values.json"

    sakal_pinecone_index_name: str = "sakal"
    sakal_pinecone_index_host: str = ""
    sakal_pinecone_namespace: str = "sakal_v1"
    sakal_bm25_encoder_path: str = "bm25_sakal_values.json"

    retrieval_top_k: int = 10
    min_retrieval_top_k: int = 1
    max_retrieval_top_k: int = 80
    date_fallback_top_k: int = 400
    hybrid_alpha: float = 0.5
    default_query_k: int = 10
    follow_up_query_k: int = 80
    max_history_messages: int = 8
    max_history_chars: int = 6000
    max_history_message_chars: int = 1200

    # Workflow configuration thresholds
    max_retrieval_attempts: int = 2
    max_answer_sources: int = 6
    max_context_chars_per_source: int = 6000
    max_story_excerpt_chars: int = 5000
    min_timeline_dated_sources: int = 2
    min_timeline_relevance_score: int = 5
    min_partial_briefing_relevance_score: int = 4
    min_negative_list_relevance_score: int = 3
    rerank_min_token_length: int = 3
    rerank_candidate_top_k: int = 80
    rerank_vector_score_weight: float = 1.0
    rerank_headline_overlap_weight: float = 2.5
    rerank_metadata_overlap_weight: float = 1.5
    rerank_chunk_overlap_weight: float = 0.75
    rerank_date_score_weight: float = 0.25
    rerank_story_evidence_weight: float = 0.15
    rerank_story_evidence_cap: int = 3
    rerank_debug_story_count: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def news_source_config(self, source: str | None = None) -> dict[str, str]:
        selected_source = (source or self.default_news_source).strip().lower()

        if selected_source == "barandbench":
            return {
                "source": "barandbench",
                "pinecone_index_host": self.barandbench_pinecone_index_host or self.pinecone_index_host,
                "pinecone_namespace": self.barandbench_pinecone_namespace,
                "bm25_encoder_path": self.barandbench_bm25_encoder_path,
            }

        if selected_source == "sakal":
            return {
                "source": "sakal",
                "pinecone_index_name": self.sakal_pinecone_index_name,
                "pinecone_index_host": self.sakal_pinecone_index_host,
                "pinecone_namespace": self.sakal_pinecone_namespace,
                "bm25_encoder_path": self.sakal_bm25_encoder_path,
            }

        raise ValueError(f"Unsupported news source: {source}")


settings = Settings()
