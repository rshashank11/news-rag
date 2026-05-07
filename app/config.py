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

    bm25_encoder_path: str = "bm25_values.json"

    retrieval_top_k: int = 10
    hybrid_alpha: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
