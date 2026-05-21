from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from app.config import settings


def use_azure_openai() -> bool:
    """
    Check whether Azure OpenAI is configured.

    If false, the app falls back to the normal OpenAI API settings.
    """
    return bool(settings.azure_openai_api_key and settings.azure_openai_endpoint)


def use_azure_chat() -> bool:
    """
    Check whether chat/completion calls should use Azure.

    Embeddings can use Azure separately, but chat needs a chat deployment name.
    """
    return bool(use_azure_openai() and settings.azure_openai_chat_deployment)


def make_azure_openai_client(api_version: str) -> AsyncAzureOpenAI:
    """
    Create an async Azure OpenAI client.

    Async clients are useful for endpoints that can await model calls.
    """
    if not use_azure_openai():
        raise RuntimeError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_API_KEY and "
            "AZURE_OPENAI_ENDPOINT."
        )

    return AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )


def make_sync_azure_openai_client(api_version: str) -> AzureOpenAI:
    """
    Create a blocking Azure OpenAI client.

    LangGraph workflow nodes here run synchronously, so they use sync clients.
    """
    if not use_azure_openai():
        raise RuntimeError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_API_KEY and "
            "AZURE_OPENAI_ENDPOINT."
        )

    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )


def make_direct_openai_client() -> AsyncOpenAI:
    """
    Create an async OpenAI client using OPENAI_API_KEY.
    """
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    return AsyncOpenAI(api_key=settings.openai_api_key)


def make_sync_direct_openai_client() -> OpenAI:
    """
    Create a blocking OpenAI client using OPENAI_API_KEY.
    """
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    return OpenAI(api_key=settings.openai_api_key)


def make_embedding_client() -> AsyncOpenAI | AsyncAzureOpenAI:
    """
    Create the async client for embedding calls.

    Example:
    If Azure credentials are set, use Azure.
    Otherwise use direct OpenAI.
    """
    if use_azure_openai():
        api_version = (
            settings.azure_openai_embedding_api_version
            or settings.azure_openai_api_version
        )
        return make_azure_openai_client(api_version=api_version)

    return make_direct_openai_client()


def make_sync_embedding_client() -> OpenAI | AzureOpenAI:
    """
    Create the blocking client for embedding calls.

    Ingestion scripts use this because they run as normal Python scripts, not
    async web handlers.
    """
    if use_azure_openai():
        api_version = (
            settings.azure_openai_embedding_api_version
            or settings.azure_openai_api_version
        )
        return make_sync_azure_openai_client(api_version=api_version)

    return make_sync_direct_openai_client()


def make_chat_client() -> AsyncOpenAI | AsyncAzureOpenAI:
    """
    Create the async client for chat/model calls.
    """
    if use_azure_chat():
        return make_azure_openai_client(
            api_version=settings.azure_openai_chat_api_version
        )

    return make_direct_openai_client()


def make_sync_chat_client() -> OpenAI | AzureOpenAI:
    """
    Create the blocking chat client used by planner and workflow steps.
    """
    if use_azure_chat():
        return make_sync_azure_openai_client(
            api_version=settings.azure_openai_chat_api_version
        )

    return make_sync_direct_openai_client()


def get_embedding_model() -> str:
    """
    Return the embedding model name.

    With Azure, this returns the deployment name.
    With direct OpenAI, this returns something like "text-embedding-3-small".
    """
    if use_azure_openai() and settings.azure_openai_embedding_deployment:
        return settings.azure_openai_embedding_deployment

    return settings.openai_embedding_model


def get_chat_model() -> str:
    """
    Return the chat model name.

    With Azure, this returns the chat deployment name.
    With direct OpenAI, this returns the configured chat model.
    """
    if use_azure_chat():
        return settings.azure_openai_chat_deployment

    return settings.openai_chat_model


def get_planner_model() -> str:
    """
    Return the model used to plan search queries.

    The planner can use a separate model if configured; otherwise it shares the
    normal chat model.
    """
    if use_azure_chat() and settings.azure_openai_planner_deployment:
        return settings.azure_openai_planner_deployment

    if settings.openai_planner_model:
        return settings.openai_planner_model

    return get_chat_model()


def get_context_judge_model() -> str:
    """
    Return the model used to judge whether retrieved sources are enough.
    """
    return get_chat_model()


def get_query_rewrite_model() -> str:
    """
    Return the model used to rewrite weak search queries.
    """
    return get_chat_model()


def get_answer_model() -> str:
    """
    Return the model used to write the final grounded answer.
    """
    return get_chat_model()
