from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from app.config import settings


def use_azure_openai() -> bool:
    return bool(settings.azure_openai_api_key and settings.azure_openai_endpoint)


def use_azure_chat() -> bool:
    return bool(use_azure_openai() and settings.azure_openai_chat_deployment)


def make_azure_openai_client(api_version: str) -> AsyncAzureOpenAI:
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
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    return AsyncOpenAI(api_key=settings.openai_api_key)


def make_sync_direct_openai_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    return OpenAI(api_key=settings.openai_api_key)


def make_embedding_client() -> AsyncOpenAI | AsyncAzureOpenAI:
    if use_azure_openai():
        api_version = (
            settings.azure_openai_embedding_api_version
            or settings.azure_openai_api_version
        )
        return make_azure_openai_client(api_version=api_version)

    return make_direct_openai_client()


def make_sync_embedding_client() -> OpenAI | AzureOpenAI:
    if use_azure_openai():
        api_version = (
            settings.azure_openai_embedding_api_version
            or settings.azure_openai_api_version
        )
        return make_sync_azure_openai_client(api_version=api_version)

    return make_sync_direct_openai_client()


def make_chat_client() -> AsyncOpenAI | AsyncAzureOpenAI:
    if use_azure_chat():
        return make_azure_openai_client(
            api_version=settings.azure_openai_chat_api_version
        )

    return make_direct_openai_client()


def make_sync_chat_client() -> OpenAI | AzureOpenAI:
    if use_azure_chat():
        return make_sync_azure_openai_client(
            api_version=settings.azure_openai_chat_api_version
        )

    return make_sync_direct_openai_client()


def get_embedding_model() -> str:
    if use_azure_openai() and settings.azure_openai_embedding_deployment:
        return settings.azure_openai_embedding_deployment

    return settings.openai_embedding_model


def get_chat_model() -> str:
    if use_azure_chat():
        return settings.azure_openai_chat_deployment

    return settings.openai_chat_model


def get_planner_model() -> str:
    if use_azure_chat() and settings.azure_openai_planner_deployment:
        return settings.azure_openai_planner_deployment

    if settings.openai_planner_model:
        return settings.openai_planner_model

    return get_chat_model()


def get_context_judge_model() -> str:
    return get_chat_model()


def get_query_rewrite_model() -> str:
    return get_chat_model()


def get_answer_model() -> str:
    return get_chat_model()
