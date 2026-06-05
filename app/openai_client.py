import functools
import os
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from app.config import settings

try:
    from langsmith.wrappers import wrap_openai
except ImportError:  # pragma: no cover - optional dependency in some deploys
    wrap_openai = None


def use_azure_openai() -> bool:
    return bool(settings.azure_openai_api_key and settings.azure_openai_endpoint)


def use_azure_chat() -> bool:
    return bool(use_azure_openai() and settings.azure_openai_chat_deployment)


def use_direct_context_judge() -> bool:
    return bool(
        settings.openai_api_key
        and getattr(settings, "openai_context_judge_model", "")
        and not getattr(settings, "azure_openai_context_judge_deployment", "")
    )


def tracing_enabled() -> bool:
    langsmith_tracing = os.getenv("LANGSMITH_TRACING", "")
    legacy_tracing = os.getenv("LANGCHAIN_TRACING_V2", "")
    return any(
        value.lower() in ("true", "1")
        for value in (langsmith_tracing, legacy_tracing)
    )


def maybe_wrap_openai_client(
    client: OpenAI | AzureOpenAI,
) -> OpenAI | AzureOpenAI:
    if tracing_enabled() and wrap_openai is not None:
        return add_default_langsmith_metadata(wrap_openai(client))
    return client


def langsmith_trace_model_name(model: str | None = None) -> str | None:
    override = (
        os.getenv("LANGSMITH_OPENAI_MODEL_NAME")
        or os.getenv("AZURE_OPENAI_MODEL_NAME")
    )
    if override:
        return override

    if use_azure_openai():
        return settings.openai_chat_model or model or None

    return model or settings.openai_chat_model or None


def langsmith_call_kwargs(model: str | None = None) -> dict[str, Any]:
    if not tracing_enabled() or wrap_openai is None:
        return {}

    trace_model = langsmith_trace_model_name(model)
    if not trace_model:
        return {}

    metadata = {
        "ls_provider": "openai",
        "ls_model_name": trace_model,
    }

    def apply_pricing_metadata(run_tree: Any) -> None:
        run_tree.add_metadata(metadata)

    return {
        "langsmith_extra": {
            "metadata": metadata,
            "_on_success": apply_pricing_metadata,
        }
    }


def add_default_langsmith_metadata(
    client: OpenAI | AzureOpenAI,
) -> OpenAI | AzureOpenAI:
    def patch_method(owner: Any, method_name: str) -> None:
        if not hasattr(owner, method_name):
            return

        original_method = getattr(owner, method_name)

        @functools.wraps(original_method)
        def wrapped_method(*args: Any, **kwargs: Any) -> Any:
            if "langsmith_extra" not in kwargs:
                kwargs.update(langsmith_call_kwargs(kwargs.get("model")))
            return original_method(*args, **kwargs)

        setattr(owner, method_name, wrapped_method)

    if hasattr(client, "responses"):
        patch_method(client.responses, "create")
        patch_method(client.responses, "parse")

    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        patch_method(client.chat.completions, "create")
        patch_method(client.chat.completions, "parse")

    if (
        hasattr(client, "beta")
        and hasattr(client.beta, "chat")
        and hasattr(client.beta.chat, "completions")
    ):
        patch_method(client.beta.chat.completions, "parse")

    return client


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
        client = make_sync_azure_openai_client(
            api_version=settings.azure_openai_chat_api_version
        )
    else:
        client = make_sync_direct_openai_client()

    return maybe_wrap_openai_client(client)


def make_sync_context_judge_client() -> OpenAI | AzureOpenAI:
    if use_direct_context_judge():
        client = make_sync_direct_openai_client()
        return maybe_wrap_openai_client(client)

    # make_sync_chat_client already applies wrap_openai when tracing is enabled
    return make_sync_chat_client()


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
    if use_azure_chat():
        azure_context_judge_deployment = getattr(
            settings, "azure_openai_context_judge_deployment", ""
        )
        openai_context_judge_model = getattr(
            settings, "openai_context_judge_model", ""
        )

        if azure_context_judge_deployment:
            return azure_context_judge_deployment

        if use_direct_context_judge():
            return openai_context_judge_model

        if settings.azure_openai_planner_deployment:
            return settings.azure_openai_planner_deployment

        return get_chat_model()

    openai_context_judge_model = getattr(settings, "openai_context_judge_model", "")
    if openai_context_judge_model:
        return openai_context_judge_model

    return get_planner_model()


def get_query_rewrite_model() -> str:
    return get_chat_model()


def get_answer_model() -> str:
    return get_chat_model()
