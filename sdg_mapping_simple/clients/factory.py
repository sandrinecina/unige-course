from __future__ import annotations

from typing import Literal

import anthropic
from openai import OpenAI

from .config import get_settings
from ._utils import require_setting


Provider = Literal["openai", "azure", "anthropic"]


def get_openai_client() -> OpenAI:
    """Create an OpenAI client using OPENAI_API_KEY."""

    settings = get_settings()
    return OpenAI(api_key=require_setting(settings.openai_api_key, name="OPENAI_API_KEY"))


def get_azure_openai_client() -> OpenAI:
    """Create an Azure OpenAI-compatible client using AZURE env vars."""

    settings = get_settings()
    return OpenAI(
        api_key=require_setting(settings.azure_openai_api_key, name="AZURE_OPENAI_API_KEY"),
        base_url=require_setting(settings.azure_api_endpoint, name="AZURE_API_ENDPOINT"),
    )


def get_anthropic_client() -> anthropic.Anthropic:
    """Create an Anthropic client using ANTHROPIC_API_KEY."""

    settings = get_settings()
    return anthropic.Anthropic(
        api_key=require_setting(settings.anthropic_api_key, name="ANTHROPIC_API_KEY"),
    )


def get_client(provider: Provider | None = None) -> OpenAI | anthropic.Anthropic:
    """Return an SDK client for the given provider."""

    settings = get_settings()
    choice = provider or settings.llm_provider
    match choice.lower():
        case "openai":
            return get_openai_client()
        case "azure":
            return get_azure_openai_client()
        case "anthropic":
            return get_anthropic_client()
    raise ValueError(f"Unsupported provider: {provider or settings.llm_provider}")


