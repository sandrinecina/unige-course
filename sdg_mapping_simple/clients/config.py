from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

"""Centralized application settings loaded from environment variables."""


class Settings(BaseSettings):
    """Typed configuration for environment-dependent values."""

    llm_provider: str = "azure"
    openai_api_key: str | None = None
    azure_openai_api_key: str | None = None
    azure_api_endpoint: str | None = None
    azure_api_key: str | None = None
    mistral_document_ai_endpoint: str | None = None
    anthropic_api_key: str | None = None
    zefix_username: str | None = None
    zefix_password: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()  # type: ignore[call-arg]


def reset_settings_cache() -> None:
    """Reset cached settings (useful in tests)."""

    get_settings.cache_clear()

