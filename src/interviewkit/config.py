"""Application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, PositiveInt, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["claude", "openai"]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and defaults."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "INTERVIEWKIT_ANTHROPIC_API_KEY"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "INTERVIEWKIT_OPENAI_API_KEY"),
    )
    database_url: str = Field(
        default="postgresql://interviewkit:interviewkit@localhost:5432/interviewkit",
        validation_alias=AliasChoices("DATABASE_URL", "INTERVIEWKIT_DATABASE_URL"),
    )
    default_provider: ProviderName = Field(
        default="openai",
        validation_alias=AliasChoices("DEFAULT_PROVIDER", "INTERVIEWKIT_DEFAULT_PROVIDER"),
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "INTERVIEWKIT_EMBEDDING_MODEL"),
    )
    chunk_size: PositiveInt = Field(
        default=800,
        validation_alias=AliasChoices("CHUNK_SIZE", "INTERVIEWKIT_CHUNK_SIZE"),
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        validation_alias=AliasChoices("CHUNK_OVERLAP", "INTERVIEWKIT_CHUNK_OVERLAP"),
    )
    langfuse_public_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY", "INTERVIEWKIT_LANGFUSE_PUBLIC_KEY"),
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_SECRET_KEY", "INTERVIEWKIT_LANGFUSE_SECRET_KEY"),
    )
    langfuse_host: str = Field(
        default="http://localhost:3000",
        validation_alias=AliasChoices("LANGFUSE_HOST", "INTERVIEWKIT_LANGFUSE_HOST"),
    )
    experiences_dir: Path = Path("experiences")
    companies_dir: Path = Path("companies")
    sessions_dir: Path = Path("sessions")

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        """Ensure overlapping chunks are smaller than the chunk body."""
        if self.chunk_overlap >= self.chunk_size:
            msg = "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
            raise ValueError(msg)
        return self

    @property
    def tracing_enabled(self) -> bool:
        """Return True only when both Langfuse keys are configured."""
        keys = (self.langfuse_public_key, self.langfuse_secret_key)
        return all(key is not None and key.get_secret_value() for key in keys)

    def provider_api_key(self, provider: ProviderName | None = None) -> SecretStr:
        """Return the configured API key for a provider or raise a clear error."""
        selected_provider = provider or self.default_provider
        key = self.anthropic_api_key if selected_provider == "claude" else self.openai_api_key
        if key is None:
            env_var = "ANTHROPIC_API_KEY" if selected_provider == "claude" else "OPENAI_API_KEY"
            msg = f"{env_var} is required when DEFAULT_PROVIDER={selected_provider!r}"
            raise ValueError(msg)
        return key


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings loaded from the environment."""
    return Settings()


settings = get_settings()
