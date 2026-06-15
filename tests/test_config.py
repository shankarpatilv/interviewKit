import pytest
from pydantic import ValidationError

from interviewkit.config import Settings


def test_settings_defaults_match_project_configuration() -> None:
    settings = Settings(_env_file=None)

    assert (
        settings.database_url
        == "postgresql://interviewkit:interviewkit@localhost:5432/interviewkit"
    )
    assert settings.default_provider == "claude"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100


def test_settings_load_unprefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5433/app")
    monkeypatch.setenv("DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("CHUNK_SIZE", "1200")
    monkeypatch.setenv("CHUNK_OVERLAP", "150")

    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql://user:pass@localhost:5433/app"
    assert settings.default_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 150


def test_settings_load_interviewkit_prefixed_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERVIEWKIT_DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("INTERVIEWKIT_DATABASE_URL", "postgresql://example.local/interviewkit")

    settings = Settings(_env_file=None)

    assert settings.default_provider == "openai"
    assert settings.database_url == "postgresql://example.local/interviewkit"


def test_default_provider_must_be_supported() -> None:
    with pytest.raises(ValidationError, match="claude|openai"):
        Settings(default_provider="local", _env_file=None)


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP must be smaller than CHUNK_SIZE"):
        Settings(chunk_size=100, chunk_overlap=100, _env_file=None)


def test_missing_selected_provider_key_raises_clear_error() -> None:
    settings = Settings(default_provider="claude", _env_file=None)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
        settings.provider_api_key()


def test_provider_api_key_returns_selected_key() -> None:
    settings = Settings(default_provider="openai", openai_api_key="sk-test", _env_file=None)

    assert settings.provider_api_key().get_secret_value() == "sk-test"
