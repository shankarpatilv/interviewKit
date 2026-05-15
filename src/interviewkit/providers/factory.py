"""Provider selection helpers."""

from interviewkit.config import ProviderName, Settings, get_settings
from interviewkit.providers.base import LLMProvider
from interviewkit.providers.claude import ClaudeProvider
from interviewkit.providers.openai import OpenAIProvider


def get_provider(
    provider: ProviderName | None = None,
    app_settings: Settings | None = None,
) -> LLMProvider:
    """Return the configured LLM provider."""
    resolved_settings = app_settings or get_settings()
    selected = provider or resolved_settings.default_provider
    if selected == "openai":
        return OpenAIProvider(app_settings=resolved_settings)
    return ClaudeProvider(app_settings=resolved_settings)
