"""Anthropic Claude chat provider."""

from importlib import import_module
from typing import Any

from interviewkit.config import Settings, get_settings
from interviewkit.providers.base import (
    Message,
    ProviderAPIError,
    ProviderConfigurationError,
    ProviderResponse,
    Tool,
)

CLAUDE_CHAT_MODEL = "claude-3-5-sonnet-20241022"


class ClaudeProvider:
    """Async Claude provider behind the shared provider interface."""

    def __init__(
        self,
        app_settings: Settings | None = None,
        client: Any | None = None,
        model: str = CLAUDE_CHAT_MODEL,
    ) -> None:
        self._settings = app_settings or get_settings()
        self._client = client or self._build_client()
        self._model = model

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> ProviderResponse:
        """Return a normalized Claude completion response."""
        request = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": messages,
        }
        if tools is not None:
            request["tools"] = tools

        try:
            response = await self._client.messages.create(**request)
        except Exception as exc:
            raise ProviderAPIError("Claude API call failed") from exc

        return ProviderResponse(
            content=_extract_text(response),
            provider="claude",
            model=self._model,
        )

    def _build_client(self) -> Any:
        try:
            anthropic = import_module("anthropic")
        except ImportError as exc:
            msg = "Anthropic SDK is required to use ClaudeProvider"
            raise ProviderConfigurationError(msg) from exc

        api_key = self._settings.provider_api_key("claude").get_secret_value()
        return anthropic.AsyncAnthropic(api_key=api_key)


def _extract_text(response: Any) -> str:
    blocks = getattr(response, "content", [])
    text_blocks = [
        getattr(block, "text", "") for block in blocks if getattr(block, "type", "") == "text"
    ]
    return "\n".join(text for text in text_blocks if text)
