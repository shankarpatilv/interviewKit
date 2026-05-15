"""OpenAI chat provider."""

from typing import Any, cast

from openai import AsyncOpenAI

from interviewkit.config import Settings, get_settings
from interviewkit.providers.base import Message, ProviderAPIError, ProviderResponse, Tool

OPENAI_CHAT_MODEL = "gpt-4o"


class OpenAIProvider:
    """Async OpenAI provider behind the shared provider interface."""

    def __init__(
        self,
        app_settings: Settings | None = None,
        client: Any | None = None,
        model: str = OPENAI_CHAT_MODEL,
    ) -> None:
        self._settings = app_settings or get_settings()
        self._client = client or AsyncOpenAI(
            api_key=self._settings.provider_api_key("openai").get_secret_value()
        )
        self._model = model

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> ProviderResponse:
        """Return a normalized OpenAI completion response."""
        request: dict[str, Any] = {
            "model": self._model,
            "messages": cast(Any, messages),
        }
        if tools is not None:
            request["tools"] = cast(Any, tools)

        try:
            response = await self._client.chat.completions.create(**request)
        except Exception as exc:
            raise ProviderAPIError("OpenAI API call failed") from exc

        content = response.choices[0].message.content or ""
        return ProviderResponse(content=content, provider="openai", model=self._model)
