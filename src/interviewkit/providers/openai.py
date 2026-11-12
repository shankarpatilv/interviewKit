"""OpenAI chat provider."""

import json
from typing import Any, cast

from openai import AsyncOpenAI

from interviewkit.config import Settings, get_settings
from interviewkit.providers.base import (
    Message,
    ProviderAPIError,
    ProviderResponse,
    Tool,
    ToolCall,
)

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
        self._client = client or self._build_client()
        self._model = model

    def _build_client(self) -> Any:
        """Build an AsyncOpenAI client, traced via Langfuse's drop-in when enabled."""
        api_key = self._settings.provider_api_key("openai").get_secret_value()
        if not self._settings.tracing_enabled:
            return AsyncOpenAI(api_key=api_key)
        from langfuse.openai import AsyncOpenAI as TracedAsyncOpenAI  # type: ignore[attr-defined]

        from interviewkit.tracing import get_langfuse_client

        get_langfuse_client(self._settings)
        return TracedAsyncOpenAI(api_key=api_key)

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

        message = response.choices[0].message
        content = message.content or ""
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            content=content,
            provider="openai",
            model=self._model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_input_tokens=getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
            )
            or 0,
            tool_calls=_tool_calls(message),
        )


def _tool_calls(message: Any) -> tuple[ToolCall, ...]:
    """Normalize OpenAI function tool calls; unparsable arguments become {}."""
    calls = []
    for call in getattr(message, "tool_calls", None) or ():
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
    return tuple(calls)
