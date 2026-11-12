"""Shared provider abstractions."""

from dataclasses import dataclass, field
from typing import Any, Protocol

# Chat messages use the OpenAI shape; tool-call turns carry non-string values.
Message = dict[str, Any]
Tool = dict[str, object]


@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized LLM provider response."""

    content: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # subset of input_tokens billed at the cache rate
    tool_calls: tuple[ToolCall, ...] = ()


class ProviderError(RuntimeError):
    """Base error for provider failures."""


class ProviderConfigurationError(ProviderError):
    """Raised when a provider cannot be configured."""


class ProviderAPIError(ProviderError):
    """Raised when a provider API call fails."""


class LLMProvider(Protocol):
    """Common async interface for LLM providers."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> ProviderResponse:
        """Complete a chat-style request."""
