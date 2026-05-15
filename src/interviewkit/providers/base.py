"""Shared provider abstractions."""

from dataclasses import dataclass
from typing import Protocol

Message = dict[str, str]
Tool = dict[str, object]


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized LLM provider response."""

    content: str
    provider: str
    model: str


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
