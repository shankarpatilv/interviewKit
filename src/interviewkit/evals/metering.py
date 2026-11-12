"""Token-counting provider wrapper and cost estimates for eval runs."""

import logging

from interviewkit.providers import LLMProvider, Message, ProviderResponse, Tool

logger = logging.getLogger(__name__)

# Estimated USD per 1M tokens: (input, cached input, output). Update when pricing changes.
PRICES_PER_MILLION: dict[str, tuple[float, float, float]] = {
    "gpt-4o": (2.50, 1.25, 10.00),
    "claude-3-5-sonnet-20241022": (3.00, 0.30, 15.00),
}


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0
) -> float:
    """Estimate USD cost; cached input tokens (a subset of input) use the cache rate."""
    prices = PRICES_PER_MILLION.get(model)
    if prices is None:
        if model:
            logger.warning("No price configured for model %s; cost reported as 0", model)
        return 0.0
    input_price, cached_price, output_price = prices
    cached = min(cached_input_tokens, input_tokens)
    fresh = input_tokens - cached
    return (fresh * input_price + cached * cached_price + output_tokens * output_price) / 1e6


class MeteredProvider:
    """Wrap a provider and accumulate token usage between resets."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.reset()

    def reset(self) -> None:
        """Clear accumulated usage before the next case."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_input_tokens = 0
        self.model = ""

    @property
    def cost_usd(self) -> float:
        """Estimated cost of usage since the last reset."""
        return estimate_cost(
            self.model, self.input_tokens, self.output_tokens, self.cached_input_tokens
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> ProviderResponse:
        """Delegate to the wrapped provider and record its token usage."""
        response = await self._inner.complete(messages, tools)
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.cached_input_tokens += response.cached_input_tokens
        self.model = response.model
        return response
