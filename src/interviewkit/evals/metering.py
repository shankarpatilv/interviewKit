"""Token-counting provider wrapper and cost estimates for eval runs."""

import logging

from interviewkit.providers import LLMProvider, Message, ProviderResponse, Tool

logger = logging.getLogger(__name__)

# Estimated USD per 1M tokens (input, output). Update when provider pricing changes.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a model's token usage; unknown models cost 0.0."""
    prices = PRICES_PER_MILLION.get(model)
    if prices is None:
        if model:
            logger.warning("No price configured for model %s; cost reported as 0", model)
        return 0.0
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


class MeteredProvider:
    """Wrap a provider and accumulate token usage between resets."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.reset()

    def reset(self) -> None:
        """Clear accumulated usage before the next case."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.model = ""

    @property
    def cost_usd(self) -> float:
        """Estimated cost of usage since the last reset."""
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> ProviderResponse:
        """Delegate to the wrapped provider and record its token usage."""
        response = await self._inner.complete(messages, tools)
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.model = response.model
        return response
