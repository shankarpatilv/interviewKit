"""OpenAI embedding request helpers."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, cast

from langchain_core.documents import Document
from openai import AsyncOpenAI, RateLimitError

from interviewkit.config import Settings
from interviewkit.ingest.embedder.models import EMBEDDING_DIMENSIONS


class _EmbeddingItem(Protocol):
    embedding: list[float]


class _EmbeddingData(Protocol):
    data: Sequence[_EmbeddingItem]


class EmbeddingsClient(Protocol):
    """Client interface for creating embeddings."""

    def create(self, *, model: str, input: list[str]) -> Awaitable[_EmbeddingData]: ...


class EmbeddingClient(Protocol):
    """Client interface for OpenAI-compatible embedding APIs."""

    embeddings: EmbeddingsClient


Sleep = Callable[[float], Awaitable[None]]


def build_openai_client(app_settings: Settings) -> EmbeddingClient:
    """Build an async OpenAI client configured for embeddings."""
    api_key = app_settings.provider_api_key("openai").get_secret_value()
    return cast(EmbeddingClient, AsyncOpenAI(api_key=api_key))


async def embed_batch(
    client: EmbeddingClient,
    batch: list[Document],
    *,
    model: str,
    max_retries: int,
    sleep: Sleep,
) -> list[list[float]]:
    """Embed one batch, retrying OpenAI rate limits with exponential backoff."""
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            response = await client.embeddings.create(
                model=model,
                input=[chunk.page_content for chunk in batch],
            )
            return _extract_embeddings(response, expected_count=len(batch))
        except RateLimitError:
            if attempt == max_retries:
                raise
            await sleep(delay)
            delay *= 2

    raise RuntimeError("Embedding retry loop exited unexpectedly")


def _extract_embeddings(response: _EmbeddingData, *, expected_count: int) -> list[list[float]]:
    embeddings = [item.embedding for item in response.data]
    if len(embeddings) != expected_count:
        msg = f"OpenAI returned {len(embeddings)} embeddings for {expected_count} chunks"
        raise ValueError(msg)
    if any(len(embedding) != EMBEDDING_DIMENSIONS for embedding in embeddings):
        msg = f"OpenAI embeddings must be {EMBEDDING_DIMENSIONS} dimensions"
        raise ValueError(msg)
    return embeddings
