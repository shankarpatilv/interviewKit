"""Embedding workflow orchestration."""

import asyncio
import logging

from langchain_core.documents import Document

from interviewkit.config import Settings, settings
from interviewkit.ingest.embedder.batching import batches, validate_batch_size
from interviewkit.ingest.embedder.models import MAX_BATCH_SIZE, EmbeddedDocument
from interviewkit.ingest.embedder.openai_client import EmbeddingClient, Sleep, build_openai_client
from interviewkit.ingest.embedder.openai_client import embed_batch

logger = logging.getLogger(__name__)


async def embed_experience_chunks(
    chunks: list[Document],
    *,
    app_settings: Settings = settings,
    client: EmbeddingClient | None = None,
    batch_size: int = MAX_BATCH_SIZE,
    max_retries: int = 3,
    sleep: Sleep = asyncio.sleep,
) -> list[EmbeddedDocument]:
    """Embed chunked documents in batches, skipping batches that keep failing."""
    validate_batch_size(batch_size)
    embedding_client = client or build_openai_client(app_settings)
    embedded: list[EmbeddedDocument] = []

    for batch_number, batch in enumerate(batches(chunks, batch_size), start=1):
        try:
            embeddings = await embed_batch(
                embedding_client,
                batch,
                model=app_settings.embedding_model,
                max_retries=max_retries,
                sleep=sleep,
            )
        except Exception:
            logger.exception("Skipping embedding batch %s after failure", batch_number)
            continue

        embedded.extend(
            EmbeddedDocument(document=document, embedding=embedding)
            for document, embedding in zip(batch, embeddings, strict=True)
        )

    return embedded
