"""Batching helpers for embedding requests."""

from langchain_core.documents import Document

from interviewkit.ingest.embedder.models import MAX_BATCH_SIZE


def batches(chunks: list[Document], batch_size: int) -> list[list[Document]]:
    """Split documents into fixed-size request batches."""
    return [chunks[index : index + batch_size] for index in range(0, len(chunks), batch_size)]


def validate_batch_size(batch_size: int) -> None:
    """Raise when an embedding batch size exceeds OpenAI request limits."""
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        msg = f"batch_size must be between 1 and {MAX_BATCH_SIZE}"
        raise ValueError(msg)
