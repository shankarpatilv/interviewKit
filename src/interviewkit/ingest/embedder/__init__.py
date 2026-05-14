"""Public embedding helpers for experience chunks."""

from interviewkit.ingest.embedder.client import embed_experience_chunks
from interviewkit.ingest.embedder.models import (
    EMBEDDING_DIMENSIONS,
    MAX_BATCH_SIZE,
    EmbeddedDocument,
)

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "MAX_BATCH_SIZE",
    "EmbeddedDocument",
    "embed_experience_chunks",
]
