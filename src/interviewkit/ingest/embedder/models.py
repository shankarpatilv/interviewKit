"""Embedding data models."""

from dataclasses import dataclass

from langchain_core.documents import Document

EMBEDDING_DIMENSIONS = 1536
MAX_BATCH_SIZE = 100


@dataclass(frozen=True)
class EmbeddedDocument:
    """A chunked document paired with its embedding vector."""

    document: Document
    embedding: list[float]
