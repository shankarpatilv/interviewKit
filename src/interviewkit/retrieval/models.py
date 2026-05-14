"""Retrieval data models."""

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievedDocument:
    """A retrieved document with its cosine similarity score."""

    document: Document
    similarity: float
