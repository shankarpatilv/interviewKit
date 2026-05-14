"""Conversion helpers for pgvector experience records."""

import hashlib
import json
from typing import cast

from langchain_core.documents import Document

from interviewkit.ingest.embedder import EmbeddedDocument
from interviewkit.retrieval.models import RetrievedDocument


def numbered_chunks(chunks: list[EmbeddedDocument]) -> list[tuple[str, EmbeddedDocument]]:
    """Pair chunks with deterministic ids based on source file and index."""
    counters: dict[str, int] = {}
    numbered: list[tuple[str, EmbeddedDocument]] = []
    for chunk in chunks:
        source_file = source_file_from(chunk.document)
        chunk_index = counters.get(source_file, 0)
        counters[source_file] = chunk_index + 1
        numbered.append((chunk_id(source_file, chunk_index), chunk))
    return numbered


def chunk_params(chunk_id_value: str, chunk: EmbeddedDocument) -> tuple[str, str, str, str, str]:
    """Convert an embedded document into SQL parameters."""
    document = chunk.document
    return (
        chunk_id_value,
        source_file_from(document),
        document.page_content,
        json.dumps(document.metadata),
        vector_literal(chunk.embedding),
    )


def row_to_retrieved_document(row: object) -> RetrievedDocument:
    """Convert a database row into a retrieved LangChain document."""
    source_file, chunk_text, chunk_meta, similarity = cast(
        tuple[str, str, dict[str, object], float],
        row,
    )
    metadata = chunk_meta.copy()
    metadata["source_file"] = source_file
    return RetrievedDocument(
        document=Document(page_content=chunk_text, metadata=metadata),
        similarity=similarity,
    )


def chunk_id(source_file: str, chunk_index: int) -> str:
    """Build a stable chunk id for idempotent re-ingestion."""
    digest = hashlib.sha256(f"{source_file}:{chunk_index}".encode("utf-8")).hexdigest()
    return f"{source_file}:{chunk_index}:{digest[:12]}"


def source_file_from(document: Document) -> str:
    """Return required source-file metadata from a document."""
    source_file = document.metadata.get("source_file")
    if not isinstance(source_file, str) or not source_file:
        msg = "Document metadata must include a non-empty source_file string"
        raise ValueError(msg)
    return source_file


def vector_literal(embedding: list[float]) -> str:
    """Convert an embedding into pgvector's text input format."""
    return "[" + ",".join(str(value) for value in embedding) + "]"
