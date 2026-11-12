"""End-to-end ingestion workflow."""

import logging
from pathlib import Path

from interviewkit.config import Settings, get_settings
from interviewkit.ingest.chunker import chunk_experience_documents
from interviewkit.ingest.embedder import embed_experience_chunks
from interviewkit.ingest.loader import load_experience_documents
from interviewkit.retrieval.vectorstore import upsert_chunks

logger = logging.getLogger(__name__)


async def ingest_experiences(
    experiences_dir: Path | None = None,
    app_settings: Settings | None = None,
) -> int:
    """Load, chunk, embed, and store experience documents.

    After a complete run, rows for chunks that no longer exist are deleted. If any
    embedding batch failed, pruning is skipped so existing rows are not lost.
    """
    resolved_settings = app_settings or get_settings()
    source_dir = experiences_dir or resolved_settings.experiences_dir
    documents = load_experience_documents(source_dir)
    chunks = chunk_experience_documents(
        documents,
        chunk_size=resolved_settings.chunk_size,
        chunk_overlap=resolved_settings.chunk_overlap,
    )
    embedded = await embed_experience_chunks(chunks, app_settings=resolved_settings)
    complete = len(embedded) == len(chunks)
    if not complete:
        logger.warning(
            "Embedded %d of %d chunks; skipping stale-chunk pruning", len(embedded), len(chunks)
        )
    if embedded:
        upsert_chunks(embedded, app_settings=resolved_settings, prune=complete)
    return len(embedded)
