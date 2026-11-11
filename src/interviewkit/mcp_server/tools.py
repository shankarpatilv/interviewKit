"""Tool implementations backing the InterviewKit MCP server."""

from pathlib import Path

from langchain_core.documents import Document

from interviewkit.config import Settings, get_settings
from interviewkit.ingest.embedder import embed_experience_chunks
from interviewkit.ingest.pipeline import ingest_experiences
from interviewkit.mcp_server.models import IngestResult, InitDatabaseResult, SearchHit
from interviewkit.retrieval.models import RetrievedDocument
from interviewkit.retrieval.vectorstore import init_db, similarity_search

MAX_SEARCH_RESULTS = 20


class McpToolError(Exception):
    """Raised when an MCP tool cannot complete its request."""


def initialize_database(app_settings: Settings | None = None) -> InitDatabaseResult:
    """Create the pgvector extension, experiences table, and indexes."""
    init_db(app_settings=app_settings or get_settings())
    return InitDatabaseResult(status="initialized")


async def ingest_experience_documents(
    experiences_dir: str | None = None,
    app_settings: Settings | None = None,
) -> IngestResult:
    """Load, chunk, embed, and store experience documents."""
    source_dir = Path(experiences_dir) if experiences_dir else None
    indexed = await ingest_experiences(
        experiences_dir=source_dir,
        app_settings=app_settings or get_settings(),
    )
    return IngestResult(indexed_chunks=indexed)


async def search_experience_chunks(
    query: str,
    k: int = 5,
    app_settings: Settings | None = None,
) -> list[SearchHit]:
    """Return the experience chunks most similar to a query."""
    if not query.strip():
        msg = "query must not be empty"
        raise McpToolError(msg)
    if not 1 <= k <= MAX_SEARCH_RESULTS:
        msg = f"k must be between 1 and {MAX_SEARCH_RESULTS}"
        raise McpToolError(msg)

    resolved_settings = app_settings or get_settings()
    embedding = await _embed_query(query, resolved_settings)
    results = similarity_search(embedding, k=k, app_settings=resolved_settings)
    return [_to_search_hit(result) for result in results]


async def _embed_query(query: str, app_settings: Settings) -> list[float]:
    embedded = await embed_experience_chunks(
        [Document(page_content=query)],
        app_settings=app_settings,
    )
    if not embedded:
        msg = "Query embedding failed"
        raise McpToolError(msg)
    return embedded[0].embedding


def _to_search_hit(result: RetrievedDocument) -> SearchHit:
    metadata = result.document.metadata
    return SearchHit(
        content=result.document.page_content,
        source_file=str(metadata.get("source_file", "")),
        header_1=metadata.get("header_1"),
        header_2=metadata.get("header_2"),
        similarity=result.similarity,
    )
