"""MCP server exposing InterviewKit ingestion and retrieval tools."""

import os
from pathlib import Path
from typing import Literal

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from interviewkit import __version__
from interviewkit.config import get_settings
from interviewkit.mcp_server.models import IngestResult, InitDatabaseResult, SearchHit
from interviewkit.mcp_server.tools import (
    ingest_experience_documents,
    initialize_database,
    search_experience_chunks,
)

Transport = Literal["stdio", "sse", "streamable-http"]

SERVER_NAME = "interviewkit"
INSTRUCTIONS = (
    "Search and index a local corpus of personal work-experience notes stored in "
    "PostgreSQL with pgvector. Call init_database once before the first ingest."
)


def build_server() -> MCPServer:
    """Return an MCP server with the InterviewKit tools registered."""
    server: MCPServer = MCPServer(
        name=SERVER_NAME,
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    @server.tool(
        name="init_database",
        description="Create the pgvector extension, experiences table, and indexes.",
        annotations=ToolAnnotations(idempotent_hint=True, destructive_hint=False),
    )
    def init_database_tool() -> InitDatabaseResult:
        return initialize_database()

    @server.tool(
        name="ingest_experiences",
        description=(
            "Load, chunk, embed, and store markdown experience documents. "
            "Re-running replaces chunks for the same source files."
        ),
        annotations=ToolAnnotations(idempotent_hint=True, destructive_hint=False),
    )
    async def ingest_experiences_tool(experiences_dir: str | None = None) -> IngestResult:
        return await ingest_experience_documents(experiences_dir)

    @server.tool(
        name="search_experiences",
        description=(
            "Retrieve the experience chunks most relevant to a question, "
            "with source files and cosine similarity scores."
        ),
        annotations=ToolAnnotations(read_only_hint=True),
    )
    async def search_experiences_tool(query: str, k: int = 5) -> list[SearchHit]:
        return await search_experience_chunks(query, k)

    return server


def use_project_dir(project_dir: Path) -> None:
    """Resolve `.env` and relative data folders against a project directory.

    MCP clients launch the server from an arbitrary working directory, so the
    cached settings must be rebuilt after moving into the project root.
    """
    if not project_dir.is_dir():
        msg = f"Project directory does not exist: {project_dir}"
        raise NotADirectoryError(msg)
    os.chdir(project_dir)
    get_settings.cache_clear()


def run_server(transport: Transport = "stdio", project_dir: Path | None = None) -> None:
    """Run the InterviewKit MCP server until the client disconnects."""
    if project_dir is not None:
        use_project_dir(project_dir)
    build_server().run(transport=transport)
