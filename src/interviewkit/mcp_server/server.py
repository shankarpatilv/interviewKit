"""MCP server exposing InterviewKit ingestion and retrieval tools."""

import os
from pathlib import Path
from typing import Literal

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from interviewkit import __version__
from interviewkit.config import RetrievalMode, StorySelection, get_settings
from interviewkit.mcp_server.models import (
    CompanyRubric,
    IngestResult,
    InitDatabaseResult,
    PrepAnswerResult,
    SearchHit,
    StoryUsage,
)
from interviewkit.mcp_server.tools import (
    get_story_chunks,
    ingest_experience_documents,
    initialize_database,
    prep_answer,
    read_company_rubric,
    read_story_usage,
    search_experience_chunks,
)

Transport = Literal["stdio", "sse", "streamable-http"]

SERVER_NAME = "interviewkit"
INSTRUCTIONS = (
    "Search and index a local corpus of personal work-experience notes stored in "
    "PostgreSQL (pgvector + full-text). Call init_database once before the first "
    "ingest. Use search_experiences and get_story to read stories, get_story_usage "
    "to avoid repeating a story to a company, and prep_answer for a grounded STAR "
    "answer; never present an answer whose grounding is 'flag' without its warning."
)
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False)


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
            "Hybrid (full-text + vector) search over experience chunks. Optional filters: "
            "source_file, company and competency (from note frontmatter). "
            "Each hit has a chunk_id and a story_id for get_story."
        ),
        annotations=READ_ONLY,
    )
    async def search_experiences_tool(
        query: str,
        k: int = 5,
        source_file: str | None = None,
        company: str | None = None,
        competency: str | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[SearchHit]:
        return await search_experience_chunks(query, k, source_file, company, competency, mode)

    @server.tool(
        name="get_story",
        description="Return every chunk of one story by its story_id, in document order.",
        annotations=READ_ONLY,
    )
    def get_story_tool(story_id: str) -> list[SearchHit]:
        return get_story_chunks(story_id)

    @server.tool(
        name="get_company_rubric",
        description="Return a company's interview principles from companies/<company>.md.",
        annotations=READ_ONLY,
    )
    def get_company_rubric_tool(company: str) -> CompanyRubric:
        return read_company_rubric(company)

    @server.tool(
        name="get_story_usage",
        description="Stories already used in interviews, newest first, optionally for one company.",
        annotations=READ_ONLY,
    )
    def get_story_usage_tool(company: str | None = None, limit: int = 20) -> list[StoryUsage]:
        return read_story_usage(company, limit)

    @server.tool(
        name="prep_answer",
        description=(
            "Draft a STAR answer from the notes and run the deterministic grounding check. "
            "Returns grounding pass/flag/refuse, chunk citations, and any unsupported numbers. "
            "Calls the LLM (costs tokens); selection='agent' lets a bounded agent pick stories."
        ),
        annotations=ToolAnnotations(
            read_only_hint=True, destructive_hint=False, open_world_hint=True
        ),
    )
    async def prep_answer_tool(
        question: str,
        principle: str,
        company: str = "",
        selection: StorySelection | None = None,
    ) -> PrepAnswerResult:
        return await prep_answer(question, principle, company, selection)

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
