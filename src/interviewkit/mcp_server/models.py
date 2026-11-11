"""Result models returned by InterviewKit MCP tools."""

from pydantic import BaseModel, Field


class IngestResult(BaseModel):
    """Outcome of an experience ingestion run."""

    indexed_chunks: int = Field(description="Number of experience chunks embedded and stored.")


class InitDatabaseResult(BaseModel):
    """Outcome of a database schema initialization."""

    status: str = Field(description="Schema initialization status.")


class SearchHit(BaseModel):
    """A single experience chunk retrieved by similarity search."""

    content: str = Field(description="Chunk text from the experience document.")
    source_file: str = Field(description="Experience file the chunk came from.")
    header_1: str | None = Field(default=None, description="Top-level markdown heading.")
    header_2: str | None = Field(default=None, description="Second-level markdown heading.")
    similarity: float = Field(description="Cosine similarity against the query, 0 to 1.")
