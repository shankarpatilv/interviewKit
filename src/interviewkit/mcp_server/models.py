"""Result models returned by InterviewKit MCP tools."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class IngestResult(BaseModel):
    """Outcome of an experience ingestion run."""

    indexed_chunks: int = Field(description="Number of experience chunks embedded and stored.")


class InitDatabaseResult(BaseModel):
    """Outcome of a database schema initialization."""

    status: str = Field(description="Schema initialization status.")


class SearchHit(BaseModel):
    """A single experience chunk returned by search or story lookup."""

    content: str = Field(description="Chunk text from the experience document.")
    source_file: str = Field(description="Experience file the chunk came from.")
    chunk_id: str | None = Field(default=None, description="Stable chunk id, used in citations.")
    story_id: str | None = Field(
        default=None, description="Story id (`<file>/<## slug>/<### slug>`) for get_story."
    )
    header_1: str | None = Field(default=None, description="Top-level markdown heading.")
    header_2: str | None = Field(default=None, description="Second-level markdown heading.")
    header_3: str | None = Field(default=None, description="Third-level (story) heading.")
    score: float = Field(
        description="Relevance score; only comparable to other scores in the same result list."
    )


class CompanyRubric(BaseModel):
    """A company's interview principles."""

    company: str = Field(description="Company key, e.g. amazon.")
    text: str = Field(description="Markdown text of companies/<company>.md.")


class StoryUsage(BaseModel):
    """One recorded use of a story with a company."""

    company: str
    story_id: str
    question: str | None = None
    used_at: datetime


class CitedChunk(BaseModel):
    """A chunk an answer cites."""

    chunk_id: str
    story_id: str | None = None
    source_file: str


class PrepAnswerResult(BaseModel):
    """A STAR answer after the deterministic grounding check."""

    text: str = Field(description="Rendered STAR answer, or the not-enough-context refusal.")
    grounding: Literal["pass", "flag", "refuse"] = Field(
        description="pass: fully grounded. flag: shown with a warning. refuse: no grounded answer."
    )
    citations: list[CitedChunk] = Field(description="Retrieved chunks the answer cites.")
    unsupported_numbers: list[str] = Field(description="Numbers found in no cited chunk.")
    invalid_citations: list[str] = Field(description="Cited ids that were not retrieved.")
    selection: Literal["pipeline", "agent"] = Field(description="How stories were picked.")
    tool_calls: int = Field(description="Agent tool calls used (0 for the pipeline).")
