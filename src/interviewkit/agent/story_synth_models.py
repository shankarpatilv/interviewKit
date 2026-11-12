"""Shared types for STAR answer synthesis."""

from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field

GroundingStatus = Literal["pass", "flag", "refuse"]


class StorySynthesisError(RuntimeError):
    """Raised when the STAR synthesis graph cannot complete."""


class ExperienceSearch(Protocol):
    """Minimal async retrieval protocol needed by the graph."""

    async def ainvoke(self, query: str) -> list[Document]:
        """Return relevant experience chunks for the query."""


class DraftAnswer(BaseModel):
    """The structured STAR answer the LLM must return, citing chunk ids."""

    refused: bool = False
    situation: str = ""
    task: str = ""
    action: str = ""
    result: str = ""
    citations: list[str] = Field(default_factory=list)

    @property
    def body(self) -> str:
        """The four STAR sections joined, for number checks."""
        return "\n".join((self.situation, self.task, self.action, self.result))


@dataclass(frozen=True)
class GroundingReport:
    """Outcome of the deterministic grounding check."""

    status: GroundingStatus
    valid_citations: tuple[str, ...] = ()
    invalid_citations: tuple[str, ...] = ()
    unsupported_numbers: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class StarAnswer:
    """A synthesized STAR answer, its cited files and chunks, and its grounding status."""

    principle: str
    text: str
    sources: tuple[str, ...]
    citations: tuple[str, ...] = ()
    grounding: GroundingStatus = "pass"


class StorySynthesisState(TypedDict, total=False):
    """State carried between STAR synthesis graph nodes."""

    question: str
    principle: str
    company: str
    retrieved_chunks: list[Document]
    tool_calls: int
    draft: DraftAnswer
    grounding: GroundingReport
    answer: StarAnswer
