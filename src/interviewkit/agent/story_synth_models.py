"""Shared types for STAR answer synthesis."""

from dataclasses import dataclass
from typing import Protocol, TypedDict

from langchain_core.documents import Document


class StorySynthesisError(RuntimeError):
    """Raised when the STAR synthesis graph cannot complete."""


class ExperienceSearch(Protocol):
    """Minimal async retrieval protocol needed by the graph."""

    async def ainvoke(self, query: str) -> list[Document]:
        """Return relevant experience chunks for the query."""


@dataclass(frozen=True)
class StarAnswer:
    """A synthesized STAR answer and its cited source files."""

    principle: str
    text: str
    sources: tuple[str, ...]


class StorySynthesisState(TypedDict, total=False):
    """State carried between STAR synthesis graph nodes."""

    question: str
    principle: str
    retrieved_chunks: list[Document]
    raw_response: str
    answer: StarAnswer
