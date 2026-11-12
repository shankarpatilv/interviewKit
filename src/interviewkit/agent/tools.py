"""Typed story tools shared by the selection agent and the MCP server."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial

from langchain_core.documents import Document

from interviewkit.config import RetrievalMode, Settings, get_settings
from interviewkit.retrieval.retriever import ExperienceRetriever
from interviewkit.stories.lookup import get_company_rubric, get_story
from interviewkit.stories.usage import StoryUse, list_uses

MAX_SEARCH_K = 10


async def search_experiences(
    query: str,
    *,
    k: int = 5,
    filters: Mapping[str, object] | None = None,
    mode: RetrievalMode | None = None,
    app_settings: Settings | None = None,
) -> list[Document]:
    """Return up to k chunks for a query, each with `chunk_id` and `story_id` metadata."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= k <= MAX_SEARCH_K:
        raise ValueError(f"k must be between 1 and {MAX_SEARCH_K}")
    retriever = ExperienceRetriever(
        k=k,
        mode=mode,
        filters=dict(filters) if filters else None,
        app_settings=app_settings or get_settings(),
    )
    return await retriever.ainvoke(query)


def get_story_usage(
    company: str, *, limit: int = 20, app_settings: Settings | None = None
) -> list[StoryUse]:
    """Return the stories already told to a company, newest first."""
    return list_uses(company, limit=limit, app_settings=app_settings)


@dataclass(frozen=True)
class StoryTools:
    """The four tool callables the agent may use; swap any of them in tests."""

    search: Callable[..., Awaitable[list[Document]]] = search_experiences
    story: Callable[[str], list[Document]] = get_story
    rubric: Callable[[str], str] = get_company_rubric
    usage: Callable[[str], list[StoryUse]] = get_story_usage

    @classmethod
    def for_settings(
        cls, app_settings: Settings, mode: RetrievalMode | None = None
    ) -> "StoryTools":
        """Bind every tool to one settings object (and search to one retrieval mode)."""
        return cls(
            search=partial(search_experiences, mode=mode, app_settings=app_settings),
            story=partial(get_story, app_settings=app_settings),
            rubric=partial(get_company_rubric, app_settings=app_settings),
            usage=partial(get_story_usage, app_settings=app_settings),
        )
