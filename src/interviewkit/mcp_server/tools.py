"""Tool implementations backing the InterviewKit MCP server."""

from pathlib import Path
from typing import Any, cast

from langchain_core.documents import Document

from interviewkit.agent.selection import agent_selector
from interviewkit.agent.story_synth import StarAnswer, build_story_graph
from interviewkit.agent.story_synth_models import ExperienceSearch, GroundingReport
from interviewkit.agent.tools import StoryTools
from interviewkit.config import RetrievalMode, Settings, StorySelection, get_settings
from interviewkit.ingest.pipeline import ingest_experiences
from interviewkit.mcp_server.models import (
    CitedChunk,
    CompanyRubric,
    IngestResult,
    InitDatabaseResult,
    PrepAnswerResult,
    SearchHit,
    StoryUsage,
)
from interviewkit.providers import LLMProvider, get_provider
from interviewkit.retrieval.retriever import ExperienceRetriever
from interviewkit.retrieval.vectorstore import init_db
from interviewkit.stories.lookup import get_company_rubric, get_story
from interviewkit.stories.usage import list_uses

MAX_SEARCH_RESULTS = 20
MAX_USAGE_RESULTS = 100


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
    source_file: str | None = None,
    company: str | None = None,
    competency: str | None = None,
    mode: RetrievalMode | None = None,
    app_settings: Settings | None = None,
) -> list[SearchHit]:
    """Return the most relevant chunks (hybrid by default), optionally filtered.

    Filters match note frontmatter: `company`, and `competency` against the
    `competencies` list; `source_file` limits the search to one file.
    """
    if not query.strip():
        raise McpToolError("query must not be empty")
    if not 1 <= k <= MAX_SEARCH_RESULTS:
        raise McpToolError(f"k must be between 1 and {MAX_SEARCH_RESULTS}")
    filters: dict[str, object] = {}
    if source_file:
        filters["source_file"] = source_file
    if company:
        filters["company"] = company
    if competency:
        filters["competencies"] = [competency]
    retriever = ExperienceRetriever(
        k=k, mode=mode, filters=filters or None, app_settings=app_settings or get_settings()
    )
    return [_to_search_hit(doc, _score(doc)) for doc in await retriever.ainvoke(query)]


def get_story_chunks(story_id: str, app_settings: Settings | None = None) -> list[SearchHit]:
    """Return every chunk of one story, in document order."""
    if not story_id.strip():
        raise McpToolError("story_id must not be empty")
    chunks = get_story(story_id, app_settings=app_settings or get_settings())
    if not chunks:
        raise McpToolError(f"No story with id {story_id!r}")
    return [_to_search_hit(doc, 1.0) for doc in chunks]


def read_company_rubric(company: str, app_settings: Settings | None = None) -> CompanyRubric:
    """Return a company's interview principles."""
    try:
        text = get_company_rubric(company, app_settings=app_settings or get_settings())
    except ValueError as exc:
        raise McpToolError(str(exc)) from exc
    return CompanyRubric(company=company.strip().lower(), text=text)


def read_story_usage(
    company: str | None = None, limit: int = 20, app_settings: Settings | None = None
) -> list[StoryUsage]:
    """Return recorded story uses, newest first, optionally for one company."""
    if not 1 <= limit <= MAX_USAGE_RESULTS:
        raise McpToolError(f"limit must be between 1 and {MAX_USAGE_RESULTS}")
    try:
        uses = list_uses(company, limit=limit, app_settings=app_settings or get_settings())
    except ValueError as exc:
        raise McpToolError(str(exc)) from exc
    return [StoryUsage(**use.__dict__) for use in uses]


async def prep_answer(
    question: str,
    principle: str,
    company: str = "",
    selection: StorySelection | None = None,
    app_settings: Settings | None = None,
    provider: LLMProvider | None = None,
    tools: StoryTools | None = None,
) -> PrepAnswerResult:
    """Run retrieval, STAR synthesis, and the grounding check for one question."""
    if not question.strip() or not principle.strip():
        raise McpToolError("question and principle must not be empty")
    settings = app_settings or get_settings()
    resolved_selection = selection or settings.story_selection
    llm = provider or get_provider()
    selector = None
    if resolved_selection == "agent":
        selector = agent_selector(llm, tools or StoryTools.for_settings(settings))
    retriever = cast(ExperienceSearch, ExperienceRetriever(k=5, app_settings=settings))
    graph = build_story_graph(provider=llm, retriever=retriever, selector=selector)
    state = await graph.ainvoke({"question": question, "principle": principle, "company": company})
    return _to_prep_result(state, resolved_selection)


def _to_prep_result(state: dict[str, Any], selection: StorySelection) -> PrepAnswerResult:
    answer = cast(StarAnswer, state["answer"])
    report = cast(GroundingReport | None, state.get("grounding"))
    by_id = {str(d.metadata.get("chunk_id")): d for d in state.get("retrieved_chunks", [])}
    citations = [
        CitedChunk(
            chunk_id=chunk_id,
            story_id=by_id[chunk_id].metadata.get("story_id"),
            source_file=str(by_id[chunk_id].metadata.get("source_file", "")),
        )
        for chunk_id in answer.citations
        if chunk_id in by_id
    ]
    return PrepAnswerResult(
        text=answer.text,
        grounding=answer.grounding,
        citations=citations,
        unsupported_numbers=list(report.unsupported_numbers) if report else [],
        invalid_citations=list(report.invalid_citations) if report else [],
        selection=selection,
        tool_calls=int(state.get("tool_calls", 0)),
    )


def _score(document: Document) -> float:
    return float(document.metadata.get("score", 0.0))


def _to_search_hit(document: Document, score: float) -> SearchHit:
    metadata = document.metadata
    return SearchHit(
        content=document.page_content,
        source_file=str(metadata.get("source_file", "")),
        chunk_id=metadata.get("chunk_id"),
        story_id=metadata.get("story_id"),
        header_1=metadata.get("header_1"),
        header_2=metadata.get("header_2"),
        header_3=metadata.get("header_3"),
        score=score,
    )
