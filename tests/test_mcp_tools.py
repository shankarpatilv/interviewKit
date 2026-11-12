import asyncio

import pytest
from langchain_core.documents import Document

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import interviewkit.mcp_server.tools as tools
from interviewkit.agent.tools import StoryTools
from interviewkit.config import Settings
from interviewkit.mcp_server.tools import (
    McpToolError,
    get_story_chunks,
    ingest_experience_documents,
    initialize_database,
    prep_answer,
    read_company_rubric,
    read_story_usage,
    search_experience_chunks,
)
from interviewkit.providers import Message, ProviderResponse, Tool, ToolCall
from interviewkit.stories.usage import StoryUse

HIT = Document(
    page_content="Led the Bedrock migration.",
    metadata={
        "source_file": "amazon.md",
        "chunk_id": "amazon.md:0:abc",
        "story_id": "amazon/impact/migration",
        "header_1": "Amazon",
        "header_2": "Impact",
        "header_3": "Migration",
        "score": 0.03,
    },
)


class FakeRetriever:
    instances: list["FakeRetriever"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        FakeRetriever.instances.append(self)

    async def ainvoke(self, query: str) -> list[Document]:
        self.query = query
        return [HIT]


@pytest.fixture
def fake_retriever(monkeypatch: pytest.MonkeyPatch) -> type[FakeRetriever]:
    FakeRetriever.instances = []
    monkeypatch.setattr(tools, "ExperienceRetriever", FakeRetriever)
    return FakeRetriever


def test_initialize_database_reports_status(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(tools, "init_db", lambda app_settings: calls.append("init"))

    result = initialize_database()

    assert result.status == "initialized"
    assert calls == ["init"]


def test_ingest_returns_indexed_chunk_count(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ingest(experiences_dir, app_settings):  # type: ignore[no-untyped-def]
        assert str(experiences_dir) == "notes"
        return 7

    monkeypatch.setattr(tools, "ingest_experiences", fake_ingest)

    result = asyncio.run(ingest_experience_documents("notes"))

    assert result.indexed_chunks == 7


def test_ingest_defaults_to_configured_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ingest(experiences_dir, app_settings):  # type: ignore[no-untyped-def]
        assert experiences_dir is None
        return 0

    monkeypatch.setattr(tools, "ingest_experiences", fake_ingest)

    assert asyncio.run(ingest_experience_documents()).indexed_chunks == 0


def test_search_is_hybrid_by_default_and_maps_ids(fake_retriever: type[FakeRetriever]) -> None:
    hits = asyncio.run(search_experience_chunks("Tell me about ownership.", k=2))

    kwargs = fake_retriever.instances[0].kwargs
    assert kwargs["k"] == 2 and kwargs["mode"] is None and kwargs["filters"] is None
    hit = hits[0]
    assert (hit.chunk_id, hit.story_id, hit.header_3) == (
        "amazon.md:0:abc",
        "amazon/impact/migration",
        "Migration",
    )
    assert hit.score == pytest.approx(0.03)
    assert hit.content == "Led the Bedrock migration."


def test_search_builds_frontmatter_filters(fake_retriever: type[FakeRetriever]) -> None:
    asyncio.run(
        search_experience_chunks(
            "q", source_file="amazon.md", company="Amazon", competency="Ownership", mode="vector"
        )
    )

    kwargs = fake_retriever.instances[0].kwargs
    assert kwargs["filters"] == {
        "source_file": "amazon.md",
        "company": "Amazon",
        "competencies": ["Ownership"],
    }
    assert kwargs["mode"] == "vector"


def test_search_rejects_empty_query() -> None:
    with pytest.raises(McpToolError, match="query must not be empty"):
        asyncio.run(search_experience_chunks("   "))


@pytest.mark.parametrize("k", [0, 21])
def test_search_rejects_out_of_range_k(k: int) -> None:
    with pytest.raises(McpToolError, match="k must be between 1 and 20"):
        asyncio.run(search_experience_chunks("ownership", k=k))


def test_get_story_returns_chunks_or_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "get_story", lambda story_id, app_settings: [HIT])
    assert get_story_chunks("amazon/impact/migration")[0].chunk_id == "amazon.md:0:abc"

    monkeypatch.setattr(tools, "get_story", lambda story_id, app_settings: [])
    with pytest.raises(McpToolError, match="No story"):
        get_story_chunks("missing")
    with pytest.raises(McpToolError, match="story_id"):
        get_story_chunks(" ")


def test_company_rubric_reads_profile_and_wraps_errors(tmp_path: Path) -> None:
    (tmp_path / "amazon.md").write_text("Ownership", encoding="utf-8")
    settings = Settings(companies_dir=tmp_path, _env_file=None)

    assert read_company_rubric("Amazon", app_settings=settings).text == "Ownership"
    with pytest.raises(McpToolError, match="No company profile"):
        read_company_rubric("acme", app_settings=settings)


def test_story_usage_maps_rows_and_checks_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    use = StoryUse("amazon", "a/b", "Q?", datetime(2026, 11, 1))
    monkeypatch.setattr(tools, "list_uses", lambda company, limit, app_settings: [use])

    assert read_story_usage("amazon")[0].story_id == "a/b"
    with pytest.raises(McpToolError, match="limit"):
        read_story_usage(limit=0)


class Scripted:
    def __init__(self, *replies: ProviderResponse) -> None:
        self.replies = iter(replies)

    async def complete(
        self, messages: list[Message], tools: list[Tool] | None = None
    ) -> ProviderResponse:
        return next(self.replies)


def _star(citations: list[str], result: str = "It shipped.") -> ProviderResponse:
    body = {"situation": "s", "task": "t", "action": "a", "result": result, "citations": citations}
    return ProviderResponse(json.dumps(body), "openai", "gpt-4o")


def test_prep_answer_pipeline_returns_grounded_citations(
    fake_retriever: type[FakeRetriever],
) -> None:
    result = asyncio.run(
        prep_answer(
            "Tell me about ownership.",
            "Ownership",
            "amazon",
            selection="pipeline",
            provider=Scripted(_star(["amazon.md:0:abc"])),
        )
    )

    assert result.grounding == "pass" and result.selection == "pipeline"
    assert result.citations[0].story_id == "amazon/impact/migration"
    assert result.unsupported_numbers == [] and result.tool_calls == 0


def test_prep_answer_never_hides_a_failed_check(fake_retriever: type[FakeRetriever]) -> None:
    flagged = asyncio.run(
        prep_answer(
            "q",
            "p",
            provider=Scripted(_star(["amazon.md:0:abc"], "Cut cost 40%.")),
            selection="pipeline",
        )
    )
    refused = asyncio.run(
        prep_answer("q", "p", provider=Scripted(_star(["ghost"])), selection="pipeline")
    )

    assert flagged.grounding == "flag" and flagged.unsupported_numbers == ["40%"]
    assert "Grounding check: flagged" in flagged.text
    assert refused.grounding == "refuse" and refused.citations == []
    assert refused.invalid_citations == ["ghost"]


def test_prep_answer_agent_selection_uses_tools() -> None:
    async def search(query: str, k: int = 5) -> list[Document]:
        return [HIT]

    select = ToolCall("1", "select_chunks", {"chunk_ids": ["amazon.md:0:abc"]})
    provider = Scripted(
        ProviderResponse("", "openai", "gpt-4o", tool_calls=(select,)),
        _star(["amazon.md:0:abc"]),
    )

    result = asyncio.run(
        prep_answer(
            "q",
            "p",
            "amazon",
            selection="agent",
            provider=provider,
            tools=StoryTools(search=search),
        )
    )

    assert result.selection == "agent" and result.grounding == "pass"
    assert result.tool_calls == 1  # the seed search


def test_prep_answer_rejects_empty_input() -> None:
    with pytest.raises(McpToolError, match="must not be empty"):
        asyncio.run(prep_answer(" ", "Ownership"))
