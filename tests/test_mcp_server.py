import asyncio
from pathlib import Path

import pytest
from langchain_core.documents import Document
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import Tool

import interviewkit.mcp_server.tools as tools
from interviewkit.config import get_settings
from interviewkit.mcp_server.server import SERVER_NAME, build_server, use_project_dir


def _tool(name: str) -> Tool:
    listed = asyncio.run(build_server().list_tools())
    return next(tool for tool in listed if tool.name == name)


def test_server_registers_expected_tools() -> None:
    server = build_server()

    registered = {tool.name for tool in asyncio.run(server.list_tools())}

    assert registered == {
        "init_database",
        "ingest_experiences",
        "search_experiences",
        "get_story",
        "get_company_rubric",
        "get_story_usage",
        "prep_answer",
    }
    assert server.name == SERVER_NAME


def test_search_tool_schema_hides_internal_settings() -> None:
    search = _tool("search_experiences")

    assert set(search.input_schema["properties"]) == {
        "query",
        "k",
        "source_file",
        "company",
        "competency",
        "mode",
    }
    assert search.input_schema["required"] == ["query"]
    assert search.annotations is not None
    assert search.annotations.read_only_hint is True


@pytest.mark.parametrize("name", ["get_story", "get_company_rubric", "get_story_usage"])
def test_story_tools_are_read_only(name: str) -> None:
    tool = _tool(name)

    assert tool.annotations is not None and tool.annotations.read_only_hint is True
    assert "app_settings" not in tool.input_schema["properties"]


def test_prep_answer_schema_hides_provider_and_tools() -> None:
    prep = _tool("prep_answer")

    assert set(prep.input_schema["properties"]) == {"question", "principle", "company", "selection"}
    assert set(prep.input_schema["required"]) == {"question", "principle"}


def test_ingest_tool_schema_exposes_optional_directory() -> None:
    ingest = _tool("ingest_experiences")

    assert set(ingest.input_schema["properties"]) == {"experiences_dir"}
    assert ingest.input_schema.get("required", []) == []


def test_search_tool_call_returns_structured_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRetriever:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["k"] == 1

        async def ainvoke(self, query: str) -> list[Document]:
            return [
                Document(
                    page_content="Owned the migration end to end.",
                    metadata={
                        "source_file": "amazon.md",
                        "chunk_id": "amazon.md:0:x",
                        "score": 0.5,
                    },
                )
            ]

    monkeypatch.setattr(tools, "ExperienceRetriever", FakeRetriever)
    server = build_server()

    result = asyncio.run(
        server.call_tool("search_experiences", {"query": "ownership", "k": 1}),
    )

    assert result.is_error is False
    assert "amazon.md:0:x" in str(result.structured_content)


def test_use_project_dir_reloads_settings_from_that_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text("CHUNK_SIZE=1234\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    get_settings.cache_clear()

    try:
        use_project_dir(tmp_path)

        assert Path.cwd() == tmp_path.resolve()
        assert get_settings().chunk_size == 1234
    finally:
        get_settings.cache_clear()


def test_use_project_dir_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError, match="Project directory does not exist"):
        use_project_dir(tmp_path / "nope")


def test_tool_error_surfaces_the_underlying_message() -> None:
    server = build_server()

    with pytest.raises(ToolError, match="query must not be empty"):
        asyncio.run(server.call_tool("search_experiences", {"query": "  "}))
