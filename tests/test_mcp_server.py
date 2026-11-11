import asyncio
from pathlib import Path

import pytest
from langchain_core.documents import Document
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import Tool

import interviewkit.mcp_server.tools as tools
from interviewkit.config import get_settings
from interviewkit.ingest.embedder import EmbeddedDocument
from interviewkit.mcp_server.server import SERVER_NAME, build_server, use_project_dir
from interviewkit.retrieval.models import RetrievedDocument


def _tool(name: str) -> Tool:
    listed = asyncio.run(build_server().list_tools())
    return next(tool for tool in listed if tool.name == name)


def test_server_registers_expected_tools() -> None:
    server = build_server()

    registered = {tool.name for tool in asyncio.run(server.list_tools())}

    assert registered == {"init_database", "ingest_experiences", "search_experiences"}
    assert server.name == SERVER_NAME


def test_search_tool_schema_hides_internal_settings() -> None:
    search = _tool("search_experiences")

    assert set(search.input_schema["properties"]) == {"query", "k"}
    assert search.input_schema["required"] == ["query"]
    assert search.annotations is not None
    assert search.annotations.read_only_hint is True


def test_ingest_tool_schema_exposes_optional_directory() -> None:
    ingest = _tool("ingest_experiences")

    assert set(ingest.input_schema["properties"]) == {"experiences_dir"}
    assert ingest.input_schema.get("required", []) == []


def test_search_tool_call_returns_structured_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_embed(documents, app_settings):  # type: ignore[no-untyped-def]
        return [EmbeddedDocument(document=documents[0], embedding=[0.3])]

    def fake_search(embedding, *, k, app_settings):  # type: ignore[no-untyped-def]
        assert k == 1
        return [
            RetrievedDocument(
                document=Document(
                    page_content="Owned the migration end to end.",
                    metadata={"source_file": "amazon.md"},
                ),
                similarity=0.88,
            )
        ]

    monkeypatch.setattr(tools, "embed_experience_chunks", fake_embed)
    monkeypatch.setattr(tools, "similarity_search", fake_search)
    server = build_server()

    result = asyncio.run(
        server.call_tool("search_experiences", {"query": "ownership", "k": 1}),
    )

    assert result.is_error is False
    assert "amazon.md" in str(result.structured_content)


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
