import asyncio

import pytest
from langchain_core.documents import Document

import interviewkit.mcp_server.tools as tools
from interviewkit.ingest.embedder import EmbeddedDocument
from interviewkit.mcp_server.tools import (
    McpToolError,
    ingest_experience_documents,
    initialize_database,
    search_experience_chunks,
)
from interviewkit.retrieval.models import RetrievedDocument


def _retrieved(source_file: str, similarity: float) -> RetrievedDocument:
    return RetrievedDocument(
        document=Document(
            page_content="Led the Bedrock migration.",
            metadata={"source_file": source_file, "header_1": "Amazon", "header_2": "Impact"},
        ),
        similarity=similarity,
    )


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


def test_search_maps_metadata_and_similarity(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_embed(documents, app_settings):  # type: ignore[no-untyped-def]
        assert documents[0].page_content == "Tell me about ownership."
        return [EmbeddedDocument(document=documents[0], embedding=[0.1, 0.2])]

    def fake_search(embedding, *, k, app_settings):  # type: ignore[no-untyped-def]
        assert embedding == [0.1, 0.2]
        assert k == 2
        return [_retrieved("amazon.md", 0.91), _retrieved("collabera.md", 0.72)]

    monkeypatch.setattr(tools, "embed_experience_chunks", fake_embed)
    monkeypatch.setattr(tools, "similarity_search", fake_search)

    hits = asyncio.run(search_experience_chunks("Tell me about ownership.", k=2))

    assert [hit.source_file for hit in hits] == ["amazon.md", "collabera.md"]
    assert hits[0].similarity == pytest.approx(0.91)
    assert hits[0].header_1 == "Amazon"
    assert hits[0].header_2 == "Impact"
    assert hits[0].content == "Led the Bedrock migration."


def test_search_rejects_empty_query() -> None:
    with pytest.raises(McpToolError, match="query must not be empty"):
        asyncio.run(search_experience_chunks("   "))


@pytest.mark.parametrize("k", [0, 21])
def test_search_rejects_out_of_range_k(k: int) -> None:
    with pytest.raises(McpToolError, match="k must be between 1 and 20"):
        asyncio.run(search_experience_chunks("ownership", k=k))


def test_search_raises_when_embedding_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_embed(documents, app_settings):  # type: ignore[no-untyped-def]
        return []

    monkeypatch.setattr(tools, "embed_experience_chunks", fake_embed)

    with pytest.raises(McpToolError, match="Query embedding failed"):
        asyncio.run(search_experience_chunks("ownership"))


def test_search_handles_missing_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_embed(documents, app_settings):  # type: ignore[no-untyped-def]
        return [EmbeddedDocument(document=documents[0], embedding=[0.1])]

    def fake_search(embedding, *, k, app_settings):  # type: ignore[no-untyped-def]
        return [RetrievedDocument(document=Document(page_content="text"), similarity=0.5)]

    monkeypatch.setattr(tools, "embed_experience_chunks", fake_embed)
    monkeypatch.setattr(tools, "similarity_search", fake_search)

    hit = asyncio.run(search_experience_chunks("ownership"))[0]

    assert hit.source_file == ""
    assert hit.header_1 is None
