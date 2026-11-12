import asyncio
from pathlib import Path

import pytest
from langchain_core.documents import Document

from interviewkit.agent import tools
from interviewkit.config import Settings
from interviewkit.retrieval.records import row_to_retrieved_document
from interviewkit.retrieval.schema import GET_STORY_SQL, KEYWORD_SEARCH_SQL, SIMILARITY_SEARCH_SQL
from interviewkit.stories.lookup import get_company_rubric, get_story
from test_vectorstore import make_connection

SETTINGS = Settings(database_url="postgresql://example.test/x", _env_file=None)


def test_search_sql_returns_chunk_ids() -> None:
    assert "AS similarity, id" in SIMILARITY_SEARCH_SQL
    assert "AS score, id" in KEYWORD_SEARCH_SQL


def test_row_with_id_sets_chunk_id_metadata() -> None:
    result = row_to_retrieved_document(("a.md", "text", {"story_id": "a/x"}, 0.5, "a.md:0:abc"))

    assert result.document.metadata == {
        "story_id": "a/x",
        "source_file": "a.md",
        "chunk_id": "a.md:0:abc",
    }


def test_get_story_queries_by_story_id_in_order() -> None:
    rows = [("a.md", "part 1", {"story_id": "a/x"}, 1.0, "a.md:2:aaa")]
    connection = make_connection(rows=rows)

    chunks = get_story(" a/x ", app_settings=SETTINGS, connect=lambda _: connection)

    statement, params = connection.cursor_instance.statements[0]
    assert statement == GET_STORY_SQL and "ORDER BY length(id), id" in statement
    assert params == ("a/x",)
    assert chunks[0].metadata["chunk_id"] == "a.md:2:aaa"


def test_get_company_rubric_reads_profile(tmp_path: Path) -> None:
    (tmp_path / "amazon.md").write_text("# Ownership\n", encoding="utf-8")
    settings = Settings(companies_dir=tmp_path, _env_file=None)

    assert get_company_rubric("Amazon", app_settings=settings) == "# Ownership"


@pytest.mark.parametrize("company", ["../secrets", "missing"])
def test_get_company_rubric_rejects_bad_or_missing(tmp_path: Path, company: str) -> None:
    settings = Settings(companies_dir=tmp_path, _env_file=None)

    with pytest.raises(ValueError):
        get_company_rubric(company, app_settings=settings)


def test_search_experiences_validates_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(tools.search_experiences(" "))
    with pytest.raises(ValueError, match="between"):
        asyncio.run(tools.search_experiences("q", k=0))


def test_search_experiences_builds_retriever_with_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeRetriever:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

        async def ainvoke(self, query: str) -> list[Document]:
            seen["query"] = query
            return [Document(page_content="hit")]

    monkeypatch.setattr(tools, "ExperienceRetriever", FakeRetriever)

    docs = asyncio.run(
        tools.search_experiences("q", k=3, filters={"source_file": "a.md"}, app_settings=SETTINGS)
    )

    assert docs[0].page_content == "hit"
    assert seen["k"] == 3 and seen["filters"] == {"source_file": "a.md"} and seen["query"] == "q"


def test_story_tools_bind_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        tools, "list_uses", lambda company, limit, app_settings: calls.append(app_settings) or []
    )

    tools.StoryTools.for_settings(SETTINGS).usage("amazon")

    assert calls == [SETTINGS]
