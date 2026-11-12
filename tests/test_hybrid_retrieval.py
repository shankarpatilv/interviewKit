import asyncio
from typing import Any

from langchain_core.documents import Document

from interviewkit.config import Settings
from interviewkit.retrieval.fusion import reciprocal_rank_fusion
from interviewkit.retrieval.hybrid import hybrid_search
from interviewkit.retrieval.keyword import keyword_search
from interviewkit.retrieval.models import RetrievedDocument
from interviewkit.retrieval.retriever import ExperienceRetriever
from interviewkit.retrieval.schema import KEYWORD_SEARCH_SQL, SCHEMA_STATEMENTS
from test_vectorstore import make_connection


def _hit(text: str, score: float = 0.0) -> RetrievedDocument:
    return RetrievedDocument(Document(text, metadata={"source_file": "a.md"}), score)


def test_rrf_rewards_documents_ranked_by_both_lists() -> None:
    vector = [_hit("both"), _hit("vector-only")]
    keyword = [_hit("keyword-only"), _hit("both")]

    fused = reciprocal_rank_fusion([vector, keyword])

    assert [r.document.page_content for r in fused][0] == "both"
    assert fused[0].similarity == 1 / 61 + 1 / 62


def test_rrf_breaks_ties_deterministically_and_limits() -> None:
    fused = reciprocal_rank_fusion([[_hit("b")], [_hit("a")]], limit=1)

    assert [r.document.page_content for r in fused] == ["a"]


def test_keyword_search_ors_terms_and_applies_filters() -> None:
    connection = make_connection(rows=[("a.md", "DynamoDB over RDS", {}, 0.4)])
    settings = Settings(database_url="postgresql://example.test/x", _env_file=None)

    results = keyword_search(
        "why DynamoDB?",
        k=3,
        filters={"role": "SDE"},
        app_settings=settings,
        connect=lambda _: connection,
    )

    statement, params = connection.cursor_instance.statements[0]
    assert statement == KEYWORD_SEARCH_SQL and "' & ', ' | '" in statement
    assert params == ("why DynamoDB?", '{"role": "SDE"}', 3)
    assert results[0].similarity == 0.4


def test_schema_adds_generated_tsvector_and_gin_index() -> None:
    sql = "\n".join(SCHEMA_STATEMENTS)

    assert "GENERATED ALWAYS AS (to_tsvector('english', chunk_text))" in sql
    assert "USING GIN (chunk_tsv)" in sql


def test_hybrid_search_fuses_candidates_from_both_searches() -> None:
    calls: list[tuple[str, Any]] = []

    def fake_vector(embedding: list[float], **kwargs: Any) -> list[RetrievedDocument]:
        calls.append(("vector", kwargs))
        return [_hit("v1"), _hit("shared")]

    def fake_keyword(query: str, **kwargs: Any) -> list[RetrievedDocument]:
        calls.append(("keyword", kwargs))
        return [_hit("shared"), _hit("k1")]

    results = hybrid_search(
        "q",
        [0.1],
        k=2,
        candidates=7,
        filters={"x": 1},
        vector_search=fake_vector,
        text_search=fake_keyword,
    )

    assert [r.document.page_content for r in results] == ["shared", "v1"]
    assert [kwargs["k"] for _, kwargs in calls] == [7, 7]
    assert all(kwargs["filters"] == {"x": 1} for _, kwargs in calls)


def test_retriever_uses_hybrid_mode_from_field_or_settings() -> None:
    seen: list[str] = []

    async def embed(query: str) -> list[float]:
        return [0.0]

    def fake_hybrid(query: str, embedding: list[float], k: int) -> list[RetrievedDocument]:
        seen.append(query)
        return [_hit("hybrid")]

    def fake_vector(embedding: list[float], k: int) -> list[RetrievedDocument]:
        return [_hit("vector")]

    settings = Settings(_env_file=None, RETRIEVAL_MODE="hybrid")  # type: ignore[call-arg]
    from_settings = ExperienceRetriever(
        app_settings=settings, query_embedder=embed, search=fake_vector, hybrid=fake_hybrid
    )
    forced_vector = ExperienceRetriever(
        mode="vector",
        app_settings=settings,
        query_embedder=embed,
        search=fake_vector,
        hybrid=fake_hybrid,
    )

    assert asyncio.run(from_settings.ainvoke("q"))[0].page_content == "hybrid"
    assert asyncio.run(forced_vector.ainvoke("q"))[0].page_content == "vector"
    assert seen == ["q"]
