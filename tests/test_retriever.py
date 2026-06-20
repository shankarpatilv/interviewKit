import asyncio

from langchain_core.documents import Document

from interviewkit.retrieval.models import RetrievedDocument
from interviewkit.ingest.embedder import EmbeddedDocument
from interviewkit.retrieval import retriever as retriever_module
from interviewkit.retrieval.retriever import ExperienceRetriever


def test_experience_retriever_invokes_search_with_query_embedding() -> None:
    calls: list[tuple[list[float], int]] = []

    async def embed_query(query: str) -> list[float]:
        assert query == "Tell me about Bedrock"
        return [0.1, 0.2, 0.3]

    def search(embedding: list[float], k: int) -> list[RetrievedDocument]:
        calls.append((embedding, k))
        return [
            RetrievedDocument(
                document=Document(
                    page_content="Built a Bedrock moderation pipeline.",
                    metadata={"source_file": "amazon.md"},
                ),
                similarity=0.92,
            )
        ]

    retriever = ExperienceRetriever(k=3, query_embedder=embed_query, search=search)

    documents = retriever.invoke("Tell me about Bedrock")

    assert calls == [([0.1, 0.2, 0.3], 3)]
    assert documents[0].page_content == "Built a Bedrock moderation pipeline."
    assert documents[0].metadata["source_file"] == "amazon.md"


def test_experience_retriever_supports_async_invoke() -> None:
    async def run_test() -> None:
        async def embed_query(query: str) -> list[float]:
            return [0.4, 0.5]

        def search(embedding: list[float], k: int) -> list[RetrievedDocument]:
            return [
                RetrievedDocument(
                    document=Document(
                        page_content="Async result.", metadata={"source_file": "a.md"}
                    ),
                    similarity=0.8,
                )
            ]

        retriever = ExperienceRetriever(k=1, query_embedder=embed_query, search=search)

        documents = await retriever.ainvoke("ownership")

        assert [document.page_content for document in documents] == ["Async result."]
        assert documents[0].metadata["source_file"] == "a.md"

    asyncio.run(run_test())


def test_experience_retriever_returns_empty_list_when_search_has_no_matches() -> None:
    async def run_test() -> None:
        async def embed_query(query: str) -> list[float]:
            return []

        retriever = ExperienceRetriever(query_embedder=embed_query, search=lambda embedding, k: [])

        documents = await retriever.ainvoke("empty")

        assert documents == []

    asyncio.run(run_test())


def test_experience_retriever_default_embedder_converts_query_to_embedding(
    monkeypatch,
) -> None:
    async def run_test() -> None:
        async def embed_chunks(chunks, *, app_settings):
            assert chunks[0].page_content == "ambiguity"
            return [EmbeddedDocument(document=chunks[0], embedding=[0.7, 0.8])]

        calls: list[tuple[list[float], int]] = []

        def search(embedding: list[float], k: int) -> list[RetrievedDocument]:
            calls.append((embedding, k))
            return []

        monkeypatch.setattr(retriever_module, "embed_experience_chunks", embed_chunks)
        retriever = ExperienceRetriever(k=2, search=search)

        documents = await retriever.ainvoke("ambiguity")

        assert documents == []
        assert calls == [([0.7, 0.8], 2)]

    asyncio.run(run_test())
