"""LangChain retriever wrapper for pgvector experience search."""

import asyncio
from collections.abc import Awaitable, Callable

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from interviewkit.config import RetrievalMode, Settings, get_settings
from interviewkit.ingest.embedder import embed_experience_chunks
from interviewkit.retrieval.hybrid import hybrid_search
from interviewkit.retrieval.models import RetrievedDocument
from interviewkit.retrieval.vectorstore import similarity_search

AsyncQueryEmbedder = Callable[[str], Awaitable[list[float]]]
SimilaritySearch = Callable[[list[float], int], list[RetrievedDocument]]
HybridSearch = Callable[[str, list[float], int], list[RetrievedDocument]]


class ExperienceRetriever(BaseRetriever):
    """Retrieve relevant experience chunks through the LangChain retriever interface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    k: int = 5
    mode: RetrievalMode | None = None  # None uses RETRIEVAL_MODE from settings
    filters: dict[str, object] | None = None
    app_settings: Settings | None = None
    query_embedder: AsyncQueryEmbedder | None = None
    search: SimilaritySearch | None = None
    hybrid: HybridSearch | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        embedding = asyncio.run(self._embed_query(query))
        return self._search_documents(query, embedding)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        embedding = await self._embed_query(query)
        return self._search_documents(query, embedding)

    async def _embed_query(self, query: str) -> list[float]:
        if self.query_embedder is not None:
            return await self.query_embedder(query)

        query_document = Document(page_content=query)
        embedded = await embed_experience_chunks([query_document], app_settings=self._settings)
        if not embedded:
            msg = "Query embedding failed"
            raise RuntimeError(msg)
        return embedded[0].embedding

    def _search_documents(self, query: str, embedding: list[float]) -> list[Document]:
        if (self.mode or self._settings.retrieval_mode) == "hybrid":
            results = self._hybrid(query, embedding)
        elif self.search is not None:
            results = self.search(embedding, self.k)
        else:
            results = similarity_search(
                embedding, k=self.k, filters=self.filters, app_settings=self._settings
            )
        return [result.document for result in results]

    def _hybrid(self, query: str, embedding: list[float]) -> list[RetrievedDocument]:
        if self.hybrid is not None:
            return self.hybrid(query, embedding, self.k)
        return hybrid_search(
            query, embedding, k=self.k, filters=self.filters, app_settings=self._settings
        )

    @property
    def _settings(self) -> Settings:
        return self.app_settings or get_settings()
