"""LangChain retriever wrapper for pgvector experience search."""

from collections.abc import Awaitable, Callable
import asyncio

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from interviewkit.config import Settings, get_settings
from interviewkit.ingest.embedder import embed_experience_chunks
from interviewkit.retrieval.models import RetrievedDocument
from interviewkit.retrieval.vectorstore import similarity_search

AsyncQueryEmbedder = Callable[[str], Awaitable[list[float]]]
SimilaritySearch = Callable[[list[float], int], list[RetrievedDocument]]


class ExperienceRetriever(BaseRetriever):
    """Retrieve relevant experience chunks through the LangChain retriever interface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    k: int = 5
    app_settings: Settings | None = None
    query_embedder: AsyncQueryEmbedder | None = None
    search: SimilaritySearch | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        embedding = asyncio.run(self._embed_query(query))
        return self._search_documents(embedding)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        embedding = await self._embed_query(query)
        return self._search_documents(embedding)

    async def _embed_query(self, query: str) -> list[float]:
        if self.query_embedder is not None:
            return await self.query_embedder(query)

        query_document = Document(page_content=query)
        embedded = await embed_experience_chunks([query_document], app_settings=self._settings)
        if not embedded:
            msg = "Query embedding failed"
            raise RuntimeError(msg)
        return embedded[0].embedding

    def _search_documents(self, embedding: list[float]) -> list[Document]:
        if self.search is not None:
            results = self.search(embedding, self.k)
        else:
            results = similarity_search(embedding, k=self.k, app_settings=self._settings)
        return [result.document for result in results]

    @property
    def _settings(self) -> Settings:
        return self.app_settings or get_settings()
