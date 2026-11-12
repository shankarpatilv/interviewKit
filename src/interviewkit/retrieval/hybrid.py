"""Hybrid retrieval: pgvector similarity plus Postgres full-text, fused by RRF."""

from collections.abc import Callable, Mapping

from interviewkit.config import Settings
from interviewkit.retrieval.fusion import reciprocal_rank_fusion
from interviewkit.retrieval.keyword import keyword_search
from interviewkit.retrieval.models import RetrievedDocument
from interviewkit.retrieval.vectorstore import similarity_search

DEFAULT_CANDIDATES = 20

VectorSearch = Callable[..., list[RetrievedDocument]]
KeywordSearch = Callable[..., list[RetrievedDocument]]


def hybrid_search(
    query: str,
    query_embedding: list[float],
    *,
    k: int = 5,
    candidates: int = DEFAULT_CANDIDATES,
    filters: Mapping[str, object] | None = None,
    app_settings: Settings | None = None,
    vector_search: VectorSearch = similarity_search,
    text_search: KeywordSearch = keyword_search,
) -> list[RetrievedDocument]:
    """Fetch `candidates` from each search, fuse them, and return the top k."""
    vector_hits = vector_search(
        query_embedding, k=candidates, filters=filters, app_settings=app_settings
    )
    keyword_hits = text_search(query, k=candidates, filters=filters, app_settings=app_settings)
    return reciprocal_rank_fusion([vector_hits, keyword_hits], limit=k)
