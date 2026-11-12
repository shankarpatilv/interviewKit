"""Reciprocal rank fusion for merging ranked retrieval lists."""

from collections.abc import Sequence

from interviewkit.retrieval.models import RetrievedDocument

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievedDocument]],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[RetrievedDocument]:
    """Merge ranked lists by summing 1 / (rrf_k + rank) for each document.

    RRF uses only ranks, so cosine similarities and full-text scores, which live
    on different scales, can be combined without normalization. Ties break on a
    stable document key so results are deterministic. The fused score replaces
    `similarity` in the returned documents.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievedDocument] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = document_key(result)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            first_seen.setdefault(key, result)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [
        RetrievedDocument(document=first_seen[key].document, similarity=scores[key])
        for key in ordered[:limit]
    ]


def document_key(result: RetrievedDocument) -> str:
    """Identify a chunk across result lists by source file and text."""
    metadata = result.document.metadata
    return f"{metadata.get('source_file', '')}\x00{result.document.page_content}"
