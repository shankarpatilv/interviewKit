"""Deterministic retrieval and answer-quality metrics (no LLM-as-judge)."""

import re
from collections.abc import Sequence

from langchain_core.documents import Document

from interviewkit.agent.story_synth import NO_MATCH_MESSAGE
from interviewkit.evals.golden import ExpectedStory, GoldenCase
from interviewkit.evals.numbers import unsupported_numbers

_SOURCES_HEADER = re.compile(r"^[ \t]*[*_]*sources[*_]*[ \t]*:", re.IGNORECASE | re.MULTILINE)
_MD_FILE = re.compile(r"[^\s()\[\]<>:,;'\"`]+\.md\b")


def _squash(text: str) -> str:
    """Lowercase and collapse whitespace for tolerant text comparison."""
    return " ".join(text.split()).lower()


def story_matches(expected: ExpectedStory, document: Document) -> bool:
    """Return True when a retrieved document is the expected story."""
    if document.metadata.get("source_file") != expected.source_file:
        return False
    if expected.header_2 is None:
        return True
    actual = document.metadata.get("header_2")
    return isinstance(actual, str) and _squash(actual) == _squash(expected.header_2)


def recall_at_k(
    expected: Sequence[ExpectedStory], retrieved: Sequence[Document], k: int
) -> float | None:
    """Fraction of expected stories matched by any of the top-``k`` retrieved documents.

    Returns None when nothing is expected. Raises ValueError if ``k < 1``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not expected:
        return None
    top = retrieved[:k]
    hits = sum(any(story_matches(story, doc) for doc in top) for story in expected)
    return hits / len(expected)


def extract_cited_sources(answer_text: str) -> tuple[str, ...]:
    """Unique ``*.md`` file names in the Sources section (or whole text if absent), in order."""
    header = _SOURCES_HEADER.search(answer_text)
    section = answer_text[header.end() :] if header else answer_text
    return tuple(dict.fromkeys(_MD_FILE.findall(section)))


def groundedness(cited: Sequence[str], retrieved_sources: Sequence[str]) -> float | None:
    """Fraction of cited sources that were actually retrieved; None when nothing is cited."""
    unique_cited = list(dict.fromkeys(cited))
    if not unique_cited:
        return None
    retrieved = set(retrieved_sources)
    return sum(source in retrieved for source in unique_cited) / len(unique_cited)


def is_refusal(answer_text: str) -> bool:
    """Return True when the answer contains the no-match refusal sentence."""
    return _squash(NO_MATCH_MESSAGE) in _squash(answer_text)


def refusal_correct(case: GoldenCase, answer_text: str) -> bool:
    """Return True when the answer refuses exactly when the case expects a refusal."""
    return is_refusal(answer_text) == case.expect_refusal


def metric_fidelity(answer_text: str, cited_chunks: Sequence[Document]) -> list[str]:
    """Numbers in the answer unsupported by the cited chunks; a refusal returns ``[]``."""
    if is_refusal(answer_text):
        return []
    return unsupported_numbers(answer_text, [chunk.page_content for chunk in cited_chunks])
