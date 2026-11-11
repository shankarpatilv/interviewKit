import pytest
from langchain_core.documents import Document

from interviewkit.agent.story_synth import NO_MATCH_MESSAGE
from interviewkit.evals.golden import ExpectedStory, GoldenCase
from interviewkit.evals.metrics import (
    extract_cited_sources,
    groundedness,
    is_refusal,
    metric_fidelity,
    recall_at_k,
    refusal_correct,
    story_matches,
)


def _doc(source: str, header_2: str | None = None, content: str = "") -> Document:
    metadata: dict[str, str] = {"source_file": source, "header_1": "Experience"}
    if header_2 is not None:
        metadata["header_2"] = header_2
    return Document(page_content=content, metadata=metadata)


def _case(expect_refusal: bool) -> GoldenCase:
    stories = () if expect_refusal else (ExpectedStory(source_file="a.md"),)
    return GoldenCase(
        id="c1",
        question="Tell me about a time.",
        company="acme",
        principle="Ownership",
        jd=None,
        expected_stories=stories,
        expect_refusal=expect_refusal,
    )


def test_story_matches_source_and_optional_header() -> None:
    assert story_matches(ExpectedStory(source_file="a.md"), _doc("a.md", "Anything"))
    assert not story_matches(ExpectedStory(source_file="a.md"), _doc("b.md"))
    expected = ExpectedStory(source_file="a.md", header_2="Cache Migration")
    assert story_matches(expected, _doc("a.md", "  cache migration "))
    assert not story_matches(expected, _doc("a.md", "Other"))
    assert not story_matches(expected, _doc("a.md"))


def test_recall_at_k() -> None:
    expected = [ExpectedStory(source_file="a.md"), ExpectedStory(source_file="b.md")]
    retrieved = [_doc("a.md"), _doc("c.md"), _doc("b.md")]
    assert recall_at_k(expected, retrieved, 1) == 0.5
    assert recall_at_k(expected, retrieved, 3) == 1.0
    assert recall_at_k(expected, [], 5) == 0.0
    assert recall_at_k([], retrieved, 3) is None


def test_recall_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k([ExpectedStory(source_file="a.md")], [], 0)


def test_extract_cited_sources_from_sources_section() -> None:
    answer = (
        "Situation: see notes.md\nResult: ok\nSources:\n"
        "- (source: b.md)\n- (source: a.md)\n- (source: b.md)\n"
    )
    assert extract_cited_sources(answer) == ("b.md", "a.md")


def test_extract_cited_sources_without_section_scans_whole_text() -> None:
    assert extract_cited_sources("Based on (source: x.md) and y.md.") == ("x.md", "y.md")
    assert extract_cited_sources("no files here") == ()


def test_groundedness() -> None:
    assert groundedness(["a.md", "b.md"], ["a.md", "c.md"]) == 0.5
    assert groundedness(["a.md"], ["a.md"]) == 1.0
    assert groundedness([], ["a.md"]) is None


def test_is_refusal_is_case_and_whitespace_tolerant() -> None:
    assert is_refusal(NO_MATCH_MESSAGE)
    assert is_refusal("Sorry.\n" + NO_MATCH_MESSAGE.upper().replace(" ", "\n  "))
    assert not is_refusal("Situation: I led a migration.")


def test_refusal_correct() -> None:
    assert refusal_correct(_case(True), NO_MATCH_MESSAGE)
    assert not refusal_correct(_case(True), "Situation: real answer")
    assert refusal_correct(_case(False), "Situation: real answer")
    assert not refusal_correct(_case(False), NO_MATCH_MESSAGE)


def test_metric_fidelity() -> None:
    chunks = [_doc("a.md", content="Cut p99 latency by 40 percent across six services.")]
    answer = "Result: cut latency 40% across 6 services, saving $1M.\nSources:\n- (source: a.md)"
    assert metric_fidelity(answer, chunks) == ["$1M"]
    assert metric_fidelity(NO_MATCH_MESSAGE + " 42", []) == []
