"""Deterministic grounding check run after STAR synthesis (no LLM judge)."""

from collections.abc import Sequence

from langchain_core.documents import Document

from interviewkit.agent.story_synth_context import chunk_id
from interviewkit.agent.story_synth_models import DraftAnswer, GroundingReport, GroundingStatus
from interviewkit.evals.numbers import unsupported_numbers


def check_grounding(draft: DraftAnswer, chunks: Sequence[Document]) -> GroundingReport:
    """Verify citations and numbers against the chunks the answer was given.

    - refuse: the model refused, wrote nothing, or cited no retrieved chunk.
    - flag: at least one real citation, but some citation is unknown or a
      number appears in no cited chunk.
    - pass: every citation was retrieved and every number is in a cited chunk.
    """
    if draft.refused:
        return GroundingReport("refuse", reason="no relevant story in the retrieved chunks")
    if not draft.body.strip():
        return GroundingReport("refuse", reason="the answer was empty")
    by_id = {chunk_id(chunk): chunk for chunk in chunks}
    cited = list(dict.fromkeys(citation.strip() for citation in draft.citations))
    valid = tuple(c for c in cited if c in by_id)
    invalid = tuple(c for c in cited if c not in by_id)
    if not valid:
        return GroundingReport(
            "refuse", invalid_citations=invalid, reason="no citation matched a retrieved chunk"
        )
    numbers = tuple(unsupported_numbers(draft.body, [by_id[c].page_content for c in valid]))
    status: GroundingStatus = "flag" if invalid or numbers else "pass"
    return GroundingReport(status, valid, invalid, numbers)


def describe_issues(report: GroundingReport) -> str:
    """One human-readable line explaining why an answer was flagged."""
    issues = []
    if report.invalid_citations:
        issues.append(f"{len(report.invalid_citations)} citation(s) matched no retrieved chunk")
    if report.unsupported_numbers:
        issues.append(f"not in cited notes: {', '.join(report.unsupported_numbers)}")
    return "; ".join(issues)
