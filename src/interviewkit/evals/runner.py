"""Run golden cases through retrieval and STAR synthesis and score them."""

import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Any, cast

from langchain_core.documents import Document

from interviewkit.agent.selection import agent_selector
from interviewkit.agent.story_synth import StarAnswer, build_story_graph
from interviewkit.agent.tools import StoryTools
from interviewkit.agent.story_synth_models import ExperienceSearch, GroundingReport
from interviewkit.config import RetrievalMode, Settings, StorySelection, get_settings
from interviewkit.evals.golden import GoldenCase
from interviewkit.evals.metering import MeteredProvider
from interviewkit.evals.metrics import is_refusal, metric_fidelity, recall_at_k
from interviewkit.evals.models import CaseResult, Scorecard
from interviewkit.providers import LLMProvider, get_provider
from interviewkit.retrieval.retriever import ExperienceRetriever
from interviewkit.tracing import flush_traces, trace_config, trace_run

logger = logging.getLogger(__name__)


async def run_eval(
    cases: Sequence[GoldenCase],
    *,
    k: int = 5,
    mode: RetrievalMode = "hybrid",
    selection: StorySelection = "pipeline",
    provider: LLMProvider | None = None,
    retriever: ExperienceSearch | None = None,
    tools: StoryTools | None = None,
    app_settings: Settings | None = None,
) -> Scorecard:
    """Run every case sequentially; a failing case is recorded, not raised."""
    settings = app_settings or get_settings()
    metered = MeteredProvider(provider or get_provider())
    resolved_retriever = retriever or cast(
        ExperienceSearch, ExperienceRetriever(k=k, mode=mode, app_settings=settings)
    )
    selector = None
    if selection == "agent":
        resolved_tools = tools or StoryTools.for_settings(settings, mode=mode)
        selector = agent_selector(metered, resolved_tools, k)
    graph = build_story_graph(provider=metered, retriever=resolved_retriever, selector=selector)
    results = []
    for case in cases:
        metered.reset()
        name = f"eval:{case.id}"
        tags = ("eval", f"mode:{mode}", f"selection:{selection}", f"k:{k}")
        metadata = {"case_id": case.id, "company": case.company}
        config = trace_config(settings, run_name=name, tags=tags, metadata=metadata)
        started = perf_counter()
        try:
            with trace_run(settings, name=name, tags=tags, metadata=metadata):
                state = await graph.ainvoke(
                    {
                        "question": case.question,
                        "principle": case.principle,
                        "company": case.company,
                    },
                    config=config,
                )
            result = score_case(case, state, k)
        except Exception as exc:  # one bad case must not abort the run
            logger.exception("Eval case %s failed", case.id)
            result = CaseResult(case_id=case.id, expect_refusal=case.expect_refusal, error=str(exc))
        results.append(_with_usage(result, perf_counter() - started, metered))
    flush_traces(settings)
    return Scorecard(mode=mode, k=k, cases=tuple(results), selection=selection)


def score_case(case: GoldenCase, state: dict[str, Any], k: int) -> CaseResult:
    """Compute deterministic metrics from a finished story-graph state.

    Groundedness is chunk-level: the share of the answer's cited chunk ids
    that were among the chunks it was given.
    """
    retrieved = cast(list[Document], state.get("retrieved_chunks", []))
    answer = cast(StarAnswer, state["answer"])
    report = cast(GroundingReport | None, state.get("grounding"))
    refused = answer.grounding == "refuse" or is_refusal(answer.text)
    cited_chunks = [d for d in retrieved if d.metadata.get("chunk_id") in answer.citations]
    return CaseResult(
        case_id=case.id,
        expect_refusal=case.expect_refusal,
        recall=recall_at_k(case.expected_stories, retrieved, k),
        groundedness=None if refused else _chunk_groundedness(report),
        unsupported_numbers=tuple(metric_fidelity(answer.text, cited_chunks)),
        refused=refused,
        grounding=answer.grounding,
        tool_calls=int(state.get("tool_calls", 0)),
    )


def _chunk_groundedness(report: GroundingReport | None) -> float | None:
    if report is None:
        return None
    cited = len(report.valid_citations) + len(report.invalid_citations)
    return len(report.valid_citations) / cited if cited else None


def _with_usage(result: CaseResult, latency_s: float, metered: MeteredProvider) -> CaseResult:
    return CaseResult(
        **{
            **result.__dict__,
            "latency_s": latency_s,
            "input_tokens": metered.input_tokens,
            "output_tokens": metered.output_tokens,
            "cached_input_tokens": metered.cached_input_tokens,
            "cost_usd": metered.cost_usd,
        }
    )
