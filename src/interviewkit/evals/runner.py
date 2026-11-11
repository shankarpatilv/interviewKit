"""Run golden cases through retrieval and STAR synthesis and score them."""

import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Any, cast

from langchain_core.documents import Document

from interviewkit.agent.story_synth import StarAnswer, build_story_graph
from interviewkit.agent.story_synth_context import source_files
from interviewkit.agent.story_synth_models import ExperienceSearch
from interviewkit.config import Settings, get_settings
from interviewkit.evals.golden import GoldenCase
from interviewkit.evals.metering import MeteredProvider
from interviewkit.evals.metrics import (
    extract_cited_sources,
    groundedness,
    is_refusal,
    metric_fidelity,
    recall_at_k,
)
from interviewkit.evals.models import CaseResult, Scorecard
from interviewkit.providers import LLMProvider, get_provider
from interviewkit.retrieval.retriever import ExperienceRetriever
from interviewkit.tracing import flush_traces, trace_config, trace_run

logger = logging.getLogger(__name__)


async def run_eval(
    cases: Sequence[GoldenCase],
    *,
    k: int = 5,
    mode: str = "vector",
    provider: LLMProvider | None = None,
    retriever: ExperienceSearch | None = None,
    app_settings: Settings | None = None,
) -> Scorecard:
    """Run every case sequentially; a failing case is recorded, not raised."""
    settings = app_settings or get_settings()
    metered = MeteredProvider(provider or get_provider())
    resolved_retriever = retriever or cast(
        ExperienceSearch, ExperienceRetriever(k=k, app_settings=settings)
    )
    graph = build_story_graph(provider=metered, retriever=resolved_retriever)
    results = []
    for case in cases:
        metered.reset()
        name = f"eval:{case.id}"
        tags = ("eval", f"mode:{mode}", f"k:{k}")
        metadata = {"case_id": case.id, "company": case.company}
        config = trace_config(settings, run_name=name, tags=tags, metadata=metadata)
        started = perf_counter()
        try:
            with trace_run(settings, name=name, tags=tags, metadata=metadata):
                state = await graph.ainvoke(
                    {"question": case.question, "principle": case.principle}, config=config
                )
            result = score_case(case, state, k)
        except Exception as exc:  # one bad case must not abort the run
            logger.exception("Eval case %s failed", case.id)
            result = CaseResult(case_id=case.id, expect_refusal=case.expect_refusal, error=str(exc))
        results.append(_with_usage(result, perf_counter() - started, metered))
    flush_traces(settings)
    return Scorecard(mode=mode, k=k, cases=tuple(results))


def score_case(case: GoldenCase, state: dict[str, Any], k: int) -> CaseResult:
    """Compute deterministic metrics from a finished story-graph state."""
    retrieved = cast(list[Document], state.get("retrieved_chunks", []))
    text = cast(StarAnswer, state["answer"]).text
    refused = is_refusal(text)
    cited = () if refused else extract_cited_sources(text)
    cited_chunks = [d for d in retrieved if d.metadata.get("source_file") in cited]
    return CaseResult(
        case_id=case.id,
        expect_refusal=case.expect_refusal,
        recall=recall_at_k(case.expected_stories, retrieved, k),
        groundedness=groundedness(cited, source_files(retrieved)),
        unsupported_numbers=tuple(metric_fidelity(text, cited_chunks)),
        refused=refused,
    )


def _with_usage(result: CaseResult, latency_s: float, metered: MeteredProvider) -> CaseResult:
    return CaseResult(
        **{
            **result.__dict__,
            "latency_s": latency_s,
            "input_tokens": metered.input_tokens,
            "output_tokens": metered.output_tokens,
            "cost_usd": metered.cost_usd,
        }
    )
