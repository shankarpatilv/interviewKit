import asyncio
import json
from dataclasses import dataclass, field

from langchain_core.documents import Document

from interviewkit.config import Settings
from interviewkit.evals.golden import ExpectedStory, GoldenCase
from interviewkit.evals.metering import MeteredProvider, estimate_cost
from interviewkit.evals.models import Scorecard
from interviewkit.evals.runner import run_eval
from interviewkit.providers import Message, ProviderResponse, Tool

ANSWER = json.dumps(
    {
        "situation": "Our payments service was slow.",
        "task": "Fix it.",
        "action": "I migrated it.",
        "result": "Latency dropped 40% for 6 teams, then 90% overall.",
        "citations": ["pay:0"],
    }
)
REFUSAL = json.dumps({"refused": True})


@dataclass
class FakeProvider:
    replies: list[str]
    calls: int = 0

    async def complete(
        self, messages: list[Message], tools: list[Tool] | None = None
    ) -> ProviderResponse:
        reply = self.replies[self.calls % len(self.replies)]
        self.calls += 1
        if reply == "boom":
            raise RuntimeError("provider exploded")
        return ProviderResponse(reply, "openai", "gpt-4o", input_tokens=1000, output_tokens=200)


@dataclass
class FakeRetriever:
    documents: list[Document] = field(default_factory=list)

    async def ainvoke(self, query: str) -> list[Document]:
        return self.documents


def _case(case_id: str, refusal: bool = False) -> GoldenCase:
    stories = () if refusal else (ExpectedStory(source_file="payments.md"),)
    return GoldenCase(
        id=case_id,
        question="Tell me about a time you took ownership.",
        company="amazon",
        principle="Ownership",
        expected_stories=stories,
        expect_refusal=refusal,
    )


def _run(provider: FakeProvider, cases: list[GoldenCase]) -> Scorecard:
    documents = [
        Document(
            "Cut latency 40% across six teams.",
            metadata={"source_file": "payments.md", "chunk_id": "pay:0"},
        ),
        Document("Unrelated on-call story.", metadata={"source_file": "oncall.md"}),
    ]
    return asyncio.run(
        run_eval(
            cases,
            k=2,
            provider=provider,
            retriever=FakeRetriever(documents),
            app_settings=Settings(_env_file=None),
        )
    )


def test_run_eval_scores_answered_case() -> None:
    scorecard = _run(FakeProvider([ANSWER]), [_case("own-1")])

    result = scorecard.cases[0]
    assert result.error is None
    assert result.recall == 1.0
    assert result.groundedness == 1.0
    assert result.unsupported_numbers == ("90%",)
    assert result.grounding == "flag"
    assert scorecard.flagged == 1
    assert result.refusal_correct
    assert (result.input_tokens, result.output_tokens) == (1000, 200)
    assert result.cost_usd == estimate_cost("gpt-4o", 1000, 200)


def test_run_eval_scores_refusal_and_false_refusal() -> None:
    scorecard = _run(FakeProvider([REFUSAL]), [_case("refuse", True), _case("miss")])

    assert [case.refusal_correct for case in scorecard.cases] == [True, False]
    assert scorecard.false_refusals == 1
    assert scorecard.refusal_accuracy == 0.5
    assert scorecard.fidelity_rate is None


def test_run_eval_records_failing_case_and_continues() -> None:
    scorecard = _run(FakeProvider(["boom", ANSWER]), [_case("bad"), _case("good")])

    assert scorecard.cases[0].error == "provider exploded"
    assert not scorecard.cases[0].refusal_correct
    assert scorecard.cases[1].error is None
    assert len(scorecard.completed) == 1


def test_metered_provider_resets_usage() -> None:
    metered = MeteredProvider(FakeProvider([ANSWER]))
    asyncio.run(metered.complete([{"role": "user", "content": "hi"}]))
    assert metered.input_tokens == 1000

    metered.reset()

    assert (metered.input_tokens, metered.output_tokens, metered.cost_usd) == (0, 0, 0.0)


def test_estimate_cost_unknown_model_is_zero() -> None:
    assert estimate_cost("mystery-model", 1000, 1000) == 0.0
    assert estimate_cost("gpt-4o", 1_000_000, 0) == 2.50


def test_estimate_cost_prices_cached_tokens_like_langfuse() -> None:
    # Real GPT-4o call observed in Langfuse: 2454 input (2304 cached), 13 output.
    assert round(estimate_cost("gpt-4o", 2454, 13, cached_input_tokens=2304), 6) == 0.003385


def test_estimate_cost_caps_cached_at_input() -> None:
    assert estimate_cost("gpt-4o", 100, 0, cached_input_tokens=500) == 100 * 1.25 / 1e6


def test_unknown_citation_lowers_chunk_groundedness() -> None:
    reply = json.loads(ANSWER) | {"citations": ["pay:0", "made-up"], "result": "Cut 40%."}
    scorecard = _run(FakeProvider([json.dumps(reply)]), [_case("own-1")])

    result = scorecard.cases[0]
    assert result.groundedness == 0.5
    assert result.grounding == "flag"


def test_run_eval_agent_selection_counts_tool_calls() -> None:
    from interviewkit.agent.tools import StoryTools
    from interviewkit.providers import ToolCall

    doc = Document(
        "Cut latency 40% across six teams.",
        metadata={"source_file": "payments.md", "chunk_id": "pay:0"},
    )

    async def search(query: str, k: int = 5) -> list[Document]:
        return [doc]

    replies = iter(
        [
            ProviderResponse(
                "",
                "openai",
                "gpt-4o",
                tool_calls=(ToolCall("1", "search_experiences", {"query": "q"}),),
            ),
            ProviderResponse(
                "",
                "openai",
                "gpt-4o",
                tool_calls=(ToolCall("2", "select_chunks", {"chunk_ids": ["pay:0"]}),),
            ),
            ProviderResponse(ANSWER.replace(", then 90% overall", ""), "openai", "gpt-4o"),
        ]
    )

    class ScriptedProvider:
        async def complete(
            self, messages: list[Message], tools: list[Tool] | None = None
        ) -> ProviderResponse:
            return next(replies)

    scorecard = asyncio.run(
        run_eval(
            [_case("own-1")],
            k=2,
            selection="agent",
            provider=ScriptedProvider(),
            tools=StoryTools(search=search),
            app_settings=Settings(_env_file=None),
        )
    )

    result = scorecard.cases[0]
    assert result.error is None and result.tool_calls == 2  # seed + one search
    assert result.recall == 1.0 and result.grounding == "pass"
    assert scorecard.selection == "agent"
