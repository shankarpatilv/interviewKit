import asyncio
from dataclasses import dataclass, field

from langchain_core.documents import Document

from interviewkit.agent.story_synth import NO_MATCH_MESSAGE
from interviewkit.config import Settings
from interviewkit.evals.golden import ExpectedStory, GoldenCase
from interviewkit.evals.metering import MeteredProvider, estimate_cost
from interviewkit.evals.models import Scorecard
from interviewkit.evals.runner import run_eval
from interviewkit.providers import Message, ProviderResponse, Tool

ANSWER = (
    "Principle: Ownership\nSituation: Our payments service was slow.\n"
    "Action: I migrated it.\nResult: Latency dropped 40% for 6 teams, then 90% overall.\n"
    "Sources:\n- (source: payments.md)"
)


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
        Document("Cut latency 40% across six teams.", metadata={"source_file": "payments.md"}),
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
    assert result.refusal_correct
    assert (result.input_tokens, result.output_tokens) == (1000, 200)
    assert result.cost_usd == estimate_cost("gpt-4o", 1000, 200)


def test_run_eval_scores_refusal_and_false_refusal() -> None:
    refusal = f"Principle: Ownership\n{NO_MATCH_MESSAGE}\nSources:\n- none"
    scorecard = _run(FakeProvider([refusal]), [_case("refuse", True), _case("miss")])

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
