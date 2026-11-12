"""Result models for eval runs."""

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class CaseResult:
    """Deterministic metrics, latency, and cost for one golden case."""

    case_id: str
    expect_refusal: bool
    recall: float | None = None
    groundedness: float | None = None
    unsupported_numbers: tuple[str, ...] = ()
    refused: bool = False
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    grounding: str | None = None  # pass, flag, or refuse from the grounding check
    tool_calls: int = 0
    error: str | None = None

    @property
    def refusal_correct(self) -> bool:
        """Return True when the case refused exactly when it should have."""
        return self.error is None and self.refused == self.expect_refusal


@dataclass(frozen=True)
class Scorecard:
    """Aggregate metrics across all cases in one eval run."""

    mode: str
    k: int
    cases: tuple[CaseResult, ...]
    selection: str = "pipeline"

    @property
    def completed(self) -> tuple[CaseResult, ...]:
        """Cases that ran without an error."""
        return tuple(case for case in self.cases if case.error is None)

    @property
    def mean_recall(self) -> float | None:
        """Mean recall@k over cases that expect a story."""
        return _mean([c.recall for c in self.completed if c.recall is not None])

    @property
    def mean_groundedness(self) -> float | None:
        """Mean share of citations that were actually retrieved."""
        return _mean([c.groundedness for c in self.completed if c.groundedness is not None])

    @property
    def unsupported_total(self) -> int:
        """Total numbers in answers that no cited chunk supports."""
        return sum(len(c.unsupported_numbers) for c in self.completed)

    @property
    def fidelity_rate(self) -> float | None:
        """Share of answered cases with zero unsupported numbers."""
        answered = [c for c in self.completed if not c.refused]
        return _mean([0.0 if c.unsupported_numbers else 1.0 for c in answered])

    @property
    def refusal_accuracy(self) -> float | None:
        """Share of completed cases whose refusal behavior was correct."""
        return _mean([1.0 if c.refusal_correct else 0.0 for c in self.completed])

    @property
    def false_refusals(self) -> int:
        """Cases that refused although a story should have been found."""
        return sum(1 for c in self.completed if c.refused and not c.expect_refusal)

    @property
    def flagged(self) -> int:
        """Answers the grounding check kept but flagged."""
        return sum(1 for c in self.completed if c.grounding == "flag")

    @property
    def mean_tool_calls(self) -> float | None:
        """Mean agent tool calls per completed case."""
        return _mean([float(c.tool_calls) for c in self.completed])

    @property
    def total_cost_usd(self) -> float:
        """Estimated LLM cost across all cases."""
        return sum(c.cost_usd for c in self.cases)


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None
