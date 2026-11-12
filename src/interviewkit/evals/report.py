"""Render and save eval scorecards."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import quantiles

from interviewkit.evals.models import Scorecard


def render_scorecard(scorecard: Scorecard) -> str:
    """Return a plain-text scorecard for the terminal."""
    latencies = [c.latency_s for c in scorecard.completed]
    rows = [
        ("Cases", f"{len(scorecard.completed)}/{len(scorecard.cases)} completed"),
        (f"Recall@{scorecard.k}", _pct(scorecard.mean_recall)),
        ("Groundedness", _pct(scorecard.mean_groundedness)),
        ("Metric fidelity", _pct(scorecard.fidelity_rate)),
        ("Unsupported numbers", str(scorecard.unsupported_total)),
        ("Refusal accuracy", _pct(scorecard.refusal_accuracy)),
        ("False refusals", str(scorecard.false_refusals)),
        ("Flagged answers", str(scorecard.flagged)),
        ("Latency mean / p95", _latency(latencies)),
        ("Estimated cost", f"${scorecard.total_cost_usd:.4f}"),
    ]
    if scorecard.selection == "agent":
        rows.insert(-2, ("Tool calls / case", _number(scorecard.mean_tool_calls)))
    width = max(len(label) for label, _ in rows)
    header = f"InterviewKit eval  mode={scorecard.mode}  selection={scorecard.selection}"
    lines = [f"{header}  k={scorecard.k}", ""]
    lines += [f"{label:<{width}}  {value}" for label, value in rows]
    lines += _problem_lines(scorecard)
    return "\n".join(lines)


def write_scorecard_json(scorecard: Scorecard, directory: Path) -> Path:
    """Write the full scorecard as JSON and return the file path."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"eval-{scorecard.mode}-{scorecard.selection}-k{scorecard.k}-{stamp}.json"
    path = directory / name
    payload = {"mode": scorecard.mode, "selection": scorecard.selection, "k": scorecard.k}
    payload_cases = [asdict(case) for case in scorecard.cases]
    path.write_text(json.dumps({**payload, "cases": payload_cases}, indent=2), encoding="utf-8")
    return path


def _problem_lines(scorecard: Scorecard) -> list[str]:
    lines = []
    for case in scorecard.cases:
        if case.error is not None:
            lines.append(f"  {case.case_id}: error: {case.error}")
        elif case.unsupported_numbers:
            lines.append(f"  {case.case_id}: unsupported {', '.join(case.unsupported_numbers)}")
        elif not case.refusal_correct:
            lines.append(
                f"  {case.case_id}: refused={case.refused}, expected={case.expect_refusal}"
            )
    return ["", "Problems:", *lines] if lines else []


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _latency(values: list[float]) -> str:
    if not values:
        return "n/a"
    p95 = quantiles(values, n=20)[-1] if len(values) > 1 else values[0]
    return f"{sum(values) / len(values):.2f}s / {p95:.2f}s"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"
