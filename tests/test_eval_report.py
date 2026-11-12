import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import interviewkit.cli
from interviewkit.cli import app
from interviewkit.evals.models import CaseResult, Scorecard
from interviewkit.evals.report import render_scorecard, write_scorecard_json

SCORECARD = Scorecard(
    mode="vector",
    k=5,
    cases=(
        CaseResult("a", False, recall=1.0, groundedness=1.0, latency_s=1.0, cost_usd=0.01),
        CaseResult("b", False, recall=0.0, groundedness=0.5, unsupported_numbers=("90%",)),
        CaseResult("c", True, refused=True, latency_s=3.0),
        CaseResult("d", False, error="provider exploded"),
    ),
)


def test_render_scorecard_shows_metrics_and_problems() -> None:
    text = render_scorecard(SCORECARD)

    assert "Recall@5" in text and "50.0%" in text
    assert "Unsupported numbers  1" in text
    assert "Cases                3/4 completed" in text
    assert "b: unsupported 90%" in text
    assert "d: error: provider exploded" in text


def test_write_scorecard_json_round_trips(tmp_path: Path) -> None:
    path = write_scorecard_json(SCORECARD, tmp_path / "sessions")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mode"] == "vector"
    assert [case["case_id"] for case in payload["cases"]] == ["a", "b", "c", "d"]


def test_eval_command_prints_scorecard(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_run_eval(cases: Any, k: int, mode: str) -> Scorecard:
        seen.update(count=len(cases), k=k, mode=mode)
        return SCORECARD

    monkeypatch.setattr(interviewkit.cli, "run_eval", fake_run_eval)
    result = CliRunner().invoke(
        app, ["eval", "--golden", "evals/golden.example.jsonl", "--k", "3", "--no-save"]
    )

    assert result.exit_code == 0, result.stdout
    assert seen == {"count": 6, "k": 3, "mode": "hybrid"}
    assert "InterviewKit eval" in result.stdout


def test_eval_command_rejects_missing_golden_set(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["eval", "--golden", str(tmp_path / "nope.jsonl")])

    assert result.exit_code != 0
