import json
from pathlib import Path
from typing import Any

import pytest

from interviewkit.evals.golden import ExpectedStory, GoldenSetError, load_golden_set

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "evals" / "golden.example.jsonl"


def _case(**overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "id": "case-1",
        "question": "Tell me about ownership.",
        "company": "amazon",
        "principle": "Ownership",
        "expected_stories": [{"source_file": "example_payments_migration.md"}],
    }
    case.update(overrides)
    return case


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_golden_set_parses_valid_cases(tmp_path: Path) -> None:
    story = {"source_file": " example_oncall_incident.md ", "header_2": "Root Cause"}
    golden = _write(
        tmp_path / "golden.jsonl",
        [
            json.dumps(_case(jd="Payments backend role.")),
            json.dumps(_case(id="case-2", expected_stories=[story])),
        ],
    )

    cases = load_golden_set(golden)

    assert [case.id for case in cases] == ["case-1", "case-2"]
    assert cases[0].jd == "Payments backend role."
    assert cases[0].expect_refusal is False
    assert cases[1].expected_stories == (
        ExpectedStory(source_file="example_oncall_incident.md", header_2="Root Cause"),
    )


def test_load_golden_set_skips_blank_lines(tmp_path: Path) -> None:
    golden = _write(tmp_path / "g.jsonl", ["", json.dumps(_case()), "   ", ""])

    assert len(load_golden_set(golden)) == 1


def test_load_golden_set_reports_invalid_json_line(tmp_path: Path) -> None:
    golden = _write(tmp_path / "g.jsonl", [json.dumps(_case()), "", "{not json"])

    with pytest.raises(GoldenSetError, match=r":3: invalid JSON"):
        load_golden_set(golden)


def test_load_golden_set_reports_validation_error_line(tmp_path: Path) -> None:
    golden = _write(tmp_path / "g.jsonl", [json.dumps(_case(company="  "))])

    with pytest.raises(GoldenSetError, match=r":1: invalid case"):
        load_golden_set(golden)


def test_load_golden_set_rejects_duplicate_ids(tmp_path: Path) -> None:
    golden = _write(tmp_path / "g.jsonl", [json.dumps(_case()), "", json.dumps(_case())])

    with pytest.raises(GoldenSetError, match=r":3: duplicate id 'case-1' .*line 1"):
        load_golden_set(golden)


def test_refusal_case_with_stories_is_rejected(tmp_path: Path) -> None:
    golden = _write(tmp_path / "g.jsonl", [json.dumps(_case(expect_refusal=True))])

    with pytest.raises(GoldenSetError, match="must not list expected_stories"):
        load_golden_set(golden)


def test_non_refusal_case_without_stories_is_rejected(tmp_path: Path) -> None:
    golden = _write(tmp_path / "g.jsonl", [json.dumps(_case(expected_stories=[]))])

    with pytest.raises(GoldenSetError, match="at least one expected story"):
        load_golden_set(golden)


def test_load_golden_set_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="not found"):
        load_golden_set(tmp_path / "missing.jsonl")


def test_load_golden_set_rejects_file_without_cases(tmp_path: Path) -> None:
    golden = _write(tmp_path / "g.jsonl", ["", "  "])

    with pytest.raises(GoldenSetError, match="no cases"):
        load_golden_set(golden)


def test_example_golden_set_loads() -> None:
    cases = load_golden_set(EXAMPLE_PATH)

    assert len(cases) == 6
    assert {case.company for case in cases} == {"amazon", "microsoft"}
    assert sum(case.expect_refusal for case in cases) >= 2
    assert any(story.header_2 for case in cases for story in case.expected_stories)
