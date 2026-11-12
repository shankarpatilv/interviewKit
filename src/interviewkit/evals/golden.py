"""Golden-set format and loader for the eval harness."""

import json
import logging
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class GoldenSetError(ValueError):
    """Raised when a golden-set file is missing, malformed, or invalid."""


class ExpectedStory(BaseModel):
    """A story the retriever is expected to surface for a golden case."""

    model_config = _MODEL_CONFIG

    source_file: str = Field(min_length=1)
    header_2: str | None = None
    story_id: str | None = None  # when set, match on chunk story_id instead of header_2


class GoldenCase(BaseModel):
    """One evaluation case: a question plus the stories expected to answer it."""

    model_config = _MODEL_CONFIG

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    company: str = Field(min_length=1)
    principle: str = Field(min_length=1)
    jd: str | None = None
    expected_stories: tuple[ExpectedStory, ...] = ()
    expect_refusal: bool = False

    @model_validator(mode="after")
    def _check_refusal_consistency(self) -> Self:
        if self.expect_refusal and self.expected_stories:
            raise ValueError("expect_refusal=true cases must not list expected_stories")
        if not self.expect_refusal and not self.expected_stories:
            raise ValueError("cases with expect_refusal=false need at least one expected story")
        return self


def load_golden_set(path: Path) -> list[GoldenCase]:
    """Load golden cases from a JSONL file (one JSON object per line; blank lines skipped).

    Raises:
        GoldenSetError: If the file is missing, empty, has invalid JSON or an invalid case
            (with its 1-based line number), or repeats a case id.
    """
    if not path.is_file():
        raise GoldenSetError(f"Golden set file not found: {path}")
    cases: list[GoldenCase] = []
    seen: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = GoldenCase.model_validate(json.loads(line))
        except json.JSONDecodeError as error:
            raise GoldenSetError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        except ValidationError as error:
            raise GoldenSetError(f"{path}:{line_number}: invalid case: {error}") from error
        if case.id in seen:
            raise GoldenSetError(
                f"{path}:{line_number}: duplicate id {case.id!r} "
                f"(first seen on line {seen[case.id]})"
            )
        seen[case.id] = line_number
        cases.append(case)
    if not cases:
        raise GoldenSetError(f"Golden set file has no cases: {path}")
    logger.info("Loaded %d golden cases from %s", len(cases), path)
    return cases
