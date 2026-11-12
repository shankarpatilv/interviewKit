import asyncio
from pathlib import Path

import pytest

import interviewkit.agent.prep as prep
from interviewkit.agent.prep import PrepError, run_prep
from interviewkit.agent.question_gen import GeneratedQuestion
from interviewkit.agent.story_synth import StarAnswer
from interviewkit.config import Settings


def test_run_prep_writes_session_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_generate_questions(
        job_description: str,
        company_principles: str,
    ) -> list[GeneratedQuestion]:
        assert "backend role" in job_description
        assert "Ownership" in company_principles
        return [GeneratedQuestion(principle="Ownership", text="Tell me about ownership.")]

    async def fake_answer(
        question: GeneratedQuestion, company: str, selector: object
    ) -> StarAnswer:
        assert company == "amazon" and selector is None
        assert question.text == "Tell me about ownership."
        return StarAnswer(
            principle="Ownership",
            text="Situation: ...\nSources:\n- (source: ownership.md)",
            sources=("ownership.md",),
        )

    jd_path = tmp_path / "job.txt"
    companies_dir = tmp_path / "companies"
    sessions_dir = tmp_path / "sessions"
    jd_path.write_text("backend role", encoding="utf-8")
    companies_dir.mkdir()
    (companies_dir / "amazon.md").write_text("Ownership", encoding="utf-8")
    settings = Settings(companies_dir=companies_dir, sessions_dir=sessions_dir)
    monkeypatch.setattr(prep, "generate_interview_questions", fake_generate_questions)
    monkeypatch.setattr(prep, "_answer_question", fake_answer)

    result = asyncio.run(run_prep(jd_path, "amazon", app_settings=settings))

    assert result.session_path.exists()
    assert result.session_path.parent == sessions_dir
    assert "Tell me about ownership." in result.markdown
    assert "ownership.md" in result.session_path.read_text(encoding="utf-8")


def test_run_prep_rejects_missing_company_profile(tmp_path: Path) -> None:
    jd_path = tmp_path / "job.txt"
    jd_path.write_text("backend role", encoding="utf-8")
    settings = Settings(companies_dir=tmp_path / "companies", sessions_dir=tmp_path)

    with pytest.raises(PrepError, match="Company profile file does not exist"):
        asyncio.run(run_prep(jd_path, "amazon", app_settings=settings))
