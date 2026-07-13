"""Company interview-prep orchestration."""

from dataclasses import dataclass
from pathlib import Path

from interviewkit.agent.question_gen import GeneratedQuestion, generate_interview_questions
from interviewkit.agent.story_synth import StarAnswer, synthesize_star_answer
from interviewkit.config import Settings, get_settings
from interviewkit.sessions import AnsweredQuestion, render_session_markdown
from interviewkit.sessions import write_session_markdown


class PrepError(RuntimeError):
    """Raised when interview prep cannot run."""


@dataclass(frozen=True)
class PrepResult:
    """Completed interview-prep output."""

    questions: list[GeneratedQuestion]
    answers: list[AnsweredQuestion]
    session_path: Path
    markdown: str


async def run_prep(
    jd_path: Path,
    company: str,
    app_settings: Settings | None = None,
) -> PrepResult:
    """Generate interview questions, STAR answers, and a session file."""
    resolved_settings = app_settings or get_settings()
    job_description = _read_required_file(jd_path, "Job description")
    company_path = resolved_settings.companies_dir / f"{company}.md"
    company_principles = _read_required_file(company_path, "Company profile")
    questions = await generate_interview_questions(job_description, company_principles)
    answers = [
        AnsweredQuestion(
            question=question,
            answer=await _answer_question(question),
        )
        for question in questions
    ]
    markdown = render_session_markdown(company, jd_path, answers)
    session_path = write_session_markdown(markdown, company, resolved_settings.sessions_dir)
    return PrepResult(
        questions=questions,
        answers=answers,
        session_path=session_path,
        markdown=markdown,
    )


async def _answer_question(question: GeneratedQuestion) -> StarAnswer:
    return await synthesize_star_answer(
        question=question.text,
        principle=question.principle,
    )


def _read_required_file(path: Path, label: str) -> str:
    if not path.exists():
        raise PrepError(f"{label} file does not exist: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise PrepError(f"{label} file is empty: {path}")
    return content
