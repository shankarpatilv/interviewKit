"""Session markdown rendering and persistence."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from interviewkit.agent.question_gen import GeneratedQuestion
from interviewkit.agent.story_synth import StarAnswer


@dataclass(frozen=True)
class AnsweredQuestion:
    """A generated interview question paired with its STAR answer."""

    question: GeneratedQuestion
    answer: StarAnswer


def render_session_markdown(
    company: str,
    jd_path: Path,
    answers: list[AnsweredQuestion],
) -> str:
    """Render interview prep output as markdown."""
    lines = [
        f"# Interview prep: {company}",
        "",
        f"Job description: `{jd_path}`",
        "",
    ]
    for index, item in enumerate(answers, start=1):
        lines.extend(
            [
                f"## {index}. [{item.question.principle}] {item.question.text}",
                "",
                item.answer.text,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_session_markdown(
    content: str,
    company: str,
    sessions_dir: Path,
    now: datetime | None = None,
) -> Path:
    """Write session markdown and return the created path."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    safe_company = company.strip().lower().replace(" ", "-")
    path = sessions_dir / f"{timestamp}-{safe_company}.md"
    path.write_text(content, encoding="utf-8")
    return path
