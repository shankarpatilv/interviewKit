"""Command-line interface for InterviewKit."""

import asyncio
from pathlib import Path

import typer

from interviewkit.agent.prep import PrepError, run_prep
from interviewkit import __version__
from interviewkit.config import Settings, get_settings
from interviewkit.evals.golden import GoldenSetError, load_golden_set
from interviewkit.evals.report import render_scorecard, write_scorecard_json
from interviewkit.evals.runner import run_eval
from interviewkit.ingest.pipeline import ingest_experiences
from interviewkit.retrieval.vectorstore import init_db

app = typer.Typer(
    help="Prepare company-specific behavioral interviews from your experience notes.",
    no_args_is_help=True,
)


def _version_callback(show_version: bool) -> None:
    if show_version:
        typer.echo(f"interviewkit {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        help="Show the installed InterviewKit version.",
    ),
) -> None:
    """Run the InterviewKit CLI."""


@app.command("init-db")
def init_db_command() -> None:
    """Initialize the local PostgreSQL schema."""
    init_db()
    typer.echo("Database schema initialized.")


@app.command()
def ingest(
    experiences_dir: Path = typer.Option(
        Path("experiences"),
        "--experiences-dir",
        help="Directory containing private markdown experience files.",
    ),
) -> None:
    """Index experience documents for retrieval."""
    settings = Settings(experiences_dir=experiences_dir)
    indexed_count = asyncio.run(ingest_experiences(app_settings=settings))
    typer.echo(f"Indexed {indexed_count} experience chunks.")


@app.command()
def prep(
    jd: Path = typer.Option(..., "--jd", help="Path to the target job description."),
    company: str = typer.Option(..., "--company", help="Company profile name."),
) -> None:
    """Generate company-specific interview prep."""
    try:
        result = asyncio.run(run_prep(jd, company))
    except PrepError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(result.markdown)
    typer.echo(f"Session written to {result.session_path}")


@app.command("eval")
def eval_command(
    golden: Path = typer.Option(
        Path("evals/golden.jsonl"),
        "--golden",
        help="JSONL golden set of questions and expected stories.",
    ),
    k: int = typer.Option(5, "--k", min=1, max=20, help="Chunks retrieved per question."),
    mode: str = typer.Option("vector", "--mode", help="Retrieval mode label: vector."),
    save: bool = typer.Option(True, "--save/--no-save", help="Write a JSON scorecard."),
) -> None:
    """Score retrieval and answers against a golden set."""
    if mode != "vector":
        raise typer.BadParameter("Only 'vector' retrieval is available.", param_hint="--mode")
    try:
        cases = load_golden_set(golden)
    except GoldenSetError as exc:
        raise typer.BadParameter(str(exc), param_hint="--golden") from exc

    scorecard = asyncio.run(run_eval(cases, k=k, mode=mode))
    typer.echo(render_scorecard(scorecard))
    if save:
        path = write_scorecard_json(scorecard, get_settings().sessions_dir)
        typer.echo(f"Scorecard written to {path}")


@app.command()
def log(
    company: str = typer.Option(..., "--company", help="Company name."),
    story: str = typer.Option(..., "--story", help="Story identifier used in an interview."),
) -> None:
    """Record that a story was used for an interview."""
    typer.echo(f"Story logging is not implemented yet: company={company}, story={story}")


@app.command()
def mock(
    jd: Path = typer.Option(..., "--jd", help="Path to the target job description."),
    company: str = typer.Option(..., "--company", help="Company profile name."),
) -> None:
    """Start a mock behavioral interview."""
    typer.echo(f"Mock interview is not implemented yet: jd={jd}, company={company}")
