"""Command-line interface for InterviewKit."""

import asyncio
from pathlib import Path
from typing import cast

import typer

from interviewkit.agent.prep import PrepError, run_prep
from interviewkit import __version__
from interviewkit.config import RetrievalMode, Settings, get_settings
from interviewkit.evals.golden import GoldenSetError, load_golden_set
from interviewkit.evals.report import render_scorecard, write_scorecard_json
from interviewkit.evals.runner import run_eval
from interviewkit.ingest.pipeline import ingest_experiences
from interviewkit.mcp_server.server import Transport, run_server
from interviewkit.retrieval.vectorstore import init_db
from interviewkit.stories.usage import list_uses, log_use

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
    mode: str = typer.Option("hybrid", "--mode", help="Retrieval mode: hybrid or vector."),
    save: bool = typer.Option(True, "--save/--no-save", help="Write a JSON scorecard."),
) -> None:
    """Score retrieval and answers against a golden set."""
    if mode not in ("vector", "hybrid"):
        raise typer.BadParameter("Use 'vector' or 'hybrid'.", param_hint="--mode")
    try:
        cases = load_golden_set(golden)
    except GoldenSetError as exc:
        raise typer.BadParameter(str(exc), param_hint="--golden") from exc

    scorecard = asyncio.run(run_eval(cases, k=k, mode=cast(RetrievalMode, mode)))
    typer.echo(render_scorecard(scorecard))
    if save:
        path = write_scorecard_json(scorecard, get_settings().sessions_dir)
        typer.echo(f"Scorecard written to {path}")


@app.command("serve-mcp")
def serve_mcp(
    transport: Transport = typer.Option(
        "stdio",
        "--transport",
        help="MCP transport to listen on.",
    ),
    project_dir: Path | None = typer.Option(
        None,
        "--project-dir",
        help="Project root holding .env, experiences/, and companies/.",
    ),
) -> None:
    """Expose ingestion and retrieval tools over MCP."""
    try:
        run_server(transport, project_dir)
    except NotADirectoryError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def log(
    company: str = typer.Option(..., "--company", help="Company name."),
    story: str | None = typer.Option(None, "--story", help="Story ID to log."),
    question: str | None = typer.Option(None, "--question", help="Question it answered."),
    show: bool = typer.Option(False, "--list", help="List past uses instead of logging."),
) -> None:
    """Record that a story was used with a company, or list past uses."""
    if show:
        for use in list_uses(company):
            typer.echo(f"{use.used_at:%Y-%m-%d}  {use.story_id}  {use.question or ''}".rstrip())
        return
    if story is None:
        raise typer.BadParameter("Pass --story, or --list to see past uses.")
    try:
        log_use(company, story, question)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Logged {story} for {company.strip().lower()}.")


@app.command()
def mock(
    jd: Path = typer.Option(..., "--jd", help="Path to the target job description."),
    company: str = typer.Option(..., "--company", help="Company profile name."),
) -> None:
    """Start a mock behavioral interview."""
    typer.echo(f"Mock interview is not implemented yet: jd={jd}, company={company}")
