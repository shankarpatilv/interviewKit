"""Command-line interface for InterviewKit."""

from pathlib import Path

import typer

from interviewkit import __version__
from interviewkit.config import Settings
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
    typer.echo(f"Experience ingestion is not implemented yet: {settings.experiences_dir}")


@app.command()
def prep(
    jd: Path = typer.Option(..., "--jd", help="Path to the target job description."),
    company: str = typer.Option(..., "--company", help="Company profile name."),
) -> None:
    """Generate company-specific interview prep."""
    typer.echo(f"Interview prep is not implemented yet: jd={jd}, company={company}")


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
