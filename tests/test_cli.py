from typer.testing import CliRunner

from interviewkit import __version__
from interviewkit.cli import app


def test_version_option_shows_package_version() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert f"interviewkit {__version__}" in result.stdout


def test_prep_placeholder_accepts_required_options() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["prep", "--jd", "job.txt", "--company", "amazon"])

    assert result.exit_code == 0
    assert "Interview prep is not implemented yet" in result.stdout
