import pytest
from typer.testing import CliRunner

import interviewkit.cli
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


def test_init_db_command_delegates_to_vectorstore(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    calls: list[bool] = []

    def fake_init_db() -> None:
        calls.append(True)

    monkeypatch.setattr(interviewkit.cli, "init_db", fake_init_db)

    result = runner.invoke(app, ["init-db"])

    assert result.exit_code == 0
    assert calls == [True]
    assert "Database schema initialized." in result.stdout
