from typing import Any

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


def test_ingest_command_runs_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    seen_dirs: list[str] = []

    async def fake_ingest_experiences(app_settings: Any) -> int:
        seen_dirs.append(str(app_settings.experiences_dir))
        return 3

    monkeypatch.setattr(interviewkit.cli, "ingest_experiences", fake_ingest_experiences)

    result = runner.invoke(app, ["ingest", "--experiences-dir", "examples/experiences"])

    assert result.exit_code == 0
    assert seen_dirs == ["examples/experiences"]
    assert "Indexed 3 experience chunks." in result.stdout


def test_prep_command_runs_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    calls: list[tuple[str, str]] = []

    async def fake_run_prep(jd: object, company: str) -> object:
        calls.append((str(jd), company))

        class FakeResult:
            markdown = "# Interview prep: amazon\n"
            session_path = "sessions/20260520-amazon.md"

        return FakeResult()

    monkeypatch.setattr(interviewkit.cli, "run_prep", fake_run_prep)

    result = runner.invoke(app, ["prep", "--jd", "job.txt", "--company", "amazon"])

    assert result.exit_code == 0
    assert calls == [("job.txt", "amazon")]
    assert "# Interview prep: amazon" in result.stdout
    assert "Session written to sessions/20260520-amazon.md" in result.stdout


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
