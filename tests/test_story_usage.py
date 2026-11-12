from datetime import datetime

import pytest
from typer.testing import CliRunner

import interviewkit.cli
from interviewkit.cli import app
from interviewkit.config import Settings
from interviewkit.retrieval.schema import SCHEMA_STATEMENTS
from interviewkit.stories import usage
from test_vectorstore import make_connection

SETTINGS = Settings(database_url="postgresql://example.test/x", _env_file=None)


def test_schema_creates_story_uses_with_company_recency_index() -> None:
    sql = "\n".join(SCHEMA_STATEMENTS)

    assert "CREATE TABLE IF NOT EXISTS story_uses" in sql
    assert "ON story_uses(company, used_at DESC)" in sql


def test_log_use_normalizes_company_and_commits() -> None:
    connection = make_connection()

    usage.log_use(" Amazon ", "a/b/c", "Q?", app_settings=SETTINGS, connect=lambda _: connection)

    _, params = connection.cursor_instance.statements[0]
    assert params == ("amazon", "a/b/c", "Q?")
    assert connection.committed and connection.closed


def test_log_use_rejects_blank_story() -> None:
    with pytest.raises(ValueError, match="story_id"):
        usage.log_use("amazon", " ", app_settings=SETTINGS, connect=lambda _: make_connection())


def test_list_uses_maps_rows_and_filters_by_company() -> None:
    when = datetime(2026, 11, 1)
    connection = make_connection(rows=[("amazon", "a/b", None, when)])

    uses = usage.list_uses("Amazon", limit=5, app_settings=SETTINGS, connect=lambda _: connection)

    _, params = connection.cursor_instance.statements[0]
    assert params == ("amazon", "amazon", 5)
    assert uses == [usage.StoryUse("amazon", "a/b", None, when)]


def test_list_uses_without_company_passes_null() -> None:
    connection = make_connection()

    usage.list_uses(app_settings=SETTINGS, connect=lambda _: connection)

    assert connection.cursor_instance.statements[0][1] == (None, None, usage.DEFAULT_USAGE_LIMIT)


def test_is_repeat_reads_exists_flag() -> None:
    yes = make_connection(rows=[(True,)])
    no = make_connection(rows=[(False,)])

    assert usage.is_repeat("amazon", "a/b", app_settings=SETTINGS, connect=lambda _: yes)
    assert not usage.is_repeat("amazon", "a/b", app_settings=SETTINGS, connect=lambda _: no)


def test_failed_write_rolls_back() -> None:
    connection = make_connection(should_fail=True)

    with pytest.raises(RuntimeError):
        usage.log_use("amazon", "a/b", app_settings=SETTINGS, connect=lambda _: connection)

    assert connection.rolled_back and connection.closed


def test_log_command_records_use(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(interviewkit.cli, "log_use", lambda *args: calls.append(args))

    result = CliRunner().invoke(app, ["log", "--company", "Amazon", "--story", "a/b"])

    assert result.exit_code == 0
    assert calls == [("Amazon", "a/b", None)]
    assert "Logged a/b for amazon." in result.output


def test_log_command_lists_uses(monkeypatch: pytest.MonkeyPatch) -> None:
    use = usage.StoryUse("amazon", "a/b", "Q?", datetime(2026, 11, 1))
    monkeypatch.setattr(interviewkit.cli, "list_uses", lambda company: [use])

    result = CliRunner().invoke(app, ["log", "--company", "amazon", "--list"])

    assert "2026-11-01  a/b  Q?" in result.output


def test_log_command_requires_story_unless_listing() -> None:
    result = CliRunner().invoke(app, ["log", "--company", "amazon"])

    assert result.exit_code != 0
