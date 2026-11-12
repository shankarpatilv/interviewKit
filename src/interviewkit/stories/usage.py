"""Track which stories were used with which company."""

from dataclasses import dataclass
from datetime import datetime
from typing import cast

from interviewkit.config import Settings, get_settings
from interviewkit.retrieval.connection import ConnectFactory, default_connect
from interviewkit.retrieval.schema import (
    INSERT_STORY_USE_SQL,
    LIST_STORY_USES_SQL,
    STORY_USE_EXISTS_SQL,
)

DEFAULT_USAGE_LIMIT = 50


@dataclass(frozen=True)
class StoryUse:
    """One recorded use of a story in an interview."""

    company: str
    story_id: str
    question: str | None
    used_at: datetime


def normalize_company(company: str) -> str:
    """Return the lowercase company key used by `companies/<name>.md`."""
    key = company.strip().lower()
    if not key:
        raise ValueError("company must not be empty")
    return key


def log_use(
    company: str,
    story_id: str,
    question: str | None = None,
    *,
    app_settings: Settings | None = None,
    connect: ConnectFactory = default_connect,
) -> None:
    """Record that `story_id` was told to `company`."""
    if not story_id.strip():
        raise ValueError("story_id must not be empty")
    params = (normalize_company(company), story_id.strip(), question)
    _execute(INSERT_STORY_USE_SQL, params, app_settings, connect, fetch=False)


def list_uses(
    company: str | None = None,
    *,
    limit: int = DEFAULT_USAGE_LIMIT,
    app_settings: Settings | None = None,
    connect: ConnectFactory = default_connect,
) -> list[StoryUse]:
    """Return recent story uses, newest first, optionally for one company."""
    key = normalize_company(company) if company is not None else None
    rows = _execute(LIST_STORY_USES_SQL, (key, key, limit), app_settings, connect, fetch=True)
    return [
        StoryUse(*cast(tuple[str, str, str | None, datetime], row)) for row in rows
    ]


def is_repeat(
    company: str,
    story_id: str,
    *,
    app_settings: Settings | None = None,
    connect: ConnectFactory = default_connect,
) -> bool:
    """Return True when `story_id` was already told to `company`."""
    params = (normalize_company(company), story_id.strip())
    rows = _execute(STORY_USE_EXISTS_SQL, params, app_settings, connect, fetch=True)
    return bool(rows and cast(tuple[bool], rows[0])[0])


def _execute(
    sql: str,
    params: tuple[object, ...],
    app_settings: Settings | None,
    connect: ConnectFactory,
    *,
    fetch: bool,
) -> list[object]:
    connection = connect((app_settings or get_settings()).database_url)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall() if fetch else []
        connection.commit()
        return rows
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
