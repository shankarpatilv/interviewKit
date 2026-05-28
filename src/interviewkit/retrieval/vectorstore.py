"""PostgreSQL and pgvector initialization helpers."""

from collections.abc import Callable
from importlib import import_module
from typing import Protocol, cast

from interviewkit.config import Settings, get_settings
from interviewkit.retrieval.schema import SCHEMA_STATEMENTS


class CursorLike(Protocol):
    """Small cursor protocol used by the initializer and tests."""

    def execute(self, query: str) -> object:
        """Execute one SQL statement."""

    def __enter__(self) -> "CursorLike":
        """Enter the cursor context manager."""

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Exit the cursor context manager."""


class ConnectionLike(Protocol):
    """Small connection protocol shared by psycopg and psycopg2."""

    def cursor(self) -> CursorLike:
        """Return a database cursor."""

    def commit(self) -> None:
        """Commit the active transaction."""

    def rollback(self) -> None:
        """Roll back the active transaction."""

    def close(self) -> None:
        """Close the database connection."""


ConnectFactory = Callable[[str], ConnectionLike]


def _default_connect(database_url: str) -> ConnectionLike:
    try:
        psycopg = import_module("psycopg")
    except ImportError:
        try:
            psycopg = import_module("psycopg2")
        except ImportError as exc:
            msg = (
                "A PostgreSQL driver is required to initialize the database. "
                "Install psycopg or psycopg2, then retry."
            )
            raise RuntimeError(msg) from exc

    return cast(ConnectionLike, psycopg.connect(database_url))


def init_db(
    app_settings: Settings | None = None,
    connect: ConnectFactory = _default_connect,
) -> None:
    """Create the pgvector extension, experiences table, and indexes."""
    resolved_settings = app_settings or get_settings()
    connection = connect(resolved_settings.database_url)

    try:
        with connection.cursor() as cursor:
            for statement in SCHEMA_STATEMENTS:
                cursor.execute(statement)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
