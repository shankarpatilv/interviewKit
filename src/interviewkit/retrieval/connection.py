"""Database connection helpers."""

from collections.abc import Callable
from importlib import import_module
from typing import Protocol, cast


class CursorLike(Protocol):
    """Small cursor protocol used by vectorstore operations and tests."""

    def execute(self, query: str, params: object | None = None) -> object:
        """Execute one SQL statement."""

    def fetchall(self) -> list[object]:
        """Return all rows from the previous query."""

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


def default_connect(database_url: str) -> ConnectionLike:
    """Open a PostgreSQL connection using an installed psycopg driver."""
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
