import pytest

from interviewkit.config import Settings
from interviewkit.retrieval.schema import SCHEMA_STATEMENTS
from interviewkit.retrieval import vectorstore


class FakeCursor:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.statements: list[str] = []

    def execute(self, query: str) -> object:
        if self.should_fail:
            raise RuntimeError("database error")
        self.statements.append(query)
        return None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_instance = cursor
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def make_connection(should_fail: bool = False) -> FakeConnection:
    return FakeConnection(FakeCursor(should_fail=should_fail))


def test_init_db_creates_pgvector_experiences_table_and_indexes() -> None:
    connection = make_connection()
    settings = Settings(database_url="postgresql://example.test/interviewkit", _env_file=None)

    vectorstore.init_db(settings, connect=lambda database_url: connection)

    assert connection.cursor_instance.statements == list(SCHEMA_STATEMENTS)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in SCHEMA_STATEMENTS[0]
    assert "embedding VECTOR(1536) NOT NULL" in SCHEMA_STATEMENTS[1]
    assert "USING ivfflat (embedding vector_cosine_ops)" in SCHEMA_STATEMENTS[2]
    assert "idx_experiences_source" in SCHEMA_STATEMENTS[3]
    assert connection.committed is True
    assert connection.rolled_back is False
    assert connection.closed is True


def test_init_db_rolls_back_and_closes_connection_on_error() -> None:
    connection = make_connection(should_fail=True)
    settings = Settings(_env_file=None)

    with pytest.raises(RuntimeError, match="database error"):
        vectorstore.init_db(settings, connect=lambda database_url: connection)

    assert connection.committed is False
    assert connection.rolled_back is True
    assert connection.closed is True
