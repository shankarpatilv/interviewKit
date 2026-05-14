import pytest
from langchain_core.documents import Document

from interviewkit.config import Settings
from interviewkit.ingest.embedder import EmbeddedDocument
from interviewkit.retrieval.schema import SCHEMA_STATEMENTS
from interviewkit.retrieval import vectorstore


class FakeCursor:
    def __init__(self, should_fail: bool = False, rows: list[object] | None = None) -> None:
        self.should_fail = should_fail
        self.statements: list[tuple[str, object | None]] = []
        self.rows = rows or []

    def execute(self, query: str, params: object | None = None) -> object:
        if self.should_fail:
            raise RuntimeError("database error")
        self.statements.append((query, params))
        return None

    def fetchall(self) -> list[object]:
        return self.rows

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


def make_connection(should_fail: bool = False, rows: list[object] | None = None) -> FakeConnection:
    return FakeConnection(FakeCursor(should_fail=should_fail, rows=rows))


def test_init_db_creates_pgvector_experiences_table_and_indexes() -> None:
    connection = make_connection()
    settings = Settings(database_url="postgresql://example.test/interviewkit", _env_file=None)

    vectorstore.init_db(settings, connect=lambda database_url: connection)

    assert connection.cursor_instance.statements == [
        (statement, None) for statement in SCHEMA_STATEMENTS
    ]
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


def test_upsert_chunks_writes_deterministic_rows_and_commits() -> None:
    connection = make_connection()
    settings = Settings(database_url="postgresql://example.test/interviewkit", _env_file=None)
    chunks = [
        EmbeddedDocument(
            document=Document(
                page_content="Built a moderation pipeline.",
                metadata={"source_file": "amazon.md", "header_1": "Amazon"},
            ),
            embedding=[0.1, 0.2, 0.3],
        ),
        EmbeddedDocument(
            document=Document(
                page_content="Reduced latency.",
                metadata={"source_file": "amazon.md", "header_1": "Amazon"},
            ),
            embedding=[0.4, 0.5, 0.6],
        ),
    ]

    vectorstore.upsert_chunks(chunks, settings, connect=lambda database_url: connection)

    first_params = connection.cursor_instance.statements[0][1]
    second_params = connection.cursor_instance.statements[1][1]
    assert isinstance(first_params, tuple)
    assert isinstance(second_params, tuple)
    assert first_params[0].startswith("amazon.md:0:")
    assert second_params[0].startswith("amazon.md:1:")
    assert first_params[1:] == (
        "amazon.md",
        "Built a moderation pipeline.",
        '{"source_file": "amazon.md", "header_1": "Amazon"}',
        "[0.1,0.2,0.3]",
    )
    assert connection.committed is True
    assert connection.rolled_back is False
    assert connection.closed is True


def test_upsert_chunks_requires_source_file_metadata() -> None:
    connection = make_connection()
    chunk = EmbeddedDocument(document=Document(page_content="Missing source"), embedding=[0.1])

    with pytest.raises(ValueError, match="source_file"):
        vectorstore.upsert_chunks([chunk], connect=lambda database_url: connection)

    assert connection.committed is False
    assert connection.rolled_back is True
    assert connection.closed is True


def test_similarity_search_returns_ranked_documents_with_metadata() -> None:
    rows = [
        (
            "amazon.md",
            "Built a moderation pipeline.",
            {"header_1": "Amazon"},
            0.91,
        )
    ]
    connection = make_connection(rows=rows)
    settings = Settings(database_url="postgresql://example.test/interviewkit", _env_file=None)

    results = vectorstore.similarity_search(
        [0.1, 0.2], k=1, app_settings=settings, connect=lambda _: connection
    )

    statement, params = connection.cursor_instance.statements[0]
    assert "ORDER BY (embedding <=>" in statement
    assert params == ("[0.1,0.2]", "[0.1,0.2]", 1)
    assert len(results) == 1
    assert results[0].document.page_content == "Built a moderation pipeline."
    assert results[0].document.metadata == {"header_1": "Amazon", "source_file": "amazon.md"}
    assert results[0].similarity == 0.91
    assert connection.committed is True
