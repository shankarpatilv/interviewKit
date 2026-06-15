"""PostgreSQL and pgvector storage helpers."""

from interviewkit.config import Settings, get_settings
from interviewkit.ingest.embedder import EmbeddedDocument
from interviewkit.retrieval.connection import ConnectFactory, default_connect
from interviewkit.retrieval.models import RetrievedDocument
from interviewkit.retrieval.records import chunk_params, numbered_chunks, row_to_retrieved_document
from interviewkit.retrieval.records import vector_literal
from interviewkit.retrieval.schema import SCHEMA_STATEMENTS, SIMILARITY_SEARCH_SQL, UPSERT_CHUNK_SQL


def init_db(
    app_settings: Settings | None = None,
    connect: ConnectFactory = default_connect,
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


def upsert_chunks(
    chunks: list[EmbeddedDocument],
    app_settings: Settings | None = None,
    connect: ConnectFactory = default_connect,
) -> None:
    """Insert or update embedded chunks in pgvector storage."""
    resolved_settings = app_settings or get_settings()
    connection = connect(resolved_settings.database_url)

    try:
        with connection.cursor() as cursor:
            for chunk_id, chunk in numbered_chunks(chunks):
                cursor.execute(
                    UPSERT_CHUNK_SQL,
                    chunk_params(chunk_id, chunk),
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def similarity_search(
    query_embedding: list[float],
    *,
    k: int = 5,
    app_settings: Settings | None = None,
    connect: ConnectFactory = default_connect,
) -> list[RetrievedDocument]:
    """Return the top-k chunks closest to a query embedding."""
    resolved_settings = app_settings or get_settings()
    connection = connect(resolved_settings.database_url)

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                SIMILARITY_SEARCH_SQL,
                (vector_literal(query_embedding), vector_literal(query_embedding), k),
            )
            rows = cursor.fetchall()
        connection.commit()
        return [row_to_retrieved_document(row) for row in rows]
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
