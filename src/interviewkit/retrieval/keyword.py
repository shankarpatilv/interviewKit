"""Postgres full-text keyword search over experience chunks."""

import json
from collections.abc import Mapping

from interviewkit.config import Settings, get_settings
from interviewkit.retrieval.connection import ConnectFactory, default_connect
from interviewkit.retrieval.models import RetrievedDocument
from interviewkit.retrieval.records import row_to_retrieved_document
from interviewkit.retrieval.schema import KEYWORD_SEARCH_SQL


def keyword_search(
    query: str,
    *,
    k: int = 5,
    filters: Mapping[str, object] | None = None,
    app_settings: Settings | None = None,
    connect: ConnectFactory = default_connect,
) -> list[RetrievedDocument]:
    """Return up to k chunks ranked by full-text relevance (`ts_rank_cd`).

    The score is stored in `RetrievedDocument.similarity`; it is a rank score,
    not a cosine similarity, so only compare it with other keyword scores.
    """
    resolved_settings = app_settings or get_settings()
    connection = connect(resolved_settings.database_url)
    try:
        with connection.cursor() as cursor:
            cursor.execute(KEYWORD_SEARCH_SQL, (query, json.dumps(dict(filters or {})), k))
            rows = cursor.fetchall()
        connection.commit()
        return [row_to_retrieved_document(row) for row in rows]
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
