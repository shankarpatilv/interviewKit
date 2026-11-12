"""Whole-story and company-rubric lookups."""

import re

from langchain_core.documents import Document

from interviewkit.config import Settings, get_settings
from interviewkit.retrieval.connection import ConnectFactory, default_connect
from interviewkit.retrieval.records import row_to_retrieved_document
from interviewkit.retrieval.schema import GET_STORY_SQL
from interviewkit.stories.usage import normalize_company

_COMPANY_KEY = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def get_story(
    story_id: str,
    *,
    app_settings: Settings | None = None,
    connect: ConnectFactory = default_connect,
) -> list[Document]:
    """Return every chunk of one story, in document order."""
    connection = connect((app_settings or get_settings()).database_url)
    try:
        with connection.cursor() as cursor:
            cursor.execute(GET_STORY_SQL, (story_id.strip(),))
            rows = cursor.fetchall()
        connection.commit()
        return [row_to_retrieved_document(row).document for row in rows]
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_company_rubric(company: str, *, app_settings: Settings | None = None) -> str:
    """Return the text of `companies/<company>.md`.

    Raises ValueError for names that are not a plain file key or have no profile.
    """
    key = normalize_company(company)
    if not _COMPANY_KEY.match(key):
        raise ValueError(f"Invalid company name: {company!r}")
    path = (app_settings or get_settings()).companies_dir / f"{key}.md"
    if not path.is_file():
        raise ValueError(f"No company profile for {key!r}")
    return path.read_text(encoding="utf-8").strip()
