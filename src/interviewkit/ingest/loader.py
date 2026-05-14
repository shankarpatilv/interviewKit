"""Load private markdown experience documents."""

from pathlib import Path
import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_experience_documents(experiences_dir: Path | str = Path("experiences")) -> list[Document]:
    """Load non-template markdown experience files as LangChain documents."""
    root = Path(experiences_dir)
    if not root.exists():
        logger.warning("Experience directory does not exist: %s", root)
        return []

    documents: list[Document] = []
    for path in sorted(root.rglob("*.md")):
        if path.name.startswith("_"):
            continue

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            logger.warning("Skipping empty experience file: %s", path)
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={"source_file": path.relative_to(root).as_posix()},
            )
        )

    return documents
