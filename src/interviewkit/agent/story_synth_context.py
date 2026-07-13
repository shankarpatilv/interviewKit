"""Prompt-context helpers for STAR answer synthesis."""

from collections.abc import Sequence

from langchain_core.documents import Document


def build_user_message(question: str, principle: str, chunks: Sequence[Document]) -> str:
    """Build the user message passed to the STAR synthesis prompt."""
    return (
        "Interview question:\n"
        f"{question}\n\n"
        "Company principle or competency:\n"
        f"{principle}\n\n"
        "Retrieved experience chunks:\n"
        f"{format_chunks(chunks)}"
    )


def format_chunks(chunks: Sequence[Document]) -> str:
    """Format retrieved chunks with source metadata for the prompt."""
    formatted_chunks = []
    for index, chunk in enumerate(chunks, start=1):
        source_file = str(chunk.metadata.get("source_file", "unknown"))
        formatted_chunks.append(
            f"Chunk {index}\nsource_file: {source_file}\ncontent:\n{chunk.page_content}"
        )
    return "\n\n".join(formatted_chunks)


def source_files(chunks: Sequence[Document]) -> tuple[str, ...]:
    """Return unique non-empty source files from retrieved chunks."""
    sources = []
    for chunk in chunks:
        source_file = str(chunk.metadata.get("source_file", "")).strip()
        if source_file and source_file not in sources:
            sources.append(source_file)
    return tuple(sources)
