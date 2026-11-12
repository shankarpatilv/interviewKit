"""Prompt-context and parsing helpers for STAR answer synthesis."""

import json
from collections.abc import Sequence

from langchain_core.documents import Document
from pydantic import ValidationError

from interviewkit.agent.story_synth_models import DraftAnswer, StorySynthesisError


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
    """Format retrieved chunks with their ids and source files for the prompt."""
    return "\n\n".join(
        f"chunk_id: {chunk_id(chunk)}\nsource_file: {chunk.metadata.get('source_file', 'unknown')}"
        f"\ncontent:\n{chunk.page_content}"
        for chunk in chunks
    )


def chunk_id(chunk: Document) -> str:
    """Return the chunk's id; `with_chunk_ids` guarantees one exists."""
    return str(chunk.metadata.get("chunk_id", ""))


def with_chunk_ids(chunks: Sequence[Document]) -> list[Document]:
    """Give chunks without a database id a stable `<source_file>#<rank>` id."""
    labeled = []
    for index, chunk in enumerate(chunks, start=1):
        if chunk.metadata.get("chunk_id"):
            labeled.append(chunk)
            continue
        source = chunk.metadata.get("source_file", "chunk")
        metadata = {**chunk.metadata, "chunk_id": f"{source}#{index}"}
        labeled.append(Document(page_content=chunk.page_content, metadata=metadata))
    return labeled


def source_files(chunks: Sequence[Document]) -> tuple[str, ...]:
    """Return unique non-empty source files from retrieved chunks."""
    sources = []
    for chunk in chunks:
        source_file = str(chunk.metadata.get("source_file", "")).strip()
        if source_file and source_file not in sources:
            sources.append(source_file)
    return tuple(sources)


def parse_draft(raw: str) -> DraftAnswer:
    """Parse the LLM's JSON answer, tolerating code fences or surrounding prose."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        raise StorySynthesisError("STAR synthesis response was not JSON")
    try:
        return DraftAnswer.model_validate(json.loads(raw[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise StorySynthesisError(f"STAR synthesis JSON was invalid: {exc}") from exc
