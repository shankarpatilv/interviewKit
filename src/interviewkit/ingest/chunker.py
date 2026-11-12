"""Split loaded experience documents into retrieval chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from interviewkit.config import settings
from interviewkit.ingest.story_ids import story_id

HEADER_KEYS = {
    "Header 1": "header_1",
    "Header 2": "header_2",
    "Header 3": "header_3",
}


def chunk_experience_documents(
    documents: list[Document],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[Document]:
    """Split markdown documents into retrieval-sized chunks with header metadata."""
    splitter = _build_splitter(chunk_size, chunk_overlap)
    chunks: list[Document] = []

    for document in documents:
        header_sections = _split_markdown_headers(document)
        for section in header_sections:
            section.metadata = _normalize_metadata(document.metadata, section.metadata)
            chunks.extend(
                _with_breadcrumb(chunk)
                for chunk in splitter.split_documents([section])
                if any(char.isalnum() for char in chunk.page_content)
            )

    return chunks


def _with_breadcrumb(chunk: Document) -> Document:
    """Prefix chunk text with its headings, which `strip_headers` removed.

    Without this, a story's title (often its strongest topical signal) never
    reaches the embedding or the LLM.
    """
    headers = [chunk.metadata.get(key) for key in HEADER_KEYS.values()]
    breadcrumb = " > ".join(h for h in headers if isinstance(h, str) and h)
    if breadcrumb:
        chunk.page_content = f"{breadcrumb}\n\n{chunk.page_content}"
    return chunk


def _split_markdown_headers(document: Document) -> list[Document]:
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ],
        strip_headers=True,
    )
    return markdown_splitter.split_text(document.page_content)


def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _normalize_metadata(
    document_metadata: dict[str, object],
    section_metadata: dict[str, object],
) -> dict[str, object]:
    metadata = document_metadata.copy()
    for source_key, target_key in HEADER_KEYS.items():
        if source_key in section_metadata:
            metadata[target_key] = section_metadata[source_key]
    metadata.setdefault("header_1", None)
    metadata.setdefault("header_2", None)
    metadata.setdefault("header_3", None)
    metadata["story_id"] = story_id(
        str(metadata.get("source_file", "")), metadata["header_2"], metadata["header_3"]
    )
    return metadata
