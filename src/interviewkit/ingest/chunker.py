"""Split loaded experience documents into retrieval chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from interviewkit.config import settings

HEADER_KEYS = {
    "Header 1": "header_1",
    "Header 2": "header_2",
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
            chunks.extend(splitter.split_documents([section]))

    return chunks


def _split_markdown_headers(document: Document) -> list[Document]:
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
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
    return metadata
