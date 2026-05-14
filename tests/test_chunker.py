from langchain_core.documents import Document
import tiktoken

from interviewkit.ingest.chunker import chunk_experience_documents


def test_chunk_experience_documents_preserves_source_and_headers() -> None:
    document = Document(
        page_content="# Amazon\n\n## Ownership\n\nBuilt a moderation pipeline.",
        metadata={"source_file": "amazon.md"},
    )

    chunks = chunk_experience_documents([document])

    assert len(chunks) == 1
    assert chunks[0].page_content == "Built a moderation pipeline."
    assert "Amazon" not in chunks[0].page_content
    assert "Ownership" not in chunks[0].page_content
    assert chunks[0].metadata["source_file"] == "amazon.md"
    assert chunks[0].metadata["header_1"] == "Amazon"
    assert chunks[0].metadata["header_2"] == "Ownership"


def test_chunk_experience_documents_splits_large_sections() -> None:
    section_text = " ".join(f"detail-{index}" for index in range(120))
    document = Document(
        page_content=f"# Project\n\n## Scale\n\n{section_text}",
        metadata={"source_file": "project.md"},
    )

    chunks = chunk_experience_documents([document], chunk_size=40, chunk_overlap=5)

    assert len(chunks) > 1
    assert all(_token_count(chunk.page_content) <= 40 for chunk in chunks)
    assert {chunk.metadata["source_file"] for chunk in chunks} == {"project.md"}
    assert {chunk.metadata["header_1"] for chunk in chunks} == {"Project"}
    assert {chunk.metadata["header_2"] for chunk in chunks} == {"Scale"}


def test_chunk_experience_documents_keeps_stable_document_order() -> None:
    documents = [
        Document(page_content="# First\n\nAlpha story.", metadata={"source_file": "first.md"}),
        Document(page_content="# Second\n\nBeta story.", metadata={"source_file": "second.md"}),
    ]

    chunks = chunk_experience_documents(documents)

    assert [chunk.metadata["source_file"] for chunk in chunks] == ["first.md", "second.md"]


def _token_count(text: str) -> int:
    return len(tiktoken.get_encoding("gpt2").encode(text))
