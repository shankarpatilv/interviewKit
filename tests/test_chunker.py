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
    assert chunks[0].page_content == "Amazon > Ownership\n\nBuilt a moderation pipeline."
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
    prefix = "Project > Scale\n\n"
    assert all(chunk.page_content.startswith(prefix) for chunk in chunks)
    assert all(_token_count(chunk.page_content.removeprefix(prefix)) <= 40 for chunk in chunks)
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


def test_chunks_get_story_ids_from_section_headers() -> None:
    document = Document(
        page_content=(
            "# Job\n\n---\n\n## Project A\n\n### Technical Decisions\n\nPicked Postgres.\n\n"
            "## Project B\n\n### Technical Decisions\n\nPicked Redis.\n\n### 2. Rollout Plan\n\nCanary."
        ),
        metadata={"source_file": "nested/job.md"},
    )

    chunks = chunk_experience_documents([document])

    assert [chunk.metadata["story_id"] for chunk in chunks] == [
        "nested/job/project-a/technical-decisions",
        "nested/job/project-b/technical-decisions",
        "nested/job/project-b/rollout-plan",
    ]
    assert chunks[2].metadata["header_3"] == "2. Rollout Plan"


def test_chunks_without_text_are_dropped() -> None:
    document = Document(
        page_content="# Job\n\n---\n\n## Role\n\nOwned it.", metadata={"source_file": "a.md"}
    )

    chunks = chunk_experience_documents([document])

    assert [chunk.page_content for chunk in chunks] == ["Job > Role\n\nOwned it."]
