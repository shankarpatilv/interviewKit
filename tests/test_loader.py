import logging

from interviewkit.ingest.loader import load_experience_documents


def test_load_experience_documents_returns_non_template_markdown_files(tmp_path) -> None:
    (tmp_path / "amazon.md").write_text("# Amazon\n\nBuilt a moderation pipeline.", encoding="utf-8")
    (tmp_path / "_template.md").write_text("# Template\n\nIgnore me.", encoding="utf-8")

    documents = load_experience_documents(tmp_path)
    
    assert len(documents) == 1
    assert documents[0].page_content == "# Amazon\n\nBuilt a moderation pipeline."
    assert documents[0].metadata["source_file"] == "amazon.md"


def test_load_experience_documents_reads_nested_files_in_stable_order(tmp_path) -> None:
    nested_dir = tmp_path / "work"
    nested_dir.mkdir()
    (nested_dir / "collabera.md").write_text("# Collabera\n\nBuilt APIs.", encoding="utf-8")
    (tmp_path / "amazon.md").write_text("# Amazon\n\nBuilt LLM systems.", encoding="utf-8")

    documents = load_experience_documents(tmp_path)

    assert [document.metadata["source_file"] for document in documents] == [
        "amazon.md",
        "work/collabera.md",
    ]


def test_load_experience_documents_skips_empty_files_and_warns(
    tmp_path,
    caplog,
) -> None:
    (tmp_path / "empty.md").write_text("  \n", encoding="utf-8")
    (tmp_path / "story.md").write_text("# Story\n\nUseful content.", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        documents = load_experience_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata["source_file"] == "story.md"
    assert "Skipping empty experience file" in caplog.text


def test_load_experience_documents_missing_directory_warns(tmp_path, caplog) -> None:
    missing_dir = tmp_path / "missing"

    with caplog.at_level(logging.WARNING):
        documents = load_experience_documents(missing_dir)

    assert documents == []
    assert "Experience directory does not exist" in caplog.text
