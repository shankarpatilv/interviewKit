from pathlib import Path

from interviewkit.ingest.frontmatter import split_frontmatter
from interviewkit.ingest.loader import load_experience_documents


def test_split_frontmatter_parses_strings_and_lists() -> None:
    text = (
        "---\nrole: SDE Intern\ncompetencies: [Ownership, 'Dive Deep']\n---\n\n# Story\n\nDid it."
    )

    metadata, body = split_frontmatter(text)

    assert metadata == {"role": "SDE Intern", "competencies": ["Ownership", "Dive Deep"]}
    assert body == "# Story\n\nDid it."


def test_split_frontmatter_ignores_text_without_fence_or_close() -> None:
    assert split_frontmatter("# Story\n\n---\n\nmore") == ({}, "# Story\n\n---\n\nmore")
    assert split_frontmatter("---\nrole: x\nno closing fence") == (
        {},
        "---\nrole: x\nno closing fence",
    )


def test_split_frontmatter_skips_reserved_keys_and_comments() -> None:
    metadata, _ = split_frontmatter("---\n# note: hi\nsource_file: evil.md\nteam: Ads\n---\nBody")

    assert metadata == {"team": "Ads"}


def test_loader_copies_frontmatter_onto_document_metadata(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("---\ntimeframe: 2024\n---\n# A\n\nBody.", encoding="utf-8")

    [document] = load_experience_documents(tmp_path)

    assert document.metadata == {"timeframe": "2024", "source_file": "a.md"}
    assert document.page_content == "# A\n\nBody."
