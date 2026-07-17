from pathlib import Path


def test_experience_template_contains_required_sections() -> None:
    content = Path("experiences/_template.md").read_text(encoding="utf-8")

    assert content.count("\n## ") == 7
    assert "## Context" in content
    assert "## Role" in content
    assert "## Technical Decisions" in content
    assert "## Metrics And Impact" in content
    assert "## Conflicts And Tradeoffs" in content
    assert "## What I Learned" in content
    assert "## Stories This Maps To" in content


def test_experience_readme_explains_private_files() -> None:
    content = Path("experiences/README.md").read_text(encoding="utf-8")

    assert "Filled-in experience files are intentionally ignored by Git" in content
    assert "_template.md" in content
    assert "Richer notes produce stronger interview answers" in content


def test_gitignore_allows_only_public_experience_files() -> None:
    content = Path(".gitignore").read_text(encoding="utf-8")

    assert "/experiences/*" in content
    assert "!/experiences/_template.md" in content
    assert "!/experiences/README.md" in content
