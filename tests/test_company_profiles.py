from pathlib import Path


def test_amazon_profile_contains_16_principles() -> None:
    content = Path("companies/amazon.md").read_text(encoding="utf-8")

    assert content.count("### ") == 16
    assert "### Customer Obsession" in content
    assert "### Success And Scale Bring Broad Responsibility" in content


def test_microsoft_profile_contains_six_core_competencies() -> None:
    content = Path("companies/microsoft.md").read_text(encoding="utf-8")

    assert content.count("### ") == 6
    assert "### Collaboration" in content
    assert "### Drive For Results" in content
    assert "### Customer Focus" in content
    assert "### Influencing For Impact" in content
    assert "### Judgment" in content
    assert "### Adaptability" in content


def test_company_readme_links_public_sources() -> None:
    content = Path("companies/README.md").read_text(encoding="utf-8")

    assert "aboutamazon.com" in content
    assert "careers.microsoft.com" in content
    assert "companies/<company>.md" in content
