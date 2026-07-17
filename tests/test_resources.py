from pathlib import Path


def test_phase1_resource_documents_checkpoint_flow() -> None:
    content = Path("resources/phase1.md").read_text(encoding="utf-8")

    assert "docker compose up -d postgres" in content
    assert "python -m interviewkit init-db" in content
    assert "python -m interviewkit ingest" in content
    assert "python -m interviewkit prep" in content
    assert "sessions/<timestamp>-amazon.md" in content
    assert "generated 8 Amazon-tagged behavioral questions" in content
