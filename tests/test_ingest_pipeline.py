import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from interviewkit.config import Settings
from interviewkit.ingest import pipeline
from interviewkit.ingest.embedder import EmbeddedDocument


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, embed_count: int) -> list[bool]:
    (tmp_path / "a.md").write_text("# A\n\n## One\n\nFirst.\n\n## Two\n\nSecond.", encoding="utf-8")
    prune_flags: list[bool] = []

    async def fake_embed(chunks: list[Document], app_settings: Any) -> list[EmbeddedDocument]:
        return [EmbeddedDocument(document=c, embedding=[0.0]) for c in chunks[:embed_count]]

    def fake_upsert(chunks: list[EmbeddedDocument], app_settings: Any, prune: bool) -> None:
        prune_flags.append(prune)

    monkeypatch.setattr(pipeline, "embed_experience_chunks", fake_embed)
    monkeypatch.setattr(pipeline, "upsert_chunks", fake_upsert)
    settings = Settings(_env_file=None, experiences_dir=tmp_path)  # type: ignore[call-arg]
    asyncio.run(pipeline.ingest_experiences(app_settings=settings))
    return prune_flags


def test_ingest_prunes_stale_chunks_after_complete_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _run(monkeypatch, tmp_path, embed_count=2) == [True]


def test_ingest_skips_pruning_when_embedding_was_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _run(monkeypatch, tmp_path, embed_count=1) == [False]
