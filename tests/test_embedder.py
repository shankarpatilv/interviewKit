from dataclasses import dataclass
import asyncio

from langchain_core.documents import Document
import pytest

from interviewkit.config import Settings
from interviewkit.ingest.embedder import EMBEDDING_DIMENSIONS, embed_experience_chunks
from interviewkit.ingest.embedder import openai_client


@dataclass
class _Embedding:
    embedding: list[float]


@dataclass
class _Response:
    data: list[_Embedding]


class _TemporaryRateLimitError(Exception):
    pass


class _FakeEmbeddingsClient:
    def __init__(self, *, fail_calls: set[int] | None = None, rate_limit_failures: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.models: list[str] = []
        self.fail_calls = fail_calls or set()
        self.rate_limit_failures = rate_limit_failures

    async def create(self, *, model: str, input: list[str]) -> _Response:
        self.calls.append(input)
        self.models.append(model)
        call_number = len(self.calls)
        if self.rate_limit_failures:
            self.rate_limit_failures -= 1
            raise _TemporaryRateLimitError("slow down")
        if call_number in self.fail_calls:
            raise RuntimeError("batch failed")
        return _Response(
            data=[_Embedding([float(call_number)] * EMBEDDING_DIMENSIONS) for _ in input]
        )


class _FakeClient:
    def __init__(self, embeddings: _FakeEmbeddingsClient) -> None:
        self.embeddings = embeddings


async def _record_sleep(seconds: float) -> None:
    _SLEEP_CALLS.append(seconds)


_SLEEP_CALLS: list[float] = []


def test_embed_experience_chunks_returns_documents_with_embeddings() -> None:
    async def run_test() -> None:
        embeddings = _FakeEmbeddingsClient()
        client = _FakeClient(embeddings)
        settings = Settings(openai_api_key="sk-test", _env_file=None)
        chunks = [
            Document(page_content="Built a moderation pipeline.", metadata={"source_file": "a.md"})
        ]

        embedded = await embed_experience_chunks(chunks, app_settings=settings, client=client)

        assert embedded[0].document is chunks[0]
        assert len(embedded[0].embedding) == EMBEDDING_DIMENSIONS
        assert embeddings.calls == [["Built a moderation pipeline."]]
        assert embeddings.models == ["text-embedding-3-small"]

    asyncio.run(run_test())


def test_embed_experience_chunks_respects_max_batch_size() -> None:
    async def run_test() -> None:
        embeddings = _FakeEmbeddingsClient()
        client = _FakeClient(embeddings)
        settings = Settings(openai_api_key="sk-test", _env_file=None)
        chunks = [Document(page_content=f"chunk {index}") for index in range(205)]

        embedded = await embed_experience_chunks(chunks, app_settings=settings, client=client)

        assert len(embedded) == 205
        assert [len(call) for call in embeddings.calls] == [100, 100, 5]

    asyncio.run(run_test())


def test_embed_experience_chunks_retries_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        monkeypatch.setattr(openai_client, "RateLimitError", _TemporaryRateLimitError)
        _SLEEP_CALLS.clear()
        embeddings = _FakeEmbeddingsClient(rate_limit_failures=2)
        client = _FakeClient(embeddings)
        settings = Settings(openai_api_key="sk-test", _env_file=None)
        chunks = [Document(page_content="retry me")]

        embedded = await embed_experience_chunks(
            chunks,
            app_settings=settings,
            client=client,
            sleep=_record_sleep,
        )

        assert len(embedded) == 1
        assert _SLEEP_CALLS == [1.0, 2.0]

    asyncio.run(run_test())


def test_embed_experience_chunks_skips_failed_batches() -> None:
    async def run_test() -> None:
        embeddings = _FakeEmbeddingsClient(fail_calls={2})
        client = _FakeClient(embeddings)
        settings = Settings(openai_api_key="sk-test", _env_file=None)
        chunks = [Document(page_content="first"), Document(page_content="second")]

        embedded = await embed_experience_chunks(
            chunks,
            app_settings=settings,
            client=client,
            batch_size=1,
        )

        assert [item.document.page_content for item in embedded] == ["first"]

    asyncio.run(run_test())
