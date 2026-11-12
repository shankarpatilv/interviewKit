import asyncio
import json
from dataclasses import dataclass

import pytest
from langchain_core.documents import Document

from interviewkit.agent.grounding import check_grounding
from interviewkit.agent.story_synth import (
    NO_MATCH_MESSAGE,
    StarAnswer,
    StorySynthesisError,
    build_story_graph,
    synthesize_star_answer,
)
from interviewkit.agent.story_synth_context import parse_draft, with_chunk_ids
from interviewkit.agent.story_synth_models import DraftAnswer
from interviewkit.providers import Message, ProviderResponse, Tool


@dataclass
class FakeProvider:
    content: str

    def __post_init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> ProviderResponse:
        self.calls.append(messages)
        return ProviderResponse(content=self.content, provider="fake", model="fake-model")


@dataclass
class FakeRetriever:
    documents: list[Document]

    def __post_init__(self) -> None:
        self.queries: list[str] = []

    async def ainvoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return self.documents


def _document(content: str, source_file: str, chunk_id: str | None = None) -> Document:
    metadata: dict[str, object] = {"source_file": source_file}
    if chunk_id:
        metadata |= {"chunk_id": chunk_id, "story_id": f"{source_file[:-3]}/story"}
    return Document(page_content=content, metadata=metadata)


def _reply(**fields: object) -> str:
    base: dict[str, object] = {
        "situation": "I joined a moderation launch.",
        "task": "I had to make the pipeline reliable.",
        "action": "I added retries and observability.",
        "result": "The service handled production traffic.",
        "citations": ["amazon.md:0:abc"],
    }
    return json.dumps(base | fields)


def _synth(reply: str, docs: list[Document]) -> StarAnswer:
    return asyncio.run(
        synthesize_star_answer(
            question="Tell me about a time you showed ownership.",
            principle="Ownership",
            provider=FakeProvider(reply),
            retriever=FakeRetriever(docs),
        )
    )


OWNED = _document("Owned a Bedrock moderation pipeline.", "amazon.md", "amazon.md:0:abc")


def test_story_graph_compiles() -> None:
    graph = build_story_graph(provider=FakeProvider(_reply()), retriever=FakeRetriever([OWNED]))

    assert graph is not None


def test_grounded_answer_passes_and_renders_star_with_story_ids() -> None:
    provider = FakeProvider(_reply())
    retriever = FakeRetriever([OWNED])

    answer = asyncio.run(
        synthesize_star_answer(
            question="Tell me about a time you showed ownership.",
            principle="Ownership",
            provider=provider,
            retriever=retriever,
        )
    )

    assert answer.grounding == "pass"
    assert answer.sources == ("amazon.md",)
    assert answer.citations == ("amazon.md:0:abc",)
    assert answer.text.startswith("Principle: Ownership\nSituation: I joined")
    assert answer.text.endswith("Sources:\n- (source: amazon.md) stories: amazon/story")
    assert retriever.queries == ["Ownership\nTell me about a time you showed ownership."]
    assert "chunk_id: amazon.md:0:abc" in provider.calls[0][1]["content"]


def test_no_retrieved_chunks_refuses_without_calling_llm() -> None:
    provider = FakeProvider("Should not be called")

    answer = asyncio.run(
        synthesize_star_answer(
            question="Tell me about conflict.",
            principle="Earn Trust",
            provider=provider,
            retriever=FakeRetriever([]),
        )
    )

    assert answer == StarAnswer("Earn Trust", NO_MATCH_MESSAGE, (), grounding="refuse")
    assert provider.calls == []


def test_synthesize_star_answer_rejects_empty_input() -> None:
    with pytest.raises(StorySynthesisError, match="Interview question text is required"):
        asyncio.run(
            synthesize_star_answer(
                question=" ",
                principle="Ownership",
                provider=FakeProvider("unused"),
                retriever=FakeRetriever([]),
            )
        )


def test_model_refusal_routes_to_refuse() -> None:
    answer = _synth(json.dumps({"refused": True}), [OWNED])

    assert answer.grounding == "refuse" and answer.text == NO_MATCH_MESSAGE


def test_citations_to_unretrieved_chunks_refuse() -> None:
    answer = _synth(_reply(citations=["other.md:9:zzz"]), [OWNED])

    assert answer.grounding == "refuse" and answer.sources == ()


def test_unsupported_number_is_flagged_not_dropped() -> None:
    answer = _synth(_reply(result="Latency fell 40%."), [OWNED])

    assert answer.grounding == "flag"
    assert "Grounding check: flagged (not in cited notes: 40%)" in answer.text


def test_number_in_cited_chunk_passes() -> None:
    doc = _document("Cut latency 40%.", "amazon.md", "amazon.md:0:abc")

    assert _synth(_reply(result="Latency fell 40%."), [doc]).grounding == "pass"


def test_chunks_without_db_ids_get_rank_ids() -> None:
    labeled = with_chunk_ids([_document("x", "a.md"), _document("y", "a.md")])

    assert [d.metadata["chunk_id"] for d in labeled] == ["a.md#1", "a.md#2"]


def test_parse_draft_tolerates_fences_and_rejects_garbage() -> None:
    assert parse_draft('```json\n{"refused": true}\n```').refused
    with pytest.raises(StorySynthesisError, match="not JSON"):
        parse_draft("Situation: prose only")
    with pytest.raises(StorySynthesisError, match="invalid"):
        parse_draft('{"citations": "not-a-list"}')


def test_check_grounding_mixed_citations_flag() -> None:
    draft = DraftAnswer(situation="s", citations=["amazon.md:0:abc", "ghost"])

    report = check_grounding(draft, [OWNED])

    assert report.status == "flag"
    assert report.valid_citations == ("amazon.md:0:abc",)
    assert report.invalid_citations == ("ghost",)


def test_check_grounding_refuses_empty_body() -> None:
    assert check_grounding(DraftAnswer(citations=["amazon.md:0:abc"]), [OWNED]).status == "refuse"
