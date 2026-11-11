import asyncio
from dataclasses import dataclass

import pytest
from langchain_core.documents import Document

from interviewkit.agent.story_synth import (
    NO_MATCH_MESSAGE,
    StarAnswer,
    StorySynthesisError,
    build_story_graph,
    synthesize_star_answer,
)
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


def test_story_graph_compiles() -> None:
    graph = build_story_graph(
        provider=FakeProvider("Principle: Ownership\nSources:\n- (source: amazon.md)"),
        retriever=FakeRetriever([_document("Built a pipeline.", "amazon.md")]),
    )

    assert graph is not None


def test_synthesize_star_answer_returns_cited_answer() -> None:
    async def run_test() -> None:
        provider = FakeProvider(
            "\n".join(
                [
                    "Principle: Ownership",
                    "Situation: I joined a moderation launch.",
                    "Task: I had to make the pipeline reliable.",
                    "Action: I added retries and observability.",
                    "Result: The service handled production traffic.",
                    "Sources:",
                    "- (source: amazon.md)",
                ]
            )
        )
        retriever = FakeRetriever([_document("Owned a Bedrock moderation pipeline.", "amazon.md")])

        answer = await synthesize_star_answer(
            question="Tell me about a time you showed ownership.",
            principle="Ownership",
            provider=provider,
            retriever=retriever,
        )

        assert answer == StarAnswer(
            principle="Ownership",
            text=provider.content,
            sources=("amazon.md",),
        )
        assert retriever.queries == ["Ownership\nTell me about a time you showed ownership."]
        assert "star_synth.txt" not in provider.calls[0][0]["content"]
        assert "source_file: amazon.md" in provider.calls[0][1]["content"]
        assert "Owned a Bedrock moderation pipeline." in provider.calls[0][1]["content"]

    asyncio.run(run_test())


def test_synthesize_star_answer_returns_no_match_without_retrieved_chunks() -> None:
    async def run_test() -> None:
        provider = FakeProvider("Should not be called")
        answer = await synthesize_star_answer(
            question="Tell me about conflict.",
            principle="Earn Trust",
            provider=provider,
            retriever=FakeRetriever([]),
        )

        assert answer == StarAnswer(principle="Earn Trust", text=NO_MATCH_MESSAGE, sources=())
        assert provider.calls == []

    asyncio.run(run_test())


def test_synthesize_star_answer_rejects_empty_input() -> None:
    async def run_test() -> None:
        with pytest.raises(StorySynthesisError, match="Interview question text is required"):
            await synthesize_star_answer(
                question=" ",
                principle="Ownership",
                provider=FakeProvider("unused"),
                retriever=FakeRetriever([]),
            )

    asyncio.run(run_test())


def test_synthesize_star_answer_requires_retrieved_source_citation() -> None:
    async def run_test() -> None:
        with pytest.raises(StorySynthesisError, match="did not cite"):
            await synthesize_star_answer(
                question="Tell me about impact.",
                principle="Deliver Results",
                provider=FakeProvider("Situation: I helped.\nSources:\n- (source: other.md)"),
                retriever=FakeRetriever([_document("Impact story.", "impact.md")]),
            )

    asyncio.run(run_test())


def _document(content: str, source_file: str) -> Document:
    return Document(page_content=content, metadata={"source_file": source_file})


def test_synthesize_star_answer_accepts_templated_refusal() -> None:
    async def run_test() -> None:
        refusal = f"Principle: Ownership\n{NO_MATCH_MESSAGE}\nSources:\n- none"
        answer = await synthesize_star_answer(
            question="Tell me about a time you led a hardware launch.",
            principle="Ownership",
            provider=FakeProvider(refusal),
            retriever=FakeRetriever([_document("Unrelated story.", "other.md")]),
        )

        assert answer.sources == ()
        assert NO_MATCH_MESSAGE in answer.text

    asyncio.run(run_test())
