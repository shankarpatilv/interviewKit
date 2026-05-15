import asyncio
from dataclasses import dataclass

import pytest

from interviewkit.agent.question_gen import (
    GeneratedQuestion,
    QuestionGenerationError,
    build_question_graph,
    generate_interview_questions,
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


def test_question_graph_compiles() -> None:
    graph = build_question_graph(provider=FakeProvider("[Ownership] Tell me about a time."))

    assert graph is not None


def test_generate_interview_questions_returns_structured_questions() -> None:
    async def run_test() -> None:
        provider = FakeProvider(
            "\n".join(
                [
                    "[Customer Obsession] Tell me about a time you worked backward.",
                    "[Ownership] Give me an example of when you owned a production issue.",
                ]
            )
        )

        questions = await generate_interview_questions(
            job_description="Backend engineer building reliable APIs.",
            company_principles="Customer Obsession\nOwnership",
            provider=provider,
        )

        assert questions == [
            GeneratedQuestion(
                principle="Customer Obsession",
                text="Tell me about a time you worked backward.",
            ),
            GeneratedQuestion(
                principle="Ownership",
                text="Give me an example of when you owned a production issue.",
            ),
        ]
        assert "question_gen.txt" not in provider.calls[0][0]["content"]
        assert "Backend engineer" in provider.calls[0][1]["content"]
        assert "Customer Obsession" in provider.calls[0][1]["content"]

    asyncio.run(run_test())


def test_generate_interview_questions_rejects_empty_input() -> None:
    async def run_test() -> None:
        with pytest.raises(QuestionGenerationError, match="Job description text is required"):
            await generate_interview_questions(
                job_description=" ",
                company_principles="Ownership",
                provider=FakeProvider("[Ownership] Tell me about a time."),
            )

    asyncio.run(run_test())


def test_generate_interview_questions_rejects_untagged_response() -> None:
    async def run_test() -> None:
        with pytest.raises(QuestionGenerationError, match="no tagged questions"):
            await generate_interview_questions(
                job_description="Backend engineer",
                company_principles="Ownership",
                provider=FakeProvider("Tell me about a time."),
            )

    asyncio.run(run_test())
