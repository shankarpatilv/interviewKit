"""LangGraph workflow for STAR answer synthesis."""

from pathlib import Path
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from interviewkit.agent.story_synth_context import build_user_message, source_files
from interviewkit.agent.story_synth_models import (
    ExperienceSearch,
    StarAnswer,
    StorySynthesisError,
    StorySynthesisState,
)
from interviewkit.providers import LLMProvider, get_provider
from interviewkit.retrieval.retriever import ExperienceRetriever

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "star_synth.txt"
NO_MATCH_MESSAGE = "I do not have enough relevant experience context to draft this answer."
__all__ = [
    "NO_MATCH_MESSAGE",
    "StarAnswer",
    "StorySynthesisError",
    "build_story_graph",
    "synthesize_star_answer",
]


def build_story_graph(
    provider: LLMProvider | None = None,
    retriever: ExperienceSearch | None = None,
) -> CompiledStateGraph[StorySynthesisState, Any, StorySynthesisState, StorySynthesisState]:
    """Build and compile the STAR synthesis LangGraph workflow."""
    resolved_provider = provider or get_provider()
    resolved_retriever = retriever or cast(ExperienceSearch, ExperienceRetriever(k=5))
    graph = StateGraph(StorySynthesisState)

    async def retrieve_node(state: StorySynthesisState) -> StorySynthesisState:
        return await _retrieve_experiences(state, resolved_retriever)

    async def synthesize_node(state: StorySynthesisState) -> StorySynthesisState:
        return await _synthesize_answer(state, resolved_provider)

    graph.add_node("parse_input", _parse_input)
    graph.add_node("retrieve_experiences", retrieve_node)
    graph.add_node("synthesize_answer", synthesize_node)
    graph.add_node("format_output", _format_output)

    graph.add_edge(START, "parse_input")
    graph.add_edge("parse_input", "retrieve_experiences")
    graph.add_edge("retrieve_experiences", "synthesize_answer")
    graph.add_edge("synthesize_answer", "format_output")
    graph.add_edge("format_output", END)
    return cast(
        CompiledStateGraph[
            StorySynthesisState,
            Any,
            StorySynthesisState,
            StorySynthesisState,
        ],
        graph.compile(),
    )


async def synthesize_star_answer(
    question: str,
    principle: str,
    provider: LLMProvider | None = None,
    retriever: ExperienceSearch | None = None,
) -> StarAnswer:
    """Generate a cited STAR answer for an interview question."""
    graph = build_story_graph(provider=provider, retriever=retriever)
    state = await graph.ainvoke({"question": question, "principle": principle})
    return cast(StarAnswer, state["answer"])


def _parse_input(state: StorySynthesisState) -> StorySynthesisState:
    question = _required_text(state, "question", "Interview question text is required")
    principle = _required_text(state, "principle", "Company principle text is required")
    return {"question": question, "principle": principle}


async def _retrieve_experiences(
    state: StorySynthesisState,
    retriever: ExperienceSearch,
) -> StorySynthesisState:
    question = _required_text(state, "question", "Interview question text is required")
    principle = _required_text(state, "principle", "Company principle text is required")
    documents = await retriever.ainvoke(f"{principle}\n{question}")
    return {"retrieved_chunks": documents}


async def _synthesize_answer(
    state: StorySynthesisState,
    provider: LLMProvider,
) -> StorySynthesisState:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"raw_response": NO_MATCH_MESSAGE}

    response = await provider.complete(
        [
            {"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8")},
            {
                "role": "user",
                "content": build_user_message(
                    question=_required_text(
                        state, "question", "Interview question text is required"
                    ),
                    principle=_required_text(
                        state, "principle", "Company principle text is required"
                    ),
                    chunks=chunks,
                ),
            },
        ]
    )
    if not response.content.strip():
        raise StorySynthesisError("STAR synthesis provider returned empty content")
    return {"raw_response": response.content}


def _format_output(state: StorySynthesisState) -> StorySynthesisState:
    raw_response = _required_text(state, "raw_response", "STAR synthesis response is missing")
    principle = _required_text(state, "principle", "Company principle text is required")
    available_sources = source_files(state.get("retrieved_chunks", []))
    cited_sources = tuple(source for source in available_sources if source in raw_response)

    if NO_MATCH_MESSAGE in raw_response and not cited_sources:
        return {"answer": StarAnswer(principle=principle, text=raw_response, sources=())}
    if not cited_sources:
        raise StorySynthesisError("STAR synthesis response did not cite a retrieved source file")
    return {"answer": StarAnswer(principle=principle, text=raw_response, sources=cited_sources)}


def _required_text(
    state: StorySynthesisState,
    key: str,
    error_message: str,
) -> str:
    value = cast(str, state.get(key, "")).strip()
    if not value:
        raise StorySynthesisError(error_message)
    return value
