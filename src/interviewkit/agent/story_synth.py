"""LangGraph workflow for grounded STAR answer synthesis.

parse_input -> select_stories -> synthesize_answer -> check_grounding
  -> accept_answer | flag_answer | refuse_answer
"""

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, cast

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from interviewkit.agent.grounding import check_grounding, describe_issues
from interviewkit.agent.story_synth_context import (
    build_user_message,
    chunk_id,
    parse_draft,
    with_chunk_ids,
)
from interviewkit.agent.story_synth_models import (
    DraftAnswer,
    ExperienceSearch,
    GroundingStatus,
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
    "StorySelector",
    "StorySynthesisError",
    "build_story_graph",
    "synthesize_star_answer",
]

StorySelector = Callable[[StorySynthesisState], Awaitable[StorySynthesisState]]
StoryGraph = CompiledStateGraph[StorySynthesisState, Any, StorySynthesisState, StorySynthesisState]


def build_story_graph(
    provider: LLMProvider | None = None,
    retriever: ExperienceSearch | None = None,
    selector: StorySelector | None = None,
) -> StoryGraph:
    """Build the STAR graph; `selector` replaces the fixed retrieve step (e.g. the agent)."""
    resolved_provider = provider or get_provider()
    resolved_retriever = retriever or cast(ExperienceSearch, ExperienceRetriever(k=5))
    graph = StateGraph(StorySynthesisState)

    async def select_node(state: StorySynthesisState) -> StorySynthesisState:
        if selector is not None:
            return await selector(state)
        return await _retrieve_experiences(state, resolved_retriever)

    async def synthesize_node(state: StorySynthesisState) -> StorySynthesisState:
        return await _synthesize_answer(state, resolved_provider)

    graph.add_node("parse_input", _parse_input)
    graph.add_node("select_stories", select_node)
    graph.add_node("synthesize_answer", synthesize_node)
    graph.add_node("check_grounding", _check_grounding)
    graph.add_node("accept_answer", _accept_answer)
    graph.add_node("flag_answer", _flag_answer)
    graph.add_node("refuse_answer", _refuse_answer)

    graph.add_edge(START, "parse_input")
    graph.add_edge("parse_input", "select_stories")
    graph.add_edge("select_stories", "synthesize_answer")
    graph.add_edge("synthesize_answer", "check_grounding")
    # The grounding verdict, not the LLM, decides which finishing node runs.
    graph.add_conditional_edges(
        "check_grounding",
        _grounding_route,
        {"pass": "accept_answer", "flag": "flag_answer", "refuse": "refuse_answer"},
    )
    for node in ("accept_answer", "flag_answer", "refuse_answer"):
        graph.add_edge(node, END)
    return cast(StoryGraph, graph.compile())


async def synthesize_star_answer(
    question: str,
    principle: str,
    provider: LLMProvider | None = None,
    retriever: ExperienceSearch | None = None,
    *,
    company: str = "",
    selector: StorySelector | None = None,
) -> StarAnswer:
    """Generate a grounded, cited STAR answer for an interview question."""
    graph = build_story_graph(provider=provider, retriever=retriever, selector=selector)
    state = await graph.ainvoke({"question": question, "principle": principle, "company": company})
    return cast(StarAnswer, state["answer"])


def _parse_input(state: StorySynthesisState) -> StorySynthesisState:
    question = _required_text(state, "question", "Interview question text is required")
    principle = _required_text(state, "principle", "Company principle text is required")
    return {"question": question, "principle": principle}


async def _retrieve_experiences(
    state: StorySynthesisState,
    retriever: ExperienceSearch,
) -> StorySynthesisState:
    query = f"{state['principle']}\n{state['question']}"
    return {"retrieved_chunks": await retriever.ainvoke(query), "tool_calls": 0}


async def _synthesize_answer(
    state: StorySynthesisState,
    provider: LLMProvider,
) -> StorySynthesisState:
    chunks = with_chunk_ids(state.get("retrieved_chunks", []))
    if not chunks:
        return {"retrieved_chunks": chunks, "draft": DraftAnswer(refused=True)}

    response = await provider.complete(
        [
            {"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8")},
            {
                "role": "user",
                "content": build_user_message(
                    question=state["question"], principle=state["principle"], chunks=chunks
                ),
            },
        ]
    )
    if not response.content.strip():
        raise StorySynthesisError("STAR synthesis provider returned empty content")
    return {"retrieved_chunks": chunks, "draft": parse_draft(response.content)}


def _check_grounding(state: StorySynthesisState) -> StorySynthesisState:
    return {"grounding": check_grounding(state["draft"], state.get("retrieved_chunks", []))}


def _grounding_route(state: StorySynthesisState) -> GroundingStatus:
    return state["grounding"].status


def _accept_answer(state: StorySynthesisState) -> StorySynthesisState:
    return {"answer": _grounded_answer(state, "pass")}


def _flag_answer(state: StorySynthesisState) -> StorySynthesisState:
    return {"answer": _grounded_answer(state, "flag")}


def _refuse_answer(state: StorySynthesisState) -> StorySynthesisState:
    answer = StarAnswer(
        principle=state["principle"], text=NO_MATCH_MESSAGE, sources=(), grounding="refuse"
    )
    return {"answer": answer}


def _grounded_answer(state: StorySynthesisState, status: GroundingStatus) -> StarAnswer:
    report = state["grounding"]
    by_id = {chunk_id(chunk): chunk for chunk in state.get("retrieved_chunks", [])}
    cited = [by_id[citation] for citation in report.valid_citations]
    issues = describe_issues(report) if status == "flag" else ""
    return StarAnswer(
        principle=state["principle"],
        text=render_answer(state["principle"], state["draft"], cited, issues),
        sources=tuple(dict.fromkeys(str(c.metadata.get("source_file", "")) for c in cited)),
        citations=report.valid_citations,
        grounding=status,
    )


def render_answer(
    principle: str, draft: DraftAnswer, cited: Sequence[Document], issues: str = ""
) -> str:
    """Render a draft as STAR text with a Sources list of cited files and stories."""
    lines = [
        f"Principle: {principle}",
        f"Situation: {draft.situation.strip()}",
        f"Task: {draft.task.strip()}",
        f"Action: {draft.action.strip()}",
        f"Result: {draft.result.strip()}",
    ]
    if issues:
        lines.append(f"Grounding check: flagged ({issues})")
    lines.append("Sources:")
    for source in dict.fromkeys(str(chunk.metadata.get("source_file", "")) for chunk in cited):
        stories = dict.fromkeys(
            str(c.metadata["story_id"])
            for c in cited
            if c.metadata.get("source_file") == source and c.metadata.get("story_id")
        )
        suffix = f" stories: {', '.join(stories)}" if stories else ""
        lines.append(f"- (source: {source}){suffix}")
    return "\n".join(lines)


def _required_text(
    state: StorySynthesisState,
    key: str,
    error_message: str,
) -> str:
    value = cast(str, state.get(key, "")).strip()
    if not value:
        raise StorySynthesisError(error_message)
    return value
