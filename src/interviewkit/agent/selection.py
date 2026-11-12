"""Bounded tool-calling subgraph that picks stories for one question.

start -> agent <-> run_tools -> finish. The agent may call search, story,
usage, and rubric tools, then finishes with `select_chunks`. A hard tool-call
budget and turn limit keep it from looping; when the budget runs out it is
offered only `select_chunks`, and `finish` falls back to the best chunks it saw.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, TypedDict, cast

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from interviewkit.agent.story_synth import StorySelector
from interviewkit.agent.story_synth_models import StorySynthesisState
from interviewkit.agent.tools import StoryTools
from interviewkit.providers import LLMProvider, Message, Tool, ToolCall

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "story_selector.txt"
MAX_TOOL_CALLS = 6
MAX_AGENT_TURNS = 5
RUBRIC_CHAR_LIMIT = 4000
SELECT_TOOL = "select_chunks"


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> Tool:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_COMPANY = {"company": {"type": "string", "description": "Company key, e.g. amazon."}}
SELECT_SCHEMA = _function(
    SELECT_TOOL,
    "Finish: the chunk_ids the answer may use, best first. Empty if nothing fits.",
    {"chunk_ids": {"type": "array", "items": {"type": "string"}}},
    ["chunk_ids"],
)
TOOL_SCHEMAS: list[Tool] = [
    _function(
        "search_experiences",
        "Hybrid search over experience note chunks.",
        {
            "query": {"type": "string"},
            "k": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        ["query"],
    ),
    _function(
        "get_story",
        "All chunks of one story.",
        {"story_id": {"type": "string"}},
        ["story_id"],
    ),
    _function("get_story_usage", "Stories already told to a company.", _COMPANY, ["company"]),
    _function("get_company_rubric", "The company's interview principles.", _COMPANY, ["company"]),
    SELECT_SCHEMA,
]


class SelectionState(TypedDict, total=False):
    """Private state of the selection subgraph."""

    question: str
    principle: str
    company: str
    messages: list[Message]
    pending: list[ToolCall]
    seen: dict[str, Document]  # chunk_id -> chunk, in the order tools returned them
    selected: list[str] | None  # None until select_chunks is called
    tool_calls: int
    turns: int
    retrieved_chunks: list[Document]


SelectionGraph = CompiledStateGraph[SelectionState, Any, SelectionState, SelectionState]


def build_selection_graph(provider: LLMProvider, tools: StoryTools, k: int = 5) -> SelectionGraph:
    """Compile the bounded story-selection subgraph."""
    graph = StateGraph(SelectionState)

    async def agent_node(state: SelectionState) -> SelectionState:
        return await _agent_turn(state, provider)

    async def tools_node(state: SelectionState) -> SelectionState:
        return await _run_tools(state, tools)

    async def finish_node(state: SelectionState) -> SelectionState:
        return await _finish(state, tools, k)

    graph.add_node("start", lambda state: _start(state, k))
    graph.add_node("agent", agent_node)
    graph.add_node("run_tools", tools_node)
    graph.add_node("finish", finish_node)
    graph.add_edge(START, "start")
    graph.add_edge("start", "agent")
    graph.add_conditional_edges("agent", _route, {"run_tools": "run_tools", "finish": "finish"})
    graph.add_edge("run_tools", "agent")
    graph.add_edge("finish", END)
    return cast(SelectionGraph, graph.compile())


def agent_selector(provider: LLMProvider, tools: StoryTools, k: int = 5) -> StorySelector:
    """Wrap the subgraph as the story graph's `select_stories` step."""
    subgraph = build_selection_graph(provider, tools, k)

    async def select(state: StorySynthesisState) -> StorySynthesisState:
        result = await subgraph.ainvoke(
            {
                "question": state["question"],
                "principle": state["principle"],
                "company": state.get("company", ""),
            }
        )
        return {"retrieved_chunks": result["retrieved_chunks"], "tool_calls": result["tool_calls"]}

    return select


def _start(state: SelectionState, k: int) -> SelectionState:
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(k=k, max_tool_calls=MAX_TOOL_CALLS)
    user = (
        f"Company: {state.get('company') or 'unknown'}\n"
        f"Principle: {state['principle']}\n"
        f"Question: {state['question']}"
    )
    return {
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user}],
        "pending": [],
        "seen": {},
        "selected": None,
        "tool_calls": 0,
        "turns": 0,
    }


async def _agent_turn(state: SelectionState, provider: LLMProvider) -> SelectionState:
    turns = state["turns"] + 1
    can_explore = state["tool_calls"] < MAX_TOOL_CALLS and turns < MAX_AGENT_TURNS
    response = await provider.complete(
        state["messages"], TOOL_SCHEMAS if can_explore else [SELECT_SCHEMA]
    )
    messages = [*state["messages"], _assistant_message(response.content, response.tool_calls)]
    for call in response.tool_calls:
        if call.name == SELECT_TOOL:
            ids = call.arguments.get("chunk_ids", [])
            selected = [str(i) for i in ids] if isinstance(ids, list) else []
            return {"messages": messages, "pending": [], "selected": selected, "turns": turns}
    pending = list(response.tool_calls) if can_explore else []
    return {"messages": messages, "pending": pending, "turns": turns}


def _route(state: SelectionState) -> str:
    if state.get("selected") is None and state.get("pending"):
        return "run_tools"
    return "finish"


async def _run_tools(state: SelectionState, tools: StoryTools) -> SelectionState:
    messages = list(state["messages"])
    seen = dict(state["seen"])
    used = state["tool_calls"]
    for call in state["pending"]:
        if used >= MAX_TOOL_CALLS:
            result: object = {"error": "tool budget exhausted; call select_chunks"}
        else:
            used += 1
            result = await _call_tool(call, state, tools, seen)
        messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)}
        )
    return {"messages": messages, "seen": seen, "tool_calls": used, "pending": []}


async def _call_tool(
    call: ToolCall, state: SelectionState, tools: StoryTools, seen: dict[str, Document]
) -> object:
    args = call.arguments
    company = str(args.get("company") or state.get("company") or "")
    try:
        if call.name == "search_experiences":
            k = int(args.get("k", 5))
            docs = await tools.search(str(args.get("query", "")), k=max(1, min(k, 10)))
            return _remember(docs, seen)
        if call.name == "get_story":
            docs = await asyncio.to_thread(tools.story, str(args.get("story_id", "")))
            return _remember(docs, seen)
        if call.name == "get_story_usage":
            uses = await asyncio.to_thread(tools.usage, company)
            return [
                {"story_id": u.story_id, "question": u.question, "used_at": u.used_at} for u in uses
            ]
        if call.name == "get_company_rubric":
            rubric = await asyncio.to_thread(tools.rubric, company)
            return rubric[:RUBRIC_CHAR_LIMIT]
    except Exception as exc:  # tool errors go back to the model, not up the stack
        return {"error": str(exc)}
    return {"error": f"unknown tool {call.name}"}


def _remember(docs: list[Document], seen: dict[str, Document]) -> list[dict[str, object]]:
    hits = []
    for doc in docs:
        chunk_id = str(doc.metadata.get("chunk_id", ""))
        if not chunk_id:
            continue
        seen.setdefault(chunk_id, doc)
        hits.append(
            {
                "chunk_id": chunk_id,
                "story_id": doc.metadata.get("story_id"),
                "source_file": doc.metadata.get("source_file"),
                "text": doc.page_content,
            }
        )
    return hits


async def _finish(state: SelectionState, tools: StoryTools, k: int) -> SelectionState:
    seen = state.get("seen", {})
    selected = state.get("selected")
    used = state.get("tool_calls", 0)
    if selected is not None:
        chunks = [seen[i] for i in dict.fromkeys(selected) if i in seen][:k]
        if chunks or not selected:
            return {"retrieved_chunks": chunks, "tool_calls": used}
    if seen:  # no usable selection: fall back to the best chunks the tools returned
        return {"retrieved_chunks": list(seen.values())[:k], "tool_calls": used}
    query = f"{state['principle']}\n{state['question']}"
    return {"retrieved_chunks": await tools.search(query, k=k), "tool_calls": used + 1}


def _assistant_message(content: str, calls: tuple[ToolCall, ...]) -> Message:
    message: Message = {"role": "assistant", "content": content or None}
    if calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in calls
        ]
    return message
