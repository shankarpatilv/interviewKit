import asyncio
from collections.abc import Iterator
from datetime import datetime

from langchain_core.documents import Document

from interviewkit.agent import selection
from interviewkit.agent.tools import StoryTools
from interviewkit.providers import Message, ProviderResponse, Tool, ToolCall
from interviewkit.stories.usage import StoryUse

A = Document(
    "Led the migration.", metadata={"chunk_id": "a:0", "story_id": "a/x", "source_file": "a.md"}
)
B = Document(
    "Fixed on-call.", metadata={"chunk_id": "b:0", "story_id": "b/y", "source_file": "b.md"}
)


class Scripted:
    def __init__(self, *turns: tuple[ToolCall, ...]) -> None:
        self.turns: Iterator[tuple[ToolCall, ...]] = iter(turns)
        self.offered: list[list[str]] = []
        self.messages: list[list[Message]] = []

    async def complete(
        self, messages: list[Message], tools: list[Tool] | None = None
    ) -> ProviderResponse:
        self.offered.append([t["function"]["name"] for t in tools or []])  # type: ignore[index]
        self.messages.append(list(messages))
        calls = next(self.turns, ())
        return ProviderResponse("", "openai", "gpt-4o", tool_calls=calls)


def _tools(log: list[str]) -> StoryTools:
    async def search(query: str, k: int = 5) -> list[Document]:
        log.append(f"search:{query}:{k}")
        return [A, B][:k]

    def usage(company: str) -> list[StoryUse]:
        log.append(f"usage:{company}")
        return [StoryUse(company, "a/x", None, datetime(2026, 1, 1))]

    def story(story_id: str) -> list[Document]:
        log.append(f"story:{story_id}")
        return [B]

    return StoryTools(search=search, story=story, usage=usage, rubric=lambda company: "rubric")


def _run(provider: Scripted, log: list[str], k: int = 5) -> dict[str, object]:
    graph = selection.build_selection_graph(provider, _tools(log), k)
    return asyncio.run(
        graph.ainvoke({"question": "Q?", "principle": "Ownership", "company": "amazon"})
    )


def call(id_: str, name: str, **args: object) -> ToolCall:
    return ToolCall(id_, name, dict(args))


def test_agent_searches_checks_usage_and_selects_unused_story() -> None:
    log: list[str] = []
    provider = Scripted(
        (
            call("1", "search_experiences", query="ownership"),
            call("2", "get_story_usage", company="amazon"),
        ),
        (call("3", "select_chunks", chunk_ids=["b:0"]),),
    )

    result = _run(provider, log)

    assert log == ["search:Ownership\nQ?:5", "search:ownership:5", "usage:amazon"]
    assert result["retrieved_chunks"] == [B]
    assert result["tool_calls"] == 3
    tool_messages = [m for m in provider.messages[1] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["seed", "1", "2"]
    assert '"story_id": "a/x"' in tool_messages[2]["content"]
    assert provider.messages[1][4]["tool_calls"][0]["function"]["name"] == "search_experiences"


def test_seed_search_is_in_context_before_first_turn() -> None:
    provider = Scripted((call("1", "select_chunks", chunk_ids=["a:0"]),))

    result = _run(provider, [])

    first = provider.messages[0]
    assert first[2]["tool_calls"][0]["id"] == "seed"
    assert '"chunk_id": "a:0"' in first[3]["content"]
    assert result["retrieved_chunks"] == [A] and result["tool_calls"] == 1


def test_budget_exhaustion_offers_only_select_then_falls_back_to_seen() -> None:
    log: list[str] = []
    searches = tuple(call(str(i), "search_experiences", query=f"q{i}", k=1) for i in range(8))
    provider = Scripted(searches, ())

    result = _run(provider, log, k=1)

    assert len(log) == selection.MAX_TOOL_CALLS
    assert result["tool_calls"] == selection.MAX_TOOL_CALLS
    assert provider.offered[-1] == [selection.SELECT_TOOL]
    budget_errors = [
        m for m in provider.messages[-1] if "budget exhausted" in str(m.get("content"))
    ]
    assert len(budget_errors) == 3  # 1 seed + 5 of 8 requested fit the budget
    assert result["retrieved_chunks"] == [A]


def test_turn_limit_ends_the_loop() -> None:
    log: list[str] = []
    provider = Scripted(*[(call(str(i), "get_story", story_id="b/y"),) for i in range(10)])

    result = _run(provider, log)

    assert len(provider.offered) == selection.MAX_AGENT_TURNS
    assert result["retrieved_chunks"] == [A, B]


def test_explicit_empty_selection_returns_no_chunks() -> None:
    provider = Scripted(
        (call("1", "search_experiences", query="q"),), (call("2", "select_chunks", chunk_ids=[]),)
    )

    assert _run(provider, [])["retrieved_chunks"] == []


def test_invented_ids_fall_back_and_no_tools_falls_back_to_search() -> None:
    invented = Scripted(
        (call("1", "search_experiences", query="q"),),
        (call("2", "select_chunks", chunk_ids=["zzz"]),),
    )
    assert _run(invented, [], k=1)["retrieved_chunks"] == [A]

    log: list[str] = []
    silent = Scripted(())
    result = _run(silent, log, k=2)
    assert log == ["search:Ownership\nQ?:2"]
    assert result["retrieved_chunks"] == [A, B] and result["tool_calls"] == 1


def test_tool_errors_are_returned_to_the_model() -> None:
    def broken_rubric(company: str) -> str:
        raise ValueError("No company profile for 'acme'")

    provider = Scripted(
        (call("1", "get_company_rubric", company="acme"),),
        (call("2", "select_chunks", chunk_ids=[]),),
    )
    tools = StoryTools(search=_tools([]).search, rubric=broken_rubric)
    graph = selection.build_selection_graph(provider, tools)

    asyncio.run(graph.ainvoke({"question": "Q?", "principle": "P", "company": "acme"}))

    assert "No company profile" in provider.messages[1][-1]["content"]


def test_agent_selector_returns_chunks_and_count() -> None:
    provider = Scripted((call("1", "select_chunks", chunk_ids=[]),))
    select = selection.agent_selector(provider, _tools([]))

    update = asyncio.run(select({"question": "Q?", "principle": "P"}))

    assert update == {"retrieved_chunks": [], "tool_calls": 1}
