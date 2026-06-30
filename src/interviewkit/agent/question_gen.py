"""LangGraph workflow for behavioral question generation."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from interviewkit.providers import LLMProvider, get_provider

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "question_gen.txt"
QUESTION_LINE_RE = re.compile(r"^\[(?P<principle>[^\]]+)\]\s*(?P<question>.+)$")


class QuestionGenerationError(RuntimeError):
    """Raised when the question-generation graph cannot complete."""


@dataclass(frozen=True)
class GeneratedQuestion:
    """A structured behavioral interview question."""

    principle: str
    text: str


class QuestionGenerationState(TypedDict, total=False):
    """State carried between LangGraph nodes."""

    job_description: str
    company_principles: str
    raw_response: str
    questions: list[GeneratedQuestion]


def build_question_graph(
    provider: LLMProvider | None = None,
) -> CompiledStateGraph[
    QuestionGenerationState, Any, QuestionGenerationState, QuestionGenerationState
]:
    """Build and compile the question-generation LangGraph workflow."""
    resolved_provider = provider or get_provider()
    graph = StateGraph(QuestionGenerationState)

    async def generate_node(state: QuestionGenerationState) -> QuestionGenerationState:
        return await _generate_questions(state, resolved_provider)

    # Each node is a normal Python function. LangGraph passes the shared state
    # through these nodes and merges the returned dictionary into that state.
    graph.add_node("parse_input", _parse_input)
    graph.add_node("generate_questions", generate_node)
    graph.add_node("format_output", _format_output)

    # Edges are the controlled order of execution for this deterministic flow.
    graph.add_edge(START, "parse_input")
    graph.add_edge("parse_input", "generate_questions")
    graph.add_edge("generate_questions", "format_output")
    graph.add_edge("format_output", END)
    return cast(
        CompiledStateGraph[
            QuestionGenerationState,
            Any,
            QuestionGenerationState,
            QuestionGenerationState,
        ],
        graph.compile(),
    )


async def generate_interview_questions(
    job_description: str,
    company_principles: str,
    provider: LLMProvider | None = None,
) -> list[GeneratedQuestion]:
    """Generate structured behavioral questions for a role and company."""
    graph = build_question_graph(provider=provider)
    state = await graph.ainvoke(
        {
            "job_description": job_description,
            "company_principles": company_principles,
        }
    )
    return cast(list[GeneratedQuestion], state["questions"])


def _parse_input(state: QuestionGenerationState) -> QuestionGenerationState:
    job_description = _required_text(state, "job_description", "Job description text is required")
    company_principles = _required_text(
        state,
        "company_principles",
        "Company principles text is required",
    )
    return {
        "job_description": job_description,
        "company_principles": company_principles,
    }


async def _generate_questions(
    state: QuestionGenerationState,
    provider: LLMProvider,
) -> QuestionGenerationState:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    job_description = _required_text(state, "job_description", "Job description text is required")
    company_principles = _required_text(
        state,
        "company_principles",
        "Company principles text is required",
    )
    response = await provider.complete(
        [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    "Job description:\n"
                    f"{job_description}\n\n"
                    "Company principles:\n"
                    f"{company_principles}"
                ),
            },
        ]
    )
    if not response.content.strip():
        raise QuestionGenerationError("Question-generation provider returned empty content")
    return {"raw_response": response.content}


def _format_output(state: QuestionGenerationState) -> QuestionGenerationState:
    raw_response = _required_text(
        state,
        "raw_response",
        "Question-generation response is missing",
    )
    questions = [_parse_question_line(line) for line in raw_response.splitlines()]
    parsed_questions = [question for question in questions if question is not None]
    if not parsed_questions:
        raise QuestionGenerationError("Question-generation response contained no tagged questions")
    return {"questions": parsed_questions}


def _parse_question_line(line: str) -> GeneratedQuestion | None:
    match = QUESTION_LINE_RE.match(line.strip())
    if match is None:
        return None
    return GeneratedQuestion(
        principle=match.group("principle").strip(),
        text=match.group("question").strip(),
    )


def _required_text(
    state: QuestionGenerationState,
    key: str,
    error_message: str,
) -> str:
    value = cast(str, state.get(key, "")).strip()
    if not value:
        raise QuestionGenerationError(error_message)
    return value
