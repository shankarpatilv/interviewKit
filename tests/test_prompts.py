from pathlib import Path

PROMPTS_DIR = Path("src/interviewkit/prompts")


def test_question_generation_prompt_defines_required_contract() -> None:
    prompt = (PROMPTS_DIR / "question_gen.txt").read_text(encoding="utf-8")

    assert "Job description text" in prompt
    assert "Company principles or competencies" in prompt
    assert "Generate 5 to 8" in prompt
    assert "[Principle or Competency] Question text" in prompt
    assert "Do not answer the questions." in prompt


def test_star_synthesis_prompt_defines_required_contract() -> None:
    prompt = (PROMPTS_DIR / "star_synth.txt").read_text(encoding="utf-8")

    assert "Interview question" in prompt
    assert "Company principle or competency" in prompt
    assert "Retrieved experience chunks" in prompt
    assert "Do not invent projects" in prompt
    assert '"refused"' in prompt
    for field in ('"situation"', '"task"', '"action"', '"result"', '"citations"'):
        assert field in prompt
    assert "chunk_id" in prompt


def test_story_selector_prompt_formats_limits() -> None:
    prompt = (PROMPTS_DIR / "story_selector.txt").read_text(encoding="utf-8")

    formatted = prompt.format(k=5, max_tool_calls=6)

    assert "at most 5 chunk_ids" in formatted and "at most 6 tool calls" in formatted
    assert "get_story_usage" in formatted
