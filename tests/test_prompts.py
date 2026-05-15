from pathlib import Path

PROMPTS_DIR = Path("src/interviewkit/prompts")


def test_question_generation_prompt_defines_required_contract() -> None:
    prompt = (PROMPTS_DIR / "question_gen.txt").read_text(encoding="utf-8")

    assert "Job description text" in prompt
    assert "Company principles or competencies" in prompt
    assert "Generate 5 to 8" in prompt
    assert "[Principle or Competency] Question text" in prompt
    assert "Do not answer the questions." in prompt
