import pytest

from interviewkit.evals.numbers import extract_numbers, unsupported_numbers


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("40%", "40 %"),
        ("40%", "40 percent"),
        ("40%", "40 per cent"),
        ("$1.2M", "1.2 million"),
        ("$1.2M", "$1,200,000"),
        ("$1.2M", "1.2m"),
        ("10k", "10K"),
        ("10k", "10,000"),
        ("10,000", "10000"),
        ("2.5B", "2.5 billion"),
        ("3x", "3X"),
        ("3x", "3×"),
        ("1.50", "1.5"),
    ],
)
def test_equivalent_forms_normalize_to_same_claim(first: str, second: str) -> None:
    assert set(extract_numbers(first)) == set(extract_numbers(second))


@pytest.mark.parametrize(
    ("text", "expected_key"),
    [
        ("cut latency 40%", "40%"),
        ("saved $1.2M", "1200000"),
        ("10k users", "10000"),
        ("2.5B rows", "2500000000"),
        ("3x faster", "3x"),
        ("over 6 weeks", "6"),
        ("in 2023", "2023"),
        ("price 1.50", "1.5"),
    ],
)
def test_extract_numbers_keys(text: str, expected_key: str) -> None:
    assert list(extract_numbers(text)) == [expected_key]


def test_multiplier_is_distinct_from_plain_number() -> None:
    assert set(extract_numbers("3x")) != set(extract_numbers("3"))


def test_range_yields_two_claims_with_shared_unit() -> None:
    assert extract_numbers("improved 20-30% overall") == {"20%": "20-30%", "30%": "20-30%"}


def test_extract_numbers_keeps_first_raw_form() -> None:
    assert extract_numbers("40% then 40 percent") == {"40%": "40%"}


def test_unsupported_numbers_reports_missing_claims_in_order() -> None:
    answer = "We grew 3x, saved $2M and cut latency 40%."
    assert unsupported_numbers(answer, ["Latency fell 40 percent."]) == ["3x", "$2M"]


def test_unsupported_numbers_dedupes_raw_forms() -> None:
    assert unsupported_numbers("9 nodes, 9 nodes", []) == ["9"]


def test_word_numbers_in_sources_support_digits() -> None:
    assert (
        unsupported_numbers("I led 6 engineers for 20 days", ["six engineers, Twenty days"]) == []
    )


def test_word_numbers_in_answer_are_not_claims() -> None:
    assert unsupported_numbers("I led six engineers", []) == []


def test_unsupported_numbers_ignores_answer_noise() -> None:
    answer = (
        "Situation: Chunk 3 described a migration.\n"
        "Action:\n"
        "1. Planned rollout in story-2024.md\n"
        "2) Shipped it\n"
        "Result: done.\n"
        "Sources:\n"
        "- (source: project-7.md)\n"
        "- 99 extra\n"
    )
    assert unsupported_numbers(answer, []) == []


def test_unsupported_numbers_partial_range_support() -> None:
    assert unsupported_numbers("20-30% gain", ["a 20% gain"]) == ["20-30%"]
    assert unsupported_numbers("20-30% gain", ["20% to 30 percent"]) == []
