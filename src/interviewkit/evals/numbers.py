"""Deterministic extraction and support checks for numeric claims in answers."""

import re
from collections.abc import Sequence
from decimal import Decimal

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_UNIT = (
    r"\s?%|\s(?i:percent|per\s?cent)\b|\s?(?i:million|billion|thousand)\b"
    r"|[kKmMbB]\b|[xX×](?!\w)"
)
_CLAIM = re.compile(
    rf"(?<![\w.,])[$€£]?(?P<a>{_NUM})(?P<ua>{_UNIT})?"
    rf"(?:\s?[-–]\s?[$€£]?(?P<b>{_NUM})(?P<ub>{_UNIT})?)?"
)
_SCALES = {"k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000}
_SCALES |= {"b": 1_000_000_000, "billion": 1_000_000_000}
_NOISE = re.compile(r"\S+\.md\b|\bchunk\s*#?\d+", re.IGNORECASE)
_LIST_MARKER = re.compile(r"^(\s*(?:[-*+]\s+)?)\d+[.)](?=\s)", re.MULTILINE)
_SOURCES_HEADER = re.compile(r"^[ \t]*[*_]*sources[*_]*[ \t]*:", re.IGNORECASE | re.MULTILINE)
_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty"
).split()
_WORD_VALUES = {word: str(index) for index, word in enumerate(_WORDS)}
_WORD_VALUES |= {"thirty": "30", "forty": "40", "fifty": "50", "hundred": "100"}
_WORD_NUMBER = re.compile(rf"\b({'|'.join(_WORD_VALUES)})\b", re.IGNORECASE)


def _normalize(number: str, unit: str | None) -> str:
    """Return a canonical key such as ``1200000``, ``40%`` or ``3x``."""
    value = Decimal(number.replace(",", ""))
    unit_key = (unit or "").strip().lower()
    suffix = ""
    if unit_key.startswith("%") or unit_key.startswith("per"):
        suffix = "%"
    elif unit_key in ("x", "×"):
        suffix = "x"
    elif unit_key in _SCALES:
        value *= _SCALES[unit_key]
    text = format(value.normalize(), "f")
    return f"{text}{suffix}"


def extract_numbers(text: str) -> dict[str, str]:
    """Map normalized numeric claim -> first raw surface form found in text."""
    claims: dict[str, str] = {}
    for match in _CLAIM.finditer(_NOISE.sub(" ", text)):
        raw = match.group(0).strip()
        first_unit, second_unit = match.group("ua"), match.group("ub")
        claims.setdefault(_normalize(match.group("a"), first_unit or second_unit), raw)
        if match.group("b") is not None:
            claims.setdefault(_normalize(match.group("b"), second_unit or first_unit), raw)
    return claims


def _clean_answer(answer: str) -> str:
    """Drop the Sources section and list/step markers from an answer."""
    header = _SOURCES_HEADER.search(answer)
    body = answer[: header.start()] if header else answer
    return _LIST_MARKER.sub(r"\1", body)


def _source_claims(source: str) -> set[str]:
    """Return normalized claims in a source, treating word numbers as digits."""
    with_digits = _WORD_NUMBER.sub(lambda m: _WORD_VALUES[m.group(1).lower()], source)
    return set(extract_numbers(source)) | set(extract_numbers(with_digits))


def unsupported_numbers(answer: str, sources: Sequence[str]) -> list[str]:
    """Raw surface forms of numbers in `answer` that appear in none of `sources`.

    Comparison happens after normalization; results keep answer order and are deduped.
    """
    supported: set[str] = set()
    for source in sources:
        supported |= _source_claims(source)
    unsupported: list[str] = []
    for key, raw in extract_numbers(_clean_answer(answer)).items():
        if key not in supported and raw not in unsupported:
            unsupported.append(raw)
    return unsupported
