"""Parse optional `---` frontmatter at the top of experience files."""

import logging

logger = logging.getLogger(__name__)

RESERVED_KEYS = frozenset({"source_file", "header_1", "header_2", "header_3", "story_id"})
FrontmatterValue = str | list[str]


def split_frontmatter(text: str) -> tuple[dict[str, FrontmatterValue], str]:
    """Return (metadata, body) for `key: value` frontmatter fenced by `---` lines.

    Supports plain strings and `[a, b]` lists, which is all experience tags need
    (role, company, timeframe, competencies) and avoids a YAML dependency. Text
    without a leading `---` fence is returned unchanged with empty metadata.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        return {}, text

    metadata: dict[str, FrontmatterValue] = {}
    for line in lines[1:end]:
        key, separator, raw = line.partition(":")
        key = key.strip()
        if not separator or not key or line.lstrip().startswith("#"):
            continue
        if key in RESERVED_KEYS:
            logger.warning("Ignoring reserved frontmatter key: %s", key)
            continue
        metadata[key] = _parse_value(raw.strip())
    return metadata, "\n".join(lines[end + 1 :]).strip()


def _parse_value(raw: str) -> FrontmatterValue:
    if raw.startswith("[") and raw.endswith("]"):
        return [item.strip().strip("'\"") for item in raw[1:-1].split(",") if item.strip()]
    return raw.strip("'\"")
