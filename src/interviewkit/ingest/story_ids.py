"""Derive stable story identifiers from chunk header metadata."""

import re
from pathlib import PurePosixPath

_NUMBERING = re.compile(r"^\s*\d+[.)]\s*")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
MAX_SEGMENT_LENGTH = 60


def slugify(text: str) -> str:
    """Return a lowercase hyphenated slug without leading list numbering."""
    without_number = _NUMBERING.sub("", text).lower()
    slug = _NON_ALNUM.sub("-", without_number).strip("-")
    return slug[:MAX_SEGMENT_LENGTH].rstrip("-")


def story_id(source_file: str, header_2: object = None, header_3: object = None) -> str:
    """Build `<file stem>/<## slug>/<### slug>` from whichever headers exist.

    Including the `##` heading keeps same-named `###` sections in different
    projects (for example two "Technical Decisions") distinct.
    """
    segments = [PurePosixPath(source_file).with_suffix("").as_posix()]
    for header in (header_2, header_3):
        if isinstance(header, str) and slugify(header):
            segments.append(slugify(header))
    if len(segments) == 1:
        segments.append("overview")
    return "/".join(segments)
