"""Parsing the generator's answer (spec 007, D7, R12): pure functions.

Citations (R12): ``[``, one or more numbers of ASCII digits separated by
commas with optional spaces, and ``]``, anywhere in the content; ``[n][m]``
is two of them. Brackets that hold anything else are not citations
(``[1-3]``, ``[2.1]``, ``[a]``, ``[]``). Numbers are read as integers, so
``[01]`` is 1. With N sources in the prompt, numbers in 1..N are valid and
all others invalid; ``valid`` and ``invalid`` list each number once, in
order of first appearance.

The abstention marker: the first line that is not empty after stripping,
with its leading whitespace, ``*``, ``_`` and ``#`` stripped in any mix,
starts with ``ONVOLDOENDE BEWIJS``, compared case-insensitively; what
follows does not matter. Empty content is null, or whitespace only.

Status (D7), checked in this order: empty content is ``error``
(``empty_answer``); the marker is ``insufficient_evidence`` with abstention
source ``prompt``, citations or not; at least one valid citation is
``answered``; otherwise ``uncited``. The answerer checks the truncation
tripwire before any of these (R11).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

STATUS_ANSWERED = "answered"
STATUS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
STATUS_UNCITED = "uncited"
STATUS_ERROR = "error"
# D7's order, which the command's batch output follows (R18).
STATUSES = (
    STATUS_ANSWERED,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_UNCITED,
    STATUS_ERROR,
)

ABSTENTION_PROMPT = "prompt"
ABSTENTION_SCORE_GATE = "score_gate"
ABSTENTION_NO_CANDIDATES = "no_candidates"

ERROR_EMPTY_ANSWER = "empty_answer"

MARKER = "ONVOLDOENDE BEWIJS"
_MARKUP = "*_#"
_CITATION = re.compile(r"\[([0-9]+(?: *, *[0-9]+)*)\]")


@dataclass(frozen=True)
class ParsedAnswer:
    """What the content says: its status, the abstention source, the cited
    numbers and, for empty content, the error type."""

    status: str
    abstention_source: Optional[str]
    valid: List[int]
    invalid: List[int]
    error_type: Optional[str] = None


def cited_numbers(content: str) -> List[int]:
    """Every cited number, in order of appearance, repeats included."""
    return [
        int(number)
        for match in _CITATION.finditer(content)
        for number in match.group(1).split(",")
    ]


def split_citations(
    numbers: Sequence[int], source_count: int
) -> Tuple[List[int], List[int]]:
    """``(valid, invalid)``: each number once, in order of first appearance;
    valid means in 1..``source_count``."""
    valid: List[int] = []
    invalid: List[int] = []
    for number in numbers:
        target = valid if 1 <= number <= source_count else invalid
        if number not in target:
            target.append(number)
    return valid, invalid


def is_empty(content: Optional[str]) -> bool:
    """Null, or whitespace only."""
    return content is None or not content.strip()


def has_marker(content: str) -> bool:
    """Whether the first non-empty line starts with :data:`MARKER`, after
    leading whitespace and markdown emphasis (R12)."""
    for line in content.splitlines():
        if not line.strip():
            continue
        start = 0
        while start < len(line) and (line[start].isspace() or line[start] in _MARKUP):
            start += 1
        return line[start : start + len(MARKER)].casefold() == MARKER.casefold()
    return False


def parse_answer(content: Optional[str], source_count: int) -> ParsedAnswer:
    """The status of ``content`` from a prompt with ``source_count`` sources
    (D7). The sources of an ``answered`` content are the chunks of ``valid``,
    in that order."""
    if is_empty(content):
        return ParsedAnswer(STATUS_ERROR, None, [], [], ERROR_EMPTY_ANSWER)
    valid, invalid = split_citations(cited_numbers(content), source_count)
    if has_marker(content):
        return ParsedAnswer(
            STATUS_INSUFFICIENT_EVIDENCE, ABSTENTION_PROMPT, valid, invalid
        )
    if valid:
        return ParsedAnswer(STATUS_ANSWERED, None, valid, invalid)
    return ParsedAnswer(STATUS_UNCITED, None, valid, invalid)
