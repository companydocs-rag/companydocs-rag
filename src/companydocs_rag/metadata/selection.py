"""Chunk selection, per document (spec 008, D6, R9, Patterns).

The stems and month names of a document come from its chunk file's top-level
``language``: that language's entries in ``keywords`` and ``month_names``,
none for a language without an entry, and the union of every configured
language when it is null, each stem keeping its own match mode.

A chunk's lines are its text split on "\\n", each hard-normalized on its own,
so nothing matches across a line break. Tokens are the maximal runs of
letters of a line. A stem matches a token by its mode: ``infix`` when it
occurs in the token, ``prefix`` when the token starts with it,
``whole_token`` when the token equals it. Stems are counted by their string:
a stem matching several tokens counts once, a token matching two stems
counts twice, and a stem listed for two languages counts once under the
union.

A line holds a valid date when one of the five patterns below matches it and
the match passes D6's range check (day 1-31, month 1-12, and the extra
conditions per pattern); overlapping matches are not resolved. This check
serves the bonus only.

A chunk's score is the number of distinct stems it matches, plus 1 when at
least one of its lines holds a valid date. Chunks with score 0 are never
selected. The rest are ranked by score, descending, then by
``chunk_index``, and fill batches in rank order: a chunk joins the current
batch while the sum of ``len(text)`` stays at most the budget; the first
chunk that does not fit starts the next batch, so a chunk longer than the
budget is a batch of its own. At most ``max_batches`` batches. Within a
batch, the chunks are ordered by ``chunk_index``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from companydocs_rag.ingestion.normalization import hard_normalize

from .settings import (
    KEYWORD_INFIX,
    KEYWORD_PREFIX,
    KEYWORD_WHOLE_TOKEN,
    MONTHS_ABBREVIATED,
    MONTHS_FULL,
)

_TOKEN = re.compile(r"[^\W\d_]+")
_LINE_BREAK = "\n"
# R4: never valid in a day-month date without a year.
_MAY = "may"
_MONTHS_PLACEHOLDER = "MONTHS"

# Patterns (D6, R9): run on one hard-normalized line, so letters are lower case
# and dashes are "-". MONTHS stands for the alternation of the month names in use.
NUMERIC_DATE = re.compile(
    r"(?<![\w.,/-])(?P<day>[0-9]{1,2}) ?(?P<sep>[-/.]) ?(?P<month>[0-9]{1,2}) ?"
    r"(?P=sep) ?(?P<year>[0-9]{4}|[0-9]{2})(?![\w]|[.,/-][0-9])"
)
ISO_DATE = re.compile(
    r"(?<![\w.,/-])(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"(?![\w]|-[0-9])"
)
_DAY_MONTH = (
    r"(?<![\w.,/-])(?P<day>[0-9]{1,2})(?:ste|de|e|st|nd|rd|th)?\.?[ -]?(?:of )?"
    r"(?P<month>MONTHS)\.?(?![\w])(?:,? ?[ -]?(?P<year>[0-9]{4})(?![0-9]))?"
)
_MONTH_DAY_YEAR = (
    r"(?<![\w])(?P<month>MONTHS)\.? (?P<day>[0-9]{1,2})(?:ste|de|e|st|nd|rd|th)?,? "
    r"(?P<year>[0-9]{4})(?![0-9])"
)
_MONTH_YEAR = r"(?<![\w])(?P<month>MONTHS)\.?,? (?P<year>[0-9]{4})(?![0-9])"


@dataclass(frozen=True)
class Vocabulary:
    """What a document's chunks are matched against: (stem, mode) pairs,
    and the month names in use."""

    stems: Tuple[Tuple[str, str], ...]
    full_months: FrozenSet[str]
    months: Tuple[str, ...]


def vocabulary(
    language: Optional[str],
    keywords: Mapping[str, Mapping[str, Sequence[str]]],
    month_names: Mapping[str, Mapping[str, Sequence[str]]],
) -> Vocabulary:
    """The stems and month names of a document in ``language`` (R9)."""
    languages = list(keywords) if language is None else [language]
    stems: List[Tuple[str, str]] = []
    for code in languages:
        for mode, words in keywords.get(code, {}).items():
            for word in words:
                if (word, mode) not in stems:
                    stems.append((word, mode))
    month_languages = list(month_names) if language is None else [language]
    full: Set[str] = set()
    every: Set[str] = set()
    for code in month_languages:
        names = month_names.get(code, {})
        full.update(names.get(MONTHS_FULL, []))
        every.update(names.get(MONTHS_FULL, []))
        every.update(names.get(MONTHS_ABBREVIATED, []))
    return Vocabulary(
        stems=tuple(stems),
        full_months=frozenset(full),
        months=tuple(sorted(every, key=lambda name: (-len(name), name))),
    )


def _matches(token: str, stem: str, mode: str) -> bool:
    if mode == KEYWORD_INFIX:
        return stem in token
    if mode == KEYWORD_PREFIX:
        return token.startswith(stem)
    if mode == KEYWORD_WHOLE_TOKEN:
        return token == stem
    raise ValueError(f"unknown match mode {mode!r}")


def _lines(text: str) -> List[str]:
    return [hard_normalize(line) for line in text.split(_LINE_BREAK)]


def matched_stems(text: str, words: Vocabulary) -> Set[str]:
    """The distinct stems that ``text`` matches (R9)."""
    found: Set[str] = set()
    for line in _lines(text):
        tokens = _TOKEN.findall(line)
        for stem, mode in words.stems:
            if stem not in found and any(_matches(t, stem, mode) for t in tokens):
                found.add(stem)
    return found


@lru_cache(maxsize=32)
def _month_patterns(months: Tuple[str, ...]) -> Tuple[re.Pattern, ...]:
    """The three patterns with month names, for ``months``, longest first."""
    if not months:
        return ()
    alternation = "|".join(re.escape(name) for name in months)
    return tuple(
        re.compile(template.replace(_MONTHS_PLACEHOLDER, alternation))
        for template in (_DAY_MONTH, _MONTH_DAY_YEAR, _MONTH_YEAR)
    )


def _in_range(text: str, low: int, high: int) -> bool:
    return low <= int(text) <= high


def _valid_numeric(match: re.Match) -> bool:
    return (
        _in_range(match["day"], 1, 31)
        and _in_range(match["month"], 1, 12)
        and (match["sep"] != "." or len(match["year"]) == 4)
    )


def _valid_iso(match: re.Match) -> bool:
    return _in_range(match["month"], 1, 12) and _in_range(match["day"], 1, 31)


def line_has_valid_date(line: str, words: Vocabulary) -> bool:
    """Whether one hard-normalized line holds a valid date (Patterns)."""
    if any(_valid_numeric(match) for match in NUMERIC_DATE.finditer(line)):
        return True
    if any(_valid_iso(match) for match in ISO_DATE.finditer(line)):
        return True
    patterns = _month_patterns(words.months)
    if not patterns:
        return False
    day_month, month_day_year, month_year = patterns
    for match in day_month.finditer(line):
        if not _in_range(match["day"], 1, 31):
            continue
        if match["year"] is not None:
            return True
        if match["month"] in words.full_months and match["month"] != _MAY:
            return True
    if any(_in_range(m["day"], 1, 31) for m in month_day_year.finditer(line)):
        return True
    return month_year.search(line) is not None


def has_valid_date(text: str, words: Vocabulary) -> bool:
    """Whether any line of ``text`` holds a valid date (D6)."""
    return any(line_has_valid_date(line, words) for line in _lines(text))


def chunk_score(text: str, words: Vocabulary) -> int:
    """D6: the distinct stems matched, plus 1 for at least one valid date."""
    return len(matched_stems(text, words)) + (1 if has_valid_date(text, words) else 0)


def plan_batches(
    chunks: Sequence[Mapping[str, Any]],
    scores: Sequence[int],
    *,
    budget: int,
    max_batches: int,
) -> List[List[Mapping[str, Any]]]:
    """D6: the batches, each in ``chunk_index`` order; ``scores`` are the
    chunks' scores, in the order of ``chunks``."""
    ranked = [
        chunk
        for chunk, _ in sorted(
            (
                (chunk, score)
                for chunk, score in zip(chunks, scores, strict=True)
                if score > 0
            ),
            key=lambda pair: (-pair[1], pair[0]["chunk_index"]),
        )
    ]
    batches: List[List[Mapping[str, Any]]] = []
    current: List[Mapping[str, Any]] = []
    total = 0
    for chunk in ranked:
        size = len(chunk["text"])
        if current and total + size > budget:
            batches.append(current)
            current, total = [], 0
            if len(batches) == max_batches:
                break
        current.append(chunk)
        total += size
    if current:
        batches.append(current)
    return [sorted(batch, key=lambda chunk: chunk["chunk_index"]) for batch in batches]


def batch_chars(batch: Sequence[Mapping[str, Any]]) -> int:
    """R9: a batch's ``chars``, the sum of ``len(text)`` of its chunks."""
    return sum(len(chunk["text"]) for chunk in batch)


def scores_of(chunks: Sequence[Mapping[str, Any]], words: Vocabulary) -> List[int]:
    return [chunk_score(chunk["text"], words) for chunk in chunks]
