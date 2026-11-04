"""Text normalization (spec 004): soft normalization of the page text store,
and hard normalization for later keyword indexing.

The ``companydocs-normalize`` command (:mod:`.normalize_cli`) reads the page
text store and, per document, writes the soft-normalized text of every page
with a record of every line it removed and every join it made. This module
holds the pure functions behind it: the soft operations, the per-document
assembly (:func:`normalize_document`) and :func:`hard_normalize`. Settings are
read from ``ingestion.normalization`` at call time.

Soft normalization
------------------
A page's text is split on the line feed only and rejoined with it, so a
trailing newline survives. Line indices are 0-based indices into the store
text split that way. The steps, in order:

1. Characters (:func:`clean_characters`). Zero-width characters (U+200B to
   U+200D, U+2060, U+FEFF) and the soft hyphen U+00AD are removed, U+00A0 and
   U+202F become a space, then NFC. Removing first keeps the step idempotent.
2. Spaces (:func:`normalize_spaces`). Within each line, trailing whitespace is
   stripped, then runs of U+0020 collapse to one. A whitespace-only line
   becomes blank. Steps 1 and 2 never change the number of lines.
3. Page labels (D8). The candidates are the first and last ``edge_lines``
   non-empty lines of a page. A candidate whose stripped text fully matches a
   configured label family is removed. A candidate that is a bare number is
   removed only as part of a run: bare numbers are grouped by offset (value
   minus page number), a group spanning at least ``min_run_pages`` distinct
   pages is page numbering, and on each of its pages the one line closest to
   an edge is removed. Distances are positions among the page's non-empty
   lines. A line nearer the first non-empty line than the last is at the
   ``top`` edge; any other line, a tie included, is at the ``bottom``.
4. Dehyphenation (D7, :func:`dehyphenate_text`). A candidate is a line ending
   in a letter followed by U+002D. Its next line is the next non-empty line on
   the same page, with at most one blank line between, read without its
   leading spaces. The first rule that applies decides: (a) the next line does
   not start with a lowercase letter: no change; (b) its first word is a
   conjunction of the page's language: no change; (c) the fragment before the
   hyphen has an uppercase letter after its first character: no change; (d)
   the hyphenated form occurs elsewhere in the document: join, keeping the
   hyphen; (e) otherwise join, removing the hyphen, and record whether the
   joined form occurs elsewhere. A joined line is examined again before moving
   on. Rules d and e search every page as it is after step 3, before any join,
   casefolded, for the form as a whole word.
5. Filler runs (:func:`collapse_filler_runs`). ``min_run`` or more identical
   filler characters collapse to ``collapse_to`` of them.

Nothing else changes: quotes, dashes, diacritics, case, digits and symbols
stay, and lines are never joined into paragraphs. Operations 1, 2, 4 and 5 are
idempotent, with one accepted exception for operation 4: a candidate kept by
rule b is joined on a second pass when its next line, a conjunction followed
by a hyphen, was itself joined by rule e.

Hard normalization
------------------
:func:`hard_normalize` (D10) is for later keyword (BM25) indexing and query
processing. The normalize command never calls it. In order: NFKC, casefold,
NFKC again (casefolding can leave text that is not NFKC); quotes to an
apostrophe or a straight double quote; U+2010 to U+2015 and U+2212 to the
hyphen-minus; private-use characters (category ``Co``) removed; every run of
whitespace (``str.isspace``) to one space, stripped at both ends. Primes and
the backtick are not mapped to quotes; NFKC decomposes U+2033 into two U+2032.
There is no diacritic stripping: the documents are mostly Dutch, with Czech,
Polish, Slovak, German and French among them.

Characters outside ASCII are written here as ``chr`` of their code point.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    MutableSet,
    Optional,
    Sequence,
    Tuple,
)

from .parse_pass import STATUS_FAILED, STATUS_PROCESSED
from .quality import TEXT_ORIGIN_NATIVE, TEXT_ORIGIN_NONE, TEXT_ORIGIN_OCR
from .settings import get_section

logger = logging.getLogger(__name__)

# --- Settings defaults (ingestion.normalization), read at call time -------------
PAGE_LABEL_EDGE_LINES = 3
PAGE_NUMBER_RUN_MIN_PAGES = 3
BARE_NUMBER_MAX_DIGITS = 3
FILLER_CHARACTERS = "._-" + chr(0x2026)  # U+2026 HORIZONTAL ELLIPSIS
FILLER_MIN_RUN = 4
FILLER_COLLAPSE_TO = 3

# --- Python constants, not settings ---------------------------------------------
# The output's schema version, bumped on any change to its keys or their
# meanings; the normalization version, bumped on any change to what an
# operation does; and the page text store schema version this module reads.
NORMALIZED_SCHEMA_VERSION = 1
NORMALIZATION_VERSION = 1
REQUIRED_PAGE_TEXT_SCHEMA_VERSION = 2

LINE_BREAK = "\n"
SPACE = " "
HYPHEN = "-"

# Step 1: U+200B ZERO WIDTH SPACE, U+200C ZERO WIDTH NON-JOINER, U+200D ZERO
# WIDTH JOINER, U+2060 WORD JOINER and U+FEFF ZERO WIDTH NO-BREAK SPACE are
# removed, and so is U+00AD SOFT HYPHEN; U+00A0 NO-BREAK SPACE and U+202F
# NARROW NO-BREAK SPACE become a space.
ZERO_WIDTH_CHARACTERS = frozenset(map(chr, (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF)))
SOFT_HYPHEN = chr(0x00AD)
NO_BREAK_SPACES = frozenset(map(chr, (0x00A0, 0x202F)))

# Hard normalization (D10): to an apostrophe, U+2018 LEFT and U+2019 RIGHT
# SINGLE QUOTATION MARK, U+201A SINGLE LOW-9 and U+201B SINGLE HIGH-REVERSED-9
# QUOTATION MARK, U+2039 and U+203A SINGLE LEFT- and RIGHT-POINTING ANGLE
# QUOTATION MARK; to a straight double quote, U+201C LEFT and U+201D RIGHT
# DOUBLE QUOTATION MARK, U+201E DOUBLE LOW-9 and U+201F DOUBLE HIGH-REVERSED-9
# QUOTATION MARK, U+00AB and U+00BB LEFT- and RIGHT-POINTING DOUBLE ANGLE
# QUOTATION MARK; to the hyphen-minus, U+2010 HYPHEN to U+2015 HORIZONTAL BAR
# and U+2212 MINUS SIGN.
HARD_SINGLE_QUOTES = "".join(map(chr, (0x2018, 0x2019, 0x201A, 0x201B, 0x2039, 0x203A)))
HARD_DOUBLE_QUOTES = "".join(map(chr, (0x201C, 0x201D, 0x201E, 0x201F, 0x00AB, 0x00BB)))
HARD_DASHES = "".join(
    map(chr, (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212))
)
PRIVATE_USE_CATEGORY = "Co"

# --- Recorded values ------------------------------------------------------------
PAGE_LABELS_APPLIED = "applied"
HEADER_FOOTER_NOT_BUILT = "not_built"
NOT_APPLICABLE = "not_applicable"
REASON_PAGE_LABEL = "page_label"
REASON_PAGE_NUMBER_RUN = "page_number_run"
EDGE_TOP = "top"
EDGE_BOTTOM = "bottom"
RULE_HYPHEN_KEPT = "d"
RULE_HYPHEN_REMOVED = "e"
TEXT_ORIGINS = (TEXT_ORIGIN_NATIVE, TEXT_ORIGIN_OCR, TEXT_ORIGIN_NONE)

# The D7 counters of a page. Every candidate increments exactly one of the
# other six; e_joined_form_in_document counts the rule-e joins whose joined
# form occurs elsewhere in the document, a subset of e_hyphen_removed.
DEHYPHENATION_COUNTERS = (
    "a_not_lowercase",
    "b_conjunction",
    "c_inner_uppercase",
    "d_hyphen_kept",
    "e_hyphen_removed",
    "e_joined_form_in_document",
    "no_next_line",
)
# The counters of operations 1, 2 and 5 in a page's counts.
OPERATION_COUNTERS = (
    "nfc_changed",
    "zero_width_removed",
    "soft_hyphens_removed",
    "nbsp_replaced",
    "space_runs_collapsed",
    "trailing_whitespace_stripped",
    "filler_runs_collapsed",
    "filler_chars_removed",
)


# --- Settings -------------------------------------------------------------------


class NormalizationConfigError(ValueError):
    """``ingestion.normalization`` lacks the label families or the conjunction
    lists, or a family regex does not compile (spec 004, section 6)."""


@dataclass
class NormalizationSettings:
    """The resolved ``ingestion.normalization`` settings."""

    edge_lines: int
    min_run_pages: int
    bare_number_max_digits: int
    families: Tuple[Tuple[str, str], ...]  # (name, regex), in configured order
    conjunctions: Mapping[str, FrozenSet[str]]  # language code -> casefolded words
    filler_characters: str
    filler_min_run: int
    filler_collapse_to: int
    family_patterns: Tuple[Tuple[str, "re.Pattern[str]"], ...]
    bare_number_pattern: "re.Pattern[str]"

    def constants(self) -> Dict[str, Any]:
        """The ``page_labels.constants`` block of the output."""
        return {
            "edge_lines": self.edge_lines,
            "min_run_pages": self.min_run_pages,
            "bare_number_max_digits": self.bare_number_max_digits,
            "families": dict(self.families),
        }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def normalization_settings(
    settings: Optional[Mapping[str, Any]] = None,
) -> NormalizationSettings:
    """Resolve ``ingestion.normalization`` at call time, YAML over defaults.

    The label families and the conjunction lists live in the YAML only, with
    no Python copy. Raises :class:`NormalizationConfigError` when either is
    missing or a family regex does not compile.
    """
    section = get_section("ingestion", "normalization", settings=settings)
    labels = _mapping(section.get("page_labels"))
    dehyphenation = _mapping(section.get("dehyphenation"))
    filler = _mapping(section.get("filler"))

    families = labels.get("families")
    if not isinstance(families, Mapping):
        raise NormalizationConfigError(
            "ingestion.normalization.page_labels.families is missing"
        )
    conjunctions = dehyphenation.get("conjunctions")
    if not isinstance(conjunctions, Mapping):
        raise NormalizationConfigError(
            "ingestion.normalization.dehyphenation.conjunctions is missing"
        )
    patterns = []
    for name, regex in families.items():
        try:
            patterns.append((str(name), re.compile(str(regex), re.IGNORECASE)))
        except re.error as error:
            raise NormalizationConfigError(
                f"page label family {str(name)!r} does not compile: {error}"
            ) from error

    max_digits = int(labels.get("bare_number_max_digits", BARE_NUMBER_MAX_DIGITS))
    return NormalizationSettings(
        edge_lines=int(labels.get("edge_lines", PAGE_LABEL_EDGE_LINES)),
        min_run_pages=int(labels.get("min_run_pages", PAGE_NUMBER_RUN_MIN_PAGES)),
        bare_number_max_digits=max_digits,
        families=tuple((str(name), str(regex)) for name, regex in families.items()),
        conjunctions={
            str(language): frozenset(str(word).casefold() for word in words or ())
            for language, words in conjunctions.items()
        },
        filler_characters=str(filler.get("characters", FILLER_CHARACTERS)),
        filler_min_run=int(filler.get("min_run", FILLER_MIN_RUN)),
        filler_collapse_to=int(filler.get("collapse_to", FILLER_COLLAPSE_TO)),
        family_patterns=tuple(patterns),
        bare_number_pattern=re.compile(f"[0-9]{{1,{max_digits}}}"),
    )


# --- The store record -------------------------------------------------------------


class StoreRecordError(ValueError):
    """A page text store file that is valid JSON but not a valid store record."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StoreRecordError(message)


def _get(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise StoreRecordError(f"{where} lacks the key {key!r}")
    return mapping[key]


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_optional_str(value: Any) -> bool:
    return value is None or isinstance(value, str)


def validate_store_record(record: Any) -> None:
    """Raise :class:`StoreRecordError` unless ``record`` holds what
    normalization reads, with the types the parse pass writes.

    The schema version is not checked here: the command refuses a record
    whose ``schema_version`` is not 2 before it gets this far. A failed
    document needs no ``pages``.
    """
    _require(isinstance(record, Mapping), "the top level is not a JSON object")
    file_info = _get(record, "file", "the record")
    _require(isinstance(file_info, Mapping), "'file' is not an object")
    _require(
        isinstance(_get(file_info, "path", "'file'"), str),
        "'file.path' is not a string",
    )
    _require(
        _is_optional_str(_get(file_info, "sha256", "'file'")),
        "'file.sha256' is not a string or null",
    )
    status = _get(record, "status", "the record")
    _require(
        status in (STATUS_PROCESSED, STATUS_FAILED),
        f"'status' is {json.dumps(status)}, not processed or failed",
    )
    _require(
        _is_int(_get(record, "page_count", "the record")),
        "'page_count' is not an integer",
    )
    if status == STATUS_FAILED:
        return
    pages = _get(record, "pages", "the record")
    _require(isinstance(pages, list), "'pages' is not a list")
    for position, page in enumerate(pages):
        where = f"pages[{position}]"
        _require(isinstance(page, Mapping), f"{where} is not an object")
        _require(
            _is_int(_get(page, "page_number", where)),
            f"{where}.page_number is not an integer",
        )
        _require(
            isinstance(_get(page, "text", where), str), f"{where}.text is not a string"
        )
        _require(
            isinstance(_get(page, "source", where), str),
            f"{where}.source is not a string",
        )
        origin = _get(page, "text_origin", where)
        _require(
            origin in TEXT_ORIGINS,
            f"{where}.text_origin is {json.dumps(origin)}, not native, ocr or none",
        )
        for key in ("no_text_reason", "language", "language_source", "tier"):
            _require(
                _is_optional_str(_get(page, key, where)),
                f"{where}.{key} is not a string or null",
            )
        confidence = _get(page, "ocr_mean_confidence", where)
        _require(
            confidence is None
            or (
                isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
            ),
            f"{where}.ocr_mean_confidence is not a number or null",
        )


# --- Operation 1: characters ------------------------------------------------------


def clean_characters(text: str) -> Tuple[str, Dict[str, Any]]:
    """Step 1: remove zero-width characters and the soft hyphen, replace the
    two no-break spaces with a space, then NFC. Returns the text and its
    counts."""
    zero_width = soft_hyphens = no_break = 0
    kept: List[str] = []
    for char in text:
        if char in ZERO_WIDTH_CHARACTERS:
            zero_width += 1
        elif char == SOFT_HYPHEN:
            soft_hyphens += 1
        elif char in NO_BREAK_SPACES:
            no_break += 1
            kept.append(SPACE)
        else:
            kept.append(char)
    removed = "".join(kept)
    composed = unicodedata.normalize("NFC", removed)
    return composed, {
        "nfc_changed": composed != removed,
        "zero_width_removed": zero_width,
        "soft_hyphens_removed": soft_hyphens,
        "nbsp_replaced": no_break,
    }


# --- Operation 2: spaces ------------------------------------------------------------

_SPACE_RUN = re.compile("  +")


def normalize_spaces(text: str) -> Tuple[str, Dict[str, int]]:
    """Step 2: per line, strip trailing whitespace, then collapse runs of
    U+0020 to one. A trailing run counts only as stripped whitespace."""
    runs = stripped = 0
    lines: List[str] = []
    for line in text.split(LINE_BREAK):
        trimmed = line.rstrip()
        if trimmed != line:
            stripped += 1
        collapsed, count = _SPACE_RUN.subn(SPACE, trimmed)
        runs += count
        lines.append(collapsed)
    return LINE_BREAK.join(lines), {
        "space_runs_collapsed": runs,
        "trailing_whitespace_stripped": stripped,
    }


# --- Pages as they go through steps 3 and 4 ----------------------------------------


@dataclass
class _Line:
    """One current line of a page, and the store lines it spans."""

    first: int  # store line index of its first line
    last: int  # store line index of the line at its end, which carries its hyphen
    text: str


def _empty_counts() -> Dict[str, Any]:
    return {
        "nfc_changed": False,
        "zero_width_removed": 0,
        "soft_hyphens_removed": 0,
        "nbsp_replaced": 0,
        "space_runs_collapsed": 0,
        "trailing_whitespace_stripped": 0,
        "page_label_lines_removed": 0,
        "page_number_run_lines_removed": 0,
        "dehyphenation": {name: 0 for name in DEHYPHENATION_COUNTERS},
        "filler_runs_collapsed": 0,
        "filler_chars_removed": 0,
    }


@dataclass
class _Page:
    store: Mapping[str, Any]
    store_lines: List[str]
    lines: List[_Line]
    counts: Dict[str, Any]
    removed_lines: List[Dict[str, Any]] = field(default_factory=list)
    joins: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def start(cls, store_page: Mapping[str, Any]) -> "_Page":
        """A store page after steps 1 and 2."""
        counts = _empty_counts()
        cleaned, character_counts = clean_characters(store_page["text"])
        spaced, space_counts = normalize_spaces(cleaned)
        counts.update(character_counts)
        counts.update(space_counts)
        return cls(
            store=store_page,
            store_lines=store_page["text"].split(LINE_BREAK),
            lines=[
                _Line(first=index, last=index, text=line)
                for index, line in enumerate(spaced.split(LINE_BREAK))
            ],
            counts=counts,
        )

    def text(self) -> str:
        return LINE_BREAK.join(line.text for line in self.lines)


# --- Step 3: page labels (D8) --------------------------------------------------------


@dataclass(frozen=True)
class _NumberLine:
    """A bare-number candidate line."""

    page: int  # index into the document's pages
    line: int  # index into that page's lines
    position: int  # among the page's non-empty lines
    distance: int  # to the nearer edge, in non-empty lines
    edge: str


def _remove_page_labels(
    pages: Sequence[_Page], config: NormalizationSettings
) -> Tuple[int, List[Dict[str, Any]]]:
    """Step 3 over a whole document. Returns the number of pages considered
    (with a non-empty line) and the accepted bare-number runs."""
    removals: Dict[int, Dict[int, Tuple[str, Optional[str], Optional[int], str]]] = {
        index: {} for index in range(len(pages))
    }
    numbers: Dict[int, List[_NumberLine]] = {}
    considered = 0

    for page_index, page in enumerate(pages):
        non_empty = [index for index, line in enumerate(page.lines) if line.text]
        count = len(non_empty)
        if not count:
            continue
        considered += 1
        reach = max(config.edge_lines, 0)
        window = sorted(
            set(range(min(reach, count))) | set(range(count - min(reach, count), count))
        )
        for position in window:
            line_index = non_empty[position]
            stripped = page.lines[line_index].text.strip()
            to_top, to_bottom = position, count - 1 - position
            edge = EDGE_TOP if to_top < to_bottom else EDGE_BOTTOM
            family = next(
                (
                    name
                    for name, pattern in config.family_patterns
                    if pattern.fullmatch(stripped)
                ),
                None,
            )
            if family is not None:
                removals[page_index][line_index] = (
                    REASON_PAGE_LABEL,
                    family,
                    None,
                    edge,
                )
            elif config.bare_number_pattern.fullmatch(stripped):
                offset = int(stripped) - page.store["page_number"]
                numbers.setdefault(offset, []).append(
                    _NumberLine(
                        page_index, line_index, position, min(to_top, to_bottom), edge
                    )
                )

    runs: List[Dict[str, Any]] = []
    for offset in sorted(numbers):
        by_page: Dict[int, List[_NumberLine]] = {}
        for candidate in numbers[offset]:
            by_page.setdefault(candidate.page, []).append(candidate)
        if len(by_page) < config.min_run_pages:
            continue
        runs.append(
            {
                "offset": offset,
                "pages": sorted(pages[index].store["page_number"] for index in by_page),
            }
        )
        for page_index, candidates in by_page.items():
            # The one closest to an edge; on a tie, the bottom one.
            chosen = min(candidates, key=lambda item: (item.distance, -item.position))
            removals[page_index][chosen.line] = (
                REASON_PAGE_NUMBER_RUN,
                None,
                offset,
                chosen.edge,
            )

    for page_index, page in enumerate(pages):
        removed = removals[page_index]
        for line_index in sorted(removed):
            reason, family, offset, edge = removed[line_index]
            store_index = page.lines[line_index].first
            page.removed_lines.append(
                {
                    "line_index": store_index,
                    "text": page.store_lines[store_index],
                    "edge": edge,
                    "reason": reason,
                    "family": family,
                    "run_offset": offset,
                }
            )
            counter = (
                "page_label_lines_removed"
                if reason == REASON_PAGE_LABEL
                else "page_number_run_lines_removed"
            )
            page.counts[counter] += 1
        page.lines = [
            line
            for line_index, line in enumerate(page.lines)
            if line_index not in removed
        ]
    return considered, runs


# --- Step 4: dehyphenation (D7) -------------------------------------------------------


def _is_candidate(text: str) -> bool:
    return len(text) >= 2 and text[-1] == HYPHEN and text[-2].isalpha()


def _next_line(lines: Sequence[_Line], index: int) -> Optional[int]:
    """The next non-empty line on the page, with at most one blank between."""
    if index + 1 < len(lines) and lines[index + 1].text:
        return index + 1
    if index + 2 < len(lines) and not lines[index + 1].text and lines[index + 2].text:
        return index + 2
    return None


def _leading_letters(text: str) -> str:
    end = 0
    while end < len(text) and text[end].isalpha():
        end += 1
    return text[:end]


def _trailing_letters(text: str) -> str:
    start = len(text)
    while start > 0 and text[start - 1].isalpha():
        start -= 1
    return text[start:]


def _occurs_as_word(form: str, corpus: Sequence[str]) -> bool:
    """Whether ``form`` (casefolded) occurs in a casefolded page of ``corpus``
    as a whole word: not preceded or followed by a letter."""
    for text in corpus:
        start = text.find(form)
        while start >= 0:
            end = start + len(form)
            if (start == 0 or not text[start - 1].isalpha()) and (
                end == len(text) or not text[end].isalpha()
            ):
                return True
            start = text.find(form, start + 1)
    return False


def _dehyphenate(
    lines: List[_Line],
    corpus: Sequence[str],
    conjunctions: FrozenSet[str],
    counts: Dict[str, int],
    joins: List[Dict[str, Any]],
) -> None:
    """Step 4 on one page's lines, in place. ``corpus`` holds the document's
    pages after step 3, casefolded; ``conjunctions`` are casefolded."""
    index = 0
    while index < len(lines):
        text = lines[index].text
        if not _is_candidate(text):
            index += 1
            continue
        next_index = _next_line(lines, index)
        if next_index is None:
            counts["no_next_line"] += 1
            index += 1
            continue
        following = lines[next_index].text.lstrip(SPACE)
        # A lowercase letter is a letter (str.isalpha) that is lowercase
        # (str.islower); str.islower alone also accepts a few characters that
        # are not letters, such as U+24D0 and U+2170.
        if not following or not (following[0].isalpha() and following[0].islower()):
            counts["a_not_lowercase"] += 1
            index += 1
            continue
        first_word = _leading_letters(following)
        if first_word.casefold() in conjunctions:
            counts["b_conjunction"] += 1
            index += 1
            continue
        fragment = _trailing_letters(text[:-1])
        if any(char.isupper() for char in fragment[1:]):
            counts["c_inner_uppercase"] += 1
            index += 1
            continue
        if _occurs_as_word(f"{fragment}{HYPHEN}{first_word}".casefold(), corpus):
            rule, joined = RULE_HYPHEN_KEPT, text + following
            counts["d_hyphen_kept"] += 1
        else:
            rule, joined = RULE_HYPHEN_REMOVED, text[:-1] + following
            counts["e_hyphen_removed"] += 1
            if _occurs_as_word(f"{fragment}{first_word}".casefold(), corpus):
                counts["e_joined_form_in_document"] += 1
        joins.append(
            {
                "line_index": lines[index].last,
                "rule": rule,
                "fragment": fragment,
                "first_word": first_word,
            }
        )
        lines[index] = _Line(
            first=lines[index].first, last=lines[next_index].last, text=joined
        )
        # The next line and the blank line between, if there was one, go; the
        # joined line is examined again before moving on (C4).
        del lines[index + 1 : next_index + 1]


def dehyphenate_text(
    text: str,
    conjunctions: Sequence[str] = (),
    corpus: Optional[Sequence[str]] = None,
) -> Tuple[str, Dict[str, int], List[Dict[str, Any]]]:
    """Step 4 on one page's text, with line indices counted in ``text``.

    ``corpus`` is the document's pages as rules d and e search them, after
    step 3 and before any join; None searches ``text`` alone. Returns the
    text, the D7 counters and the joins.
    """
    lines = [
        _Line(first=index, last=index, text=line)
        for index, line in enumerate(text.split(LINE_BREAK))
    ]
    searched = [page.casefold() for page in (corpus if corpus is not None else [text])]
    counts = {name: 0 for name in DEHYPHENATION_COUNTERS}
    joins: List[Dict[str, Any]] = []
    _dehyphenate(
        lines,
        searched,
        frozenset(word.casefold() for word in conjunctions),
        counts,
        joins,
    )
    return LINE_BREAK.join(line.text for line in lines), counts, joins


def _conjunctions_for(
    language: Optional[str],
    config: NormalizationSettings,
    warned: MutableSet[Optional[str]],
) -> FrozenSet[str]:
    """The page language's conjunction list; an empty list, with one WARNING
    per language per run, when there is none (C11)."""
    words = config.conjunctions.get(language) if language is not None else None
    if words is not None:
        return words
    if language not in warned:
        warned.add(language)
        logger.warning(
            "No conjunction list for page language %r under "
            "ingestion.normalization.dehyphenation.conjunctions; rule b uses an "
            "empty list",
            language,
        )
    return frozenset()


# --- Operation 5: filler runs ---------------------------------------------------------


def collapse_filler_runs(
    text: str,
    characters: str = FILLER_CHARACTERS,
    min_run: int = FILLER_MIN_RUN,
    collapse_to: int = FILLER_COLLAPSE_TO,
) -> Tuple[str, Dict[str, int]]:
    """Step 5: ``min_run`` or more identical characters from ``characters``
    collapse to ``collapse_to`` of that character."""
    counts = {"filler_runs_collapsed": 0, "filler_chars_removed": 0}
    if not characters:
        return text, counts
    allowed = "".join(re.escape(char) for char in characters)
    pattern = re.compile(f"(?P<char>[{allowed}])(?P=char){{{min_run - 1},}}")

    def collapse(match: "re.Match[str]") -> str:
        counts["filler_runs_collapsed"] += 1
        counts["filler_chars_removed"] += len(match.group(0)) - collapse_to
        return match.group("char") * collapse_to

    return pattern.sub(collapse, text), counts


# --- The document ---------------------------------------------------------------------


def text_sha256(text: str) -> str:
    """Hex sha256 of the text's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_sha256(pages: Sequence[Mapping[str, Any]]) -> str:
    """The document's ``normalized_sha256``: the hash of the JSON array of
    ``[page_number, normalized_sha256]`` pairs, compact and ASCII."""
    pairs = [[page["page_number"], page["normalized_sha256"]] for page in pages]
    return text_sha256(json.dumps(pairs, separators=(",", ":"), ensure_ascii=True))


def _page_output(page: _Page, config: NormalizationSettings) -> Dict[str, Any]:
    text, filler_counts = collapse_filler_runs(
        page.text(),
        config.filler_characters,
        config.filler_min_run,
        config.filler_collapse_to,
    )
    page.counts.update(filler_counts)
    store = page.store
    return {
        "page_number": store["page_number"],
        "source": store["source"],
        "text_origin": store["text_origin"],
        "no_text_reason": store["no_text_reason"],
        "language": store["language"],
        "language_source": store["language_source"],
        "tier": store["tier"],
        "ocr_mean_confidence": store["ocr_mean_confidence"],
        "input_sha256": text_sha256(store["text"]),
        "text": text,
        "normalized_sha256": text_sha256(text),
        "removed_lines": page.removed_lines,
        "joins": page.joins,
        "counts": page.counts,
    }


def normalize_document(
    record: Mapping[str, Any],
    settings: Optional[Mapping[str, Any]] = None,
    *,
    config: Optional[NormalizationSettings] = None,
    warned_languages: Optional[MutableSet[Optional[str]]] = None,
) -> Dict[str, Any]:
    """Soft-normalize one page text store record into its output document.

    Args:
        record: the store record, with ``schema_version`` 2 already checked.
        settings: a settings mapping overriding ``config/settings.yaml``,
            read at call time; ignored when ``config`` is given.
        config: settings already resolved by :func:`normalization_settings`.
        warned_languages: the page languages without a conjunction list
            already warned about in this run; None warns afresh.

    Raises :class:`StoreRecordError` when the record is not a valid store
    record, and :class:`NormalizationConfigError` for missing or broken
    settings. A failed store document gives an output without pages.
    """
    config = config if config is not None else normalization_settings(settings)
    warned = warned_languages if warned_languages is not None else set()
    validate_store_record(record)

    pages: List[Dict[str, Any]] = []
    labels_status = header_footer_status = NOT_APPLICABLE
    labels_considered = header_footer_considered = 0
    runs: List[Dict[str, Any]] = []
    if record["status"] == STATUS_PROCESSED:
        work = [_Page.start(store_page) for store_page in record["pages"]]
        labels_considered, runs = _remove_page_labels(work, config)
        header_footer_considered = sum(
            1 for page in work if any(line.text for line in page.lines)
        )
        corpus = [page.text().casefold() for page in work]
        for page in work:
            _dehyphenate(
                page.lines,
                corpus,
                _conjunctions_for(page.store["language"], config, warned),
                page.counts["dehyphenation"],
                page.joins,
            )
        pages = [_page_output(page, config) for page in work]
        labels_status = PAGE_LABELS_APPLIED
        header_footer_status = HEADER_FOOTER_NOT_BUILT

    return {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "unicode_version": unicodedata.unidata_version,
        "file": {"path": record["file"]["path"], "sha256": record["file"]["sha256"]},
        "status": record["status"],
        "page_count": record["page_count"],
        "page_labels": {
            "status": labels_status,
            "pages_considered": labels_considered,
            "constants": config.constants(),
            "runs": runs,
        },
        "header_footer": {
            "status": header_footer_status,
            "pages_considered": header_footer_considered,
            "constants": {},
        },
        "normalized_sha256": document_sha256(pages),
        "pages": pages,
    }


# --- Hard normalization (D10) ---------------------------------------------------------

_HARD_TRANSLATION = str.maketrans(
    {
        **{char: "'" for char in HARD_SINGLE_QUOTES},
        **{char: '"' for char in HARD_DOUBLE_QUOTES},
        **{char: HYPHEN for char in HARD_DASHES},
    }
)


def hard_normalize(text: str) -> str:
    """Hard-normalize ``text`` for keyword indexing and query processing.

    See the module docstring for the steps and their order. Pure: the same
    input always gives the same output, and nothing is read or written.
    """
    folded = unicodedata.normalize(
        "NFKC", unicodedata.normalize("NFKC", text).casefold()
    )
    mapped = folded.translate(_HARD_TRANSLATION)
    kept = "".join(
        char for char in mapped if unicodedata.category(char) != PRIVATE_USE_CATEGORY
    )
    return " ".join(kept.split())
