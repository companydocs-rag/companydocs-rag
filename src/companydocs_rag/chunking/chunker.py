"""Chunking (spec 005): cut each normalized document into chunks.

The ``companydocs-chunk`` command (:mod:`.cli`) reads the files
``companydocs-normalize`` writes and, per document, writes its chunks. This
module holds the pure functions behind it: the settings
(:func:`chunking_settings`), the record check
(:func:`validate_normalized_record`), the heading runs (:func:`decompose_runs`),
the line splitter (:func:`split_lines`) and the per-document assembly
(:func:`chunk_document`). Settings are read from the top-level ``chunking``
section at call time. Nothing here changes text: every chunk's ``text`` is a
slice of the document's stream.

Stream and lines (K2)
---------------------
A document's stream is the soft-normalized text of its pages in page order,
with the pages whose text is "" left out and the others joined with one line
feed. That line feed belongs to no page. Offsets are Python ``str`` indices
into the stream, end exclusive, and sizes are lengths in ``str`` characters,
line feeds included. A line is a stretch of the stream that ends with its line
feed, which belongs to it; a non-empty rest after the last line feed is the
last line. A line boundary is the start of a line or the end of the stream,
and every chunk starts and ends at one.

Headings and runs (K3-K5)
-------------------------
A line, without its line feed, is tested against the keyword pattern (the
configured ``heading_keywords``, case-insensitive) and then against
:data:`DECIMAL_RE`. A match makes the line a heading candidate of that family,
with the number as written and, as its chain value, the integer before the
number's first ".". A standalone number line matches neither pattern.

Per family, :func:`decompose_runs` splits the candidates, in stream order,
into runs: chains whose members step +1 (up to +1 + ``heading_max_gap``) and
that together cover the most candidates, where every run after the first
starts at 1 and directly follows a run of at least ``min_heading_run_length``
members. The family with the longest single run is chosen, keyword on a tie;
every other candidate is body text.

Sections and the route (K6, K7)
-------------------------------
A section runs from a member's heading line to the next member's heading line;
the last section of a run ends at the next run's first heading, or at the end
of the stream. A run is accepted when it has at least
``min_heading_run_length`` members (else ``run_too_short``) and the median of
its section lengths, the lower middle value for an even count, is at least
``min_run_median_section_chars`` (else ``sections_too_short``). The route is
``structured`` when a run is accepted, else ``generic``. Every maximal stretch
outside the sections of accepted runs is a generic region.

Strategy (spec 002)
-------------------
``strategy`` is ``structure_aware`` by default: the route is chosen as above
(architecture decision 2). ``fixed_length`` is an experiment that uses no run:
every document takes the generic route, so the whole stream is one generic
region, cut by the splitter at ``chunk_target_chars`` with
``chunk_overlap_chars`` of overlap, without clause fields. The heading
candidates and runs are still found and assessed, and recorded in the heading
evidence; they decide nothing.

Chunks (K8-K13)
---------------
The splitter packs whole lines into a piece while it stays at most
``chunk_target_chars``; a longer line is a piece of its own. Each later piece
of the same region or section starts with the longest run of the previous
piece's trailing lines that totals at most ``chunk_overlap_chars`` and leaves
room for the piece's next line. A piece whose text is whitespace only is not
emitted.

A generic region is cut into ``generic`` chunks. A section of at most
``chunk_max_chars`` is one ``heading_section`` chunk; a longer one is cut by
the splitter, and its pieces that start less than ``max_label_inherit_chars``
after the heading are ``heading_section_part`` chunks with the clause fields,
the later ones ``heading_section_overflow`` chunks without them. Each chunk
records the pages it covers, each with that page's ``text_origin``, ``tier``
and ``ocr_mean_confidence``; there is no chunk-level tier or confidence.

Characters outside ASCII are written here as ``chr`` of their code point.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from companydocs_rag.ingestion.normalization import TEXT_ORIGINS
from companydocs_rag.ingestion.parse_pass import STATUS_FAILED, STATUS_PROCESSED
from companydocs_rag.ingestion.settings import get_section

# --- Settings defaults (chunking), read at call time ----------------------------------
# How every document is cut (spec 002): one of STRATEGIES below.
# "structure_aware" is the baseline.
STRATEGY = "structure_aware"
# The keyword family's keywords (K3), tried in this order. "$" is recognition
# only: a "$ 3" heading keeps its "$" in the text and in its clause label.
HEADING_KEYWORDS: Tuple[str, ...] = (
    "Artikel",
    "Article",
    "Art.",
    "Artyku" + chr(0x0142),  # U+0142 LATIN SMALL LETTER L WITH STROKE
    # U+010C LATIN CAPITAL LETTER C WITH CARON and U+00E1 LATIN SMALL LETTER A
    # WITH ACUTE
    chr(0x010C) + "l" + chr(0x00E1) + "nek",
    chr(0x010C) + "l" + chr(0x00E1) + "nok",
    chr(0x010C) + "l.",
    "Section",
    "Clause",
    chr(0x00A7),  # U+00A7 SECTION SIGN
    "$",
)
HEADING_MAX_GAP = 0
MIN_HEADING_RUN_LENGTH = 3
MIN_RUN_MEDIAN_SECTION_CHARS = 400
MAX_LABEL_INHERIT_CHARS = 6000
CHUNK_TARGET_CHARS = 1200
CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP_CHARS = 150
CLAUSE_LABEL_MAX_CHARS = 80

# --- Python constants, not settings ---------------------------------------------------
# The output's schema version, bumped on any change to its keys or their
# meanings; the chunker version, bumped on any change to what the chunker does;
# and the normalized files' schema version this module reads.
CHUNKS_SCHEMA_VERSION = 1
CHUNKER_VERSION = 1
REQUIRED_NORMALIZED_SCHEMA_VERSION = 1

# The decimal family (K3): a number such as "8" or "1.2", then ".", ")" or
# whitespace, then a title that starts with neither a digit nor whitespace.
DECIMAL_RE = re.compile(
    r"^\s*(?P<num>[0-9]+(?:\.[0-9]+)*)(?:[.)]\s*|\s+)(?P<rest>[^0-9\s].*)$"
)

STRUCTURE_HINT_UNKNOWN = "unknown"
STRATEGY_STRUCTURE_AWARE = "structure_aware"
STRATEGY_FIXED_LENGTH = "fixed_length"
ROUTE_STRUCTURED = "structured"
ROUTE_GENERIC = "generic"
HEADING_FAMILY_KEYWORD = "keyword"
HEADING_FAMILY_DECIMAL = "decimal"
CHUNK_METHOD_HEADING_SECTION = "heading_section"
CHUNK_METHOD_HEADING_SECTION_PART = "heading_section_part"
CHUNK_METHOD_HEADING_SECTION_OVERFLOW = "heading_section_overflow"
CHUNK_METHOD_GENERIC = "generic"
REJECT_RUN_TOO_SHORT = "run_too_short"
REJECT_SECTIONS_TOO_SHORT = "sections_too_short"
CHUNK_ID_HEX_CHARS = 20

# The values above, in the order the output and the command's summary use.
STRATEGIES = (STRATEGY_STRUCTURE_AWARE, STRATEGY_FIXED_LENGTH)
HEADING_FAMILIES = (HEADING_FAMILY_KEYWORD, HEADING_FAMILY_DECIMAL)
ROUTES = (ROUTE_STRUCTURED, ROUTE_GENERIC)
CHUNK_METHODS = (
    CHUNK_METHOD_HEADING_SECTION,
    CHUNK_METHOD_HEADING_SECTION_PART,
    CHUNK_METHOD_HEADING_SECTION_OVERFLOW,
    CHUNK_METHOD_GENERIC,
)
REJECT_REASONS = (REJECT_RUN_TOO_SHORT, REJECT_SECTIONS_TOO_SHORT)

LINE_BREAK = "\n"


# --- Settings -------------------------------------------------------------------------


class ChunkingConfigError(ValueError):
    """The ``chunking`` settings are invalid (R6)."""


# The integer settings: (key, default, minimum), in the YAML's order (R6).
_INTEGER_SETTINGS = (
    ("heading_max_gap", HEADING_MAX_GAP, 0),
    ("min_heading_run_length", MIN_HEADING_RUN_LENGTH, 1),
    ("min_run_median_section_chars", MIN_RUN_MEDIAN_SECTION_CHARS, 0),
    ("max_label_inherit_chars", MAX_LABEL_INHERIT_CHARS, 1),
    ("chunk_target_chars", CHUNK_TARGET_CHARS, 1),
    ("chunk_max_chars", CHUNK_MAX_CHARS, 1),
    ("chunk_overlap_chars", CHUNK_OVERLAP_CHARS, 0),
    ("clause_label_max_chars", CLAUSE_LABEL_MAX_CHARS, 1),
)


@dataclass(frozen=True)
class ChunkingSettings:
    """The resolved ``chunking`` settings, with the keyword pattern built
    from ``heading_keywords``."""

    strategy: str
    heading_keywords: Tuple[str, ...]
    heading_max_gap: int
    min_heading_run_length: int
    min_run_median_section_chars: int
    max_label_inherit_chars: int
    chunk_target_chars: int
    chunk_max_chars: int
    chunk_overlap_chars: int
    clause_label_max_chars: int
    keyword_pattern: "re.Pattern[str]"


def keyword_heading_pattern(keywords: Sequence[str]) -> "re.Pattern[str]":
    """The keyword family's pattern (K3): the keywords, each escaped, as
    alternatives in the given order, then optional whitespace and a number.
    Anything may follow the number. Case-insensitive."""
    alternatives = "|".join(re.escape(keyword) for keyword in keywords)
    return re.compile(
        r"^\s*(?P<kw>" + alternatives + r")\s*(?P<num>[0-9]+)(?P<rest>.*)$",
        re.IGNORECASE,
    )


def chunking_settings(
    settings: Optional[Mapping[str, Any]] = None,
) -> ChunkingSettings:
    """Resolve the top-level ``chunking`` section at call time, YAML over the
    defaults above; ``settings`` replaces ``config/settings.yaml``.

    Raises :class:`ChunkingConfigError` (R6) when ``strategy`` is not one of
    :data:`STRATEGIES`, ``heading_keywords`` is not a non-empty list of
    non-empty strings, or another key is not an integer (a bool is not one)
    or is below its minimum: 0 for ``heading_max_gap``,
    ``min_run_median_section_chars`` and ``chunk_overlap_chars``, 1 for the
    rest. A key that is absent takes its default.
    """
    section = get_section("chunking", settings=settings)
    strategy = section.get("strategy", STRATEGY)
    if not (isinstance(strategy, str) and strategy in STRATEGIES):
        raise ChunkingConfigError(
            f"chunking.strategy is {strategy!r}, not one of " + ", ".join(STRATEGIES)
        )
    keywords = section.get("heading_keywords", list(HEADING_KEYWORDS))
    if not (
        isinstance(keywords, list)
        and keywords
        and all(isinstance(keyword, str) and keyword for keyword in keywords)
    ):
        raise ChunkingConfigError(
            "chunking.heading_keywords is not a non-empty list of non-empty strings"
        )
    values: Dict[str, int] = {}
    for key, default, minimum in _INTEGER_SETTINGS:
        value = section.get(key, default)
        if type(value) is not int:
            raise ChunkingConfigError(f"chunking.{key} is {value!r}, not an integer")
        if value < minimum:
            raise ChunkingConfigError(
                f"chunking.{key} is {value}, below its minimum of {minimum}"
            )
        values[key] = value
    return ChunkingSettings(
        strategy=strategy,
        heading_keywords=tuple(keywords),
        keyword_pattern=keyword_heading_pattern(keywords),
        **values,
    )


# --- The normalized record (R7) -------------------------------------------------------


class NormalizedRecordError(ValueError):
    """A normalized file that is valid JSON but not a valid normalized record
    (R7)."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NormalizedRecordError(message)


def _get(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise NormalizedRecordError(f"{where} lacks the key {key!r}")
    return mapping[key]


def _is_optional_str(value: Any) -> bool:
    return value is None or isinstance(value, str)


def validate_normalized_record(record: Any) -> None:
    """Raise :class:`NormalizedRecordError` unless ``record`` holds what the
    chunker reads, with the types ``companydocs-normalize`` writes (R7).

    ``file.sha256`` may be null only on a failed document. The schema version
    is not checked here: the command refuses a record whose
    ``schema_version`` is not 1 before it gets this far.
    """
    _require(isinstance(record, Mapping), "the top level is not a JSON object")
    file_info = _get(record, "file", "the record")
    _require(isinstance(file_info, Mapping), "'file' is not an object")
    _require(
        isinstance(_get(file_info, "path", "'file'"), str),
        "'file.path' is not a string",
    )
    status = _get(record, "status", "the record")
    _require(
        status in (STATUS_PROCESSED, STATUS_FAILED),
        f"'status' is {json.dumps(status)}, not processed or failed",
    )
    sha256 = _get(file_info, "sha256", "'file'")
    if status == STATUS_FAILED:
        _require(_is_optional_str(sha256), "'file.sha256' is not a string or null")
    else:
        _require(
            isinstance(sha256, str),
            "'file.sha256' is not a string, and only a failed document may have none",
        )
    _require(
        isinstance(_get(record, "normalized_sha256", "the record"), str),
        "'normalized_sha256' is not a string",
    )
    pages = _get(record, "pages", "the record")
    _require(isinstance(pages, list), "'pages' is not a list")
    for position, page in enumerate(pages):
        where = f"pages[{position}]"
        _require(isinstance(page, Mapping), f"{where} is not an object")
        _require(
            type(_get(page, "page_number", where)) is int,
            f"{where}.page_number is not an integer",
        )
        _require(
            isinstance(_get(page, "text", where), str), f"{where}.text is not a string"
        )
        origin = _get(page, "text_origin", where)
        _require(
            origin in TEXT_ORIGINS,
            f"{where}.text_origin is {json.dumps(origin)}, not native, ocr or none",
        )
        for key in ("tier", "language", "language_source"):
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


# --- The stream and its lines (K2) ----------------------------------------------------


@dataclass(frozen=True)
class _Page:
    """A page whose text is not "", and its stretch of the stream."""

    record: Mapping[str, Any]
    start: int
    end: int


def _place_pages(pages: Sequence[Mapping[str, Any]]) -> Tuple[str, List[_Page]]:
    """The stream of ``pages``, given in page order, and where each page with
    text lies in it. Pages whose text is "" are left out (K11)."""
    placed: List[_Page] = []
    offset = 0
    for page in pages:
        text = page["text"]
        if text == "":
            continue
        if placed:
            offset += len(LINE_BREAK)  # the line feed between two pages
        placed.append(_Page(page, offset, offset + len(text)))
        offset += len(text)
    return LINE_BREAK.join(page.record["text"] for page in placed), placed


def line_bounds(stream: str) -> List[Tuple[int, int]]:
    """The ``(start, end)`` of every line of ``stream``: a line ends after its
    line feed, and a non-empty rest after the last line feed is the last
    line. There is no empty last line."""
    bounds: List[Tuple[int, int]] = []
    start = 0
    while start < len(stream):
        newline = stream.find(LINE_BREAK, start)
        end = len(stream) if newline < 0 else newline + len(LINE_BREAK)
        bounds.append((start, end))
        start = end
    return bounds


# --- Heading candidates (K3) ----------------------------------------------------------


@dataclass(frozen=True)
class HeadingCandidate:
    """A line that matches a heading pattern."""

    family: str
    start: int  # stream offset of the line
    text: str  # the line without its line feed
    number: str  # the matched number, as written
    value: int  # the chain value: the integer before the number's first "."


def find_heading_candidates(
    stream: str,
    lines: Sequence[Tuple[int, int]],
    keyword_pattern: "re.Pattern[str]",
) -> Dict[str, List[HeadingCandidate]]:
    """The heading candidates of each family, in stream order. Each line is
    tested against the keyword pattern, then against :data:`DECIMAL_RE`, and
    is a candidate of the first family it matches."""
    found: Dict[str, List[HeadingCandidate]] = {
        family: [] for family in HEADING_FAMILIES
    }
    patterns = (
        (HEADING_FAMILY_KEYWORD, keyword_pattern),
        (HEADING_FAMILY_DECIMAL, DECIMAL_RE),
    )
    for start, end in lines:
        text = stream[start:end]
        if text.endswith(LINE_BREAK):
            text = text[: -len(LINE_BREAK)]
        for family, pattern in patterns:
            match = pattern.match(text)
            if match is not None:
                number = match.group("num")
                found[family].append(
                    HeadingCandidate(
                        family=family,
                        start=start,
                        text=text,
                        number=number,
                        value=int(number.split(".")[0]),
                    )
                )
                break
    return found


# --- Runs (K4) ------------------------------------------------------------------------

# A state's score in decompose_runs: (members covered, minus the number of runs).
_Score = Tuple[int, int]
# A back pointer: (index, capped run length, whether a new run starts after it).
_Back = Tuple[int, int, bool]


def decompose_runs(
    values: Sequence[int],
    min_run_length: int = MIN_HEADING_RUN_LENGTH,
    max_gap: int = HEADING_MAX_GAP,
) -> List[List[int]]:
    """Decompose one family's candidates into runs (K4, R10), as probe M5 does.

    ``values`` are the candidates' chain values, in stream order. A run is a
    chain of candidates in stream order in which each member's value is one
    to ``1 + max_gap`` more than the previous member's. The runs do not
    overlap and together cover the most candidates. The first run may start
    at any value; every later run starts at 1 and directly follows a run of
    at least ``min_run_length`` members, so only the last run can be shorter.
    Ties: fewer runs; then the decomposition whose last member comes
    earliest; where candidates of the same value can take the same place,
    the earliest one; and between steps that score the same, the smaller
    step.

    Returns the runs in stream order, each as a list of indices into
    ``values``; no runs for no values.

    A dynamic programme over the candidates. A state is a candidate as the
    last member so far, with the length of its run capped at
    ``min_run_length``; its score is (members covered, minus the number of
    runs), higher being better.
    """
    cap_full = min_run_length
    # (index, capped length) -> (score, back pointer).
    states: Dict[Tuple[int, int], Tuple[_Score, Optional[_Back]]] = {}
    # (value, capped length) -> (score, index): the best state so far.
    best: Dict[Tuple[int, int], Tuple[_Score, int]] = {}
    # The best state so far whose run has at least min_run_length members.
    closed: Optional[Tuple[_Score, int]] = None

    for index, value in enumerate(values):
        options: Dict[int, Tuple[_Score, Optional[_Back]]] = {
            1: ((1, -1), None)  # a first run that starts here
        }
        for step in range(1, max_gap + 2):
            for cap in range(1, cap_full + 1):
                previous = best.get((value - step, cap))
                if previous is None:
                    continue
                (covered, runs), at = previous
                score = (covered + 1, runs)
                new_cap = min(cap + 1, cap_full)
                if new_cap not in options or score > options[new_cap][0]:
                    options[new_cap] = (score, (at, cap, False))
        if value == 1 and closed is not None:
            (covered, runs), at = closed
            score = (covered + 1, runs - 1)
            if score > options[1][0]:
                options[1] = (score, (at, cap_full, True))
        for cap, option in options.items():
            states[(index, cap)] = option
        for cap, (score, _) in options.items():
            held = best.get((value, cap))
            if held is None or score > held[0]:
                best[(value, cap)] = (score, index)
            if cap == cap_full and (closed is None or score > closed[0]):
                closed = (score, index)

    if not states:
        return []
    key = max(states, key=lambda item: (states[item][0], -item[0]))
    runs_found: List[List[int]] = [[]]
    while True:
        runs_found[-1].append(key[0])
        back = states[key][1]
        if back is None:
            break
        at, cap, new_run = back
        if new_run:
            runs_found.append([])
        key = (at, cap)
    return [run[::-1] for run in runs_found[::-1]]


def _choose_family(runs: Mapping[str, Sequence[Sequence[Any]]]) -> Optional[str]:
    """K5: the family with the longest single run, keyword on a tie; None
    when neither family has a candidate."""
    longest = {
        family: max((len(run) for run in runs[family]), default=0)
        for family in HEADING_FAMILIES
    }
    if not any(longest.values()):
        return None
    if longest[HEADING_FAMILY_KEYWORD] >= longest[HEADING_FAMILY_DECIMAL]:
        return HEADING_FAMILY_KEYWORD
    return HEADING_FAMILY_DECIMAL


# --- Sections and acceptance (K6) -----------------------------------------------------


@dataclass(frozen=True)
class _Run:
    """A run of the chosen family, with its sections."""

    index: int  # 1-based, in stream order
    members: Tuple[HeadingCandidate, ...]
    sections: Tuple[Tuple[int, int], ...]  # (start, end), one per member
    median_section_chars: int
    reject_reason: Optional[str]

    @property
    def accepted(self) -> bool:
        return self.reject_reason is None


def _assess_runs(
    members_per_run: Sequence[Sequence[HeadingCandidate]],
    stream_length: int,
    config: ChunkingSettings,
) -> List[_Run]:
    firsts = [members[0].start for members in members_per_run]
    runs: List[_Run] = []
    for position, members in enumerate(members_per_run):
        # The last section ends at the next run's first heading, accepted or
        # not, or at the end of the stream.
        end = firsts[position + 1] if position + 1 < len(firsts) else stream_length
        starts = [member.start for member in members]
        sections = tuple(zip(starts, starts[1:] + [end]))
        lengths = sorted(stop - start for start, stop in sections)
        median = lengths[(len(lengths) - 1) // 2]  # the lower middle value (Q3)
        if len(members) < config.min_heading_run_length:
            reason: Optional[str] = REJECT_RUN_TOO_SHORT  # checked first (R2)
        elif median < config.min_run_median_section_chars:
            reason = REJECT_SECTIONS_TOO_SHORT
        else:
            reason = None
        runs.append(
            _Run(
                index=position + 1,
                members=tuple(members),
                sections=sections,
                median_section_chars=median,
                reject_reason=reason,
            )
        )
    return runs


# --- The splitter (K8) ----------------------------------------------------------------


def _trailing_lines(
    lines: Sequence[Tuple[int, int]], limit: int
) -> List[Tuple[int, int]]:
    """The longest run of ``lines``' trailing lines that totals at most
    ``limit``; none when the last line alone is longer."""
    taken = count = 0
    for start, end in reversed(lines):
        if taken + (end - start) > limit:
            break
        taken += end - start
        count += 1
    return list(lines[len(lines) - count :])


def split_lines(
    lines: Sequence[Tuple[int, int]], target: int, overlap: int
) -> List[Tuple[int, int]]:
    """K8 over consecutive ``lines`` of one region or section: the ``(start,
    end)`` of each piece, in order.

    Whole lines are added to a piece while it stays at most ``target``; a
    line longer than ``target`` is a piece of its own, without overlap. Every
    other piece after the first starts with the longest run of the previous
    piece's trailing lines that totals at most ``overlap`` and leaves room
    for its own first line within ``target`` (Q2). Each piece holds at least
    one line of its own.
    """
    pieces: List[Tuple[int, int]] = []
    previous: List[Tuple[int, int]] = []
    position = 0
    while position < len(lines):
        first_start, first_end = lines[position]
        first_size = first_end - first_start
        position += 1
        if first_size > target:
            piece = [(first_start, first_end)]
        else:
            carried = _trailing_lines(previous, min(overlap, target - first_size))
            piece = [*carried, (first_start, first_end)]
            size = first_end - piece[0][0]
            while position < len(lines):
                start, end = lines[position]
                if size + (end - start) > target:
                    break
                piece.append((start, end))
                size += end - start
                position += 1
        pieces.append((piece[0][0], piece[-1][1]))
        previous = piece
    return pieces


# --- Chunks (K7, K9-K13) --------------------------------------------------------------


@dataclass(frozen=True)
class _Piece:
    """One chunk to be: its stretch of the stream and how it was made."""

    start: int
    end: int
    method: str
    heading: Optional[HeadingCandidate] = None  # set when it has clause fields
    run_index: Optional[int] = None
    part: Optional[int] = None
    part_count: Optional[int] = None


class _Planner:
    """Cuts one document's stream into pieces, in stream order."""

    def __init__(
        self, stream: str, lines: Sequence[Tuple[int, int]], config: ChunkingSettings
    ) -> None:
        self.stream = stream
        self.lines = lines
        self.line_starts = [start for start, _ in lines]
        self.config = config

    def _lines_in(self, start: int, end: int) -> Sequence[Tuple[int, int]]:
        """The lines from line boundary ``start`` up to line boundary ``end``."""
        return self.lines[
            bisect_left(self.line_starts, start) : bisect_left(self.line_starts, end)
        ]

    def _split(self, start: int, end: int) -> List[Tuple[int, int]]:
        """The splitter's pieces of ``start``..``end``, without the pieces
        whose text is whitespace only (Q4)."""
        return [
            (low, high)
            for low, high in split_lines(
                self._lines_in(start, end),
                self.config.chunk_target_chars,
                self.config.chunk_overlap_chars,
            )
            if not self.stream[low:high].isspace()
        ]

    def generic(self, start: int, end: int) -> List[_Piece]:
        return [
            _Piece(low, high, CHUNK_METHOD_GENERIC)
            for low, high in self._split(start, end)
        ]

    def section(
        self, heading: HeadingCandidate, run_index: int, start: int, end: int
    ) -> List[_Piece]:
        if end - start <= self.config.chunk_max_chars:
            return [
                _Piece(start, end, CHUNK_METHOD_HEADING_SECTION, heading, run_index)
            ]
        split = self._split(start, end)
        # R9: a piece's start is its char_start, overlap included.
        labelled = sum(
            1 for low, _ in split if low - start < self.config.max_label_inherit_chars
        )
        pieces = [
            _Piece(
                low,
                high,
                CHUNK_METHOD_HEADING_SECTION_PART,
                heading,
                run_index,
                part=number,
                part_count=labelled,
            )
            for number, (low, high) in enumerate(split[:labelled], start=1)
        ]
        pieces.extend(
            _Piece(low, high, CHUNK_METHOD_HEADING_SECTION_OVERFLOW)
            for low, high in split[labelled:]
        )
        return pieces

    def plan(self, runs: Sequence[_Run]) -> List[_Piece]:
        """The sections of the accepted runs, and the generic regions: the
        maximal stretches outside them (R1)."""
        pieces: List[_Piece] = []
        cursor = 0
        for run in runs:
            if not run.accepted:
                continue
            pieces.extend(self.generic(cursor, run.sections[0][0]))
            for heading, (start, end) in zip(run.members, run.sections):
                pieces.extend(self.section(heading, run.index, start, end))
            cursor = run.sections[-1][1]
        pieces.extend(self.generic(cursor, len(self.stream)))
        return pieces


def _sha256_hex(text: str) -> str:
    """Hex sha256 of the text's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _page_spans(placed: Sequence[_Page], start: int, end: int) -> List[Dict[str, Any]]:
    """K13: one entry per page that contributes characters to
    ``start``..``end``, in page order, with offsets into that page's text."""
    spans: List[Dict[str, Any]] = []
    for page in placed:
        low, high = max(start, page.start), min(end, page.end)
        if low >= high:
            continue
        spans.append(
            {
                "page_number": page.record["page_number"],
                "char_start": low - page.start,
                "char_end": high - page.start,
                "text_origin": page.record["text_origin"],
                "tier": page.record["tier"],
                "ocr_mean_confidence": page.record["ocr_mean_confidence"],
            }
        )
    return spans


def _chunk_output(
    index: int,
    piece: _Piece,
    stream: str,
    placed: Sequence[_Page],
    file_sha256: Optional[str],
    config: ChunkingSettings,
) -> Dict[str, Any]:
    text = stream[piece.start : piece.end]
    text_sha256 = _sha256_hex(text)
    spans = _page_spans(placed, piece.start, piece.end)
    heading = piece.heading
    return {
        "chunk_id": _sha256_hex(f"{file_sha256}:{piece.start}:{text_sha256}")[
            :CHUNK_ID_HEX_CHARS
        ],
        "chunk_index": index,
        "file_sha256": file_sha256,
        "text": text,
        "text_sha256": text_sha256,
        "char_start": piece.start,
        "char_end": piece.end,
        "page_start": spans[0]["page_number"],
        "page_end": spans[-1]["page_number"],
        "text_origins": sorted({span["text_origin"] for span in spans}),
        "page_spans": spans,
        "chunk_method": piece.method,
        "clause_label": (
            None
            if heading is None
            else heading.text.strip()[: config.clause_label_max_chars]
        ),
        "clause_number": None if heading is None else heading.number,
        "numbering_run": piece.run_index,
        "heading_family": None if heading is None else heading.family,
        "part": piece.part,
        "part_count": piece.part_count,
        "exceeds_max_chars": len(text) > config.chunk_max_chars,
    }


def _page_at(placed: Sequence[_Page], offset: int) -> Mapping[str, Any]:
    """The page whose stretch of the stream holds ``offset``."""
    return placed[bisect_right([page.start for page in placed], offset) - 1].record


def _run_output(run: _Run, placed: Sequence[_Page]) -> Dict[str, Any]:
    first = run.members[0]
    return {
        "run_index": run.index,
        "first_number": first.value,  # R3
        "length": len(run.members),
        "start_page": _page_at(placed, first.start)["page_number"],
        "median_section_chars": run.median_section_chars,
        "accepted": run.accepted,
        "reject_reason": run.reject_reason,
    }


def _shared(pages: Sequence[Mapping[str, Any]], key: str) -> Any:
    """The value of ``key`` all pages share; None if they disagree or there
    are no pages."""
    values = [page[key] for page in pages]
    if values and all(value == values[0] for value in values):
        return values[0]
    return None


def chunk_document(
    record: Mapping[str, Any],
    settings: Optional[Mapping[str, Any]] = None,
    *,
    config: Optional[ChunkingSettings] = None,
    structure_hint: str = STRUCTURE_HINT_UNKNOWN,
) -> Dict[str, Any]:
    """Chunk one normalized record into its output document.

    Args:
        record: the normalized record, with ``schema_version`` 1 already
            checked.
        settings: a settings mapping replacing ``config/settings.yaml``, read
            at call time; ignored when ``config`` is given.
        config: settings already resolved by :func:`chunking_settings`.
        structure_hint: reserved for a later structure detector
            (architecture decision 2); its only value is ``unknown``.

    Raises ``ValueError`` for any other structure hint (R5),
    :class:`NormalizedRecordError` when the record is not a valid normalized
    record, and :class:`ChunkingConfigError` for invalid settings. A failed
    document is chunked like any other; with no pages it has no chunks. Under
    the ``fixed_length`` strategy no run cuts the document (spec 002).
    """
    if structure_hint != STRUCTURE_HINT_UNKNOWN:
        raise ValueError(
            f"structure_hint is {structure_hint!r}; its only value is "
            f"{STRUCTURE_HINT_UNKNOWN!r}"
        )
    config = config if config is not None else chunking_settings(settings)
    validate_normalized_record(record)

    # Page order; the sort is stable, and normalize writes pages in order.
    pages = sorted(record["pages"], key=lambda page: page["page_number"])
    stream, placed = _place_pages(pages)
    lines = line_bounds(stream)
    candidates = find_heading_candidates(stream, lines, config.keyword_pattern)
    runs_by_family = {
        family: [
            [found[index] for index in run]
            for run in decompose_runs(
                [candidate.value for candidate in found],
                config.min_heading_run_length,
                config.heading_max_gap,
            )
        ]
        for family, found in candidates.items()
    }
    family = _choose_family(runs_by_family)
    runs = (
        _assess_runs(runs_by_family[family], len(stream), config)
        if family is not None
        else []
    )
    # The runs that cut the document: none under fixed_length (spec 002), where
    # the runs are evidence only.
    used = runs if config.strategy == STRATEGY_STRUCTURE_AWARE else []
    pieces = _Planner(stream, lines, config).plan(used)
    file_sha256 = record["file"]["sha256"]
    chunks = [
        _chunk_output(index, piece, stream, placed, file_sha256, config)
        for index, piece in enumerate(pieces)
    ]

    return {
        "schema_version": CHUNKS_SCHEMA_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "file": copy.deepcopy(record["file"]),
        "status": record["status"],
        "normalized_hash": record["normalized_sha256"],
        "language": _shared(pages, "language"),
        "language_source": _shared(pages, "language_source"),
        "structure_hint": structure_hint,
        "route": (
            ROUTE_STRUCTURED if any(run.accepted for run in used) else ROUTE_GENERIC
        ),
        "empty_pages": [page["page_number"] for page in pages if page["text"] == ""],
        "heading_evidence": {
            "family": family,
            "candidate_counts": {
                name: len(candidates[name]) for name in HEADING_FAMILIES
            },
            "runs": [_run_output(run, placed) for run in runs],
        },
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
