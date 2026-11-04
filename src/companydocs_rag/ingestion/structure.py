"""Formatting consistency (layer 2, part 2 -- quality criterion 3).

Each document is tested against a small set of GENERIC structural patterns and
the best match wins. There is deliberately no per-type rule set: maintaining
separate structural rules for lease agreements, articles of association, KvK
extracts and permits does not scale, and every new type would need its own.

The three patterns:

(a) ``numbered_clauses``   -- numbered articles and clauses, covering Arabic
                              ("Artikel 1", "1.1"), Roman ("Article I") and
                              lettered/nested subclauses ("(a)", "2.1(a)").
(b) ``field_label_pairs``  -- fixed label/value pairs, either colon-separated
                              ("KvK-nummer: 12345678") or column-aligned, as
                              KvK extracts are.
(c) ``headings``           -- recognisable section headings without numbering.

IMPORTANT -- what this actually measures
----------------------------------------
This is a PRESENCE check: does the document have recognisable structure
*anywhere*. That is a deliberate simplification of formatting *consistency* in
the fuller sense, which would mean internal consistency (does the numbering run
unbroken from article 1 to article 24, or does it collapse halfway through
where extraction failed) or deviation from sibling documents of the same type
(does this lease look structurally like the other forty leases).

The fuller version is deferred until several real documents per type can be
compared, because "consistent" is only definable against a reference set we do
not have yet. This proxy is an intentional interim measure, not the final
implementation of this criterion.

Like the type keyword lists, the pattern set below is a FIRST VERSION meant to
be refined once the real document subset can be inspected -- particularly the
heading heuristic, which is the softest of the three.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .extraction import PageText
from .settings import parse_pass_section

# --- Pattern defaults -------------------------------------------------------
# Overridable via ingestion.parse_pass.structure in settings.yaml.
# Hit counts giving a pattern full credit (score 1.0).
DEFAULT_SATURATION: Dict[str, int] = {
    "numbered_clauses": 6,
    "field_label_pairs": 8,
    "headings": 5,
}
# A pattern needs at least this many hits before it counts as present at all.
DEFAULT_MIN_HITS = 2

# (a) Numbered clauses and articles.
_NUMBERED_CLAUSE_PATTERNS = (
    # "Artikel 1", "Article 12", "Art. 3", "§ 5" -- Arabic top-level articles.
    re.compile(
        r"^\s*(?:artikel|article|art\.?|clausule|paragraaf|§)\s*\d+", re.IGNORECASE
    ),
    # "Artikel I", "Article IV" -- Roman top-level articles.
    re.compile(r"^\s*(?:artikel|article|art\.?)\s+[IVXLC]{1,6}\b", re.IGNORECASE),
    # "1.1", "2.3.4" followed by text -- decimal nested numbering.
    re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){1,3}\.?\s+\S"),
    # "1." / "12)" followed by text -- plain numbered items.
    re.compile(r"^\s*\d{1,2}[.)]\s+\S"),
    # "IV." / "VII)" -- bare Roman numerals as top-level items.
    re.compile(r"^\s*[IVXLC]{1,6}[.)]\s+\S"),
    # "(a)" / "a)" -- lettered subclauses at the start of a line.
    re.compile(r"^\s*\(?[a-z]\)\s+\S"),
    # "2.1(a)" -- lettered subclause nested inside decimal numbering.
    re.compile(r"\b\d{1,2}\.\d{1,2}\([a-z]\)"),
)

# (b) Fixed field-label/value pairs.
_FIELD_LABEL_PATTERNS = (
    # "Statutaire naam: Voorbeeld B.V." -- colon-separated label and value.
    re.compile(r"^\s*[^\W\d_][\w \-/()'.]{1,40}:\s+\S"),
    # KNOWN-BROKEN (spec 001), kept but marked rather than rewritten -- see
    # test_structure.py's skipped test for the failing case.
    #
    # This is meant to catch "Statutaire naam        Voorbeeld B.V.", a
    # column-aligned label/value pair with wide whitespace between them, the
    # way a two-column layout would visually appear on the page. It can never
    # match this codebase's extraction, though: extraction.py calls
    # `Page.extract_text()` with no kwargs, which pdfplumber routes to
    # `WordMap.to_textmap(layout=False)`, where `x_dist = 0`
    # (pdfplumber/utils/text.py:388). That makes `num_spaces_prepend` exactly
    # `min(1, line_len)` -- one space between words, zero at line start
    # (text.py:390) -- so the wide gaps this regex looks for are never
    # produced; end-of-line padding and `blank_line` are layout-only
    # (text.py:401-402, :323-325) and equally absent.
    #
    # The regex is not the real gap, in other words: the real gap is that a
    # two-column KvK row arrives as "KvK-nummer 12345678" -- no colon, a
    # single space -- so NEITHER pattern above fires on the most standardised
    # of the four document types. Rewriting this regex against a second
    # untested assumption (what wide-gap text would look like, if pdfplumber
    # ever produced it) is not an improvement while no real documents exist to
    # check it against. Left in place, known-broken, pending real documents.
    re.compile(r"^\s*[^\W\d_][\w \-/()'.]{1,40}?\s{3,}\S"),
)

# (c) Heading-driven sections without numbering.
# Headings are short, carry no terminal sentence punctuation, and are either
# all-caps or capitalised word by word.
_MAX_HEADING_CHARS = 60
_MAX_HEADING_WORDS = 8
_HEADING_TRAILING_PUNCTUATION = ".,;:"


@dataclass
class PatternMatch:
    """How well one structural pattern matched the document."""

    name: str
    hits: int  # lines matching this pattern
    score: float  # hits scaled against the pattern's saturation, capped at 1.0
    first_page: Optional[int] = None  # page where this pattern first appeared


@dataclass
class StructureResult:
    """Outcome of the formatting-consistency check for one document."""

    matched_pattern: Optional[str]  # winning pattern, None if none reached min_hits
    score: float  # winning pattern's score, 0.0 if none matched
    first_page: Optional[int]  # page where the winning pattern was first seen
    patterns: List[PatternMatch] = field(
        default_factory=list
    )  # all patterns, for recalibration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched_pattern": self.matched_pattern,
            "score": self.score,
            "first_page": self.first_page,
            # Kept in full so thresholds can be re-tuned from stored results
            # without reprocessing the documents.
            "patterns": [asdict(pattern) for pattern in self.patterns],
        }


def _matches_numbered_clause(line: str) -> bool:
    return any(pattern.search(line) for pattern in _NUMBERED_CLAUSE_PATTERNS)


def _matches_field_label(line: str) -> bool:
    return any(pattern.search(line) for pattern in _FIELD_LABEL_PATTERNS)


def _matches_heading(line: str) -> bool:
    """Recognise an unnumbered section heading.

    Numbered lines are excluded on purpose: pattern (c) is specifically for
    documents whose structure is carried by headings *instead of* numbering,
    so a numbered heading belongs to pattern (a), not here.
    """
    stripped = line.strip()
    if not (3 <= len(stripped) <= _MAX_HEADING_CHARS):
        return False
    if stripped[-1] in _HEADING_TRAILING_PUNCTUATION:
        return False
    if _matches_numbered_clause(stripped) or stripped[0].isdigit():
        return False

    words = stripped.split()
    if not words or len(words) > _MAX_HEADING_WORDS:
        return False
    if not any(char.isalpha() for char in stripped):
        return False

    letters = [char for char in stripped if char.isalpha()]
    if letters and all(char.isupper() for char in letters):
        return True  # ALL CAPS heading

    # Otherwise require every substantial word to be capitalised, which
    # distinguishes "Duur, Verlenging En Opzegging" from a wrapped sentence
    # fragment. Short words (articles, prepositions) are ignored.
    substantial = [word for word in words if len(word) > 3 and word[0].isalpha()]
    if not substantial:
        return False
    return all(word[0].isupper() for word in substantial)


_PATTERN_MATCHERS = {
    "numbered_clauses": _matches_numbered_clause,
    "field_label_pairs": _matches_field_label,
    "headings": _matches_heading,
}

# Each line is attributed to AT MOST ONE pattern: the most specific one that
# matches, in this order. Without this, the patterns overlap and the softest
# one wins on saturation alone -- a KvK line like "KvK-nummer: 12345678" is
# both a field-label pair and, structurally, a short capitalised line, and
# "Artikel 1    Definities" is both a numbered clause and a wide-gap label
# pair. Single attribution keeps the winning pattern an honest description of
# how the document is actually organised.
_PATTERN_ORDER = ("numbered_clauses", "field_label_pairs", "headings")


def analyse_structure(
    pages: Sequence[PageText],
    settings: Optional[Mapping[str, Any]] = None,
) -> StructureResult:
    """Test the document against each generic pattern and take the best match.

    Records the page on which the winning pattern was first detected, so the
    structural evidence is located inside the document rather than reduced to a
    match/no-match boolean.
    """
    section = parse_pass_section("structure", settings=settings)
    configured_saturation = section.get("saturation")
    saturation = dict(DEFAULT_SATURATION)
    if isinstance(configured_saturation, Mapping):
        for name, value in configured_saturation.items():
            if name in saturation:
                saturation[str(name)] = int(value)
    min_hits = int(section.get("min_hits", DEFAULT_MIN_HITS))

    hits: Dict[str, int] = {name: 0 for name in _PATTERN_MATCHERS}
    first_page: Dict[str, Optional[int]] = {name: None for name in _PATTERN_MATCHERS}

    for page in pages:
        for line in page.lines:
            for name in _PATTERN_ORDER:
                # One hit per line, credited to the most specific matching
                # pattern, so a single dense line cannot inflate several
                # counts at once.
                if _PATTERN_MATCHERS[name](line):
                    hits[name] += 1
                    if first_page[name] is None:
                        first_page[name] = page.page_number
                    break

    matches: List[PatternMatch] = []
    for name in _PATTERN_MATCHERS:
        limit = saturation.get(name) or 1
        score = min(1.0, hits[name] / limit) if hits[name] else 0.0
        matches.append(
            PatternMatch(
                name=name,
                hits=hits[name],
                score=round(score, 4),
                first_page=first_page[name],
            )
        )

    matches.sort(key=lambda match: (match.score, match.hits), reverse=True)
    best = matches[0]

    # Below min_hits there is no structural signal worth acting on. This is a
    # weak/no match, which pushes the document towards Decent -- it is never on
    # its own a reason to call a document Garbage; that decision rests on the
    # preocr extraction signals.
    if best.hits < min_hits:
        return StructureResult(
            matched_pattern=None,
            score=0.0,
            first_page=None,
            patterns=matches,
        )

    return StructureResult(
        matched_pattern=best.name,
        score=best.score,
        first_page=best.first_page,
        patterns=matches,
    )


# Human-readable names for the CSV explanation column.
PATTERN_DESCRIPTIONS = {
    "numbered_clauses": "article/clause numbering",
    "field_label_pairs": "fixed field-label pairs",
    "headings": "unnumbered section headings",
}


def describe_pattern(result: StructureResult) -> str:
    """Render the structural outcome as a short human-readable phrase."""
    if result.matched_pattern is None:
        return "no clear structural pattern"
    description = PATTERN_DESCRIPTIONS.get(
        result.matched_pattern, result.matched_pattern
    )
    return f"{description} detected starting page {result.first_page}"
