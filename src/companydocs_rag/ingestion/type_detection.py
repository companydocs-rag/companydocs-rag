"""Document type detection (layer 2, part 1).

Four known types -- lease/rental/hire agreements, articles of association,
Chamber of Commerce extracts and permits/licences -- recognised by matching
sourced keyword lists against the page text it is given. In the parse pass
that is the downstream view built by
:func:`companydocs_rag.ingestion.parse_pass.classify_and_assess`: per page, the
native text from :mod:`companydocs_rag.ingestion.extraction`, OCR text, or
nothing.

The keyword lists themselves live in ``config/settings.yaml``. They are a FIRST
VERSION derived from public sources (ROZ model lease agreements, BW Boek 2, the
KvK Handelsregister layout) and are explicitly meant to be refined through
error analysis once the real documents can be inspected. Nothing
here assumes they are complete or final.

Scoring combines two things, as required: how MANY type-specific terms matched
and how MANY DISTINCT ones did. A document matching five different lease terms
is a far stronger signal than one matching a single term five times, so
diversity carries the larger weight. Documents matching nothing well enough
fall back to "unknown/other" with low confidence -- never a forced guess.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .extraction import DocumentText, PageText
from .settings import parse_pass_section

# Label used when no type matches well enough. Downstream code should treat
# this as "route to generic handling", not as an error.
UNKNOWN_TYPE = "unknown/other"

# --- Scoring defaults -------------------------------------------------------
# Overridable via ingestion.parse_pass.type_detection in settings.yaml.
# These are first assumptions, not validated thresholds.
DEFAULT_DIVERSITY_SATURATION = 5  # distinct terms giving full diversity credit
DEFAULT_FREQUENCY_SATURATION = 20  # total occurrences giving full frequency credit
DEFAULT_DIVERSITY_WEIGHT = 0.70
DEFAULT_FREQUENCY_WEIGHT = 0.30
DEFAULT_STRONG_SIGNAL_DIVERSITY_UNITS = 2.0
DEFAULT_MIN_CONFIDENCE = 0.25
DEFAULT_MIN_DISTINCT_TERMS = 2
DEFAULT_MARGIN_FLOOR = 0.40


@dataclass(frozen=True)
class TypeEvidence:
    """One term matched on one page.

    Recorded per page rather than per document so the evidence is *located*
    inside the document and can be checked by eye during calibration.
    """

    term: str  # the configured keyword or the name of the regex signal
    language: str  # "nl", "en", "de"
    page: int  # 1-indexed page number where the match occurred
    occurrences: int  # times the term occurs on that page
    kind: str = "keyword"  # "keyword" or "regex"
    strong: bool = False  # counts extra towards diversity


@dataclass
class TypeScore:
    """Scoring breakdown for a single candidate type."""

    type_key: str
    label: str
    score: float
    diversity_component: float
    frequency_component: float
    distinct_terms: int
    distinct_units: float  # distinct terms, with strong signals weighted up
    total_occurrences: int
    evidence: List[TypeEvidence] = field(default_factory=list)


@dataclass
class TypeDetectionResult:
    """Outcome of type detection for one document."""

    detected_type: str
    label: str
    confidence: float
    evidence: List[TypeEvidence] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    runner_up: Optional[str] = None
    fallback_reason: Optional[str] = None  # why it fell back to unknown/other

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_type": self.detected_type,
            "label": self.label,
            "confidence": self.confidence,
            "evidence": [asdict(item) for item in self.evidence],
            "scores": self.scores,
            "runner_up": self.runner_up,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class _Matcher:
    """A compiled term matcher: either a keyword or a configured regex."""

    term: str
    language: str
    regex: re.Pattern[str]
    kind: str
    strong: bool


def _compile_keyword(keyword: str) -> re.Pattern[str]:
    """Compile a keyword into a case-insensitive whole-term matcher.

    Lookarounds are used rather than ``\\b`` so terms that begin or end with a
    non-word character (``KvK-nummer``, ``SBI-code``) still anchor correctly.
    Interior whitespace is made flexible even though matching runs against
    whitespace-collapsed text, so the matchers stay correct if a caller passes
    raw text instead.
    """
    parts = [re.escape(part) for part in keyword.split()]
    body = r"\s+".join(parts)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def _build_matchers(type_config: Mapping[str, Any]) -> List[_Matcher]:
    """Build the matcher list for one type from its configuration.

    Keywords from all configured languages are merged: a document is scored
    against the union of the Dutch, English and German variants, because type
    detection is not given the document's language -- the parse pass resolves
    it before detection runs but does not pass it on -- and a minority of
    documents are not in Dutch.
    """
    matchers: List[_Matcher] = []

    keywords = type_config.get("keywords") or {}
    if isinstance(keywords, Mapping):
        for language, terms in keywords.items():
            for term in terms or []:
                matchers.append(
                    _Matcher(
                        term=str(term),
                        language=str(language),
                        regex=_compile_keyword(str(term)),
                        kind="keyword",
                        strong=False,
                    )
                )

    for signal in type_config.get("regex_signals") or []:
        if not isinstance(signal, Mapping) or not signal.get("pattern"):
            continue
        matchers.append(
            _Matcher(
                term=str(signal.get("name") or signal["pattern"]),
                language=str(signal.get("language", "nl")),
                regex=re.compile(str(signal["pattern"]), re.IGNORECASE),
                kind="regex",
                strong=bool(signal.get("strong", False)),
            )
        )

    return matchers


def load_type_definitions(
    settings: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return the configured type definitions from settings.yaml.

    The keyword lists live only in the YAML file -- there is no Python copy to
    drift out of sync with it. An absent or empty configuration is not an
    error: detection then yields "unknown/other" for every document, with the
    missing configuration named as the reason.
    """
    section = parse_pass_section("type_detection", settings=settings)
    types = section.get("types")
    if not isinstance(types, Mapping):
        return {}
    return {
        str(key): dict(value)
        for key, value in types.items()
        if isinstance(value, Mapping)
    }


def _score_type(
    type_key: str,
    type_config: Mapping[str, Any],
    pages: Sequence[PageText],
    *,
    diversity_saturation: float,
    frequency_saturation: float,
    diversity_weight: float,
    frequency_weight: float,
    strong_units: float,
) -> TypeScore:
    """Score one candidate type against the document's pages."""
    matchers = _build_matchers(type_config)
    evidence: List[TypeEvidence] = []
    distinct_terms: set[str] = set()
    distinct_units = 0.0
    total_occurrences = 0

    for page in pages:
        haystack = page.flat_text
        if not haystack:
            continue
        for matcher in matchers:
            occurrences = len(matcher.regex.findall(haystack))
            if occurrences == 0:
                continue
            evidence.append(
                TypeEvidence(
                    term=matcher.term,
                    language=matcher.language,
                    page=page.page_number,
                    occurrences=occurrences,
                    kind=matcher.kind,
                    strong=matcher.strong,
                )
            )
            total_occurrences += occurrences
            if matcher.term not in distinct_terms:
                distinct_terms.add(matcher.term)
                # A term flagged `strong` (e.g. a "artikel 2:242" citation)
                # counts for more towards diversity than an ordinary keyword.
                distinct_units += strong_units if matcher.strong else 1.0

    diversity = (
        min(1.0, distinct_units / diversity_saturation)
        if diversity_saturation > 0
        else 0.0
    )
    if frequency_saturation > 0 and total_occurrences > 0:
        frequency = min(
            1.0, math.log1p(total_occurrences) / math.log1p(frequency_saturation)
        )
    else:
        frequency = 0.0

    score = diversity_weight * diversity + frequency_weight * frequency

    return TypeScore(
        type_key=type_key,
        label=str(type_config.get("label", type_key)),
        score=round(score, 4),
        diversity_component=round(diversity, 4),
        frequency_component=round(frequency, 4),
        distinct_terms=len(distinct_terms),
        distinct_units=round(distinct_units, 2),
        total_occurrences=total_occurrences,
        evidence=evidence,
    )


def detect_type(
    document: DocumentText,
    settings: Optional[Mapping[str, Any]] = None,
) -> TypeDetectionResult:
    """Detect the document type from already-extracted page text.

    Takes a :class:`DocumentText` rather than a path, so detection never reads
    the file itself. In the parse pass it gets the downstream view: the
    pdfplumber extraction that quality assessment reads, with OCR text on
    pages whose ``text_origin`` is ``ocr`` and no text on those whose
    ``text_origin`` is ``none`` (see
    :func:`companydocs_rag.ingestion.parse_pass.classify_and_assess`).
    """
    section = parse_pass_section("type_detection", settings=settings)
    diversity_saturation = float(
        section.get("diversity_saturation", DEFAULT_DIVERSITY_SATURATION)
    )
    frequency_saturation = float(
        section.get("frequency_saturation", DEFAULT_FREQUENCY_SATURATION)
    )
    diversity_weight = float(section.get("diversity_weight", DEFAULT_DIVERSITY_WEIGHT))
    frequency_weight = float(section.get("frequency_weight", DEFAULT_FREQUENCY_WEIGHT))
    strong_units = float(
        section.get(
            "strong_signal_diversity_units", DEFAULT_STRONG_SIGNAL_DIVERSITY_UNITS
        )
    )
    min_confidence = float(section.get("min_confidence", DEFAULT_MIN_CONFIDENCE))
    min_distinct_terms = int(
        section.get("min_distinct_terms", DEFAULT_MIN_DISTINCT_TERMS)
    )
    margin_floor = float(section.get("margin_floor", DEFAULT_MARGIN_FLOOR))

    definitions = load_type_definitions(settings=settings)
    if not definitions:
        return TypeDetectionResult(
            detected_type=UNKNOWN_TYPE,
            label=UNKNOWN_TYPE,
            confidence=0.0,
            fallback_reason="no document type definitions configured in settings.yaml",
        )

    scored = [
        _score_type(
            type_key,
            type_config,
            document.pages,
            diversity_saturation=diversity_saturation,
            frequency_saturation=frequency_saturation,
            diversity_weight=diversity_weight,
            frequency_weight=frequency_weight,
            strong_units=strong_units,
        )
        for type_key, type_config in definitions.items()
    ]
    scored.sort(key=lambda item: item.score, reverse=True)

    all_scores = {item.type_key: item.score for item in scored}
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None

    if best.score <= 0.0:
        return TypeDetectionResult(
            detected_type=UNKNOWN_TYPE,
            label=UNKNOWN_TYPE,
            confidence=0.0,
            evidence=[],
            scores=all_scores,
            runner_up=second.type_key if second else None,
            fallback_reason="no type-specific terms matched",
        )

    # Confidence is the winner's score, suppressed when a runner-up scores
    # almost as well: two types matching equally is genuinely ambiguous, and
    # articles of association and KvK extracts do share vocabulary.
    separation = (best.score - second.score) / best.score if second else 1.0
    separation = max(0.0, min(1.0, separation))
    confidence = best.score * (margin_floor + (1.0 - margin_floor) * separation)
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    fallback_reason = None
    if best.distinct_terms < min_distinct_terms:
        fallback_reason = (
            f"only {best.distinct_terms} distinct term(s) matched for "
            f"{best.type_key}, minimum is {min_distinct_terms}"
        )
    elif confidence < min_confidence:
        fallback_reason = (
            f"best candidate {best.type_key} scored {confidence:.2f}, "
            f"below the minimum confidence of {min_confidence:.2f}"
        )

    if fallback_reason is not None:
        # Keep the evidence that was found: even a rejected candidate tells
        # the analyst what weak signals the document did carry.
        return TypeDetectionResult(
            detected_type=UNKNOWN_TYPE,
            label=UNKNOWN_TYPE,
            confidence=confidence,
            evidence=best.evidence,
            scores=all_scores,
            runner_up=best.type_key,
            fallback_reason=fallback_reason,
        )

    return TypeDetectionResult(
        detected_type=best.type_key,
        label=best.label,
        confidence=confidence,
        evidence=best.evidence,
        scores=all_scores,
        runner_up=second.type_key if second and second.score > 0 else None,
    )


def summarise_evidence(evidence: Iterable[TypeEvidence], limit: int = 3) -> str:
    """Render a few evidence items as a short human-readable phrase."""
    items = sorted(evidence, key=lambda item: (item.page, item.term))
    if not items:
        return "no type keywords matched"
    shown = [f"'{item.term}' (p{item.page})" for item in items[:limit]]
    remainder = len(items) - len(shown)
    text = ", ".join(shown)
    if remainder > 0:
        text += f" +{remainder} more"
    return text
