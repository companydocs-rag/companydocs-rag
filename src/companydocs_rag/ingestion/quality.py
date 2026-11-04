"""Quality assessment: Clean / Decent / Garbage (criteria 1-3).

Criteria 1 and 2 -- text extraction quality and OCR artifacts -- are decided by
this module's own per-page signals: character count against
``MIN_CHARS_NATIVE_PAGE`` for the source axis (native vs. no text layer), and
each native page's own ``non_printable_ratio``, ``unicode_noise_ratio`` and
``avg_word_length`` for its Clean/Decent/Garbage tier. The underlying
per-page heuristics are computed with the ``preocr`` library's own text-quality
function (https://github.com/yuvaraj-kannan/preocr), re-run per page rather
than reimplemented, but preocr's own ``needs_ocr()`` verdict is recorded as
evidence only and never decides either axis -- see unvalidated assumption (2)
below.

Criterion 3 -- formatting consistency -- comes from
:mod:`companydocs_rag.ingestion.structure`, computed over the downstream text:
the same extraction, with OCR text for the pages that got it (spec 003).
It is recorded alongside the tier for the CSV explanation, as evidence, but
does not decide it: no tier is gated on structure at any level.

The chunker (:mod:`companydocs_rag.chunking`, spec 005) selects its route per
document from its own heading recognition; neither the quality tier nor the
structure score selects it, and it does not read the structure score. Each
page's tier travels with the chunks as provenance: structure decides where to
cut, the tier decides how much to trust the result (architecture decision 2
in docs/DEVELOPMENT.md).

Every signal feeding the tier decision is preserved in the result alongside the
label, so thresholds can be recalibrated from stored results without
reprocessing any document.

=============================================================================
UNVALIDATED ASSUMPTIONS -- read before trusting any number in this module
=============================================================================

(1) preocr's default thresholds have NOT been validated on Dutch legal and
    financial documents. They should not be treated as trustworthy out of the
    box. There is a known failure pattern from a comparable project: a
    text-quality ratio tuned on plain prose misfires on documents dominated by
    numbered clauses, monetary amounts and field-label formatting, because
    those reduce the proportion of "normal running text" even in a perfectly
    clean document. Our documents are structurally exactly that -- numbered
    articles, amounts, KvK field labels.

    Concretely, the signals most at risk here are ``avg_word_length`` (a KvK
    extract page is largely short labels, numbers and dates, which drags the
    average down without any extraction problem) and preocr's internal
    ``text_norm``, which saturates at 200 characters per document. Both need
    checking against real Clean-labelled examples before the constants below
    are treated as anything but placeholders.

(2) preocr's ``needs_ocr()`` is a scan-versus-digital GATE, not a text-quality
    grade, and it is much coarser than it looks. In preocr 1.11.0,
    ``core/detector.py`` applies a "hard digital guard": any PDF yielding at
    least ``HARD_DIGITAL_TEXT_THRESHOLD`` (20) characters returns
    ``needs_ocr=False`` at confidence 0.9 and forces every page to
    ``needs_ocr=False``, before any noise or layout analysis runs. A badly
    garbled 40-page scan that yields 25 characters of nonsense is therefore
    reported as a clean digital PDF.

    Since spec 001, preocr's verdict is recorded as evidence only -- in
    ``page_stats`` (``preocr_needs_ocr``, ``preocr_confidence``,
    ``preocr_confidence_band``) and the raw ``preocr`` block -- and is never
    branched on. The source axis (native vs no_text_layer, below) is decided
    from this module's own per-page character count, and the Decent/Garbage
    separation from this module's own per-page noise statistics, both
    independently of preocr's verdict. A badly garbled scan that preocr
    reports as a clean digital PDF is therefore still caught, if at all, by
    those per-page signals alone.

(3) preocr's own ``non_printable_ratio`` and ``unicode_noise_ratio`` in its
    ``signals`` block are computed over a TRUNCATED sample: ``probes/
    pdf_probe.py`` returns only the first 1000 characters of the document
    (roughly one page). For a 30-page lease that is page 1 only. This module
    therefore re-runs preocr's own text-quality function over each page of the
    full extraction to get per-page ratios. The function is preocr's, not a
    reimplementation; only the input sample differs.

(4) The low/zero-character page -- resolved by spec 001's Axis 2, see
    MIN_CHARS_NATIVE_PAGE below: such a page is routed to OCR (`no_text_layer`)
    rather than guessed at from the text alone, and image coverage is
    recorded but deliberately never decides it. Still unvalidated against
    real documents, in the sense that MIN_CHARS_NATIVE_PAGE's value is a
    first assumption, not that the routing decision itself is in question.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

from .extraction import DocumentText
from .settings import parse_pass_section
from .structure import StructureResult

if TYPE_CHECKING:
    # Annotation only: ocr.py imports this module, not the other way round.
    from .ocr import OcrOutcome

# preocr (and the python-magic it pulls in) is imported lazily, inside the
# functions that actually need it (`run_preocr`, `_page_quality`,
# `ocr_text_signals`), rather than
# here at module scope. Importing python-magic can hang indefinitely on
# Windows when Git's own msys-magic-1.dll shadows the one python-magic
# expects, because a directory containing it (Git's usr/bin) is on PATH.
# Deferring the import means merely importing this module -- or the
# `companydocs_rag.ingestion` package -- never risks that hang; only actually
# running the parse pass does.
logger = logging.getLogger(__name__)

# Quality tier labels. These are the values written to the CSV report and
# consumed by the chunking step.
TIER_CLEAN = "Clean"
TIER_DECENT = "Decent"
TIER_GARBAGE = "Garbage"

# =============================================================================
# TIER DETERMINATION CONSTANTS
#
# Two independent per-page decisions, not one document-wide ladder (spec 001):
#
#   Axis 2 (source, per page)       -- `native` vs `no_text_layer`, from
#                                       character count alone. See
#                                       MIN_CHARS_NATIVE_PAGE below.
#   Axis 3 (quality tier, per page) -- Clean / Decent / Garbage, computed only
#                                       for `native` pages, from that page's
#                                       own noise and word-length signals.
#                                       A `no_text_layer` page has no tier:
#                                       its text, if any, comes from OCR,
#                                       which is never tiered (spec 003).
#
# The document-level label in the CSV is a ROLL-UP over per-page tiers
# (MAX_GARBAGE_PAGE_RATIO_DOCUMENT below), not a decision in its own right:
# routing happens per page (the `pages` list, each carrying its own `source`
# and `tier`), and the document label exists for the CSV and for a human
# skimming a few hundred rows. Worst-tier-wins is rejected -- see that
# constant for why.
#
# A file that failed to open at all, or that yielded zero pages, is not
# assessed here at all -- it is `status=failed` (Axis 1, see parse_pass.py),
# never a quality tier.
#
# Every value below is overridable via ingestion.parse_pass.quality in
# config/settings.yaml, so it can be tuned once the real document set is
# available without touching this code. All of them are FIRST ASSUMPTIONS, not
# validated thresholds.
# =============================================================================

# --- Axis 2 (source, per page) ----------------------------------------------
# A page whose extracted character count is BELOW this many characters is
# `no_text_layer` -- it carries no reliable text layer and is routed towards
# OCR (spec 003; see the routing in ocr.py). A page AT OR ABOVE this
# threshold is `native`. The comparison is strictly
# less-than, deliberately: a page sitting exactly on the threshold is
# `native`, the same choice as the roll-up comparison below and for the same
# reason -- state it, don't let it slip.
#
# Character count is the ONLY deciding signal on this axis. Image coverage
# (see extraction.py's per-page image geometry) is computed and recorded per
# page alongside this, but decides nothing: there is no calibration data yet
# to fix a coverage threshold against, the cost of the two kinds of mistake is
# asymmetric (an occasional blank page routed to OCR costs a fraction of a
# run; missed content is irreversible), and on this corpus a native title page
# can legitimately carry heavy image coverage (letterheads), so the signal
# does not cleanly separate the case it would be used for. See spec
# 001's Approach for the full reasoning, including the two rejected
# font-based alternatives.
#
# ASSUMPTION (known failure mode from a comparable project): a page this thin
# carries too little text to say anything about extraction quality either --
# signature pages, stamp pages, attachment cover sheets and near-blank
# backsides of KvK extracts all land here. preocr's own per-page verdict is
# literally `page_text_length < MIN_TEXT_LENGTH` (50) -- see
# analysis/page_detection.py -- and its document-level aggregate used to be
# `overall_needs_ocr = pages_needing_ocr > 0`, so a single blank signature
# page was enough to flip an otherwise perfect 30-page lease to "needs OCR"
# under the old single-label design. Deliberately aligned with preocr's own
# value of 50. This axis is what actually fixes that inflated-flagging
# failure mode now: a thin page is routed to OCR on its own, individually,
# rather than folded into a document-wide ratio.
#
# The zero-character page is genuinely ambiguous -- it may be an unscanned
# blank backside, or a scanned page whose text layer is missing and which
# really does need OCR. Nothing in the text alone can tell these apart; both
# are `no_text_layer` here (see unvalidated assumption (4) at the top of this
# module). Verify the threshold value against real documents before relying
# on it.
MIN_CHARS_NATIVE_PAGE = 50

# --- Axis 3 (quality tier, per page) ----------------------------------------
# Applied to ONE page's own signals, not a document-wide average -- see
# _tier_for_page. A page only reaches this axis if Axis 2 already placed it as
# `native` (char_count >= MIN_CHARS_NATIVE_PAGE), so these thresholds are only
# ever compared against a page that already carries a substantive amount of
# text.

# Mean of preocr's non_printable_ratio and unicode_noise_ratio, for this page.
MAX_NOISE_RATIO_CLEAN = 0.02
MAX_NOISE_RATIO_GARBAGE = 0.10

# Characters on this page expected of a cleanly extracted text-layer page.
MIN_CHARS_PER_SUBSTANTIVE_PAGE_CLEAN = 250

# preocr's avg_word_length. Garbled extractions collapse into 1-2 character
# fragments or fuse words together into very long ones.
# SUSPECT ON OUR CORPUS -- see unvalidated assumption (1) at the top of this
# module: KvK extracts and clause-numbered contracts legitimately depress this
# average. The lower bound is set permissively for that reason.
MIN_AVG_WORD_LENGTH_CLEAN = 3.0
MAX_AVG_WORD_LENGTH_CLEAN = 12.0

# --- Roll-up: per-page tiers -> one document reporting label ----------------
# Fraction of TIERED pages (source == native) that may be Garbage before the
# document as a whole is reported Garbage. Below it, the document is Decent
# and the offending pages are named in the reasons -- worst-tier-wins is
# rejected: one noisy signature page must not drag an otherwise clean 20-page
# lease to Garbage, which is precisely the problem spec 001 exists to fix.
#
# The comparison is strictly greater-than: a ratio exactly equal to this
# value is NOT Garbage. This differs DELIBERATELY from the (removed)
# sparse-page ratio's >=; do not carry that comparison across.
#
# Starting value 0.50 is explicitly provisional, pending calibration against
# the real document subset -- and lenient: nine garbled pages out of twenty
# still report Decent. It is NOT justified by the numeric coincidence with the
# old (removed) sparse-page ratio's own 0.50: that rule counted sparse pages,
# this one counts Garbage-TIERED pages, and thin/sparse pages now become
# `no_text_layer` and are excluded from this roll-up altogether. The two ran
# over different populations; the parallel was cosmetic and carried no
# evidence about the right value here.
MAX_GARBAGE_PAGE_RATIO_DOCUMENT = 0.50

# preocr's confidence bands (constants.HIGH/MEDIUM/LOW_CONFIDENCE), used to
# label how much weight its verdict deserves.
PREOCR_HIGH_CONFIDENCE = 0.9
PREOCR_MEDIUM_CONFIDENCE = 0.7
PREOCR_LOW_CONFIDENCE = 0.5

# Axis 2 (source, per page) values.
SOURCE_NATIVE = "native"
SOURCE_NO_TEXT_LAYER = "no_text_layer"

# Where a page's downstream text comes from (spec 003), beside the unchanged
# source: the pdfplumber text, OCR text, or none at all. Source is decided
# before OCR runs and never from OCR text.
TEXT_ORIGIN_NATIVE = "native"
TEXT_ORIGIN_OCR = "ocr"
TEXT_ORIGIN_NONE = "none"

# Why a page has `text_origin` none, in this order: its extraction failed and
# it was not routed to OCR; it was not routed to OCR for another reason (only
# on a direct `assess_quality` call, which runs no OCR); OCR failed; OCR
# returned whitespace only.
NO_TEXT_REASON_EXTRACTION_FAILED = "extraction_failed"
NO_TEXT_REASON_OCR_SKIPPED = "ocr_skipped"
NO_TEXT_REASON_OCR_ERROR = "ocr_error"
NO_TEXT_REASON_OCR_EMPTY = "ocr_empty"
NO_TEXT_REASONS = (
    NO_TEXT_REASON_EXTRACTION_FAILED,
    NO_TEXT_REASON_OCR_SKIPPED,
    NO_TEXT_REASON_OCR_ERROR,
    NO_TEXT_REASON_OCR_EMPTY,
)

# The `awaiting_ocr` marker (document level), computed from `text_origin`
# (spec 003): "all" when every page is still without text (`text_origin`
# none), "partial" when some are, `None` when none are. It is independent of
# the label: the label is None exactly when the document has no native pages,
# and a `status=failed` document (Axis 1) carries neither.
AWAITING_OCR_ALL = "all"
AWAITING_OCR_PARTIAL = "partial"


@dataclass
class PageQuality:
    """Per-page extraction statistics, from preocr's own heuristics.

    ``source`` and ``tier`` are the two page-level axes of spec 001: source is
    decided by character count alone (see MIN_CHARS_NATIVE_PAGE); tier is
    computed only for pages carrying a text layer (source == SOURCE_NATIVE)
    and stays None on a ``no_text_layer`` page, whose text, if any, comes from
    OCR and is never tiered.

    These fields, and the image evidence, describe the pdfplumber extraction.
    ``text_origin`` and the ``ocr_*`` fields (spec 003) record where the
    downstream text comes from and the OCR evidence behind it; they decide no
    tier and no source.
    """

    page_number: int
    char_count: int
    source: str = SOURCE_NO_TEXT_LAYER  # SOURCE_NATIVE | SOURCE_NO_TEXT_LAYER
    tier: Optional[str] = None  # Clean / Decent / Garbage; None if not tiered
    is_empty: bool = False  # no extractable text at all
    extraction_failed: bool = False  # the page raised during extraction (extraction.py)
    non_printable_ratio: Optional[float] = None
    unicode_noise_ratio: Optional[float] = None
    avg_word_length: Optional[float] = None
    # Image coverage/count, carried through from extraction.py. Recorded as
    # evidence only -- see MIN_CHARS_NATIVE_PAGE above for why it never
    # decides the source axis.
    image_area_ratio: Optional[float] = None
    image_count: int = 0
    # --- Text origin and OCR evidence (spec 003) ---------------------------
    text_origin: Optional[str] = None  # TEXT_ORIGIN_NATIVE | _OCR | _NONE
    no_text_reason: Optional[str] = None  # a NO_TEXT_REASONS value; None unless none
    # PageText.extraction_error, verbatim (also carries image-geometry errors).
    extraction_error: Optional[str] = None
    ocr_routed: bool = False  # the routing decision (ocr.py)
    ocr_error: Optional[str] = None  # message of a failed OCR attempt, bounded
    ocr_mean_confidence: Optional[float] = None  # 0-100, from Tesseract's TSV
    # preocr's three text-quality signals on the OCR text. Evidence only.
    ocr_non_printable_ratio: Optional[float] = None
    ocr_unicode_noise_ratio: Optional[float] = None
    ocr_avg_word_length: Optional[float] = None
    language: Optional[str] = None  # the document's (language.py)
    language_source: Optional[str] = None


@dataclass
class QualityAssessment:
    """Quality tier (a roll-up over per-page tiers) plus every signal behind it.

    ``label`` is None exactly when the document has no native pages, so no
    page carries a tier at all -- whether or not its pages got OCR text; it is
    never ``Garbage`` for that reason. This is a reporting label only: the
    per-page ``source``, ``tier`` and ``text_origin`` in ``pages`` are the
    actual routing input.
    """

    label: Optional[str]  # Clean / Decent / Garbage, or None (see above)
    awaiting_ocr: Optional[str] = None  # AWAITING_OCR_ALL | AWAITING_OCR_PARTIAL | None
    reasons: List[str] = field(default_factory=list)
    preocr: Dict[str, Any] = field(default_factory=dict)  # raw preocr output
    pages: List[PageQuality] = field(default_factory=list)
    page_stats: Dict[str, Any] = field(default_factory=dict)
    structure: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)  # overrides and caveats applied

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "awaiting_ocr": self.awaiting_ocr,
            "reasons": self.reasons,
            # Raw preocr output is kept whole, including its per-page
            # classification, so nothing is lost for later recalibration.
            # Evidence only: no code path here branches on it.
            "preocr": self.preocr,
            "pages": [asdict(page) for page in self.pages],
            "page_stats": self.page_stats,
            "structure": self.structure,
            "notes": self.notes,
        }


def _thresholds(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
    """Resolve tier thresholds, YAML overriding the module constants."""
    section = parse_pass_section("quality", settings=settings)
    return {
        "min_chars_native_page": float(
            section.get("min_chars_native_page", MIN_CHARS_NATIVE_PAGE)
        ),
        "max_noise_ratio_clean": float(
            section.get("max_noise_ratio_clean", MAX_NOISE_RATIO_CLEAN)
        ),
        "max_noise_ratio_garbage": float(
            section.get("max_noise_ratio_garbage", MAX_NOISE_RATIO_GARBAGE)
        ),
        "min_chars_per_substantive_page_clean": float(
            section.get(
                "min_chars_per_substantive_page_clean",
                MIN_CHARS_PER_SUBSTANTIVE_PAGE_CLEAN,
            )
        ),
        "min_avg_word_length_clean": float(
            section.get("min_avg_word_length_clean", MIN_AVG_WORD_LENGTH_CLEAN)
        ),
        "max_avg_word_length_clean": float(
            section.get("max_avg_word_length_clean", MAX_AVG_WORD_LENGTH_CLEAN)
        ),
        "max_garbage_page_ratio_document": float(
            section.get(
                "max_garbage_page_ratio_document", MAX_GARBAGE_PAGE_RATIO_DOCUMENT
            )
        ),
    }


def preocr_confidence_band(confidence: Optional[float]) -> str:
    """Name the preocr confidence band a score falls into."""
    if confidence is None:
        return "unknown"
    if confidence >= PREOCR_HIGH_CONFIDENCE:
        return "high"
    if confidence >= PREOCR_MEDIUM_CONFIDENCE:
        return "medium"
    if confidence >= PREOCR_LOW_CONFIDENCE:
        return "low"
    return "very_low"


def run_preocr(pdf_path: Union[str, Path]) -> Dict[str, Any]:
    """Run preocr's OCR-need detection with page-level output.

    ``page_level=True`` is requested so preocr's own per-page classification is
    preserved in the result. Never raises: a failure is returned as an ``error``
    key so a folder run does not stop on one bad file.
    """
    try:
        # Lazy import -- see the note at the top of this module. Imported
        # inside the try so an ImportError (preocr missing or broken) is
        # caught by the same handler as any other preocr failure, matching
        # this function's "never raises" contract.
        from preocr import needs_ocr as preocr_needs_ocr

        return dict(preocr_needs_ocr(str(pdf_path), page_level=True))
    except Exception as error:  # noqa: BLE001 - preocr raises varied errors per file type
        logger.error(
            "preocr failed for %s: %s: %s", pdf_path, type(error).__name__, error
        )
        return {"error": f"{type(error).__name__}: {error}"}


def _page_quality(
    document: DocumentText, min_chars_native_page: float
) -> List[PageQuality]:
    """Apply preocr's text-quality heuristics to each page, and decide Axis 2.

    Uses preocr's own ``_compute_text_quality_signals`` rather than a local
    copy, so the ratios match the ones preocr uses internally. The difference
    is the input: full page text instead of the first 1000 characters of the
    document (unvalidated assumption (3) at the top of this module).

    Source (Axis 2) is decided here, from character count alone: a page below
    ``min_chars_native_page`` is ``no_text_layer``, strictly less-than -- see
    MIN_CHARS_NATIVE_PAGE. Per-page quality tier (Axis 3) is computed
    separately, only for ``native`` pages.
    """
    # Lazy import -- see the note at the top of this module. preocr exposes
    # its per-page text-quality heuristics only through this private helper.
    # It is imported rather than copied so the noise ratios stay identical to
    # the ones preocr uses for its own decisions; if a future preocr release
    # renames it, this raises immediately and loudly at call time rather than
    # silently diverging. What actually pins this dependency: `preocr==1.11.0`
    # in pyproject.toml (reflected in uv.lock), plus
    # tests/ingestion/test_preocr_contract.py, which pins this function's
    # name, its exact output keys and values, and its 10-character cut-off.
    # An upgrade means bumping that pin and deliberately re-deriving the
    # contract test's values -- never carrying the old numbers forward.
    from preocr.core import signals as preocr_signals

    _compute_text_quality_signals = preocr_signals._compute_text_quality_signals

    pages: List[PageQuality] = []
    for page in document.pages:
        char_count = page.char_count
        quality = _compute_text_quality_signals(page.text) or {}
        source = (
            SOURCE_NATIVE
            if char_count >= min_chars_native_page
            else SOURCE_NO_TEXT_LAYER
        )
        pages.append(
            PageQuality(
                page_number=page.page_number,
                char_count=char_count,
                source=source,
                is_empty=char_count == 0,
                extraction_failed=page.extraction_failed,
                extraction_error=page.extraction_error,
                non_printable_ratio=quality.get("non_printable_ratio"),
                unicode_noise_ratio=quality.get("unicode_noise_ratio"),
                avg_word_length=quality.get("avg_word_length"),
                image_area_ratio=page.image_area_ratio,
                image_count=page.image_count,
            )
        )
    return pages


def ocr_text_signals(text: str) -> Dict[str, Optional[float]]:
    """preocr's three text-quality signals on OCR text (spec 003), as evidence.

    The same private preocr function, behind the same lazy import, as
    :func:`_page_quality` -- see the comment there. Nothing decides on these
    values: OCR text gets no tier.
    """
    from preocr.core import signals as preocr_signals

    quality = preocr_signals._compute_text_quality_signals(text) or {}
    return {
        "non_printable_ratio": quality.get("non_printable_ratio"),
        "unicode_noise_ratio": quality.get("unicode_noise_ratio"),
        "avg_word_length": quality.get("avg_word_length"),
    }


def resolve_text_origins(
    pages: Sequence[PageQuality],
    document: DocumentText,
    ocr_outcomes: Optional[Mapping[int, "OcrOutcome"]] = None,
) -> List[str]:
    """Set ``text_origin``, the no-text reason and OCR evidence on every page.

    Returns each page's downstream text: the pdfplumber text for a
    ``native`` origin, the OCR text for ``ocr``, and "" for ``none``. Neither
    ``source`` nor ``tier`` changes here: both were decided from the
    pdfplumber extraction before OCR ran, and OCR text is never tiered.

    A routed page's OCR outcome is "text" when the OCR text is non-empty after
    stripping whitespace, "empty" when it is whitespace only, and "error" when
    OCR failed. A routed native page (tiered Garbage, routed by
    ``ocr_garbage_tier_pages``) keeps its native text on an empty or failed
    OCR outcome. A page that was not routed is ``native`` if its source is,
    and otherwise ``none``: ``extraction_failed`` if its extraction failed,
    else ``ocr_skipped``. With ``ocr_outcomes`` None -- a direct
    :func:`assess_quality` call, which runs no routing and no OCR -- no page
    is routed.
    """
    texts: List[str] = []
    for page, page_text in zip(pages, document.pages):
        native_text = page_text.text
        page.text_origin = None
        page.no_text_reason = None

        if not page.ocr_routed:
            if page.source == SOURCE_NATIVE:
                page.text_origin = TEXT_ORIGIN_NATIVE
                texts.append(native_text)
            else:
                page.text_origin = TEXT_ORIGIN_NONE
                page.no_text_reason = (
                    NO_TEXT_REASON_EXTRACTION_FAILED
                    if page.extraction_failed
                    else NO_TEXT_REASON_OCR_SKIPPED
                )
                texts.append("")
            continue

        outcome = (ocr_outcomes or {}).get(page.page_number)
        if outcome is None:
            raise ValueError(
                f"page {page.page_number} was routed to OCR but has no OCR outcome"
            )
        page.ocr_error = outcome.error
        has_text = (
            outcome.error is None
            and outcome.text is not None
            and outcome.text.strip() != ""
        )
        if has_text:
            signals = ocr_text_signals(outcome.text)
            page.text_origin = TEXT_ORIGIN_OCR
            page.ocr_mean_confidence = outcome.mean_confidence
            page.ocr_non_printable_ratio = signals["non_printable_ratio"]
            page.ocr_unicode_noise_ratio = signals["unicode_noise_ratio"]
            page.ocr_avg_word_length = signals["avg_word_length"]
            texts.append(outcome.text)
        elif page.source == SOURCE_NATIVE:
            page.text_origin = TEXT_ORIGIN_NATIVE
            texts.append(native_text)
        else:
            page.text_origin = TEXT_ORIGIN_NONE
            page.no_text_reason = (
                NO_TEXT_REASON_OCR_ERROR
                if outcome.error is not None
                else NO_TEXT_REASON_OCR_EMPTY
            )
            texts.append("")
    return texts


def _tier_for_page(page: PageQuality, limits: Mapping[str, float]) -> str:
    """Axis 3: this page's own quality tier, from its own signals alone.

    Only ever called for a ``native`` page (source is Axis 2's decision, made
    before this runs). Reuses the same named thresholds the document-level
    ladder used to apply as an average -- MAX_NOISE_RATIO_*,
    MIN_CHARS_PER_SUBSTANTIVE_PAGE_CLEAN, MIN/MAX_AVG_WORD_LENGTH_CLEAN --
    now applied to one page's own char count and noise ratios. Source must not
    determine quality: nothing about how this page came to be ``native``
    (image coverage, page position) is used here.
    """
    noise_values = [
        value
        for value in (page.non_printable_ratio, page.unicode_noise_ratio)
        if value is not None
    ]
    noise = fmean(noise_values) if noise_values else None

    if noise is not None and noise > limits["max_noise_ratio_garbage"]:
        return TIER_GARBAGE

    is_clean = (
        (noise is None or noise <= limits["max_noise_ratio_clean"])
        and page.char_count >= limits["min_chars_per_substantive_page_clean"]
        and (
            page.avg_word_length is None
            or limits["min_avg_word_length_clean"]
            <= page.avg_word_length
            <= limits["max_avg_word_length_clean"]
        )
    )
    return TIER_CLEAN if is_clean else TIER_DECENT


def _page_numbers(pages: Sequence[PageQuality]) -> List[int]:
    return [page.page_number for page in pages]


def _no_native_pages_reason(
    page_count: int,
    ocr_pages: Sequence[PageQuality],
    no_text_pages: Sequence[PageQuality],
) -> str:
    """The label-None reason: the OCR'd pages, then the pages still without
    text, grouped by no-text reason in NO_TEXT_REASONS order.

    "Awaiting OCR" is said of ``ocr_skipped`` pages only (spec 004, D3); the
    other reasons say why the page has no text.
    """
    if (
        page_count
        and len(no_text_pages) == page_count
        and all(
            page.no_text_reason == NO_TEXT_REASON_OCR_SKIPPED for page in no_text_pages
        )
    ):
        return "document has no pages with a text layer; entirely awaiting OCR"
    wording = {
        NO_TEXT_REASON_EXTRACTION_FAILED: (
            "have no text because text extraction failed and they were not sent to OCR"
        ),
        NO_TEXT_REASON_OCR_SKIPPED: "still have no text and are awaiting OCR",
        NO_TEXT_REASON_OCR_ERROR: "have no text because OCR failed",
        NO_TEXT_REASON_OCR_EMPTY: "have no text because OCR ran and returned no text",
    }
    parts = ["document has no pages with a text layer"]
    if ocr_pages:
        parts.append(
            f"{len(ocr_pages)} page(s) carry OCR text "
            f"(page(s) {_page_numbers(ocr_pages)})"
        )
    for reason in NO_TEXT_REASONS:
        pages = [page for page in no_text_pages if page.no_text_reason == reason]
        if pages:
            parts.append(
                f"{len(pages)} page(s) {wording[reason]} "
                f"(page(s) {_page_numbers(pages)})"
            )
    return "; ".join(parts)


def _no_text_layer_reason(
    page_count: int, no_text_layer_pages: Sequence[PageQuality]
) -> str:
    """The ladder's no-text-layer reason: the OCR'd pages, then the rest
    grouped by no-text reason in NO_TEXT_REASONS order.

    "Awaiting OCR" is said of ``ocr_skipped`` pages only (spec 004, D3); the
    other reasons say why the page has no text.
    """
    ocr_pages = [
        page for page in no_text_layer_pages if page.text_origin == TEXT_ORIGIN_OCR
    ]
    without_text = [
        page for page in no_text_layer_pages if page.text_origin != TEXT_ORIGIN_OCR
    ]
    head = f"{len(no_text_layer_pages)} of {page_count} page(s) have no text layer"
    if not without_text:
        return f"{head} and carry OCR text (page(s) {_page_numbers(ocr_pages)})"
    if not ocr_pages and all(
        page.no_text_reason == NO_TEXT_REASON_OCR_SKIPPED for page in without_text
    ):
        return f"{head} and are awaiting OCR (page(s) {_page_numbers(without_text)})"
    wording = {
        NO_TEXT_REASON_EXTRACTION_FAILED: (
            "have no text because text extraction failed and they were not sent to OCR"
        ),
        NO_TEXT_REASON_OCR_SKIPPED: "are awaiting OCR",
        NO_TEXT_REASON_OCR_ERROR: "have no text because OCR failed",
        NO_TEXT_REASON_OCR_EMPTY: "have no text because OCR ran and returned no text",
    }
    clauses = []
    if ocr_pages:
        clauses.append(
            f"{len(ocr_pages)} carry OCR text (page(s) {_page_numbers(ocr_pages)})"
        )
    for reason in NO_TEXT_REASONS:
        pages = [page for page in without_text if page.no_text_reason == reason]
        if pages:
            clauses.append(
                f"{len(pages)} {wording[reason]} (page(s) {_page_numbers(pages)})"
            )
    return f"{head}: " + ", ".join(clauses)


def _roll_up(pages: List[PageQuality], limits: Mapping[str, float]) -> Dict[str, Any]:
    """Roll per-page tiers up into one document reporting label.

    Worst-tier-wins is rejected (see MAX_GARBAGE_PAGE_RATIO_DOCUMENT): one
    noisy page must not sink an otherwise good document. The roll-up runs over
    TIERED pages only (source == native); a ``no_text_layer`` page is never
    scored, and still rules out Clean whether or not it got OCR text.
    ``awaiting_ocr`` counts the pages with ``text_origin`` none, whatever
    their source.

    Returns a dict with the resolved ``label`` (``None`` exactly when the
    document has no native pages, so no page carries a tier at all),
    ``awaiting_ocr``, ``reasons`` and the full ``page_stats`` aggregate -- the
    per-page evidence this decision rests on, kept so it can be recalibrated
    later.
    """
    page_count = len(pages)
    tiered = [page for page in pages if page.source == SOURCE_NATIVE]
    no_text_layer_pages = [
        page for page in pages if page.source == SOURCE_NO_TEXT_LAYER
    ]
    failed_pages = [page for page in pages if page.extraction_failed]
    ocr_pages = [page for page in pages if page.text_origin == TEXT_ORIGIN_OCR]
    no_text_pages = [page for page in pages if page.text_origin == TEXT_ORIGIN_NONE]

    clean_pages = [page for page in tiered if page.tier == TIER_CLEAN]
    decent_pages = [page for page in tiered if page.tier == TIER_DECENT]
    garbage_pages = [page for page in tiered if page.tier == TIER_GARBAGE]

    if page_count and len(no_text_pages) == page_count:
        awaiting_ocr = AWAITING_OCR_ALL
    elif no_text_pages:
        awaiting_ocr = AWAITING_OCR_PARTIAL
    else:
        awaiting_ocr = None

    noise_values = [
        ((page.non_printable_ratio or 0.0) + (page.unicode_noise_ratio or 0.0)) / 2.0
        for page in tiered
    ]
    word_lengths = [page.avg_word_length for page in tiered if page.avg_word_length]

    stats: Dict[str, Any] = {
        "page_count": page_count,
        "native_page_count": len(tiered),
        "no_text_layer_page_count": len(no_text_layer_pages),
        "no_text_layer_page_numbers": [
            page.page_number for page in no_text_layer_pages
        ],
        "empty_page_count": sum(1 for page in pages if page.is_empty),
        # Pages that raised during extraction (extraction.py). These are also
        # `no_text_layer` (a failed page carries no text), but are tracked
        # separately so a genuine extraction failure is never conflated with
        # a page that merely has no text layer.
        "failed_page_count": len(failed_pages),
        "failed_page_numbers": [page.page_number for page in failed_pages],
        "clean_page_count": len(clean_pages),
        "clean_page_numbers": [page.page_number for page in clean_pages],
        "decent_page_count": len(decent_pages),
        "decent_page_numbers": [page.page_number for page in decent_pages],
        "garbage_page_count": len(garbage_pages),
        "garbage_page_numbers": [page.page_number for page in garbage_pages],
        "garbage_page_ratio": (
            round(len(garbage_pages) / len(tiered), 4) if tiered else None
        ),
        "mean_noise_ratio": round(fmean(noise_values), 5) if noise_values else None,
        "mean_avg_word_length": round(fmean(word_lengths), 2) if word_lengths else None,
        "chars_per_native_page": (
            round(sum(page.char_count for page in tiered) / len(tiered), 1)
            if tiered
            else 0.0
        ),
        # By text_origin (spec 003): pages whose downstream text is OCR text,
        # and pages still without any text, with the reason per page.
        "ocr_page_count": len(ocr_pages),
        "ocr_page_numbers": [page.page_number for page in ocr_pages],
        "no_text_page_count": len(no_text_pages),
        "no_text_page_numbers": [page.page_number for page in no_text_pages],
        "no_text_reason_counts": {
            reason: sum(1 for page in no_text_pages if page.no_text_reason == reason)
            for reason in NO_TEXT_REASONS
        },
    }

    reasons: List[str] = []

    # A page that failed to extract at all carries no reliable quality
    # signals. Named explicitly so it is never mistaken for a page that
    # merely has no text layer, and so the document can never be tiered Clean
    # on its account -- see the Clean branch below.
    if failed_pages:
        reasons.append(
            f"text extraction failed on page(s) {stats['failed_page_numbers']}; "
            "those pages carry no quality signals"
        )

    if not tiered:
        # The fully scanned document: not a corner case on this corpus (spec
        # 001's Approach). status is still `processed` -- the file opened and
        # every page was read -- but there is nothing to roll up into a tier,
        # OCR text or not (OCR text is never tiered). The reason names OCR'd
        # pages and pages still without text separately.
        reasons.append(_no_native_pages_reason(page_count, ocr_pages, no_text_pages))
        return {
            "label": None,
            "awaiting_ocr": awaiting_ocr,
            "reasons": reasons,
            "page_stats": stats,
        }

    garbage_ratio = stats["garbage_page_ratio"]

    # The comparison is strictly greater-than: a ratio exactly equal to the
    # threshold is NOT Garbage. Deliberately different from the removed
    # sparse-page ratio's >=; see MAX_GARBAGE_PAGE_RATIO_DOCUMENT.
    if garbage_ratio > limits["max_garbage_page_ratio_document"]:
        label = TIER_GARBAGE
        reasons.append(
            f"{len(garbage_pages)} of {len(tiered)} tiered pages are Garbage "
            f"({garbage_ratio:.0%}, page(s) {stats['garbage_page_numbers']})"
        )
    elif garbage_pages:
        label = TIER_DECENT
        reasons.append(
            f"page(s) {stats['garbage_page_numbers']} are Garbage "
            f"({garbage_ratio:.0%} of tiered pages)"
        )
    elif failed_pages:
        # Reason already recorded above.
        label = TIER_DECENT
    elif no_text_layer_pages:
        # A no_text_layer page rules out Clean whether or not it got OCR text
        # (OCR text has no tier); the reason tells the two apart.
        label = TIER_DECENT
        reasons.append(_no_text_layer_reason(page_count, no_text_layer_pages))
    elif decent_pages:
        label = TIER_DECENT
        reasons.append(f"page(s) {stats['decent_page_numbers']} rated Decent")
    else:
        # Clean only when every tiered page is Clean, with no failed pages and
        # no no_text_layer page, OCR text or not -- all three already ruled
        # out above.
        label = TIER_CLEAN
        reasons.append("every tiered page is Clean")

    return {
        "label": label,
        "awaiting_ocr": awaiting_ocr,
        "reasons": reasons,
        "page_stats": stats,
    }


def assess_pages(
    document: DocumentText, settings: Optional[Mapping[str, Any]] = None
) -> List[PageQuality]:
    """The per-page step: source (Axis 2) and, for native pages, tier (Axis 3).

    Runs on the pdfplumber extraction, before any OCR: OCR text never reaches
    this step, and nothing after it recomputes source.
    """
    limits = _thresholds(settings)
    pages = _page_quality(document, limits["min_chars_native_page"])
    for page in pages:
        if page.source == SOURCE_NATIVE:
            page.tier = _tier_for_page(page, limits)
    return pages


def roll_up_quality(
    pages: List[PageQuality],
    structure: StructureResult,
    preocr_result: Optional[Dict[str, Any]] = None,
    settings: Optional[Mapping[str, Any]] = None,
) -> QualityAssessment:
    """The roll-up step: per-page results into a document label and evidence.

    Expects every page's ``text_origin`` to be set already (see
    :func:`resolve_text_origins`), since ``awaiting_ocr`` and the reasons
    read it.
    """
    limits = _thresholds(settings)
    preocr_result = preocr_result if preocr_result is not None else {}
    notes: List[str] = []

    preocr_error = preocr_result.get("error")
    if preocr_error:
        notes.append(
            f"preocr did not run ({preocr_error}); tiering on per-page statistics alone"
        )

    rollup = _roll_up(pages, limits)
    stats = rollup["page_stats"]

    # preocr's verdict becomes evidence-only (spec 001's Approach): recorded
    # for later recalibration, never branched on.
    stats["preocr_needs_ocr"] = bool(preocr_result.get("needs_ocr", False))
    stats["preocr_confidence"] = preocr_result.get("confidence")
    stats["preocr_confidence_band"] = preocr_confidence_band(
        preocr_result.get("confidence")
    )

    return QualityAssessment(
        label=rollup["label"],
        awaiting_ocr=rollup["awaiting_ocr"],
        reasons=rollup["reasons"],
        preocr=preocr_result,
        pages=pages,
        page_stats=stats,
        structure=structure.to_dict(),
        notes=notes,
    )


def assess_quality(
    document: DocumentText,
    structure: StructureResult,
    preocr_result: Optional[Dict[str, Any]] = None,
    settings: Optional[Mapping[str, Any]] = None,
) -> QualityAssessment:
    """Per-page source and tier (Axes 2 and 3), rolled up into a document label.

    ``preocr_result`` is passed in rather than computed here so the caller can
    run it once per document alongside the single text extraction. Its
    verdict is recorded as evidence only (unvalidated assumption (2) at the
    top of this module): no branch here depends on it.

    ``structure`` is recorded in the result for the CSV explanation and for
    later recalibration, but -- like ``preocr_result`` -- does not decide the
    tier: Axis 3 is computed from each page's own noise and word-length
    signals alone (see ``_tier_for_page``).

    This runs no routing and no OCR: every ``no_text_layer`` page ends with
    ``text_origin`` none, with reason ``extraction_failed`` if its extraction
    failed and ``ocr_skipped`` otherwise. The parse pass itself calls
    :func:`assess_pages`, the OCR step, :func:`resolve_text_origins` and
    :func:`roll_up_quality` instead (spec 003).

    Precondition: ``document.extraction_error`` is ``None`` and
    ``document.page_count > 0``. Those two hard-failure cases are Axis 1
    (status), decided upstream in :func:`parse_pass.classify_and_assess`
    before this function is ever called -- a document that failed to open, or
    that yielded zero pages, gets ``status=failed`` and no quality tier at
    all, never ``Garbage``. This function no longer handles them.
    """
    pages = assess_pages(document, settings=settings)
    resolve_text_origins(pages, document)
    return roll_up_quality(
        pages, structure, preocr_result=preocr_result, settings=settings
    )
