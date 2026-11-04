"""Tier-determination tests (Clean / Decent / Garbage).

``assess_quality`` takes preocr's ``needs_ocr`` output as a parameter rather
than calling it, so these tests stub that output with plain dictionaries: no
PDF is opened. preocr's text-quality function does run, though: ``_page_quality``
calls it for real, per page, on the fabricated text below. The stubs below
document the shape this code relies on.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any, Dict

import pytest

from companydocs_rag.ingestion.extraction import PageText
from companydocs_rag.ingestion.quality import (
    AWAITING_OCR_ALL,
    MIN_CHARS_NATIVE_PAGE,
    SOURCE_NATIVE,
    SOURCE_NO_TEXT_LAYER,
    TIER_CLEAN,
    TIER_DECENT,
    TIER_GARBAGE,
    PageQuality,
    QualityAssessment,
    _no_native_pages_reason,
    _no_text_layer_reason,
    _thresholds,
    assess_quality,
    preocr_confidence_band,
    run_preocr,
)
from companydocs_rag.ingestion.settings import load_settings, parse_pass_section
from companydocs_rag.ingestion.structure import StructureResult
from tests.ingestion._helpers import make_document, make_page

# --- Stubbed preocr output --------------------------------------------------
# Mirrors the relevant keys of preocr.needs_ocr(page_level=True).

DIGITAL_PREOCR: Dict[str, Any] = {
    "needs_ocr": False,
    "file_type": "pdf",
    "category": "structured",
    "confidence": 0.9,
    "reason": "Digital PDF with extractable text (hard digital guard)",
    "reason_code": "PDF_DIGITAL",
    "signals": {
        "text_length": 4200,
        "non_printable_ratio": 0.0,
        "unicode_noise_ratio": 0.0,
    },
    "pages": [{"page_number": 1, "needs_ocr": False, "text_length": 1400}],
}

SCANNED_PREOCR: Dict[str, Any] = {
    "needs_ocr": True,
    "file_type": "pdf",
    "category": "unstructured",
    "confidence": 0.95,
    "reason": "PDF appears to be scanned (hard scan shortcut)",
    "reason_code": "PDF_SCANNED",
    "signals": {
        "text_length": 4,
        "non_printable_ratio": 0.0,
        "unicode_noise_ratio": 0.0,
    },
    "pages": [{"page_number": 1, "needs_ocr": True, "text_length": 0}],
}

# --- Inline page text -------------------------------------------------------

CLEAN_PAGE = (
    "Artikel 1 Definities\n"
    "In deze huurovereenkomst wordt verstaan onder het gehuurde de kantoorruimte "
    "gelegen aan de Voorbeeldweg te Amsterdam, met inbegrip van de bijbehorende "
    "parkeerplaatsen en de gemeenschappelijke ruimten zoals nader omschreven in "
    "de bij deze overeenkomst behorende tekening.\n"
    "Artikel 2 Duur en verlenging\n"
    "De overeenkomst wordt aangegaan voor de duur van vijf jaar en wordt daarna "
    "voortgezet voor aansluitende perioden van telkens vijf jaar.\n"
)

THIN_PAGE = (
    "Artikel 1 Definities\nDe huurder en de verhuurder komen het volgende overeen.\n"
)

# A signature or stamp page: a little text, and nothing at all wrong with it.
SIGNATURE_PAGE = "Aldus overeengekomen en getekend te Amsterdam.\n"

GARBLED_PAGE = "�" * 150 + (
    "Artikel 1 Definities In deze overeenkomst wordt verstaan onder het gehuurde "
    "de kantoorruimte gelegen aan de Voorbeeldweg te Amsterdam met inbegrip van "
    "de bijbehorende parkeerplaatsen en gemeenschappelijke ruimten."
)

STRONG_STRUCTURE = StructureResult(
    matched_pattern="numbered_clauses", score=1.0, first_page=1, patterns=[]
)
NO_STRUCTURE = StructureResult(
    matched_pattern=None, score=0.0, first_page=None, patterns=[]
)


# --- Clean ------------------------------------------------------------------


def test_strong_extraction_plus_clear_structure_is_clean():
    result = assess_quality(
        make_document(CLEAN_PAGE, CLEAN_PAGE, CLEAN_PAGE),
        STRONG_STRUCTURE,
        preocr_result=DIGITAL_PREOCR,
    )
    assert result.label == TIER_CLEAN


# --- Decent -----------------------------------------------------------------


def test_structure_does_not_gate_the_quality_tier():
    """Axis 3 depends only on each page's own extraction signals.

    Structure is a document-level, informational signal only (still recorded
    in the result for the CSV explanation and later recalibration): unlike
    the old single ladder, it no longer decides Clean vs. Decent.
    """
    result = assess_quality(
        make_document(CLEAN_PAGE, CLEAN_PAGE, CLEAN_PAGE),
        NO_STRUCTURE,
        preocr_result=DIGITAL_PREOCR,
    )
    assert result.label == TIER_CLEAN
    # Structure is still recorded as evidence, just not decisive.
    assert result.structure["matched_pattern"] is None


def test_thin_pages_are_tiered_decent_not_clean():
    result = assess_quality(
        make_document(THIN_PAGE, THIN_PAGE, THIN_PAGE),
        STRONG_STRUCTURE,
        preocr_result=DIGITAL_PREOCR,
    )
    assert result.label == TIER_DECENT
    assert result.page_stats["decent_page_count"] == 3
    assert any("rated Decent" in reason for reason in result.reasons)


# --- Garbage ----------------------------------------------------------------


def test_preocr_needs_ocr_is_evidence_only_and_never_forces_a_tier():
    """preocr's verdict becomes evidence-only (spec 001): nothing branches on it."""
    result = assess_quality(
        make_document(CLEAN_PAGE, CLEAN_PAGE, CLEAN_PAGE),
        STRONG_STRUCTURE,
        preocr_result=SCANNED_PREOCR,  # needs_ocr=True
    )
    assert result.label == TIER_CLEAN
    assert result.page_stats["preocr_needs_ocr"] is True
    assert not any("needs OCR" in reason for reason in result.reasons)


def test_noisy_extraction_is_garbage():
    result = assess_quality(
        make_document(GARBLED_PAGE),
        STRONG_STRUCTURE,
        preocr_result=DIGITAL_PREOCR,
    )
    assert result.label == TIER_GARBAGE
    assert result.pages[0].source == SOURCE_NATIVE
    assert result.pages[0].tier == TIER_GARBAGE
    assert result.page_stats["garbage_page_numbers"] == [1]


# A document that failed to open, or yielded zero pages, is no longer
# assess_quality's concern: that is Axis 1 (status), decided upstream in
# parse_pass.classify_and_assess before assess_quality is ever called (see
# tests/ingestion/test_parse_pass.py). assess_quality is never called with
# such a document any more, so there is nothing to test here.


# --- Roll-up: per-page tiers -> document label (spec 001) -------------------


def _clean_pages(count: int):
    return [CLEAN_PAGE] * count


def _garbage_pages(count: int):
    return [GARBLED_PAGE] * count


def test_one_garbage_page_out_of_twenty_gives_decent_naming_the_page():
    document = make_document(*(_garbage_pages(1) + _clean_pages(19)))
    result = assess_quality(document, NO_STRUCTURE, preocr_result=DIGITAL_PREOCR)

    assert result.label == TIER_DECENT
    assert result.page_stats["garbage_page_count"] == 1
    assert result.page_stats["garbage_page_numbers"] == [1]
    assert any("[1]" in reason for reason in result.reasons)


def test_eleven_garbage_pages_out_of_twenty_gives_document_garbage():
    document = make_document(*(_garbage_pages(11) + _clean_pages(9)))
    result = assess_quality(document, NO_STRUCTURE, preocr_result=DIGITAL_PREOCR)

    assert result.label == TIER_GARBAGE
    assert result.page_stats["garbage_page_count"] == 11


def test_exactly_ten_of_twenty_garbage_pages_is_decent_not_garbage():
    """The roll-up comparison is strictly greater-than the 0.50 threshold."""
    document = make_document(*(_garbage_pages(10) + _clean_pages(10)))
    result = assess_quality(document, NO_STRUCTURE, preocr_result=DIGITAL_PREOCR)

    assert result.page_stats["garbage_page_ratio"] == 0.50
    assert result.label == TIER_DECENT


def test_garbage_ratio_threshold_is_read_from_settings():
    """A stricter threshold configured in settings.yaml changes the verdict."""
    document = make_document(*(_garbage_pages(2) + _clean_pages(18)))  # ratio 0.10

    default_result = assess_quality(
        document, NO_STRUCTURE, preocr_result=DIGITAL_PREOCR
    )
    assert default_result.label == TIER_DECENT  # 0.10 <= default 0.50 threshold

    strict = {
        "ingestion": {
            "parse_pass": {"quality": {"max_garbage_page_ratio_document": 0.05}}
        }
    }
    strict_result = assess_quality(
        document, NO_STRUCTURE, preocr_result=DIGITAL_PREOCR, settings=strict
    )
    assert strict_result.label == TIER_GARBAGE  # 0.10 > 0.05 threshold


def test_garbage_ratio_threshold_key_is_present_in_the_real_settings_yaml(monkeypatch):
    """Guards against a misspelled/missing ``max_garbage_page_ratio_document``
    key in config/settings.yaml going unnoticed.

    A test that only checks the tier outcome on the real settings would pass
    even if the key were dropped or misspelled -- the code would silently
    fall back to the identical Python default and no behavioural difference
    would show. This test instead loads the real, unpatched settings file,
    asserts the exact key quality.py reads is present at the path it reads it
    from (so a missing/misspelled key still fails this test), and checks that
    ``_thresholds`` -- the function the roll-up gets its limits from --
    resolves that same value, whatever it currently is. It intentionally does
    not hard-code the threshold's numeric value, which is expected to be
    recalibrated over time (see the module docstring's unvalidated
    assumptions).
    """
    monkeypatch.delenv("COMPANYDOCS_RAG_SETTINGS", raising=False)
    load_settings.cache_clear()
    try:
        quality_section = parse_pass_section("quality")
        assert "max_garbage_page_ratio_document" in quality_section
        yaml_value = quality_section["max_garbage_page_ratio_document"]

        limits = _thresholds()
        assert limits["max_garbage_page_ratio_document"] == yaml_value
    finally:
        load_settings.cache_clear()


def test_a_fully_scanned_document_has_no_tier_and_is_marked_awaiting_ocr_all():
    """Not a corner case on this corpus (spec 001): resolved, not left open."""
    document = make_document("", "", "")
    result = assess_quality(document, NO_STRUCTURE, preocr_result=DIGITAL_PREOCR)

    assert result.label is None
    assert result.label != TIER_GARBAGE
    assert result.awaiting_ocr == AWAITING_OCR_ALL


def test_source_and_tier_are_independent_axes():
    """A native page can be Garbage; a no_text_layer page carries no tier at all."""
    document = make_document(GARBLED_PAGE, "kort")
    result = assess_quality(document, NO_STRUCTURE, preocr_result=DIGITAL_PREOCR)

    assert result.pages[0].source == SOURCE_NATIVE
    assert result.pages[0].tier == TIER_GARBAGE
    assert result.pages[1].source == SOURCE_NO_TEXT_LAYER
    assert result.pages[1].tier is None


# --- The (removed) sparse-page override: replacement test -------------------


def test_a_single_signature_page_does_not_sink_a_good_document_to_garbage():
    """Regression test for the known inflated-flagging failure mode.

    preocr used to flip a whole document to needs_ocr as soon as ONE page fell
    below 50 characters. A signature page at the end of an otherwise perfect
    lease is exactly that case: it must not be treated the same as a garbled
    page. Under the per-page model the signature page becomes `no_text_layer`
    (awaiting OCR) rather than being scored -- which keeps the document out of
    Garbage, but (deliberately, per the roll-up's Clean gate) also keeps it
    out of Clean: it genuinely has one page with no text layer.
    """
    document = make_document(*([CLEAN_PAGE] * 7), SIGNATURE_PAGE)
    result = assess_quality(document, STRONG_STRUCTURE, preocr_result=SCANNED_PREOCR)

    assert result.label == TIER_DECENT
    assert result.label != TIER_GARBAGE
    assert result.pages[7].source == SOURCE_NO_TEXT_LAYER
    assert result.page_stats["no_text_layer_page_numbers"] == [8]
    assert any("awaiting OCR" in reason for reason in result.reasons)


def test_no_text_layer_pages_never_pollute_the_native_noise_average():
    """A short garbled page becomes no_text_layer, excluded from mean_noise_ratio."""
    document = make_document(CLEAN_PAGE, CLEAN_PAGE, CLEAN_PAGE, "�" * 12)
    result = assess_quality(document, STRONG_STRUCTURE, preocr_result=DIGITAL_PREOCR)

    assert result.pages[3].source == SOURCE_NO_TEXT_LAYER
    assert result.page_stats["mean_noise_ratio"] < 0.01
    # The page's own ratio is still preserved as evidence, even though it
    # never enters the native-page average.
    assert result.pages[3].unicode_noise_ratio == 1.0


def test_no_text_layer_and_empty_pages_are_counted_separately():
    document = make_document(CLEAN_PAGE, SIGNATURE_PAGE, "")
    result = assess_quality(document, STRONG_STRUCTURE, preocr_result=DIGITAL_PREOCR)
    assert result.page_stats["no_text_layer_page_count"] == 2
    assert result.page_stats["empty_page_count"] == 1


# --- Axis 2: source, per page (spec 001) ------------------------------------


def test_a_page_exactly_at_min_chars_native_page_is_native():
    """The comparison is strictly less-than: exactly on the threshold is native."""
    text = "x" * int(MIN_CHARS_NATIVE_PAGE)
    result = assess_quality(
        make_document(text), NO_STRUCTURE, preocr_result=DIGITAL_PREOCR
    )
    assert result.pages[0].char_count == MIN_CHARS_NATIVE_PAGE
    assert result.pages[0].source == SOURCE_NATIVE


def test_a_page_one_character_below_the_threshold_is_no_text_layer():
    text = "x" * (int(MIN_CHARS_NATIVE_PAGE) - 1)
    result = assess_quality(
        make_document(text), NO_STRUCTURE, preocr_result=DIGITAL_PREOCR
    )
    assert result.pages[0].source == SOURCE_NO_TEXT_LAYER


def test_heavy_image_coverage_above_the_threshold_stays_native():
    """Coverage decides nothing: a native page stays native regardless."""
    page = make_page(1, CLEAN_PAGE, image_area_ratio=0.95, image_count=3)
    result = assess_quality(
        make_document(page), NO_STRUCTURE, preocr_result=DIGITAL_PREOCR
    )
    assert result.pages[0].source == SOURCE_NATIVE
    assert result.pages[0].image_area_ratio == 0.95


def test_no_images_at_all_below_the_threshold_is_still_no_text_layer():
    """Coverage decides nothing: a thin page isn't rescued by having no images."""
    page = make_page(1, "too little text", image_area_ratio=0.0, image_count=0)
    result = assess_quality(
        make_document(page), NO_STRUCTURE, preocr_result=DIGITAL_PREOCR
    )
    assert result.pages[0].source == SOURCE_NO_TEXT_LAYER
    assert result.pages[0].image_count == 0


# --- Signals are preserved --------------------------------------------------


def test_raw_signals_are_returned_alongside_the_label():
    result = assess_quality(
        make_document(CLEAN_PAGE, CLEAN_PAGE, CLEAN_PAGE),
        STRONG_STRUCTURE,
        preocr_result=DIGITAL_PREOCR,
    )
    signals = result.to_dict()

    # preocr's own output, including its per-page classification.
    assert signals["preocr"]["pages"] == DIGITAL_PREOCR["pages"]
    assert signals["preocr"]["reason_code"] == "PDF_DIGITAL"
    # The winning structural pattern and where it was found.
    assert signals["structure"]["matched_pattern"] == "numbered_clauses"
    assert signals["structure"]["first_page"] == 1
    # Per-page statistics behind the tier.
    assert len(signals["pages"]) == 3
    assert signals["page_stats"]["preocr_confidence_band"] == "high"


def test_each_page_tier_is_present_in_the_serialised_result():
    """Per-page tier must be reachable for chunking, not just the document label."""
    result = assess_quality(
        make_document(CLEAN_PAGE, GARBLED_PAGE),
        NO_STRUCTURE,
        preocr_result=DIGITAL_PREOCR,
    )
    signals = result.to_dict()

    assert signals["pages"][0]["tier"] == TIER_CLEAN
    assert signals["pages"][0]["source"] == SOURCE_NATIVE
    assert signals["pages"][1]["tier"] == TIER_GARBAGE
    assert signals["awaiting_ocr"] is None


def test_preocr_failure_is_recorded_and_tiering_continues():
    result = assess_quality(
        make_document(CLEAN_PAGE, CLEAN_PAGE, CLEAN_PAGE),
        STRONG_STRUCTURE,
        preocr_result={"error": "PDFProcessingError: encrypted file"},
    )
    assert result.label == TIER_CLEAN
    assert any("preocr did not run" in note for note in result.notes)


def test_thresholds_can_be_overridden_through_settings():
    strict = {
        "ingestion": {
            "parse_pass": {"quality": {"min_chars_per_substantive_page_clean": 10000}}
        }
    }
    result = assess_quality(
        make_document(CLEAN_PAGE, CLEAN_PAGE, CLEAN_PAGE),
        STRONG_STRUCTURE,
        preocr_result=DIGITAL_PREOCR,
        settings=strict,
    )
    assert result.label == TIER_DECENT


def test_preocr_confidence_bands():
    assert preocr_confidence_band(0.95) == "high"
    assert preocr_confidence_band(0.75) == "medium"
    assert preocr_confidence_band(0.55) == "low"
    assert preocr_confidence_band(0.2) == "very_low"
    assert preocr_confidence_band(None) == "unknown"


# --- Failed pages (extraction.py PageText.extraction_failed) ----------------


def test_a_failed_page_prevents_clean_and_is_named_distinctly_in_the_reasons():
    """A document that would otherwise be Clean must not be Clean if a page failed.

    A failed page is always `no_text_layer` too (it carries no text), which on
    its own would already rule out Clean -- but the failure is named
    distinctly ("text extraction failed on page(s) [5]"), not folded into the
    generic "awaiting OCR" wording, so a genuine extraction bug is never
    mistaken for an ordinary scanned page.
    """
    document = make_document(*([CLEAN_PAGE] * 9))
    document.pages[4] = PageText(
        page_number=5,
        text="",
        extraction_failed=True,
        extraction_error="ValueError: boom",
    )

    result = assess_quality(document, STRONG_STRUCTURE, preocr_result=DIGITAL_PREOCR)

    assert result.label == TIER_DECENT
    assert result.pages[4].source == SOURCE_NO_TEXT_LAYER
    assert any("page(s) [5]" in reason for reason in result.reasons)
    assert any("text extraction failed" in reason for reason in result.reasons)
    assert result.page_stats["failed_page_count"] == 1
    assert result.page_stats["failed_page_numbers"] == [5]
    assert result.pages[4].extraction_failed is True


def test_an_ordinary_no_text_layer_page_is_distinct_from_a_failed_page():
    """A page that is merely awaiting OCR is not reported as an extraction failure."""
    document = make_document(*([CLEAN_PAGE] * 9), SIGNATURE_PAGE)

    result = assess_quality(document, STRONG_STRUCTURE, preocr_result=DIGITAL_PREOCR)

    assert result.label == TIER_DECENT
    assert result.page_stats["failed_page_count"] == 0
    assert not any("extraction failed" in reason for reason in result.reasons)
    assert any("awaiting OCR" in reason for reason in result.reasons)


def test_failed_pages_are_tracked_separately_from_ordinary_no_text_layer_pages():
    """A failed page has no text layer, but must not be reported as merely
    awaiting OCR."""
    document = make_document(CLEAN_PAGE, CLEAN_PAGE, CLEAN_PAGE, SIGNATURE_PAGE)
    document.pages[1] = PageText(page_number=2, text="", extraction_failed=True)

    result = assess_quality(document, STRONG_STRUCTURE, preocr_result=DIGITAL_PREOCR)

    # The failed page and the signature page both have no text layer...
    assert result.page_stats["no_text_layer_page_count"] == 2
    # ...but only one of them actually failed to extract.
    assert result.page_stats["failed_page_count"] == 1
    assert result.page_stats["failed_page_numbers"] == [2]


def test_multiple_failed_pages_are_all_named_in_the_reasons():
    document = make_document(CLEAN_PAGE, CLEAN_PAGE, CLEAN_PAGE)
    document.pages[0] = PageText(page_number=1, text="", extraction_failed=True)
    document.pages[2] = PageText(page_number=3, text="", extraction_failed=True)

    result = assess_quality(document, STRONG_STRUCTURE, preocr_result=DIGITAL_PREOCR)

    assert result.label != TIER_CLEAN
    assert result.page_stats["failed_page_numbers"] == [1, 3]
    assert any("[1, 3]" in reason for reason in result.reasons)


def test_a_document_without_any_failed_page_is_unaffected():
    """No extraction_failed pages: the failed-page guard must not fire spuriously."""
    result = assess_quality(
        make_document(CLEAN_PAGE, CLEAN_PAGE, CLEAN_PAGE),
        STRONG_STRUCTURE,
        preocr_result=DIGITAL_PREOCR,
    )
    assert result.label == TIER_CLEAN
    assert result.page_stats["failed_page_count"] == 0
    assert result.page_stats["failed_page_numbers"] == []
    assert not any("extraction failed on page" in reason for reason in result.reasons)


# --- preocr failures are logged (run_preocr) ---------------------------------


def test_preocr_failure_is_logged_with_path_and_exception(monkeypatch, caplog):
    """run_preocr must log the failure, not just swallow it into the result dict.

    A fake ``preocr`` module is injected into ``sys.modules`` rather than
    monkeypatching the real package: importing the real ``preocr`` package can
    hang on Windows when python-magic's loader finds a conflicting DLL on
    PATH (see quality.py's module docstring), and this test must not depend on
    that succeeding.
    """
    fake_preocr = types.ModuleType("preocr")

    def _boom(pdf_path, page_level=True):
        raise RuntimeError("preocr exploded")

    fake_preocr.needs_ocr = _boom
    monkeypatch.setitem(sys.modules, "preocr", fake_preocr)

    with caplog.at_level(logging.ERROR, logger="companydocs_rag.ingestion.quality"):
        result = run_preocr("fabricated-lease.pdf")

    assert result == {"error": "RuntimeError: preocr exploded"}

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    message = error_records[0].getMessage()
    assert "fabricated-lease.pdf" in message
    assert "RuntimeError" in message
    assert "preocr exploded" in message


def test_run_preocr_never_raises_when_preocr_itself_cannot_be_imported(
    monkeypatch, caplog
):
    """The lazy `import preocr` lives inside the try, so an ImportError is caught too.

    ``sys.modules["preocr"] = None`` makes any ``import preocr`` raise
    ImportError, the same injection style used above, without needing the real
    package (or python-magic) to be importable.
    """
    monkeypatch.setitem(sys.modules, "preocr", None)

    with caplog.at_level(logging.ERROR, logger="companydocs_rag.ingestion.quality"):
        result = run_preocr("fabricated-lease.pdf")

    # `sys.modules["preocr"] = None` makes the import raise ModuleNotFoundError,
    # a subclass of ImportError; either way the point is it is caught, not raised.
    assert "error" in result
    assert "ModuleNotFoundError" in result["error"]

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    message = error_records[0].getMessage()
    assert "fabricated-lease.pdf" in message
    assert "ModuleNotFoundError" in message


# --- Text origin on a direct call (spec 003) ----------------------------------
# A direct assess_quality call runs no routing and no OCR: every no_text_layer
# page ends with text_origin none. The pipeline with OCR is covered in
# test_parse_pass.py, test_ocr.py and test_cli_pipeline.py.


def test_a_direct_call_gives_no_text_layer_pages_no_text_and_a_reason():
    document = make_document(CLEAN_PAGE, SIGNATURE_PAGE, CLEAN_PAGE)
    document.pages[2] = PageText(
        page_number=3,
        text="",
        extraction_failed=True,
        extraction_error="ValueError: fabricated page failure",
    )

    result = assess_quality(document, STRONG_STRUCTURE, preocr_result=DIGITAL_PREOCR)

    assert [page.text_origin for page in result.pages] == ["native", "none", "none"]
    assert [page.no_text_reason for page in result.pages] == [
        None,
        "ocr_skipped",
        "extraction_failed",
    ]
    assert [page.ocr_routed for page in result.pages] == [False, False, False]
    assert result.pages[2].extraction_error == "ValueError: fabricated page failure"
    assert result.awaiting_ocr == "partial"
    stats = result.page_stats
    assert stats["ocr_page_count"] == 0
    assert stats["ocr_page_numbers"] == []
    assert stats["no_text_page_count"] == 2
    assert stats["no_text_page_numbers"] == [2, 3]
    assert stats["no_text_reason_counts"] == {
        "extraction_failed": 1,
        "ocr_skipped": 1,
        "ocr_error": 0,
        "ocr_empty": 0,
    }


def test_the_page_record_carries_the_new_fields_with_defaults():
    result = assess_quality(
        make_document(CLEAN_PAGE), NO_STRUCTURE, preocr_result=DIGITAL_PREOCR
    )
    page = result.to_dict()["pages"][0]

    assert page["text_origin"] == "native"
    for key in (
        "no_text_reason",
        "extraction_error",
        "ocr_error",
        "ocr_mean_confidence",
        "ocr_non_printable_ratio",
        "ocr_unicode_noise_ratio",
        "ocr_avg_word_length",
        "language",
        "language_source",
    ):
        assert page[key] is None
    assert page["ocr_routed"] is False


def test_the_label_is_none_exactly_when_there_are_no_native_pages():
    """A8: QualityAssessment's docstring states the rule."""
    docstring = " ".join(QualityAssessment.__doc__.split())
    assert "None exactly when the document has no native pages" in docstring


# --- The reason wording per no-text reason (spec 004, D3; A5) ------------------------
# Pages fabricated as PageQuality records with their text origin and no-text
# reason already set, as resolve_text_origins leaves them.

NO_NATIVE = "document has no pages with a text layer"
EXTRACTION_FAILED_WORDS = (
    "have no text because text extraction failed and they were not sent to OCR"
)


def _no_text_layer_page(number, origin, reason=None):
    return PageQuality(
        page_number=number,
        char_count=0,
        source=SOURCE_NO_TEXT_LAYER,
        text_origin=origin,
        no_text_reason=reason,
    )


def _ocr(number):
    return _no_text_layer_page(number, "ocr")


def _none(number, reason):
    return _no_text_layer_page(number, "none", reason)


def _label_none_reason(pages):
    ocr_pages = [page for page in pages if page.text_origin == "ocr"]
    no_text_pages = [page for page in pages if page.text_origin == "none"]
    return _no_native_pages_reason(len(pages), ocr_pages, no_text_pages)


@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        (
            [_ocr(1), _ocr(2)],
            f"{NO_NATIVE}; 2 page(s) carry OCR text (page(s) [1, 2])",
        ),
        (
            [_none(1, "extraction_failed")],
            f"{NO_NATIVE}; 1 page(s) {EXTRACTION_FAILED_WORDS} (page(s) [1])",
        ),
        (
            [_none(1, "ocr_skipped"), _none(2, "ocr_skipped")],
            f"{NO_NATIVE}; entirely awaiting OCR",
        ),
        (
            [_none(1, "ocr_error")],
            f"{NO_NATIVE}; 1 page(s) have no text because OCR failed (page(s) [1])",
        ),
        (
            [_none(1, "ocr_empty"), _none(2, "ocr_empty")],
            f"{NO_NATIVE}; 2 page(s) have no text because OCR ran and returned no "
            "text (page(s) [1, 2])",
        ),
    ],
    ids=["ocr", "extraction_failed", "ocr_skipped", "ocr_error", "ocr_empty"],
)
def test_the_label_none_reason_per_no_text_reason(pages, expected):
    assert _label_none_reason(pages) == expected


def test_the_label_none_reason_for_mixed_pages_follows_the_reason_order():
    pages = [
        _ocr(1),
        _none(2, "ocr_empty"),
        _none(3, "ocr_skipped"),
        _none(4, "extraction_failed"),
        _none(5, "ocr_error"),
        _ocr(6),
        _none(7, "ocr_empty"),
    ]

    assert _label_none_reason(pages) == (
        f"{NO_NATIVE}; 2 page(s) carry OCR text (page(s) [1, 6]); "
        f"1 page(s) {EXTRACTION_FAILED_WORDS} (page(s) [4]); "
        "1 page(s) still have no text and are awaiting OCR (page(s) [3]); "
        "1 page(s) have no text because OCR failed (page(s) [5]); "
        "2 page(s) have no text because OCR ran and returned no text "
        "(page(s) [2, 7])"
    )


def test_ocr_skipped_next_to_another_reason_is_not_entirely_awaiting_ocr():
    pages = [_none(1, "ocr_skipped"), _none(2, "extraction_failed")]

    assert _label_none_reason(pages) == (
        f"{NO_NATIVE}; 1 page(s) {EXTRACTION_FAILED_WORDS} (page(s) [2]); "
        "1 page(s) still have no text and are awaiting OCR (page(s) [1])"
    )


@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        (
            [_ocr(4), _ocr(5)],
            "2 of 5 page(s) have no text layer and carry OCR text (page(s) [4, 5])",
        ),
        (
            [_none(5, "extraction_failed")],
            f"1 of 5 page(s) have no text layer: 1 {EXTRACTION_FAILED_WORDS} "
            "(page(s) [5])",
        ),
        (
            [_none(4, "ocr_skipped"), _none(5, "ocr_skipped")],
            "2 of 5 page(s) have no text layer and are awaiting OCR (page(s) [4, 5])",
        ),
        (
            [_none(5, "ocr_error")],
            "1 of 5 page(s) have no text layer: 1 have no text because OCR failed "
            "(page(s) [5])",
        ),
        (
            [_none(5, "ocr_empty")],
            "1 of 5 page(s) have no text layer: 1 have no text because OCR ran and "
            "returned no text (page(s) [5])",
        ),
    ],
    ids=["ocr", "extraction_failed", "ocr_skipped", "ocr_error", "ocr_empty"],
)
def test_the_no_text_layer_reason_per_no_text_reason(pages, expected):
    assert _no_text_layer_reason(5, pages) == expected


def test_the_no_text_layer_reason_for_mixed_pages_follows_the_reason_order():
    pages = [
        _ocr(2),
        _none(3, "ocr_empty"),
        _none(4, "ocr_skipped"),
        _none(5, "extraction_failed"),
        _none(6, "ocr_error"),
        _ocr(7),
    ]

    assert _no_text_layer_reason(8, pages) == (
        "6 of 8 page(s) have no text layer: 2 carry OCR text (page(s) [2, 7]), "
        f"1 {EXTRACTION_FAILED_WORDS} (page(s) [5]), "
        "1 are awaiting OCR (page(s) [4]), "
        "1 have no text because OCR failed (page(s) [6]), "
        "1 have no text because OCR ran and returned no text (page(s) [3])"
    )


def test_ocr_pages_next_to_ocr_skipped_pages_keep_the_earlier_wording():
    pages = [_ocr(2), _none(3, "ocr_skipped")]

    assert _no_text_layer_reason(3, pages) == (
        "2 of 3 page(s) have no text layer: 1 carry OCR text (page(s) [2]), "
        "1 are awaiting OCR (page(s) [3])"
    )
