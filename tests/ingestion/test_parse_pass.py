"""End-to-end tests for classify_and_assess, with preocr and extraction stubbed.

No PDF is read. The extraction step is replaced with an inline
:class:`DocumentText` and the preocr call with a plain dictionary, which is
what makes it possible to test the orchestration without any sample document.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from companydocs_rag.ingestion import classify_and_assess
from companydocs_rag.ingestion import parse_pass as parse_pass_module
from companydocs_rag.ingestion.extraction import PageText
from companydocs_rag.ingestion.ocr import TesseractOutput
from companydocs_rag.ingestion.quality import assess_quality
from companydocs_rag.ingestion.structure import StructureResult
from tests.ingestion._helpers import (
    make_document,
    make_tsv,
    stub_ocr,
    tesseract_output,
)
from tests.ingestion.test_quality import CLEAN_PAGE, DIGITAL_PREOCR, SCANNED_PREOCR

LEASE_PAGE = (
    "HUUROVEREENKOMST KANTOORRUIMTE\n"
    "Artikel 1 Definities\n"
    "De verhuurder verhuurt aan de huurder het gehuurde, gelegen aan de "
    "Voorbeeldweg te Amsterdam, zoals nader omschreven in de bij deze "
    "overeenkomst behorende tekening en de daarbij horende bijlagen.\n"
    "Artikel 2 Huurprijs\n"
    "De huurprijs bedraagt EUR 4.500 per maand, jaarlijks te indexeren "
    "overeenkomstig de algemene bepalingen.\n"
    "Artikel 3 Waarborgsom\n"
    "De huurder stelt een waarborgsom van drie maandtermijnen.\n"
)


def _stub(monkeypatch, document, preocr_result):
    monkeypatch.setattr(parse_pass_module, "extract_pages", lambda path: document)
    monkeypatch.setattr(parse_pass_module, "run_preocr", lambda path: preocr_result)
    # Default OCR stub (spec 003): every routed page gets empty OCR text, and
    # no page is rendered and no Tesseract process started.
    stub_ocr(monkeypatch)


def test_returns_type_and_quality_from_a_single_extraction(monkeypatch):
    _stub(monkeypatch, make_document(LEASE_PAGE, LEASE_PAGE), DIGITAL_PREOCR)
    result = classify_and_assess("lease.pdf")

    assert result["detected_type"] == "lease_agreement"
    assert result["type_confidence"] > 0.5
    assert result["quality_label"] == "Clean"


def test_result_carries_the_evidence_behind_both_outcomes(monkeypatch):
    _stub(monkeypatch, make_document(LEASE_PAGE, LEASE_PAGE), DIGITAL_PREOCR)
    result = classify_and_assess("lease.pdf")

    # Type evidence: which keyword, which language, which page.
    first = result["type_evidence"][0]
    assert set(first) >= {"term", "language", "page", "occurrences"}
    assert first["page"] in (1, 2)

    # Quality signals: raw preocr output plus the structural match.
    signals = result["quality_signals"]
    assert signals["preocr"]["reason_code"] == "PDF_DIGITAL"
    assert signals["structure"]["matched_pattern"] == "numbered_clauses"
    assert signals["structure"]["first_page"] == 1
    assert signals["page_stats"]["page_count"] == 2


def test_explanation_is_short_and_mentions_both_the_tier_and_the_type(monkeypatch):
    _stub(monkeypatch, make_document(LEASE_PAGE, LEASE_PAGE), DIGITAL_PREOCR)
    explanation = classify_and_assess("lease.pdf")["explanation"]

    assert explanation.startswith("Clean:")
    assert "article/clause numbering detected starting page 1" in explanation
    assert "lease_agreement" in explanation


def test_unrecognisable_document_is_unknown_but_still_assessed(monkeypatch):
    """No type match must not stop the quality assessment or force a guess."""
    _stub(
        monkeypatch, make_document(CLEAN_PAGE, CLEAN_PAGE, CLEAN_PAGE), DIGITAL_PREOCR
    )
    result = classify_and_assess("mystery.pdf")

    assert result["detected_type"] == "unknown/other"
    assert result["quality_label"] == "Clean"
    assert result["type_fallback_reason"]


def test_fully_scanned_document_is_awaiting_ocr_not_garbage_and_type_falls_back(
    monkeypatch,
):
    """Both pages are empty -- no page has a text layer -- so status stays
    processed, there is no document tier at all (not Garbage), and it is
    marked entirely awaiting OCR (spec 001's "fully scanned document")."""
    _stub(monkeypatch, make_document("", ""), SCANNED_PREOCR)
    result = classify_and_assess("scan.pdf")

    assert result["status"] == "processed"
    assert result["quality_label"] is None
    assert result["awaiting_ocr"] == "all"
    assert result["quality_signals"]["awaiting_ocr"] == "all"
    assert result["detected_type"] == "unknown/other"
    assert result["type_confidence"] == 0.0


def test_missing_file_is_reported_rather_than_raised():
    """Runs for real, no stubs: a folder run must survive a bad path.

    Axis 1: a document that could not even be opened is status=failed, with
    no quality tier at all -- not Garbage.
    """
    result = classify_and_assess("this-file-does-not-exist.pdf")

    assert result["status"] == "failed"
    assert result["status_reason"]
    assert result["quality_label"] is None
    assert result["awaiting_ocr"] is None
    assert result["detected_type"] == "unknown/other"
    assert result["file"]["extraction_error"]
    assert result["file"]["page_count"] == 0


def test_zero_page_document_is_failed_status_not_garbage(monkeypatch):
    """A document that opened but yielded zero pages is also status=failed."""
    _stub(monkeypatch, make_document(), DIGITAL_PREOCR)
    result = classify_and_assess("empty.pdf")

    assert result["status"] == "failed"
    assert "no pages" in result["status_reason"]
    assert result["quality_label"] is None


def test_an_exception_inside_classify_and_assess_yields_failed_status_never_a_tier(
    monkeypatch,
):
    """A bug in our own code must be reported as status=failed, never Garbage.

    This is the exact bug the Goal of spec 001 calls out: an internal
    exception must not be mislabelled as a property of the document.
    """
    _stub(monkeypatch, make_document(LEASE_PAGE), DIGITAL_PREOCR)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom from inside the parse pass")

    monkeypatch.setattr(parse_pass_module, "detect_type", _boom)

    result = classify_and_assess("lease.pdf")

    assert result["status"] == "failed"
    assert result["quality_label"] is None
    assert "RuntimeError" in result["status_reason"]
    assert "boom from inside the parse pass" in result["status_reason"]


def test_a_mixed_document_reports_per_page_sources_and_a_non_garbage_tier(
    monkeypatch,
):
    """3 native pages + 1 image-only page (spec 001's acceptance criterion).

    Status stays processed, the 4th page is marked for OCR (source
    no_text_layer), and the document tier is not Garbage -- one page awaiting
    OCR must not sink an otherwise good document.
    """
    document = make_document(LEASE_PAGE, LEASE_PAGE, LEASE_PAGE, "")
    _stub(monkeypatch, document, DIGITAL_PREOCR)

    result = classify_and_assess("mixed.pdf")

    assert result["status"] == "processed"
    sources = [page["source"] for page in result["quality_signals"]["pages"]]
    assert sources == ["native", "native", "native", "no_text_layer"]
    assert result["quality_label"] != "Garbage"
    assert result["awaiting_ocr"] == "partial"
    assert result["quality_signals"]["awaiting_ocr"] == "partial"


def test_failed_pages_are_surfaced_in_the_result(monkeypatch):
    """A per-page extraction failure must reach the signals JSON, not just the label."""
    document = make_document(LEASE_PAGE, LEASE_PAGE)
    document.pages[1] = PageText(
        page_number=2,
        text="",
        extraction_failed=True,
        extraction_error="ValueError: boom",
    )
    _stub(monkeypatch, document, DIGITAL_PREOCR)

    result = classify_and_assess("lease.pdf")

    assert result["file"]["failed_pages"] == [2]
    assert result["quality_label"] != "Clean"


def test_result_is_json_serialisable():
    """The result is stored for later recalibration, so it must serialise."""
    import json

    result = classify_and_assess("this-file-does-not-exist.pdf")
    reloaded = json.loads(json.dumps(result))
    assert reloaded["status"] == "failed"
    assert reloaded["quality_label"] is None


# --- Spec 003: OCR text, the page axes and the order -------------------------
# The OCR seams are stubbed with fabricated text (tests/ingestion/_helpers.py):
# no page is rendered and no Tesseract process is started.

OCR_TEXT = (
    "Fictieve OCR tekst: de huurder betaalt de huurprijs maandelijks vooruit "
    "aan de verhuurder."
)
SHORT_FRAGMENT = "los fragment"  # below MIN_CHARS_NATIVE_PAGE: no_text_layer
GARBAGE_PAGE = "�" * 150 + LEASE_PAGE  # native, tiered Garbage
FAILED_PAGE_ERROR = "ValueError: fabricated page failure"


def _failed_page(number):
    return PageText(
        page_number=number,
        text="",
        extraction_failed=True,
        extraction_error=FAILED_PAGE_ERROR,
    )


def _ocr_flags(**flags):
    return {"ingestion": {"parse_pass": {"ocr": flags}}}


def _capture_downstream(monkeypatch):
    """Record the page texts that structure analysis and type detection get."""
    captured = {}
    real_structure = parse_pass_module.analyse_structure
    real_type = parse_pass_module.detect_type

    def structure_spy(pages, settings=None):
        captured["structure"] = [page.text for page in pages]
        return real_structure(pages, settings=settings)

    def type_spy(document, settings=None):
        captured["type"] = [page.text for page in document.pages]
        return real_type(document, settings=settings)

    monkeypatch.setattr(parse_pass_module, "analyse_structure", structure_spy)
    monkeypatch.setattr(parse_pass_module, "detect_type", type_spy)
    return captured


def _direct_assessment(document):
    """A direct assess_quality call: no routing and no OCR outcomes."""
    return assess_quality(
        document,
        StructureResult(matched_pattern=None, score=0.0, first_page=None),
        preocr_result=DIGITAL_PREOCR,
    )


def test_a_page_without_native_text_gets_ocr_text_and_keeps_its_source(monkeypatch):
    """A2."""
    _stub(monkeypatch, make_document(""), SCANNED_PREOCR)
    stub_ocr(monkeypatch, {1: OCR_TEXT})

    result = classify_and_assess("fictief.pdf")

    page = result["quality_signals"]["pages"][0]
    assert len(OCR_TEXT) >= 50
    assert page["source"] == "no_text_layer"
    assert page["char_count"] == 0
    assert page["text_origin"] == "ocr"
    assert page["tier"] is None
    assert result["page_text"]["pages"][0]["text"] == OCR_TEXT


_GARBAGE_FLAG = _ocr_flags(ocr_garbage_tier_pages=True)
_FAILED_FLAG = _ocr_flags(ocr_extraction_failed_pages=True)
_OCR_FAILURE = RuntimeError("fabricated OCR failure")

# (id, page, settings, OCR output or None if not routed, text_origin,
#  no_text_reason, downstream and store text)
_ROWS = [
    ("native", CLEAN_PAGE, None, None, "native", None, CLEAN_PAGE),
    ("garbage-flag-text", GARBAGE_PAGE, _GARBAGE_FLAG, OCR_TEXT, "ocr", None, OCR_TEXT),
    (
        "garbage-flag-error",
        GARBAGE_PAGE,
        _GARBAGE_FLAG,
        _OCR_FAILURE,
        "native",
        None,
        GARBAGE_PAGE,
    ),
    (
        "garbage-flag-empty",
        GARBAGE_PAGE,
        _GARBAGE_FLAG,
        "",
        "native",
        None,
        GARBAGE_PAGE,
    ),
    ("no-text-layer-text", SHORT_FRAGMENT, None, OCR_TEXT, "ocr", None, OCR_TEXT),
    (
        "no-text-layer-error",
        SHORT_FRAGMENT,
        None,
        _OCR_FAILURE,
        "none",
        "ocr_error",
        "",
    ),
    ("no-text-layer-empty", SHORT_FRAGMENT, None, " \n", "none", "ocr_empty", ""),
    ("failed-default", _failed_page(1), None, None, "none", "extraction_failed", ""),
    (
        "failed-flag-text",
        _failed_page(1),
        _FAILED_FLAG,
        OCR_TEXT,
        "ocr",
        None,
        OCR_TEXT,
    ),
    (
        "failed-flag-error",
        _failed_page(1),
        _FAILED_FLAG,
        _OCR_FAILURE,
        "none",
        "ocr_error",
        "",
    ),
    ("failed-flag-empty", _failed_page(1), _FAILED_FLAG, "", "none", "ocr_empty", ""),
]


@pytest.mark.parametrize(
    ("page", "settings", "output", "origin", "reason", "text"),
    [row[1:] for row in _ROWS],
    ids=[row[0] for row in _ROWS],
)
def test_each_reachable_row_of_the_page_axes_table(
    monkeypatch, page, settings, output, origin, reason, text
):
    """A3: text_origin, the no-text reason, and the downstream and store text."""
    _stub(monkeypatch, make_document(page), DIGITAL_PREOCR)
    stub = stub_ocr(monkeypatch, {} if output is None else {1: output})
    downstream = _capture_downstream(monkeypatch)

    result = classify_and_assess("fictief.pdf", settings=settings)

    record = result["quality_signals"]["pages"][0]
    assert stub.pages == ([] if output is None else [1])
    assert record["text_origin"] == origin
    assert record["no_text_reason"] == reason
    assert downstream["structure"] == [text]
    assert downstream["type"] == [text]
    store_page = result["page_text"]["pages"][0]
    assert store_page["text"] == text
    assert store_page["text_origin"] == origin
    assert store_page["no_text_reason"] == reason


@pytest.mark.parametrize(
    ("page", "settings", "output", "origin"),
    [row[1:5] for row in _ROWS],
    ids=[row[0] for row in _ROWS],
)
def test_the_store_tier_is_the_page_tier_exactly_on_native_origin_pages(
    monkeypatch, page, settings, output, origin
):
    """Spec 004, D2 (A4), per reachable row: the store's `tier` equals the
    signals' tier when `text_origin` is native and is null otherwise, also for
    a native Garbage page that got OCR text."""
    _stub(monkeypatch, make_document(page), DIGITAL_PREOCR)
    stub_ocr(monkeypatch, {} if output is None else {1: output})

    result = classify_and_assess("fictief.pdf", settings=settings)

    evidence = result["quality_signals"]["pages"][0]
    store_page = result["page_text"]["pages"][0]
    assert result["page_text"]["schema_version"] == 2
    assert store_page["text_origin"] == origin
    if origin == "native":
        assert evidence["tier"] is not None
        assert store_page["tier"] == evidence["tier"]
    else:
        assert store_page["tier"] is None


def test_the_last_row_is_reached_by_a_direct_assess_quality_call():
    """A3's last row (Q5): without OCR outcomes a no_text_layer page is `none`
    with `ocr_skipped`."""
    page = _direct_assessment(make_document(SHORT_FRAGMENT)).pages[0]

    assert page.source == "no_text_layer"
    assert page.ocr_routed is False
    assert page.text_origin == "none"
    assert page.no_text_reason == "ocr_skipped"


def _mentions_awaiting_ocr(result):
    texts = [*result["quality_signals"]["reasons"], result["explanation"]]
    return any("awaiting ocr" in text.lower() for text in texts)


@pytest.mark.parametrize(
    ("pages", "outputs", "label", "awaiting", "mentions_awaiting"),
    [
        (["", ""], {1: OCR_TEXT, 2: OCR_TEXT}, None, None, False),
        (["", ""], {1: "", 2: ""}, None, "all", False),
        ([CLEAN_PAGE] * 3 + [""], {4: OCR_TEXT}, "Decent", None, False),
        ([CLEAN_PAGE] * 3 + [""], {4: _OCR_FAILURE}, "Decent", "partial", False),
    ],
    ids=["scan-ocr-text", "scan-ocr-empty", "mixed-ocr-text", "mixed-ocr-error"],
)
def test_awaiting_ocr_and_the_label_follow_text_origin(
    monkeypatch, pages, outputs, label, awaiting, mentions_awaiting
):
    """A7, and C13: the label is the same with and without OCR outcomes."""
    document = make_document(*pages)
    _stub(monkeypatch, document, DIGITAL_PREOCR)
    stub_ocr(monkeypatch, outputs)

    result = classify_and_assess("fictief.pdf")

    assert result["status"] == "processed"
    assert result["quality_label"] == label
    assert result["awaiting_ocr"] == awaiting
    assert result["quality_signals"]["awaiting_ocr"] == awaiting
    if label is None:
        assert result["explanation"].startswith("No text layer:")
    assert _mentions_awaiting_ocr(result) is mentions_awaiting
    assert result["quality_label"] == _direct_assessment(document).label


# --- Spec 004, D3: "awaiting OCR" only for ocr_skipped (A5) --------------------------

# (id, pages, OCR output per page). No page has a native Garbage tier or a
# failed extraction next to native pages, so every document's reasons carry
# the label-None reason (no native pages) or the no-text-layer reason.
_NO_TEXT_DOCUMENTS = [
    ("scan-ocr-text", ["", ""], {1: OCR_TEXT, 2: OCR_TEXT}),
    ("scan-ocr-empty", ["", ""], {1: "", 2: ""}),
    ("scan-ocr-error", ["", ""], {1: _OCR_FAILURE, 2: _OCR_FAILURE}),
    ("scan-extraction-failed", [_failed_page(1), _failed_page(2)], {}),
    (
        "scan-mixed",
        ["", "", _failed_page(3), "", ""],
        {1: OCR_TEXT, 2: "", 4: _OCR_FAILURE, 5: ""},
    ),
    ("native-ocr-text", [CLEAN_PAGE, ""], {2: OCR_TEXT}),
    ("native-ocr-empty", [CLEAN_PAGE, ""], {2: ""}),
    ("native-ocr-error", [CLEAN_PAGE, ""], {2: _OCR_FAILURE}),
    (
        "native-mixed",
        [CLEAN_PAGE, "", "", ""],
        {2: OCR_TEXT, 3: "", 4: _OCR_FAILURE},
    ),
]


def _has_ocr_skipped_page(pages):
    return any(page["no_text_reason"] == "ocr_skipped" for page in pages)


@pytest.mark.parametrize(
    ("pages", "outputs"),
    [row[1:] for row in _NO_TEXT_DOCUMENTS],
    ids=[row[0] for row in _NO_TEXT_DOCUMENTS],
)
def test_the_parse_pass_mentions_awaiting_ocr_exactly_when_a_page_is_ocr_skipped(
    monkeypatch, pages, outputs
):
    """The parse pass routes every no_text_layer page whose extraction did not
    fail, so no page ends ocr_skipped and neither the reasons nor the
    explanation say "awaiting OCR", whatever the other no-text reasons."""
    _stub(monkeypatch, make_document(*pages), DIGITAL_PREOCR)
    stub_ocr(monkeypatch, outputs)

    result = classify_and_assess("fictief.pdf")

    page_records = result["quality_signals"]["pages"]
    assert any(page["text_origin"] != "native" for page in page_records)
    assert _has_ocr_skipped_page(page_records) is False
    assert _mentions_awaiting_ocr(result) is False


@pytest.mark.parametrize(
    ("pages", "skipped"),
    [
        (["", ""], True),
        ([_failed_page(1), _failed_page(2)], False),
        (["", _failed_page(2)], True),
        ([CLEAN_PAGE, CLEAN_PAGE, ""], True),
    ],
    ids=["all-skipped", "all-failed", "skipped-and-failed", "native-and-skipped"],
)
def test_a_direct_call_mentions_awaiting_ocr_exactly_when_a_page_is_ocr_skipped(
    pages, skipped
):
    """A direct assess_quality call runs no OCR: its no_text_layer pages are
    ocr_skipped unless their extraction failed. For documents whose reasons
    reach the label-None or the no-text-layer reason, they say "awaiting OCR"
    exactly when a page is ocr_skipped."""
    assessment = _direct_assessment(make_document(*pages))

    reasons = " ".join(assessment.reasons).lower()
    has_skipped = any(page.no_text_reason == "ocr_skipped" for page in assessment.pages)
    assert has_skipped is skipped
    assert ("awaiting ocr" in reasons) is skipped


def test_a_higher_priority_reason_may_leave_an_ocr_skipped_page_unmentioned():
    """Spec 004, M1: when the ladder reports a Garbage page first, the
    ocr_skipped page is not named in the reasons, and awaiting_ocr still marks
    it."""
    assessment = _direct_assessment(make_document(GARBAGE_PAGE, ""))

    assert [page.no_text_reason for page in assessment.pages] == [None, "ocr_skipped"]
    assert assessment.label == "Garbage"
    assert assessment.awaiting_ocr == "partial"
    assert not any("awaiting ocr" in reason.lower() for reason in assessment.reasons)


def test_ocr_evidence_is_recorded_on_the_page_and_decides_nothing(monkeypatch):
    """A9."""
    from preocr.core.signals import _compute_text_quality_signals

    tsv = make_tsv(("Fictieve", 90), ("tekst", 80), ("", 12))
    _stub(monkeypatch, make_document(""), SCANNED_PREOCR)
    stub_ocr(monkeypatch, {1: TesseractOutput(text=OCR_TEXT + "\f", tsv=tsv)})

    page = classify_and_assess("fictief.pdf")["quality_signals"]["pages"][0]

    expected = _compute_text_quality_signals(OCR_TEXT)
    assert page["ocr_mean_confidence"] == 85.0
    assert page["ocr_non_printable_ratio"] == expected["non_printable_ratio"]
    assert page["ocr_unicode_noise_ratio"] == expected["unicode_noise_ratio"]
    assert page["ocr_avg_word_length"] == expected["avg_word_length"]
    assert page["tier"] is None


def test_confidence_decides_nothing(monkeypatch):
    """A10: every OCR word at conf 5, then at conf 95."""
    document = make_document("", CLEAN_PAGE, CLEAN_PAGE, "")

    def run(conf):
        _stub(monkeypatch, document, DIGITAL_PREOCR)
        stub_ocr(monkeypatch, {1: tesseract_output(OCR_TEXT, conf), 4: ""})
        return classify_and_assess("fictief.pdf")

    def decisions(result):
        pages = result["quality_signals"]["pages"]
        return (
            [page["text_origin"] for page in pages],
            [page["no_text_reason"] for page in pages],
            result["awaiting_ocr"],
            result["quality_label"],
            result["quality_signals"]["reasons"],
            result["explanation"],
        )

    low, high = run(5), run(95)

    assert low["quality_signals"]["pages"][0]["ocr_mean_confidence"] == 5.0
    assert high["quality_signals"]["pages"][0]["ocr_mean_confidence"] == 95.0
    assert decisions(low) == decisions(high)


def test_the_ocr_roll_up_over_ocr_pages(monkeypatch):
    """A11."""
    _stub(monkeypatch, make_document("", "", CLEAN_PAGE), DIGITAL_PREOCR)
    stub_ocr(
        monkeypatch,
        {1: tesseract_output(OCR_TEXT, 80), 2: tesseract_output(OCR_TEXT, 90)},
    )

    block = classify_and_assess("fictief.pdf")["ocr"]

    assert block["page_count"] == 2
    assert block["mean_page_confidence"] == 85.0
    assert block["min_page_confidence"] == 80.0
    assert block["engine_version"] == "5.5.3"


def test_the_ocr_roll_up_without_ocr_pages(monkeypatch):
    """A11."""
    _stub(monkeypatch, make_document(CLEAN_PAGE, CLEAN_PAGE), DIGITAL_PREOCR)

    block = classify_and_assess("fictief.pdf")["ocr"]

    assert block["page_count"] == 0
    assert block["mean_page_confidence"] is None
    assert block["min_page_confidence"] is None
    assert block["engine_version"] is None


def test_type_and_structure_read_the_ocr_text(monkeypatch):
    """A20: OCR runs before type detection and structure analysis."""
    _stub(monkeypatch, make_document("", ""), SCANNED_PREOCR)
    stub_ocr(monkeypatch, default=LEASE_PAGE)

    result = classify_and_assess("fictief.pdf")

    assert result["detected_type"] == "lease_agreement"
    structure = result["quality_signals"]["structure"]
    assert structure["matched_pattern"] == "numbered_clauses"
    pages = result["quality_signals"]["pages"]
    assert [page["source"] for page in pages] == ["no_text_layer", "no_text_layer"]


def test_file_sha256_is_the_digest_of_the_file_bytes(tmp_path, monkeypatch):
    """A22."""
    path = tmp_path / "fictief.pdf"
    path.write_bytes(b"fabricated bytes, not a real document")
    _stub(monkeypatch, make_document("", LEASE_PAGE), DIGITAL_PREOCR)
    stub_ocr(monkeypatch, {1: OCR_TEXT})

    result = classify_and_assess(path)

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["file"]["sha256"] == expected
    assert result["page_text"]["file"]["sha256"] == expected
    assert result["quality_signals"]["pages"][0]["text_origin"] == "ocr"
    assert json.loads(json.dumps(result)) == result


def test_file_sha256_is_none_for_a_missing_file(tmp_path, monkeypatch):
    """A22: for a failed document, and for a stubbed one whose file is absent."""
    failed = classify_and_assess(tmp_path / "bestaat-niet.pdf")
    assert failed["file"]["sha256"] is None
    assert failed["page_text"]["file"]["sha256"] is None
    assert failed["ocr"] == {}
    assert failed["page_text"]["status"] == "failed"
    assert failed["page_text"]["pages"] == []

    _stub(monkeypatch, make_document(LEASE_PAGE), DIGITAL_PREOCR)
    processed = classify_and_assess(tmp_path / "ook-niet.pdf")
    assert processed["status"] == "processed"
    assert processed["file"]["sha256"] is None
