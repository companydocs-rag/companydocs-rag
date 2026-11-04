"""OCR (spec 003): routing, engine failures, TSV and text, rendering, the cache.

Unit tests replace the seams in ``ocr.py`` -- the page seam (render and
Tesseract, :func:`ocr.ocr_page`), the version query, or the single process
call and the render behind the page seam -- so no unit test starts Tesseract
or renders a real document. The rendering tests render PDFs generated here
from a Pillow image. Exactly one integration test runs the real binary, on a
Pillow-drawn image of fabricated Dutch words; it is skipped when Tesseract or
its ``nld`` data is absent. Page texts are fabricated.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from companydocs_rag.ingestion import cli, ocr
from companydocs_rag.ingestion import parse_pass as parse_pass_module
from companydocs_rag.ingestion.language import LanguageResolver
from companydocs_rag.ingestion.ocr import (
    LazyPdf,
    OcrError,
    OcrRun,
    TesseractOutput,
    mean_word_confidence,
    ocr_text,
    version_token,
)
from companydocs_rag.ingestion.parse_pass import classify_and_assess, file_sha256
from tests.ingestion._helpers import (
    make_document,
    make_page,
    make_tsv,
    stub_ocr,
    tesseract_output,
    write_client_csv,
)
from tests.ingestion.test_quality import CLEAN_PAGE, DIGITAL_PREOCR, GARBLED_PAGE

_LOGGER = "companydocs_rag.ingestion.ocr"
FABRICATED_FAILURE = "ValueError: fabricated page failure"
VERSION_LINE = b"tesseract v5.5.3.20260724\n leptonica-1.87.0\n"


def _four_page_document():
    """[native Clean, native Garbage, no_text_layer, extraction failed]."""
    return make_document(
        CLEAN_PAGE,
        GARBLED_PAGE,
        "",
        make_page(4, "", extraction_failed=True, extraction_error=FABRICATED_FAILURE),
    )


def _stub_pipeline(monkeypatch, document):
    monkeypatch.setattr(parse_pass_module, "extract_pages", lambda path: document)
    monkeypatch.setattr(parse_pass_module, "run_preocr", lambda path: DIGITAL_PREOCR)


def _ocr_settings(**values):
    return {"ingestion": {"parse_pass": {"ocr": values}}}


def _pages(result):
    return result["quality_signals"]["pages"]


def _store_texts(result):
    return [page["text"] for page in result["page_text"]["pages"]]


def _placeholder(path: Path, data: bytes = b"fabricated bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _fake_render(pdf, page_number, dpi, grayscale):
    return Image.new("L" if grayscale else "RGB", (40, 20), 255)


class FakeTesseract:
    """Stands in for the process call: records every call and, for an OCR
    call, writes the two output files Tesseract would write."""

    def __init__(
        self,
        text="Fictieve regel een\n\nFictieve regel twee\n\f",
        tsv=None,
        ocr_failure=None,
        version_failure=None,
    ):
        self.text = text
        self.tsv = tsv if tsv is not None else make_tsv(("Fictieve", 91), ("regel", 89))
        self.ocr_failure = ocr_failure
        self.version_failure = version_failure
        self.calls = []

    def __call__(self, args, timeout_seconds):
        args = list(args)
        self.calls.append(args)
        if args[1] == "--version":
            if self.version_failure is not None:
                raise self.version_failure
            return subprocess.CompletedProcess(args, 0, VERSION_LINE, b"")
        if self.ocr_failure == "nonzero":
            return subprocess.CompletedProcess(
                args, 1, b"", b"fabricated tesseract complaint"
            )
        if self.ocr_failure == "timeout":
            raise subprocess.TimeoutExpired(args, timeout_seconds)
        if isinstance(self.ocr_failure, BaseException):
            raise self.ocr_failure
        output_base = Path(args[2])
        output_base.with_suffix(".txt").write_text(self.text, encoding="utf-8")
        output_base.with_suffix(".tsv").write_text(self.tsv, encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    @property
    def ocr_calls(self):
        return [call for call in self.calls if call[1] != "--version"]

    @property
    def version_calls(self):
        return [call for call in self.calls if call[1] == "--version"]


def _make_pdf(path: Path, width: int, height: int, pages: int = 1) -> Path:
    """A PDF of ``width`` x ``height`` points, generated from Pillow images."""
    images = []
    for number in range(pages):
        image = Image.new("RGB", (width, height), "white")
        ImageDraw.Draw(image).rectangle((20, 20, 80 + number, 60), fill="black")
        images.append(image)
    images[0].save(
        path, "PDF", resolution=72.0, save_all=True, append_images=images[1:]
    )
    return path


# --- Routing (A4, A5) -------------------------------------------------------------


def test_default_routing_sends_only_the_no_text_layer_page(monkeypatch):
    _stub_pipeline(monkeypatch, _four_page_document())
    stub = stub_ocr(monkeypatch, {3: "Fictieve OCR tekst op pagina drie"})

    result = classify_and_assess("fictief.pdf")

    pages = _pages(result)
    assert stub.pages == [3]
    assert [page["ocr_routed"] for page in pages] == [False, False, True, False]
    assert pages[3]["text_origin"] == "none"
    assert pages[3]["no_text_reason"] == "extraction_failed"
    assert pages[3]["extraction_error"] == FABRICATED_FAILURE


def test_the_garbage_flag_routes_the_garbage_page_which_keeps_source_and_tier(
    monkeypatch,
):
    _stub_pipeline(monkeypatch, _four_page_document())
    ocr_page_two = "Fictieve heldere OCR tekst voor pagina twee van dit document."
    stub = stub_ocr(monkeypatch, {2: ocr_page_two, 3: "Fictieve tekst drie"})

    result = classify_and_assess(
        "fictief.pdf", settings=_ocr_settings(ocr_garbage_tier_pages=True)
    )

    page = _pages(result)[1]
    assert stub.pages == [2, 3]
    assert page["ocr_routed"] is True
    assert (page["source"], page["tier"]) == ("native", "Garbage")
    assert page["text_origin"] == "ocr"
    assert _store_texts(result)[1] == ocr_page_two


def test_a_flag_routed_garbage_page_whose_ocr_fails_stays_native(monkeypatch):
    _stub_pipeline(monkeypatch, _four_page_document())
    stub_ocr(monkeypatch, {2: RuntimeError("fabricated OCR failure")})

    result = classify_and_assess(
        "fictief.pdf", settings=_ocr_settings(ocr_garbage_tier_pages=True)
    )

    page = _pages(result)[1]
    assert (page["source"], page["tier"]) == ("native", "Garbage")
    assert page["text_origin"] == "native"
    assert "fabricated OCR failure" in page["ocr_error"]
    assert _store_texts(result)[1] == GARBLED_PAGE


def test_the_extraction_failed_flag_routes_the_failed_page(monkeypatch):
    _stub_pipeline(monkeypatch, _four_page_document())
    stub = stub_ocr(monkeypatch, {4: "Fictieve tekst op de mislukte pagina"})

    result = classify_and_assess(
        "fictief.pdf", settings=_ocr_settings(ocr_extraction_failed_pages=True)
    )

    page = _pages(result)[3]
    assert stub.pages == [3, 4]
    assert page["text_origin"] == "ocr"
    assert page["extraction_failed"] is True


def test_both_flags_route_pages_two_three_and_four(monkeypatch):
    _stub_pipeline(monkeypatch, _four_page_document())
    stub = stub_ocr(monkeypatch)

    classify_and_assess(
        "fictief.pdf",
        settings=_ocr_settings(
            ocr_garbage_tier_pages=True, ocr_extraction_failed_pages=True
        ),
    )

    assert stub.pages == [2, 3, 4]


def test_a_failed_page_with_ocr_text_still_rules_out_clean(monkeypatch):
    document = make_document(
        CLEAN_PAGE,
        CLEAN_PAGE,
        make_page(3, "", extraction_failed=True, extraction_error=FABRICATED_FAILURE),
    )
    _stub_pipeline(monkeypatch, document)
    stub_ocr(monkeypatch, {3: "Fictieve tekst op de mislukte pagina"})

    result = classify_and_assess(
        "fictief.pdf", settings=_ocr_settings(ocr_extraction_failed_pages=True)
    )

    assert _pages(result)[2]["text_origin"] == "ocr"
    assert result["quality_label"] == "Decent"


# --- Engine failures (A6) -----------------------------------------------------------


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("seam", "RuntimeError: fabricated seam failure"),
        ("render", "render failed: RuntimeError: fabricated render failure"),
        ("nonzero", "exited with code 1: fabricated tesseract complaint"),
        ("timeout", "timed out"),
        ("missing", "could not be started"),
    ],
)
def test_each_engine_failure_gives_ocr_error_a_message_and_a_warning(
    monkeypatch, caplog, failure, expected
):
    _stub_pipeline(monkeypatch, make_document(""))
    if failure == "seam":
        stub_ocr(monkeypatch, {1: RuntimeError("fabricated seam failure")})
    else:

        def failing_render(pdf, page_number, dpi, grayscale):
            raise RuntimeError("fabricated render failure")

        monkeypatch.setattr(
            ocr, "render_page", failing_render if failure == "render" else _fake_render
        )
        fake = FakeTesseract(
            ocr_failure=failure if failure in ("nonzero", "timeout") else None,
            version_failure=(
                FileNotFoundError(2, "fabricated: no such binary")
                if failure == "missing"
                else None
            ),
        )
        monkeypatch.setattr(ocr, "_run_process", fake)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = classify_and_assess("fictief_scan.pdf")

    page = _pages(result)[0]
    assert result["status"] == "processed"
    assert page["text_origin"] == "none"
    assert page["no_text_reason"] == "ocr_error"
    assert expected in page["ocr_error"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("fictief_scan.pdf" in m and "page 1" in m for m in warnings)


def test_whitespace_only_ocr_text_is_ocr_empty(monkeypatch):
    _stub_pipeline(monkeypatch, make_document(""))
    stub_ocr(monkeypatch, {1: TesseractOutput(text=" \n\t \n\f", tsv=make_tsv())})

    page = _pages(classify_and_assess("fictief.pdf"))[0]

    assert page["text_origin"] == "none"
    assert page["no_text_reason"] == "ocr_empty"
    assert page["ocr_error"] is None


def test_an_error_message_is_bounded_in_length(monkeypatch):
    _stub_pipeline(monkeypatch, make_document(""))
    stub_ocr(monkeypatch, {1: RuntimeError("x" * 5000)})

    page = _pages(classify_and_assess("fictief.pdf"))[0]

    assert len(page["ocr_error"]) <= ocr.MAX_ERROR_CHARS


def test_without_tesseract_a_run_keeps_every_document_and_logs_one_error(
    tmp_path, monkeypatch, caplog
):
    folder = tmp_path / "invoer"
    _placeholder(folder / "fictief_a.pdf")
    _placeholder(folder / "fictief_b.pdf")
    _stub_pipeline(monkeypatch, make_document("", "x" * 300, ""))
    monkeypatch.setattr(ocr, "render_page", _fake_render)
    fake = FakeTesseract(version_failure=FileNotFoundError(2, "fabricated: missing"))
    monkeypatch.setattr(ocr, "_run_process", fake)

    with caplog.at_level(logging.WARNING):
        results = cli.process_folder(folder, tmp_path / "report.csv")

    assert [result["status"] for result in results] == ["processed", "processed"]
    for result in results:
        routed = [page for page in _pages(result) if page["ocr_routed"]]
        assert [page["page_number"] for page in routed] == [1, 3]
        assert all(page["text_origin"] == "none" for page in routed)
        assert result["ocr"]["engine_version"] is None
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "Tesseract is unavailable" in errors[0]
    assert len(fake.version_calls) == 1
    assert fake.ocr_calls == []


# --- One Tesseract run per page (A13) -----------------------------------------------


def test_one_tesseract_process_per_routed_page_with_text_tsv_language_and_dpi(
    tmp_path, monkeypatch
):
    pdf = _placeholder(tmp_path / "invoer" / "fictief_scan.pdf")
    overrides = write_client_csv(
        tmp_path / "overrides.csv", ("Bestandsnaam", "Taal"), [(pdf.name, "SK")]
    )
    _stub_pipeline(monkeypatch, make_document("", "x" * 300, ""))
    monkeypatch.setattr(ocr, "render_page", _fake_render)
    fake = FakeTesseract(text="Fictieve regel een\n\nFictieve regel twee\n\f")
    monkeypatch.setattr(ocr, "_run_process", fake)
    cache_dir = tmp_path / "cache"

    result = classify_and_assess(
        pdf,
        language_resolver=LanguageResolver(overrides),
        ocr_run=OcrRun(cache_dir),
    )

    assert len(fake.ocr_calls) == 2
    for args in fake.ocr_calls:
        assert args[-2:] == ["txt", "tsv"]
        assert args[args.index("-l") + 1] == "slk"
        assert args[args.index("--dpi") + 1] == "300"
        assert not Path(args[1]).parent.exists()
    assert list((cache_dir / ocr.TEMP_SUBDIR).iterdir()) == []
    texts = _store_texts(result)
    assert texts[0] == texts[2] == "Fictieve regel een\n\nFictieve regel twee\n"


def test_without_a_cache_dir_temporary_files_go_to_system_temp_and_are_removed(
    tmp_path, monkeypatch
):
    _stub_pipeline(monkeypatch, make_document(""))
    monkeypatch.setattr(ocr, "render_page", _fake_render)
    fake = FakeTesseract()
    monkeypatch.setattr(ocr, "_run_process", fake)

    classify_and_assess(_placeholder(tmp_path / "fictief.pdf"))

    (args,) = fake.ocr_calls
    temp_folder = Path(args[1]).parent
    assert temp_folder.parent == Path(tempfile.gettempdir())
    assert not temp_folder.exists()


# --- TSV and text (A9, A13) ---------------------------------------------------------


def test_page_confidence_is_the_mean_of_word_rows_with_text():
    tsv = make_tsv(("verhuurder", 90), ("huurder", 80), ("", 5))
    assert mean_word_confidence(tsv) == 85.0


def test_page_confidence_is_rounded_to_two_decimals():
    assert mean_word_confidence(make_tsv(("een", 90.123), ("twee", 80))) == 85.06


def test_page_confidence_is_none_without_word_rows():
    assert mean_word_confidence(make_tsv()) is None
    assert mean_word_confidence("") is None


def test_ocr_text_drops_only_the_trailing_page_separator():
    assert ocr_text("Regel een\n\nRegel twee\n\f") == "Regel een\n\nRegel twee\n"
    assert ocr_text("Geen scheidingsteken") == "Geen scheidingsteken"
    assert ocr_text("Twee\f\f") == "Twee\f"


# --- Rendering (A12) ----------------------------------------------------------------


def test_a_page_renders_to_grayscale_at_300_dpi(tmp_path):
    pdf = LazyPdf(_make_pdf(tmp_path / "gegenereerd.pdf", 200, 300))
    try:
        image = ocr.render_page(pdf, 1, 300, True)
    finally:
        pdf.close()

    assert image.mode == "L"
    assert image.size == (math.ceil(200 * 300 / 72), math.ceil(300 * 300 / 72))


def test_dpi_and_grayscale_are_read_from_settings_at_call_time(tmp_path, monkeypatch):
    pdf = _make_pdf(tmp_path / "gegenereerd.pdf", 200, 300)
    captured = []

    def capture(image, **kwargs):
        captured.append((image.mode, image.size, kwargs["dpi"]))
        return tesseract_output("Fictief")

    monkeypatch.setattr(ocr, "run_tesseract", capture)
    monkeypatch.setattr(ocr, "query_tesseract_version", lambda cmd, timeout: "5.5.3")
    run = OcrRun()

    for settings in ({}, _ocr_settings(dpi=150), _ocr_settings(grayscale=False)):
        run.ocr_document(pdf, [1], sha256=None, language="nld", settings=settings)

    full = (math.ceil(200 * 300 / 72), math.ceil(300 * 300 / 72))
    assert captured[0] == ("L", full, 300)
    mode, (width, height), dpi = captured[1]
    assert (mode, dpi) == ("L", 150)
    assert abs(width - full[0] / 2) <= 1
    assert abs(height - full[1] / 2) <= 1
    assert captured[2][0] == "RGB"


def test_the_pdf_is_opened_once_per_document_and_closed(tmp_path, monkeypatch):
    pdf = _make_pdf(tmp_path / "gegenereerd.pdf", 200, 300, pages=2)
    opened = []
    real_document = ocr.pdfium.PdfDocument

    def counting_document(*args, **kwargs):
        document = real_document(*args, **kwargs)
        opened.append(document)
        return document

    monkeypatch.setattr(ocr.pdfium, "PdfDocument", counting_document)
    monkeypatch.setattr(
        ocr, "run_tesseract", lambda image, **kwargs: tesseract_output("Fictief")
    )
    monkeypatch.setattr(ocr, "query_tesseract_version", lambda cmd, timeout: "5.5.3")
    closed = []
    monkeypatch.setattr(ocr.LazyPdf, "close", lambda self: closed.append(self))

    result = OcrRun().ocr_document(pdf, [1, 2], sha256=None, language="nld")

    assert len(opened) == 1
    assert len(closed) == 1
    assert [result.outcomes[n].text for n in (1, 2)] == ["Fictief", "Fictief"]
    opened[0].close()


# --- Version --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "token"),
    [
        ("tesseract v5.5.3.20260724", "v5.5.3.20260724"),
        ("tesseract 5.3.0", "5.3.0"),
        ("iets onverwachts", "iets onverwachts"),
    ],
)
def test_the_version_line_is_reduced_to_its_token(line, token):
    assert version_token(line) == token


def test_the_version_query_reads_the_first_line(monkeypatch):
    monkeypatch.setattr(ocr, "_run_process", FakeTesseract())
    assert ocr.query_tesseract_version("tesseract", 10) == "v5.5.3.20260724"


def test_a_failing_version_query_raises(monkeypatch):
    def nonzero(args, timeout_seconds):
        return subprocess.CompletedProcess(args, 3, b"", b"fabricated")

    monkeypatch.setattr(ocr, "_run_process", nonzero)
    with pytest.raises(OcrError):
        ocr.query_tesseract_version("tesseract", 10)


def test_the_version_is_never_queried_for_a_document_without_routed_pages(
    monkeypatch,
):
    _stub_pipeline(monkeypatch, make_document(CLEAN_PAGE, CLEAN_PAGE))
    stub = stub_ocr(monkeypatch)

    result = classify_and_assess("fictief.pdf")

    assert stub.version_calls == 0
    assert result["ocr"]["engine_version"] is None
    assert result["page_text"]["ocr_engine"] is None


def test_the_version_is_queried_once_per_run(tmp_path, monkeypatch):
    folder = tmp_path / "invoer"
    for name in ("fictief_a.pdf", "fictief_b.pdf", "fictief_c.pdf"):
        _placeholder(folder / name)
    _stub_pipeline(monkeypatch, make_document("", ""))
    stub = stub_ocr(monkeypatch, default="Fictieve tekst")

    results = cli.process_folder(folder, tmp_path / "report.csv")

    assert stub.version_calls == 1
    assert len(stub.page_calls) == 6
    assert {result["ocr"]["engine_version"] for result in results} == {"5.5.3"}


# --- Cache (A15, A16) --------------------------------------------------------


def test_a_second_run_is_served_from_the_cache(tmp_path, monkeypatch, caplog):
    pdf = _placeholder(tmp_path / "invoer" / "fictief.pdf")
    _stub_pipeline(monkeypatch, make_document("", "x" * 300, ""))
    renders = []

    def counting_render(pdf, page_number, dpi, grayscale):
        renders.append(page_number)
        return _fake_render(pdf, page_number, dpi, grayscale)

    monkeypatch.setattr(ocr, "render_page", counting_render)
    fake = FakeTesseract(
        text="Fictieve tekst uit de cache\n\f",
        tsv=make_tsv(("Fictieve", 80), ("tekst", 90)),
    )
    monkeypatch.setattr(ocr, "_run_process", fake)
    cache_dir = tmp_path / "cache"

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        first = classify_and_assess(pdf, ocr_run=OcrRun(cache_dir))
        assert renders == [1, 3]
        assert len(fake.ocr_calls) == 2

        entries = sorted(cache_dir.glob("*.json"))
        assert len(entries) == 2
        stored = [json.loads(entry.read_text(encoding="utf-8")) for entry in entries]
        assert all(set(entry) == {"key", "text", "tsv"} for entry in stored)
        assert sorted(entry["key"]["page_number"] for entry in stored) == [1, 3]
        assert all(
            entry["text"] == "Fictieve tekst uit de cache\n\f" for entry in stored
        )
        assert all("Fictieve" in entry["tsv"] for entry in stored)

        second = classify_and_assess(pdf, ocr_run=OcrRun(cache_dir))

    # Neither rendering nor a Tesseract OCR run the second time. The version
    # is still queried once per run: it is part of the cache key.
    assert renders == [1, 3]
    assert len(fake.ocr_calls) == 2
    assert len(fake.version_calls) == 2
    assert second["page_text"] == first["page_text"]
    assert [p["ocr_mean_confidence"] for p in _pages(second)] == [
        p["ocr_mean_confidence"] for p in _pages(first)
    ]
    assert _pages(second)[0]["ocr_mean_confidence"] == 85.0
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any(
        "2 page(s) routed, 0 served from the cache, 2 OCR'd now" in m for m in info
    )
    assert any(
        "2 page(s) routed, 2 served from the cache, 0 OCR'd now" in m for m in info
    )


KEY_CHANGES = [
    "file bytes",
    "page number",
    "dpi",
    "grayscale",
    "language",
    "tesseract version",
    "cache format version",
]


@pytest.mark.parametrize("change", KEY_CHANGES)
def test_changing_any_one_key_component_is_a_miss(tmp_path, monkeypatch, change):
    pdf = _placeholder(tmp_path / "fictief.pdf", b"fabricated bytes")
    stub = stub_ocr(monkeypatch, default="Fictieve tekst")
    cache_dir = tmp_path / "cache"

    def run(page=1, settings=None, language="nld"):
        OcrRun(cache_dir).ocr_document(
            pdf,
            [page],
            sha256=file_sha256(pdf),
            language=language,
            settings=settings if settings is not None else {},
        )

    run()
    run()
    assert len(stub.page_calls) == 1  # the same key is a hit

    if change == "file bytes":
        pdf.write_bytes(b"other fabricated bytes")
        run()
    elif change == "page number":
        run(page=2)
    elif change == "dpi":
        run(settings=_ocr_settings(dpi=150))
    elif change == "grayscale":
        run(settings=_ocr_settings(grayscale=False))
    elif change == "language":
        run(language="eng")
    elif change == "tesseract version":
        monkeypatch.setattr(
            ocr, "query_tesseract_version", lambda cmd, timeout: "5.5.4"
        )
        run()
    elif change == "cache format version":
        monkeypatch.setattr(ocr, "CACHE_FORMAT_VERSION", ocr.CACHE_FORMAT_VERSION + 1)
        run()

    assert len(stub.page_calls) == 2
    assert len(list(cache_dir.glob("*.json"))) == 2


def test_a_corrupt_entry_is_a_miss_logged_and_rewritten(tmp_path, monkeypatch, caplog):
    pdf = _placeholder(tmp_path / "fictief.pdf")
    stub = stub_ocr(monkeypatch, default="Fictieve tekst")
    cache_dir = tmp_path / "cache"

    def run():
        return OcrRun(cache_dir).ocr_document(
            pdf, [1], sha256=file_sha256(pdf), language="nld", settings={}
        )

    run()
    (entry,) = cache_dir.glob("*.json")
    entry.write_text("{geen geldige json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = run()

    assert len(stub.page_calls) == 2
    assert result.outcomes[1].text == "Fictieve tekst"
    assert any(entry.name in r.getMessage() for r in caplog.records)
    assert json.loads(entry.read_text(encoding="utf-8"))["text"] == "Fictieve tekst\f"


def test_an_ocr_error_is_not_cached_and_is_retried(tmp_path, monkeypatch):
    pdf = _placeholder(tmp_path / "fictief.pdf")
    cache_dir = tmp_path / "cache"

    def run():
        return OcrRun(cache_dir).ocr_document(
            pdf, [1], sha256=file_sha256(pdf), language="nld", settings={}
        )

    stub_ocr(monkeypatch, {1: RuntimeError("fabricated failure")})
    assert run().outcomes[1].error is not None
    assert list(cache_dir.glob("*.json")) == []

    retry = stub_ocr(monkeypatch, default="Fictieve tekst")
    assert run().outcomes[1].text == "Fictieve tekst"
    assert len(retry.page_calls) == 1
    assert len(list(cache_dir.glob("*.json"))) == 1


def test_an_empty_ocr_result_is_cached(tmp_path, monkeypatch):
    pdf = _placeholder(tmp_path / "fictief.pdf")
    stub = stub_ocr(monkeypatch)  # empty text
    cache_dir = tmp_path / "cache"

    for _ in range(2):
        OcrRun(cache_dir).ocr_document(
            pdf, [1], sha256=file_sha256(pdf), language="nld", settings={}
        )

    assert len(stub.page_calls) == 1


def test_without_a_cache_dir_nothing_is_cached_or_written(tmp_path, monkeypatch):
    pdf = _placeholder(tmp_path / "fictief.pdf")
    stub_ocr(monkeypatch, default="Fictieve tekst")

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("the cache must not be used without a cache directory")

    monkeypatch.setattr(ocr, "read_cache_entry", must_not_be_called)
    monkeypatch.setattr(ocr, "write_cache_entry", must_not_be_called)
    monkeypatch.chdir(tmp_path)

    result = OcrRun().ocr_document(
        pdf, [1], sha256=file_sha256(pdf), language="nld", settings={}
    )

    assert result.outcomes[1].text == "Fictieve tekst"
    assert [path.name for path in tmp_path.iterdir()] == ["fictief.pdf"]


def test_a_failed_cache_write_is_logged_and_changes_nothing(
    tmp_path, monkeypatch, caplog
):
    pdf = _placeholder(tmp_path / "fictief.pdf")
    monkeypatch.setattr(ocr, "render_page", _fake_render)
    monkeypatch.setattr(ocr, "_run_process", FakeTesseract(text="Fictieve tekst\f"))

    def run(cache_dir):
        return OcrRun(cache_dir).ocr_document(
            pdf, [1], sha256=file_sha256(pdf), language="nld", settings={}
        )

    expected = run(tmp_path / "werkende-cache").outcomes

    # A usable cache directory: rendering, Tesseract and its temporary folder
    # inside the cache all work. Only moving the written entry into place
    # fails, as when another process holds the file.
    cache_dir = tmp_path / "cache"
    real_replace = os.replace

    def replace(src, dst):
        if Path(dst).parent == cache_dir:
            raise PermissionError(13, "fabricated: entry locked", str(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = run(cache_dir)

    assert result.outcomes == expected
    assert result.outcomes[1].text == "Fictieve tekst"
    assert result.outcomes[1].error is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Could not write OCR cache entry" in warnings[0]
    # No entry and no leftover temporary file: only the emptied temp folder.
    assert [path.name for path in cache_dir.iterdir()] == [ocr.TEMP_SUBDIR]
    assert list((cache_dir / ocr.TEMP_SUBDIR).iterdir()) == []


# --- Integration: the real binary (A14) ---------------------------------------


def test_integration_real_tesseract_reads_fabricated_dutch_words(tmp_path):
    """The only test that starts Tesseract. Skipped without it or its nld data."""
    binary = shutil.which(ocr.TESSERACT_CMD)
    if binary is None:
        pytest.skip("Tesseract is not on PATH")
    languages = subprocess.run(
        [binary, "--list-langs"], capture_output=True, text=True, timeout=60
    )
    if "nld" not in languages.stdout.split():
        pytest.skip("Tesseract has no nld language data")

    lines = ["Fictieve huurovereenkomst", "De verhuurder woont aan de voorbeeldstraat"]
    image = Image.new("L", (2000, 420), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=72)
    for number, line in enumerate(lines):
        draw.text((80, 80 + 160 * number), line, font=font, fill=0)
    pdf = tmp_path / "gegenereerd.pdf"
    image.save(pdf, "PDF", resolution=200.0)

    result = OcrRun(tmp_path / "cache").ocr_document(
        pdf, [1], sha256=file_sha256(pdf), language="nld", settings={}
    )

    outcome = result.outcomes[1]
    assert outcome.error is None, outcome.error
    text = outcome.text.lower()
    for word in ("fictieve", "huurovereenkomst", "verhuurder", "voorbeeldstraat"):
        assert word in text
    assert 0 <= outcome.mean_confidence <= 100
    assert result.engine_version
