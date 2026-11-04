"""Tests for extract_pages: failure handling, logging and error propagation.

Only the failure paths are covered here. No real PDF is opened for the
per-page failure case: pdfplumber is monkeypatched instead, so a fabricated
page-text stub is all that is needed.
"""

from __future__ import annotations

import logging

from companydocs_rag.ingestion import extraction

_LOGGER_NAME = "companydocs_rag.ingestion.extraction"


class _FakePage:
    def __init__(
        self,
        text: str = "",
        raises: Exception | None = None,
        images: list | None = None,
        images_raises: Exception | None = None,
        width: float = 600.0,
        height: float = 800.0,
    ):
        self._text = text
        self._raises = raises
        self._images = images if images is not None else []
        self._images_raises = images_raises
        self.width = width
        self.height = height

    def extract_text(self):
        if self._raises is not None:
            raise self._raises
        return self._text

    @property
    def images(self):
        """A property (not a plain attribute) so a test can make it raise,
        independently of ``extract_text``, to exercise the image-geometry
        failure path."""
        if self._images_raises is not None:
            raise self._images_raises
        return self._images


class _FakePDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


# --- Document-level failures -------------------------------------------------


def test_missing_file_is_reported_and_logged(tmp_path, caplog):
    missing = tmp_path / "does-not-exist.pdf"

    with caplog.at_level(logging.ERROR, logger=_LOGGER_NAME):
        document = extraction.extract_pages(missing)

    assert document.extraction_error == f"File not found: {missing}"
    assert document.page_count == 0

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    message = error_records[0].getMessage()
    assert str(missing) in message


def test_corrupt_non_pdf_file_is_reported_and_logged(tmp_path, caplog):
    """A .pdf file that is not actually a PDF, run through the real extract_pages."""
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a pdf")

    with caplog.at_level(logging.ERROR, logger=_LOGGER_NAME):
        document = extraction.extract_pages(corrupt)

    assert document.extraction_error is not None
    assert document.page_count == 0

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    message = error_records[0].getMessage()
    assert str(corrupt) in message


# --- Page-level failures ------------------------------------------------------


def test_a_failing_page_is_logged_flagged_and_does_not_sink_the_document(
    monkeypatch, tmp_path, caplog
):
    pdf_path = tmp_path / "lease.pdf"
    pdf_path.write_bytes(b"stand-in bytes; pdfplumber.open is monkeypatched below")

    fake_pages = [
        _FakePage(text="Artikel 1 Definities\nFabricated clause text, page one.\n"),
        _FakePage(raises=ValueError("corrupted content stream")),
        _FakePage(
            text="Artikel 3 Slotbepalingen\nFabricated clause text, page three.\n"
        ),
    ]
    monkeypatch.setattr(
        extraction.pdfplumber, "open", lambda path: _FakePDF(fake_pages)
    )

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        document = extraction.extract_pages(pdf_path)

    # The document as a whole is not sunk: all three pages are present.
    assert document.extraction_error is None
    assert document.page_count == 3

    failed_page = document.pages[1]
    assert failed_page.page_number == 2
    assert failed_page.extraction_failed is True
    assert failed_page.extraction_error == "ValueError: corrupted content stream"
    assert failed_page.text == ""

    # The surrounding pages extracted normally.
    assert document.pages[0].extraction_failed is False
    assert document.pages[2].extraction_failed is False

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    record = warning_records[0]
    # Format string is "Failed to extract text from %s page %d: %s: %s"
    # (extraction.py); check the page number via %-args rather than via the
    # rendered message, which could also contain a stray "2" from the tmp
    # path pytest generates. This fails if the page number were wrong.
    assert record.args[1] == 2
    message = record.getMessage()
    assert str(pdf_path) in message
    assert "page 2: " in message
    assert "ValueError" in message
    assert "corrupted content stream" in message


# --- Per-page image geometry (spec 001, Axis 2 evidence) ---------------------


def _extract_single_page(monkeypatch, tmp_path, page):
    pdf_path = tmp_path / "lease.pdf"
    pdf_path.write_bytes(b"stand-in bytes; pdfplumber.open is monkeypatched below")
    monkeypatch.setattr(extraction.pdfplumber, "open", lambda path: _FakePDF([page]))
    return extraction.extract_pages(pdf_path).pages[0]


def test_image_coverage_is_computed_from_page_images(monkeypatch, tmp_path):
    """One image covering a quarter of a 600x800 page -> ratio 0.25."""
    page = _FakePage(
        text="Artikel 1 Definities",
        images=[{"x0": 0.0, "x1": 300.0, "y0": 0.0, "y1": 400.0}],
        width=600.0,
        height=800.0,
    )
    result = _extract_single_page(monkeypatch, tmp_path, page)

    assert result.image_count == 1
    assert result.image_area_ratio == 0.25


def test_a_page_with_no_images_has_zero_coverage(monkeypatch, tmp_path):
    page = _FakePage(text="Artikel 1 Definities")
    result = _extract_single_page(monkeypatch, tmp_path, page)

    assert result.image_count == 0
    assert result.image_area_ratio == 0.0


def test_multiple_images_are_summed_and_counted(monkeypatch, tmp_path):
    page = _FakePage(
        text="",
        images=[
            {"x0": 0.0, "x1": 100.0, "y0": 0.0, "y1": 100.0},
            {"x0": 0.0, "x1": 100.0, "y0": 0.0, "y1": 100.0},
        ],
        width=200.0,
        height=200.0,
    )
    result = _extract_single_page(monkeypatch, tmp_path, page)

    assert result.image_count == 2
    assert result.image_area_ratio == 0.5


def test_image_coverage_is_none_when_page_area_is_unknown(monkeypatch, tmp_path):
    page = _FakePage(
        text="",
        images=[{"x0": 0, "x1": 10, "y0": 0, "y1": 10}],
        width=0.0,
        height=0.0,
    )
    result = _extract_single_page(monkeypatch, tmp_path, page)

    assert result.image_area_ratio is None


def test_a_failed_page_carries_no_image_geometry(monkeypatch, tmp_path):
    """A page that raised during extraction has nothing to measure."""
    page = _FakePage(raises=ValueError("boom"))
    result = _extract_single_page(monkeypatch, tmp_path, page)

    assert result.extraction_failed is True
    assert result.image_area_ratio is None
    assert result.image_count == 0


def test_geometry_failure_preserves_text_and_leaves_extraction_failed_false(
    monkeypatch, tmp_path, caplog
):
    """Image geometry is evidence only (spec 001): a failure computing it must
    not discard already-extracted text, must not flip ``extraction_failed``
    (which would incorrectly sway the source axis in quality.py), and must be
    logged under its own cause rather than misattributed to text extraction.
    """
    page = _FakePage(
        text="Artikel 1 Definities\nFabricated clause text.",
        images_raises=RuntimeError("images unavailable"),
    )
    pdf_path = tmp_path / "lease.pdf"
    pdf_path.write_bytes(b"stand-in bytes; pdfplumber.open is monkeypatched below")
    monkeypatch.setattr(extraction.pdfplumber, "open", lambda path: _FakePDF([page]))

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        document = extraction.extract_pages(pdf_path)

    result = document.pages[0]
    assert result.text == "Artikel 1 Definities\nFabricated clause text."
    assert result.extraction_failed is False
    assert result.image_area_ratio is None
    assert result.image_count == 0
    assert result.extraction_error is not None
    assert "image geometry" in result.extraction_error

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    message = warning_records[0].getMessage()
    assert "image geometry" in message
    assert "page 1" in message
    assert "Failed to extract text" not in message
