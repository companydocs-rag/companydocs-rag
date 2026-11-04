"""Single page-level text extraction shared by both parse-pass analyses.

Document type detection and formatting-consistency analysis are two readings of
*one* extraction, not two passes over the file: whether a type can be
recognised at all depends on how well the text came out. This module produces
that one extraction.

Why pdfplumber directly
-----------------------
This calls ``pdfplumber``'s ``page.extract_text()`` -- byte for byte the same
call preocr's own ``probes/pdf_probe.py`` makes to decide whether a document
needs OCR. On pages with native text, layer 2 therefore reasons about exactly
the text preocr judged, not a second opinion from a different engine; pages
that get OCR text (spec 003) contribute Tesseract's text instead.

Two things prevent us from simply reading the text back out of preocr:

1. ``preocr.needs_ocr()`` opens and extracts the PDF internally and offers no
   hook to inject or retrieve the page text; its page-level output carries
   character counts only, no text.
2. ``pdf_probe.extract_pdf_text()`` truncates the text it returns to the first
   1000 characters (see ``_extract_with_pdfplumber``), which is roughly one
   page. That is enough for preocr's own scan/digital gate but far too little
   for keyword and structure analysis over a 30-page lease agreement.

So preocr does re-open the file once for its own decision. That duplicated read
is the price of not reimplementing preocr's decision engine, which is
deliberately out of scope for this layer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import pdfplumber

logger = logging.getLogger(__name__)

_WHITESPACE_RUN = re.compile(r"\s+")


def normalise_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces.

    Multi-word keywords such as "algemene bepalingen" or "datum van
    oprichting" are regularly broken across a line wrap in extracted PDF text.
    Matching against the whitespace-collapsed form keeps those matches instead
    of silently losing them.
    """
    return _WHITESPACE_RUN.sub(" ", text).strip()


@dataclass
class PageText:
    """The extracted text of a single page."""

    page_number: int  # 1-indexed, matching preocr's page numbering
    text: str  # raw extracted text, line breaks preserved
    extraction_failed: bool = False  # this page raised during extraction
    # Set when extraction_failed is True (the page's own text extraction
    # raised). Also set, with a "image geometry failed: " prefix, when only
    # the image-geometry computation below raised for an otherwise
    # successfully extracted page -- in that case extraction_failed stays
    # False, since the page's text is intact and the source axis (quality.py)
    # must not be swayed by a geometry-only failure.
    extraction_error: Optional[str] = None
    # Image coverage (summed image bbox area / page area) and count, from
    # `page.images` during this same extraction pass. Recorded as evidence for
    # the quality layer's source axis; it decides nothing on its own -- see
    # quality.py for why. None when the page area could not be determined
    # (e.g. a page that failed to extract).
    image_area_ratio: Optional[float] = None
    image_count: int = 0

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def flat_text(self) -> str:
        """Whitespace-collapsed text, for keyword matching."""
        return normalise_whitespace(self.text)

    @property
    def lines(self) -> List[str]:
        """Non-empty, right-stripped lines, for structural pattern matching."""
        return [line.rstrip() for line in self.text.splitlines() if line.strip()]


@dataclass
class DocumentText:
    """The full page-level extraction of one document."""

    path: str
    pages: List[PageText] = field(default_factory=list)
    extraction_error: Optional[str] = None  # set when the file could not be opened

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n".join(page.text for page in self.pages)

    @property
    def total_chars(self) -> int:
        return sum(page.char_count for page in self.pages)


def _page_image_geometry(page: Any) -> Tuple[Optional[float], int]:
    """Per-page image coverage and count, from the same pdfplumber page object.

    Coverage is (summed image bounding-box area / page area), computed from
    ``page.images`` during this same single extraction pass -- no new
    dependency, no second file open. Verified: ``images`` is a first-class
    per-page accessor (``pdfplumber/container.py:46-48``), image objects carry
    ``x0``/``x1``/``y0``/``y1`` (``pdfplumber/page.py:53-56`` in ``ALL_ATTRS``),
    and page area comes from ``Page.width``/``Page.height``
    (``pdfplumber/page.py:239-245``).

    This is evidence only: nothing in this codebase compares the ratio
    returned here against anything (see quality.py's Axis 2 rationale for
    why coverage is recorded but never decisive). Overlapping images can push
    the ratio above 1.0; that is not clamped, since the raw number is more
    useful for later calibration than a value that has been made to look
    like a clean percentage.
    """
    images = page.images
    width = page.width
    height = page.height
    page_area = float(width) * float(height) if width and height else 0.0

    image_area = 0.0
    for image in images:
        x0, x1 = float(image.get("x0", 0.0)), float(image.get("x1", 0.0))
        y0, y1 = float(image.get("y0", 0.0)), float(image.get("y1", 0.0))
        image_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)

    image_area_ratio = round(image_area / page_area, 4) if page_area > 0 else None
    return image_area_ratio, len(images)


def extract_pages(pdf_path: Union[str, Path]) -> DocumentText:
    """Extract per-page text from a PDF.

    Never raises for a damaged or unreadable file: a failure is reported as an
    empty :class:`DocumentText` carrying ``extraction_error``, which
    ``parse_pass.classify_and_assess`` turns into ``status=failed`` before the
    quality layer ever runs -- see ``build_failed_result`` in parse_pass.py. A
    parse pass over a folder of client documents must not stop on the first
    bad file.
    """
    path = Path(pdf_path)
    document = DocumentText(path=str(path))

    if not path.is_file():
        document.extraction_error = f"File not found: {path}"
        logger.error("Text extraction failed for %s: file not found", path)
        return document

    try:
        with pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as page_error:  # noqa: BLE001 - one bad page must not sink the document
                    # Mirrors preocr's own per-page behaviour: record the page
                    # as empty and carry on with the rest of the document. The
                    # image-geometry call below is skipped entirely here: a
                    # page whose own text extraction raised carries no image
                    # geometry either (see
                    # test_a_failed_page_carries_no_image_geometry).
                    logger.warning(
                        "Failed to extract text from %s page %d: %s: %s",
                        path,
                        page_number,
                        type(page_error).__name__,
                        page_error,
                    )
                    error_text = f"{type(page_error).__name__}: {page_error}"
                    document.pages.append(
                        PageText(
                            page_number=page_number,
                            text="",
                            extraction_failed=True,
                            extraction_error=error_text,
                        )
                    )
                    continue

                try:
                    image_area_ratio, image_count = _page_image_geometry(page)
                    geometry_error: Optional[str] = None
                except Exception as geometry_error_exc:  # noqa: BLE001 - image geometry is evidence only (spec 001); its failure must not discard already-extracted text or affect the source axis
                    logger.warning(
                        "Failed to compute image geometry for %s page %d: %s: %s",
                        path,
                        page_number,
                        type(geometry_error_exc).__name__,
                        geometry_error_exc,
                    )
                    image_area_ratio = None
                    image_count = 0
                    geometry_error = (
                        f"image geometry failed: {type(geometry_error_exc).__name__}: "
                        f"{geometry_error_exc}"
                    )

                document.pages.append(
                    PageText(
                        page_number=page_number,
                        text=text,
                        image_area_ratio=image_area_ratio,
                        image_count=image_count,
                        extraction_error=geometry_error,
                    )
                )
    except Exception as error:  # noqa: BLE001 - encrypted, truncated or non-PDF input
        document.extraction_error = f"{type(error).__name__}: {error}"
        logger.error(
            "Text extraction failed for %s: %s: %s", path, type(error).__name__, error
        )

    return document
