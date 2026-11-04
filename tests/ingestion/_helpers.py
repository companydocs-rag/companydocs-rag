"""Shared helpers for the ingestion tests.

The tests here run on short inline text strings. No real document is read:
the parse pass is deliberately structured so that everything after text
extraction operates on a :class:`DocumentText`, which can be built by hand,
and the OCR seams are replaced with fabricated output (:func:`stub_ocr`). The
only PDFs are a few generated inside a test from a Pillow image. Real
documents are validated separately by the project owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from companydocs_rag.ingestion import ocr as ocr_module
from companydocs_rag.ingestion.extraction import DocumentText, PageText
from companydocs_rag.ingestion.ocr import TesseractOutput


def make_page(
    page_number: int,
    text: str,
    *,
    image_area_ratio: Optional[float] = None,
    image_count: int = 0,
    extraction_failed: bool = False,
    extraction_error: Optional[str] = None,
) -> PageText:
    """Build one fabricated page, with optional image evidence.

    A plain string is enough for most tests; this is for the ones that also
    need to fabricate the per-page image geometry (coverage, count) that
    ``extraction.py`` would normally capture from a real PDF.
    """
    return PageText(
        page_number=page_number,
        text=text,
        extraction_failed=extraction_failed,
        extraction_error=extraction_error,
        image_area_ratio=image_area_ratio,
        image_count=image_count,
    )


def make_document(
    *pages: Union[str, PageText], path: str = "inline-test.pdf"
) -> DocumentText:
    """Build a DocumentText from one string per page, or from PageText objects.

    A plain string becomes a page with no image evidence; a :class:`PageText`
    built with :func:`make_page` is used as-is, so a test can fabricate image
    coverage or count without touching the others.
    """
    built: List[PageText] = []
    for number, item in enumerate(pages, start=1):
        if isinstance(item, PageText):
            built.append(item)
        else:
            built.append(PageText(page_number=number, text=item))
    return DocumentText(path=path, pages=built)


def make_pages(*pages: Union[str, PageText]) -> List[PageText]:
    """Build just the page list, for the structure analyser."""
    return make_document(*pages).pages


# --- Fabricated OCR output (spec 003) ----------------------------------------

TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
    "left\ttop\twidth\theight\tconf\ttext"
)


def make_tsv(*words: tuple) -> str:
    """A fabricated Tesseract TSV: one word row per ``(text, conf)``.

    Page, block, paragraph and line rows come first, at conf -1 with empty
    text, as Tesseract writes them.
    """
    rows = [
        TSV_HEADER,
        "1\t1\t0\t0\t0\t0\t0\t0\t2480\t3508\t-1\t",
        "2\t1\t1\t0\t0\t0\t100\t100\t2000\t200\t-1\t",
        "3\t1\t1\t1\t0\t0\t100\t100\t2000\t200\t-1\t",
        "4\t1\t1\t1\t1\t0\t100\t100\t2000\t50\t-1\t",
    ]
    for number, (text, conf) in enumerate(words, start=1):
        rows.append(
            f"5\t1\t1\t1\t1\t{number}\t{100 * number}\t100\t90\t50\t{conf}\t{text}"
        )
    return "\n".join(rows) + "\n"


def tesseract_output(text: str, conf: float = 90.0) -> TesseractOutput:
    """Raw Tesseract output for ``text``: a trailing form feed, and a TSV with
    one word row per whitespace-separated token at ``conf``."""
    return TesseractOutput(
        text=text + "\f",
        tsv=make_tsv(*((word, conf) for word in text.split())),
    )


@dataclass
class OcrStub:
    """What the stubbed OCR seams were asked to do."""

    page_calls: List[Dict[str, Any]] = field(default_factory=list)
    version_calls: int = 0

    @property
    def pages(self) -> List[int]:
        return [call["page_number"] for call in self.page_calls]


def stub_ocr(
    monkeypatch,
    outputs: Optional[Mapping[int, Union[str, TesseractOutput, BaseException]]] = None,
    *,
    default: Union[str, TesseractOutput, BaseException] = "",
    version: Union[str, BaseException] = "5.5.3",
) -> OcrStub:
    """Replace both OCR seams: no page is rendered and no process started.

    ``outputs`` maps a page number to its OCR text (see
    :func:`tesseract_output`), to raw :class:`TesseractOutput`, or to an
    exception the page seam raises. Other pages get ``default`` -- empty text
    unless given. ``version`` is what the version seam returns, or raises.
    """
    stub = OcrStub()

    def _as_output(value):
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, TesseractOutput):
            return value
        return tesseract_output(value)

    def fake_ocr_page(pdf, page_number, **kwargs):
        stub.page_calls.append({"page_number": page_number, "path": pdf.path, **kwargs})
        return _as_output((outputs or {}).get(page_number, default))

    def fake_version(tesseract_cmd, timeout_seconds):
        stub.version_calls += 1
        if isinstance(version, BaseException):
            raise version
        return version

    monkeypatch.setattr(ocr_module, "ocr_page", fake_ocr_page)
    monkeypatch.setattr(ocr_module, "query_tesseract_version", fake_version)
    return stub


# --- Fabricated client files (spec 003) ---------------------------------------


def write_client_csv(
    path: Path, header: Sequence[str], rows: Iterable[Sequence[str]]
) -> Path:
    """Write a fabricated file in the client index's conventions.

    ``;``-separated, UTF-8 with BOM, CRLF line endings, a header row.
    """
    lines = [";".join(header), *(";".join(row) for row in rows)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8-sig"))
    return path


# --- Fabricated page text store records (spec 004) ------------------------------

# The provenance the parse pass gives a store page of each text origin.
_STORE_PAGE_PROVENANCE = {
    "native": ("native", "Clean", None, None),
    "ocr": ("no_text_layer", None, 90.0, None),
    "none": ("no_text_layer", None, None, "ocr_empty"),
}


def store_page(
    page_number: int,
    text: str = "",
    *,
    text_origin: str = "native",
    language: Optional[str] = "nld",
    **overrides: Any,
) -> Dict[str, Any]:
    """One fabricated page text store page (``schema_version`` 2), with the
    source, tier, OCR confidence and no-text reason the parse pass gives a
    page of that text origin. ``overrides`` replaces any of its keys."""
    source, tier, confidence, reason = _STORE_PAGE_PROVENANCE[text_origin]
    page = {
        "page_number": page_number,
        "text": text,
        "source": source,
        "tier": tier,
        "text_origin": text_origin,
        "language": language,
        "language_source": "index",
        "ocr_mean_confidence": confidence,
        "no_text_reason": reason,
    }
    page.update(overrides)
    return page


def store_record(
    *pages: Union[str, Dict[str, Any]],
    path: str = "invoer/fictief.pdf",
    status: str = "processed",
    page_count: Optional[int] = None,
    language: str = "nld",
) -> Dict[str, Any]:
    """A fabricated page text store record (``schema_version`` 2).

    A plain string is a native page, numbered by its position; a dict from
    :func:`store_page` is used as it is. A failed record has no pages and no
    language, as the parse pass writes it.
    """
    built = [
        store_page(number, item, language=language) if isinstance(item, str) else item
        for number, item in enumerate(pages, start=1)
    ]
    processed = status == "processed"
    return {
        "schema_version": 2,
        "file": {
            "path": path,
            "filename": path.rsplit("/", 1)[-1],
            "sha256": "ab" * 32,
        },
        "status": status,
        "page_count": len(built) if page_count is None else page_count,
        "language": language if processed else None,
        "language_source": "index" if processed else None,
        "ocr_engine": None,
        "pages": built,
    }
