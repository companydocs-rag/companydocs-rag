"""The parse pass: document type and quality tier, from one text extraction and OCR.

This is step 1 of the pipeline. A single call produces both outcomes, the
quality tier and the document type. The quality tier comes from a single read
of the page text; pages without usable native text are then OCR'd (spec 003),
and type detection and structure analysis read the native text together with
that OCR text.

Axis 1 -- status. Every result carries a document-level ``status``:
``"processed"`` or ``"failed"``, with an explicit reason for the latter. A
failed document carries no quality tier at all -- never ``Garbage``. This is
what separates "our code failed" (a bug, a corrupt file, an unreadable path)
from "the document is bad" (which is what a quality tier actually means).
:func:`classify_and_assess` therefore never raises: any failure, whether in
extraction or anywhere else in this pass, is caught and reported as
``status=failed``.

Usage from Python:

    from companydocs_rag.ingestion import classify_and_assess

    result = classify_and_assess("some_lease.pdf")
    if result["status"] == "failed":
        ...  # route for manual triage; there is no quality tier to act on
    elif result["awaiting_ocr"] == "all":
        ...  # no page has any text: no text layer, and OCR gave none
    else:
        # Each page's own tier -- result["quality_signals"]["pages"][i]["tier"],
        # None where ["source"] is "no_text_layer" -- is provenance. The page
        # text store keeps it only where ["text_origin"] is "native", and None
        # otherwise: a native page tiered Garbage that got OCR text has no
        # tier there. The chunker (companydocs_rag.chunking) copies the store's
        # tier, by way of the normalized files, into its chunks' page spans;
        # neither the tier nor the structure score selects its route. See
        # quality.py's module docstring.
        ...
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .extraction import DocumentText, PageText, extract_pages
from .language import LanguageResolver
from .ocr import ENGINE_NAME, DocumentOcr, OcrRun, route_pages
from .quality import (
    TEXT_ORIGIN_NATIVE,
    TEXT_ORIGIN_NONE,
    TEXT_ORIGIN_OCR,
    PageQuality,
    assess_pages,
    resolve_text_origins,
    roll_up_quality,
    run_preocr,
)
from .structure import analyse_structure, describe_pattern
from .type_detection import UNKNOWN_TYPE, detect_type, summarise_evidence

logger = logging.getLogger(__name__)

# Axis 1 -- status (document level). See the module docstring.
STATUS_PROCESSED = "processed"
STATUS_FAILED = "failed"

# type_fallback_reason for a failed document: type detection never actually
# ran, so this is not a type-detection verdict, just a placeholder explaining
# why the field is empty.
_TYPE_FALLBACK_PROCESSING_FAILED = (
    "document processing failed before type detection could run"
)

# The page text store's schema version (spec 003): bumped on any change to its
# keys or their meanings. Version 2 (spec 004) adds the page-level `tier`.
PAGE_TEXT_SCHEMA_VERSION = 2


def file_sha256(path: Path) -> Optional[str]:
    """Hex sha256 of the file's bytes; None if it cannot be read."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except FileNotFoundError:
        return None
    except OSError as error:
        logger.warning(
            "Could not read %s to hash it: %s: %s", path, type(error).__name__, error
        )
        return None
    return digest.hexdigest()


def _file_info(
    path: Path, document: DocumentText, sha256: Optional[str]
) -> Dict[str, Any]:
    return {
        "path": str(path),
        "filename": path.name,
        "sha256": sha256,
        "page_count": document.page_count,
        "extraction_error": document.extraction_error,
        # Pages that raised during per-page extraction (extraction.py), kept
        # alongside the document-level error so a failure on a single page of
        # an otherwise-openable document also reaches the signals JSON, not
        # just the quality reasons.
        "failed_pages": [
            page.page_number for page in document.pages if page.extraction_failed
        ],
    }


def _page_text_record(
    path: Path,
    sha256: Optional[str],
    status: str,
    page_count: int,
    language: Optional[str],
    language_source: Optional[str],
    ocr_engine: Optional[Dict[str, Any]],
    pages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """The page text store record for one document (spec 003).

    The contract for downstream stages: every page's text, native or OCR,
    with its provenance. Every key is always present, null where it does not
    apply.
    """
    return {
        "schema_version": PAGE_TEXT_SCHEMA_VERSION,
        "file": {"path": str(path), "filename": path.name, "sha256": sha256},
        "status": status,
        "page_count": page_count,
        "language": language,
        "language_source": language_source,
        "ocr_engine": ocr_engine,
        "pages": pages,
    }


def _store_pages(
    pages: Sequence[PageQuality], texts: Sequence[str]
) -> List[Dict[str, Any]]:
    """The store's page objects.

    ``tier`` is the tier :func:`quality.assess_pages` gave the page when its
    ``text_origin`` is ``native``, and null otherwise (spec 004, D2): OCR text
    is never tiered, so a native page tiered Garbage that got OCR text has no
    tier here. The signals JSON keeps every page's tier as evidence.
    """
    return [
        {
            "page_number": page.page_number,
            "text": text,
            "source": page.source,
            "tier": page.tier if page.text_origin == TEXT_ORIGIN_NATIVE else None,
            "text_origin": page.text_origin,
            "language": page.language,
            "language_source": page.language_source,
            "ocr_mean_confidence": (
                page.ocr_mean_confidence
                if page.text_origin == TEXT_ORIGIN_OCR
                else None
            ),
            "no_text_reason": (
                page.no_text_reason if page.text_origin == TEXT_ORIGIN_NONE else None
            ),
        }
        for page, text in zip(pages, texts)
    ]


def _ocr_block(pages: Sequence[PageQuality], ocr: DocumentOcr) -> Dict[str, Any]:
    """The result's ``ocr`` block: the engine, and the per-document roll-up.

    ``page_count`` counts the pages with OCR text; the confidences are the
    mean and minimum of their page confidences, None without such pages.
    """
    ocr_pages = [page for page in pages if page.text_origin == TEXT_ORIGIN_OCR]
    confidences = [
        page.ocr_mean_confidence
        for page in ocr_pages
        if page.ocr_mean_confidence is not None
    ]
    return {
        "engine_version": ocr.engine_version,
        "dpi": ocr.dpi,
        "grayscale": ocr.grayscale,
        "page_count": len(ocr_pages),
        "mean_page_confidence": round(fmean(confidences), 2) if confidences else None,
        "min_page_confidence": min(confidences) if confidences else None,
    }


def _ocr_engine(ocr: DocumentOcr) -> Optional[Dict[str, Any]]:
    """The store's ``ocr_engine``: null when no page was routed."""
    if not ocr.routed:
        return None
    return {
        "name": ENGINE_NAME,
        "version": ocr.engine_version,
        "dpi": ocr.dpi,
        "grayscale": ocr.grayscale,
    }


def build_failed_result(
    pdf_path: Union[str, Path],
    document: Optional[DocumentText],
    reason: str,
) -> Dict[str, Any]:
    """Build the Axis-1 "failed" result shape: no quality tier, ever.

    Exposed (not module-private) so a caller's own outer safety net -- e.g.
    the CLI's guard against a bug inside this module itself -- can report a
    failure in exactly this shape too, rather than mislabelling it Garbage.
    That mislabelling is precisely the bug this spec exists to fix (see the
    module docstring and spec 001's Goal): a bug in our own code must never be
    reported as if it were a property of the document.
    """
    path = Path(pdf_path)
    document = document if document is not None else DocumentText(path=str(path))
    logger.error("classify_and_assess: %s: %s", path, reason)
    sha256 = file_sha256(path)
    return {
        "file": _file_info(path, document, sha256),
        "status": STATUS_FAILED,
        "status_reason": reason,
        "detected_type": UNKNOWN_TYPE,
        "type_label": UNKNOWN_TYPE,
        "type_confidence": 0.0,
        "type_evidence": [],
        "type_scores": {},
        "type_runner_up": None,
        "type_fallback_reason": _TYPE_FALLBACK_PROCESSING_FAILED,
        "quality_label": None,
        "awaiting_ocr": None,
        "language": None,
        "language_source": None,
        "quality_signals": {},
        "ocr": {},
        "explanation": f"Failed: {reason}.",
        "page_text": _page_text_record(
            path,
            sha256,
            STATUS_FAILED,
            document.page_count,
            language=None,
            language_source=None,
            ocr_engine=None,
            pages=[],
        ),
    }


def _build_explanation(
    type_result: Any,
    quality_label: Optional[str],
    quality_reasons: list[str],
    structure_description: str,
) -> str:
    """One short, readable sentence covering both the type and the tier.

    Written for the CSV report and for a human skimming a few hundred rows,
    e.g. "Clean: strong extraction signals (1840 chars/page, noise 0.000);
    article/clause numbering detected starting page 1. Type
    lease_agreement from 'verhuurder' (p1), 'huurprijs' (p2) +7 more."

    ``quality_label`` is ``None`` for a document without native pages (every
    page ``no_text_layer``), whatever its ``awaiting_ocr``: there is no tier
    to report, only the reasons explaining why, under the prefix "No text
    layer:".
    """
    if quality_label is None:
        quality_part = "No text layer: " + "; ".join(
            quality_reasons or ["entire document has no text layer"]
        )
    else:
        quality_part = f"{quality_label}: " + "; ".join(
            quality_reasons or ["no signals recorded"]
        )

    if type_result.detected_type == UNKNOWN_TYPE:
        reason = type_result.fallback_reason or "no type matched"
        type_part = f"Type unknown/other ({reason})"
    else:
        type_part = (
            f"Type {type_result.detected_type} "
            f"(confidence {type_result.confidence:.2f}) from "
            f"{summarise_evidence(type_result.evidence)}"
        )

    return f"{quality_part}. {structure_description}. {type_part}."


def classify_and_assess(
    pdf_path: Union[str, Path],
    settings: Optional[Mapping[str, Any]] = None,
    *,
    language_resolver: Optional[LanguageResolver] = None,
    ocr_run: Optional[OcrRun] = None,
) -> Dict[str, Any]:
    """Determine a document's type and quality tier in one pass.

    Args:
        pdf_path: Path to the PDF to analyse.
        settings: Optional settings mapping, overriding ``config/settings.yaml``.
            Mainly useful for tests and for calibration experiments.
        language_resolver: The run's language resolution (override file and
            client index cache). None: a fresh one without an override file.
        ocr_run: The run's OCR state (cache directory, Tesseract version).
            None: a fresh one without a cache.

    Returns:
        A structured result containing:

        - ``status``          -- ``"processed"`` or ``"failed"`` (Axis 1).
        - ``status_reason``   -- set when ``status`` is ``"failed"``.
        - ``detected_type``   -- one of the configured type keys, or
          ``"unknown/other"``. Never a forced guess.
        - ``type_confidence`` -- 0.0-1.0.
        - ``type_evidence``   -- every matched keyword, with its language and
          the page number it was found on.
        - ``quality_label``   -- ``"Clean"``, ``"Decent"`` or ``"Garbage"``,
          or ``None`` when ``status`` is ``"failed"`` or the document has no
          native pages, whether or not they got OCR text.
        - ``awaiting_ocr``    -- ``None``, ``"partial"`` or ``"all"``: whether
          any page is still without text after OCR (``text_origin`` none).
        - ``language``, ``language_source`` -- the document's language for
          OCR and where it came from (language.py).
        - ``quality_signals`` -- the raw preocr output including its per-page
          classification, the per-page statistics computed here, and the
          structural match with the page it was first detected on. This is the
          evidence behind the label, kept so thresholds can be recalibrated
          from stored results without reprocessing documents.
        - ``ocr``             -- the OCR engine and the per-document OCR
          roll-up.
        - ``explanation``     -- a short readable summary of both outcomes.
        - ``page_text``       -- the page text store record: every page's
          text, native or OCR, with its provenance.

        Never raises: a failure anywhere in this pass -- an unreadable or
        damaged file, a zero-page document, or an unexpected exception in this
        module's own code -- is caught and reported as ``status=failed`` with
        the reason recorded, never as a quality tier.
    """
    path = Path(pdf_path)

    # extract_pages never raises (see extraction.py): a damaged or unreadable
    # file comes back as an empty DocumentText carrying extraction_error.
    document = extract_pages(path)

    try:
        if document.extraction_error:
            return build_failed_result(
                path,
                document,
                f"text extraction failed ({document.extraction_error})",
            )
        if document.page_count == 0:
            return build_failed_result(path, document, "document contains no pages")

        sha256 = file_sha256(path)

        # preocr covers quality criteria 1 and 2 (extraction quality, OCR
        # artifacts); it is not reimplemented here.
        preocr_result = run_preocr(path)

        resolver = language_resolver or LanguageResolver()
        language, language_source = resolver.resolve(path, settings=settings)

        # The per-page step, on the pdfplumber extraction only: source, the
        # native signals and a tier for native pages. OCR text never reaches
        # it, and nothing after it recomputes source.
        pages = assess_pages(document, settings=settings)
        for page in pages:
            page.language = language
            page.language_source = language_source

        route_pages(pages, settings=settings)
        ocr = (ocr_run or OcrRun()).ocr_document(
            path,
            [page.page_number for page in pages if page.ocr_routed],
            sha256=sha256,
            language=language,
            settings=settings,
        )
        texts = resolve_text_origins(pages, document, ocr.outcomes)

        # The downstream view: native text, OCR text or "" per page. Type
        # detection and structure analysis read this, not the extraction.
        downstream = DocumentText(
            path=document.path,
            pages=[
                PageText(page_number=page.page_number, text=text)
                for page, text in zip(pages, texts)
            ],
        )
        structure_result = analyse_structure(downstream.pages, settings=settings)
        type_result = detect_type(downstream, settings=settings)
        quality = roll_up_quality(
            pages,
            structure_result,
            preocr_result=preocr_result,
            settings=settings,
        )

        explanation = _build_explanation(
            type_result,
            quality.label,
            quality.reasons,
            describe_pattern(structure_result),
        )

        return {
            "file": _file_info(path, document, sha256),
            "status": STATUS_PROCESSED,
            "status_reason": None,
            "detected_type": type_result.detected_type,
            "type_label": type_result.label,
            "type_confidence": type_result.confidence,
            "type_evidence": [asdict(item) for item in type_result.evidence],
            # Scores for every candidate type, not just the winner: the runner-up
            # margin is what makes a confidence recalibration possible later.
            "type_scores": type_result.scores,
            "type_runner_up": type_result.runner_up,
            "type_fallback_reason": type_result.fallback_reason,
            "quality_label": quality.label,
            # Surfaced at the top level (not just inside quality_signals) so a
            # caller can tell pages still without text apart from an ordinary
            # tier without reaching into the nested evidence -- see the
            # roll-up's `awaiting_ocr` marker in quality.py.
            "awaiting_ocr": quality.awaiting_ocr,
            "language": language,
            "language_source": language_source,
            "quality_signals": quality.to_dict(),
            "ocr": _ocr_block(pages, ocr),
            "explanation": explanation,
            "page_text": _page_text_record(
                path,
                sha256,
                STATUS_PROCESSED,
                document.page_count,
                language=language,
                language_source=language_source,
                ocr_engine=_ocr_engine(ocr),
                pages=_store_pages(pages, texts),
            ),
        }
    except Exception as error:  # noqa: BLE001 - a bug in our own code must not
        # sink the batch, and per Axis 1 must never be reported as a property
        # of the document (that was the `Garbage` mislabelling this spec
        # exists to fix) -- so it is reported as status=failed instead.
        logger.exception("classify_and_assess failed unexpectedly for %s", path)
        return build_failed_result(path, document, f"{type(error).__name__}: {error}")
