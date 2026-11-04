"""Command-line wrapper around the parse pass.

Thin by design: it walks a folder, calls
:func:`companydocs_rag.ingestion.parse_pass.classify_and_assess` per PDF and
writes a CSV report. All logic lives in the importable functions, and the
chunking step's input comes through this command: ``companydocs-normalize``
turns the page text store written here into the normalized files that
``companydocs-chunk`` reads.

    companydocs-parse-pass <folder> -o report.csv

The CSV carries the summary columns for eyeballing a batch. Since a CSV cannot
hold the nested signal evidence, the full result per document, except its page
text, is also written as JSON, one file per document, under a signals folder
-- that is what makes threshold recalibration possible later without
reprocessing every file. This is a default output (spec 001), not opt-in:
``--signals-dir`` only relocates it. Each document's path relative to the
folder being processed is mirrored under the signals folder (``a/lease.pdf``
-> ``<signals_dir>/a/lease.json``), so a ``--recursive`` run over subfolders
that happen to share a filename never has one document's signals JSON
silently overwrite another's.

The page text (spec 003) goes to the page text store instead, one JSON file
per document at the same relative path under ``--page-text-dir``: every page's
text, native or OCR, with its provenance. OCR results are cached under
``--ocr-cache-dir``, and document languages can be overridden by file name in
the ``--language-overrides`` file. All three are default locations under the
git-ignored ``data/``; the flags only relocate them.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .language import (
    LANGUAGE_SOURCE_DEFAULT,
    LANGUAGE_SOURCE_INDEX,
    LANGUAGE_SOURCE_OVERRIDE,
    LanguageResolver,
)
from .ocr import OcrRun
from .parse_pass import (
    STATUS_FAILED,
    STATUS_PROCESSED,
    build_failed_result,
    classify_and_assess,
)
from .quality import (
    NO_TEXT_REASONS,
    TEXT_ORIGIN_NATIVE,
    TEXT_ORIGIN_NONE,
    TEXT_ORIGIN_OCR,
)
from .settings import find_project_root

logger = logging.getLogger(__name__)

# Where the CSV report lands unless -o/--output overrides it. data/interim/ is
# git-ignored (see .gitignore) -- contract filenames must never land in the
# repository, and the repo root itself is not git-ignored.
DEFAULT_REPORT_RELATIVE_PATH = Path("data") / "interim" / "parse_pass_report.csv"

# Where the per-document signals JSON lands unless --signals-dir overrides it.
# Same reasoning as DEFAULT_REPORT_RELATIVE_PATH above.
DEFAULT_SIGNALS_RELATIVE_PATH = Path("data") / "interim" / "parse_pass_signals"

# Where the page text store and the OCR cache land unless --page-text-dir and
# --ocr-cache-dir override them (spec 003). Same reasoning as above: both hold
# contract text.
DEFAULT_PAGE_TEXT_RELATIVE_PATH = Path("data") / "interim" / "page_text"
DEFAULT_OCR_CACHE_RELATIVE_PATH = Path("data") / "interim" / "ocr_cache"

# The hand-kept language override file (spec 003): under data/, apart from
# generated data, because its keys name client files. Never in config/.
DEFAULT_LANGUAGE_OVERRIDES_RELATIVE_PATH = (
    Path("data") / "manual" / "language_overrides.csv"
)

CSV_COLUMNS = [
    "filename",
    "status",
    "detected_type",
    "type_confidence",
    "quality_label",
    "awaiting_ocr",
    "explanation",
    "language",
    "language_source",
    "ocr_pages",
    "no_text_pages",
    "ocr_mean_confidence",
    "no_text_reasons",
]

# The result key the page text store record travels under; left out of the
# signals JSON so page text is not duplicated into the evidence.
PAGE_TEXT_KEY = "page_text"


def find_pdfs(folder: Path, recursive: bool = False) -> List[Path]:
    """List the PDFs in a folder, sorted for reproducible reports."""
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(
        path
        for path in folder.glob(pattern)
        if path.is_file() and not path.name.startswith(".")
    )


def _cell(value: Any) -> str:
    return "" if value is None else str(value)


def _no_text_reasons_cell(counts: Dict[str, int]) -> str:
    """Non-zero reason counts as ``reason:count``, in D4's order, ``;``-joined."""
    return ";".join(
        f"{reason}:{counts[reason]}" for reason in NO_TEXT_REASONS if counts.get(reason)
    )


def _row_for(result: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "filename": result["file"]["filename"],
        "status": result["status"],
        "detected_type": result["detected_type"],
        "type_confidence": f"{result['type_confidence']:.2f}",
        # Empty cell rather than a word that looks like a verdict: None for a
        # failed document (Axis 1: no quality tier at all) or for a document
        # without native pages, whether or not they got OCR text.
        "quality_label": result["quality_label"] or "",
        # "", "partial" or "all": whether pages are still without text after
        # OCR (see quality.py's roll-up). An empty quality_label no longer
        # implies "all": a fully OCR'd scan shows "" here.
        "awaiting_ocr": result["awaiting_ocr"] or "",
        "explanation": result["explanation"],
        "language": "",
        "language_source": "",
        "ocr_pages": "",
        "no_text_pages": "",
        "ocr_mean_confidence": "",
        "no_text_reasons": "",
    }
    if result["status"] == STATUS_FAILED:
        return row

    page_stats = (result.get("quality_signals") or {}).get("page_stats") or {}
    mean_confidence = (result.get("ocr") or {}).get("mean_page_confidence")
    row.update(
        {
            "language": _cell(result.get("language")),
            "language_source": _cell(result.get("language_source")),
            "ocr_pages": _cell(page_stats.get("ocr_page_count")),
            "no_text_pages": _cell(page_stats.get("no_text_page_count")),
            "ocr_mean_confidence": (
                "" if mean_confidence is None else f"{mean_confidence:.2f}"
            ),
            "no_text_reasons": _no_text_reasons_cell(
                page_stats.get("no_text_reason_counts") or {}
            ),
        }
    )
    return row


def write_report(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    """Write the CSV report.

    Encoded as UTF-8 with a BOM so Dutch document names and accented characters
    survive being opened directly in Excel.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _format_counts(counts: Dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _log_run_summary(results: Sequence[Dict[str, Any]]) -> None:
    """One INFO line: documents per status, and page totals per text_origin,
    per no-text reason and per language_source."""
    processed = sum(1 for result in results if result.get("status") == STATUS_PROCESSED)
    failed = sum(1 for result in results if result.get("status") == STATUS_FAILED)
    origins = {
        origin: 0 for origin in (TEXT_ORIGIN_NATIVE, TEXT_ORIGIN_OCR, TEXT_ORIGIN_NONE)
    }
    reasons = {reason: 0 for reason in NO_TEXT_REASONS}
    language_sources = {
        source: 0
        for source in (
            LANGUAGE_SOURCE_OVERRIDE,
            LANGUAGE_SOURCE_INDEX,
            LANGUAGE_SOURCE_DEFAULT,
        )
    }
    for result in results:
        pages = (result.get("quality_signals") or {}).get("pages") or []
        for page in pages:
            origin = page.get("text_origin")
            if origin is not None:
                origins[origin] = origins.get(origin, 0) + 1
            reason = page.get("no_text_reason")
            if reason is not None:
                reasons[reason] = reasons.get(reason, 0) + 1
            source = page.get("language_source")
            if source is not None:
                language_sources[source] = language_sources.get(source, 0) + 1
    logger.info(
        "Run summary: %d document(s) processed, %d failed; pages by text_origin: "
        "%s; pages by no_text_reason: %s; pages by language_source: %s",
        processed,
        failed,
        _format_counts(origins),
        _format_counts(reasons),
        _format_counts(language_sources),
    )


def process_folder(
    folder: Path,
    output_path: Path,
    recursive: bool = False,
    signals_dir: Optional[Path] = None,
    page_text_dir: Optional[Path] = None,
    ocr_cache_dir: Optional[Path] = None,
    language_overrides: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run the parse pass over every PDF in a folder and write the report.

    ``signals_dir=None`` (the default for this function) means no signals
    JSON is written -- used by callers that only want the CSV/the returned
    results. The CLI itself always passes a resolved path (see
    :func:`default_signals_dir` and ``build_parser``'s ``--signals-dir``
    default): the per-document signals JSON is a default *CLI* output, not
    opt-in, but this function does not bake a real filesystem path in as its
    own default so that calling it directly (e.g. from tests) never writes
    anywhere unless asked to.

    The same holds for ``page_text_dir`` (the page text store),
    ``ocr_cache_dir`` (the OCR cache) and ``language_overrides`` (the
    override file): None writes, caches or reads nothing. Their directories
    are created when their first file is written. At the end of the run,
    unmatched or ambiguous override entries are logged, and one INFO summary.
    """
    pdfs = find_pdfs(folder, recursive=recursive)
    if not pdfs:
        logger.warning("No PDF files found in %s", folder)

    rows: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    if signals_dir is not None:
        signals_dir.mkdir(parents=True, exist_ok=True)

    language_resolver = LanguageResolver(language_overrides)
    ocr_run = OcrRun(ocr_cache_dir)

    for index, pdf_path in enumerate(pdfs, start=1):
        logger.info("[%d/%d] %s", index, len(pdfs), pdf_path.name)
        try:
            result = classify_and_assess(
                pdf_path, language_resolver=language_resolver, ocr_run=ocr_run
            )
        except Exception as error:  # noqa: BLE001 - classify_and_assess already
            # guards against this itself, so this branch should be unreachable
            # in normal operation. It exists as an outer safety net only: if
            # it is ever hit, a bug in our own code must still not be reported
            # as if it were a property of the document (Axis 1) or silently
            # drop the file from the batch -- both bugs this spec fixes.
            logger.exception("Failed to process %s", pdf_path.name)
            result = build_failed_result(
                pdf_path, None, f"{type(error).__name__}: {error}"
            )

        results.append(result)
        rows.append(_row_for(result))

        relative = pdf_path.relative_to(folder)
        if signals_dir is not None:
            _write_json(
                (signals_dir / relative).with_suffix(".json"),
                {key: value for key, value in result.items() if key != PAGE_TEXT_KEY},
            )
        if page_text_dir is not None:
            _write_json(
                (page_text_dir / relative).with_suffix(".json"),
                result[PAGE_TEXT_KEY],
            )

    write_report(rows, output_path)
    logger.info("Wrote %d row(s) to %s", len(rows), output_path)
    language_resolver.report_end_of_run()
    _log_run_summary(results)
    return results


def default_report_path() -> Path:
    """Resolve the default CSV report location under the git-ignored data/interim/.

    Resolved relative to the project root (found via ``pyproject.toml``, same
    as ``ingestion.settings.find_project_root``), not the current working
    directory, so the CLI's default does not depend on where it is invoked
    from. Falls back to a path relative to the current directory when no
    project root can be found, e.g. from an installed wheel.
    """
    root = find_project_root()
    base = root if root is not None else Path.cwd()
    return base / DEFAULT_REPORT_RELATIVE_PATH


def default_signals_dir() -> Path:
    """Resolve the default per-document signals-JSON folder, under data/interim/.

    Same resolution as :func:`default_report_path`. The per-document signals
    JSON is a default output as of spec 001 (the stated purpose -- recalibrating
    thresholds without reprocessing -- cannot be served by a report that skips
    it by default): a plain ``companydocs-parse-pass <folder>`` run, with no
    ``--signals-dir`` override, writes one JSON file per document here.
    ``--signals-dir`` only relocates it.
    """
    root = find_project_root()
    base = root if root is not None else Path.cwd()
    return base / DEFAULT_SIGNALS_RELATIVE_PATH


def _project_base() -> Path:
    root = find_project_root()
    return root if root is not None else Path.cwd()


def default_page_text_dir() -> Path:
    """Resolve the default page text store folder, under data/interim/.

    Same resolution as :func:`default_report_path`.
    """
    return _project_base() / DEFAULT_PAGE_TEXT_RELATIVE_PATH


def default_ocr_cache_dir() -> Path:
    """Resolve the default OCR cache folder, under data/interim/.

    Same resolution as :func:`default_report_path`.
    """
    return _project_base() / DEFAULT_OCR_CACHE_RELATIVE_PATH


def default_language_overrides_path() -> Path:
    """Resolve the default language override file, under data/manual/.

    Same resolution as :func:`default_report_path`. A missing file is not an
    error.
    """
    return _project_base() / DEFAULT_LANGUAGE_OVERRIDES_RELATIVE_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companydocs-parse-pass",
        description=(
            "Parse pass over a folder of PDFs: detect the document type and "
            "assign a quality tier (Clean/Decent/Garbage) for downstream chunking."
        ),
    )
    parser.add_argument(
        "folder", type=Path, help="Folder containing the PDFs to process"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_report_path(),
        help=(
            "Path of the CSV report to write (default: data/interim/"
            "parse_pass_report.csv under the project root -- git-ignored, "
            "since the report carries contract filenames)"
        ),
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Also process PDFs in subfolders",
    )
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=default_signals_dir(),
        help=(
            "Folder to write the full per-document result, except its page "
            "text, as JSON, including all quality signals and type evidence "
            "(default: data/interim/"
            "parse_pass_signals under the project root -- git-ignored). This "
            "is a default output, not opt-in: pass this flag only to "
            "relocate it, e.g. when collecting data to recalibrate thresholds."
        ),
    )
    parser.add_argument(
        "--page-text-dir",
        type=Path,
        default=default_page_text_dir(),
        help=(
            "Folder for the page text store: one JSON file per document with "
            "every page's text, native or OCR, and its provenance (default: "
            "data/interim/page_text under the project root -- git-ignored)"
        ),
    )
    parser.add_argument(
        "--ocr-cache-dir",
        type=Path,
        default=default_ocr_cache_dir(),
        help=(
            "Folder for the OCR cache and OCR temporary files (default: "
            "data/interim/ocr_cache under the project root -- git-ignored)"
        ),
    )
    parser.add_argument(
        "--language-overrides",
        type=Path,
        default=default_language_overrides_path(),
        help=(
            "Language override file: ';'-separated, UTF-8 with BOM, columns "
            "Bestandsnaam and Taal (default: data/manual/language_overrides.csv "
            "under the project root -- git-ignored; a missing file is not an "
            "error)"
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only report warnings and errors",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.folder.is_dir():
        logger.error("Not a folder: %s", args.folder)
        return 2

    process_folder(
        args.folder,
        args.output,
        recursive=args.recursive,
        signals_dir=args.signals_dir,
        page_text_dir=args.page_text_dir,
        ocr_cache_dir=args.ocr_cache_dir,
        language_overrides=args.language_overrides,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
