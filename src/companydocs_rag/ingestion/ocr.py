"""OCR for pages without usable native text (spec 003).

Routing, rendering, Tesseract, TSV parsing, the Tesseract version and the OCR
cache. Everything runs locally: pypdfium2 renders, and the Tesseract binary
installed on this machine reads the rendered page.

Routing (spec 001's routing point, now a recorded value, ``ocr_routed`` per
page). A page is routed to OCR when any of these holds:

- its source is ``no_text_layer`` and its extraction did not fail;
- its tier is Garbage and ``ocr_garbage_tier_pages`` is true;
- its extraction failed and ``ocr_extraction_failed_pages`` is true.

Both flags default to false. Routing reads the tier, so it runs after the
per-page quality step, and it never changes ``source``: source is decided
before OCR runs, from the pdfplumber extraction alone.

Engine. Each routed page is rendered whole with pypdfium2 at ``dpi`` (scale
``dpi / 72``), grayscale by default; embedded images are not extracted. One
Tesseract run per page writes both plain text and TSV. The page's text is
Tesseract's plain text minus the trailing page-separator form feed; its
confidence is the unweighted mean ``conf`` of the TSV's word rows. Neither
decides anything: OCR text gets no tier, and there is no confidence
threshold.

Seams. The render-and-Tesseract work for one page is :func:`ocr_page`, and the
version query is :func:`query_tesseract_version`. Unit tests replace these.
The version is queried once per run, lazily, at the first routed page; it is
part of the cache key, so it is known before the cache is consulted. The
cache sits in front of :func:`ocr_page`: a hit skips rendering and Tesseract.

Cache. One JSON entry per page and key (cache format version, file sha256,
page number, dpi, grayscale, language, Tesseract version), holding the key,
Tesseract's raw text and its raw TSV. Only successful runs are cached, empty
ones included; an error is retried on the next run.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

import pypdfium2 as pdfium

from .quality import SOURCE_NO_TEXT_LAYER, TIER_GARBAGE, PageQuality
from .settings import parse_pass_section

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# --- Settings defaults (ingestion.parse_pass.ocr) ---------------------------
DPI = 300
GRAYSCALE = True
OCR_GARBAGE_TIER_PAGES = False
OCR_EXTRACTION_FAILED_PAGES = False
TESSERACT_CMD = "tesseract"  # looked up on PATH
# Per-page limit on one Tesseract run, in seconds. An operational limit, so a
# hung binary ends as an explicit error; not a quality threshold.
TIMEOUT_SECONDS = 300

# --- Python constants ---------------------------------------------------------
ENGINE_NAME = "tesseract"
# Bump when the entry format or the Tesseract invocation changes, or when the
# pypdfium2 pin changes.
CACHE_FORMAT_VERSION = 1
POINTS_PER_INCH = 72
# Tesseract writes this after every page of its plain-text output.
PAGE_SEPARATOR = "\f"
# TSV rows at this level are words.
TSV_WORD_LEVEL = "5"
# Bounds on stored error messages.
MAX_ERROR_CHARS = 500
MAX_STDERR_CHARS = 300
# Temporary files live in this folder inside the OCR cache directory.
TEMP_SUBDIR = "tmp"

PathLike = Union[str, Path]


@dataclass(frozen=True)
class OcrSettings:
    dpi: int
    grayscale: bool
    ocr_garbage_tier_pages: bool
    ocr_extraction_failed_pages: bool
    tesseract_cmd: str
    timeout_seconds: float


def ocr_settings(settings: Optional[Mapping[str, Any]] = None) -> OcrSettings:
    """Resolve the OCR settings at call time, YAML overriding the defaults."""
    section = parse_pass_section("ocr", settings=settings)
    return OcrSettings(
        dpi=int(section.get("dpi", DPI)),
        grayscale=bool(section.get("grayscale", GRAYSCALE)),
        ocr_garbage_tier_pages=bool(
            section.get("ocr_garbage_tier_pages", OCR_GARBAGE_TIER_PAGES)
        ),
        ocr_extraction_failed_pages=bool(
            section.get("ocr_extraction_failed_pages", OCR_EXTRACTION_FAILED_PAGES)
        ),
        tesseract_cmd=str(section.get("tesseract_cmd", TESSERACT_CMD)),
        timeout_seconds=float(section.get("timeout_seconds", TIMEOUT_SECONDS)),
    )


# --- Routing ------------------------------------------------------------------


def is_routed(page: PageQuality, config: OcrSettings) -> bool:
    """Whether one page goes to OCR (see the module docstring)."""
    if page.source == SOURCE_NO_TEXT_LAYER and not page.extraction_failed:
        return True
    if page.tier == TIER_GARBAGE and config.ocr_garbage_tier_pages:
        return True
    if page.extraction_failed and config.ocr_extraction_failed_pages:
        return True
    return False


def route_pages(
    pages: Sequence[PageQuality], settings: Optional[Mapping[str, Any]] = None
) -> None:
    """Record the routing decision on every page, as ``ocr_routed``."""
    config = ocr_settings(settings)
    for page in pages:
        page.ocr_routed = is_routed(page, config)


# --- Results --------------------------------------------------------------------


@dataclass
class TesseractOutput:
    """Tesseract's raw output for one page, as run or as cached."""

    text: str
    tsv: str


@dataclass
class OcrOutcome:
    """What OCR gave one routed page: text (possibly empty), or an error."""

    text: Optional[str] = None  # plain text minus the page separator; None on error
    error: Optional[str] = None  # message of a failed attempt, bounded in length
    mean_confidence: Optional[float] = None  # 0-100; None without word rows


@dataclass
class DocumentOcr:
    """OCR outcomes for one document's routed pages, plus the engine used."""

    outcomes: Dict[int, OcrOutcome] = field(default_factory=dict)
    engine_version: Optional[str] = None  # None if nothing was routed or no Tesseract
    dpi: int = DPI
    grayscale: bool = GRAYSCALE
    routed: bool = False


class OcrError(Exception):
    """A render or Tesseract failure on one page, or a failed version query."""


# --- Text and TSV ---------------------------------------------------------------


def ocr_text(raw_text: str) -> str:
    """Tesseract's plain text minus the trailing page-separator form feed."""
    if raw_text.endswith(PAGE_SEPARATOR):
        return raw_text[: -len(PAGE_SEPARATOR)]
    return raw_text


def mean_word_confidence(tsv: str) -> Optional[float]:
    """Unweighted mean ``conf`` of the TSV's word rows that have text.

    Word rows are level 5; rows with empty text are skipped, and ``conf`` is
    used as Tesseract reports it (0-100). Rounded to 2 decimals; None if there
    are no such rows.
    """
    reader = csv.reader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    header = next(reader, None)
    if header is None:
        return None
    try:
        level_at = header.index("level")
        conf_at = header.index("conf")
        text_at = header.index("text")
    except ValueError:
        logger.warning("Tesseract TSV has no level/conf/text header: %r", header)
        return None

    confidences: List[float] = []
    for row in reader:
        if len(row) <= max(level_at, conf_at):
            continue
        text = row[text_at] if len(row) > text_at else ""
        if row[level_at].strip() != TSV_WORD_LEVEL or not text.strip():
            continue
        try:
            confidences.append(float(row[conf_at]))
        except ValueError:
            continue
    if not confidences:
        return None
    return round(fmean(confidences), 2)


def _bounded(message: str, limit: int = MAX_ERROR_CHARS) -> str:
    return message if len(message) <= limit else message[: limit - 3] + "..."


def _decode(output: Optional[bytes]) -> str:
    if not output:
        return ""
    if isinstance(output, str):
        return output
    return output.decode("utf-8", errors="replace")


# --- Rendering and Tesseract ----------------------------------------------------


class LazyPdf:
    """A PDF opened with pypdfium2 on first use, once per document."""

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        self._pdf: Optional[pdfium.PdfDocument] = None
        self._error: Optional[BaseException] = None

    def get(self) -> pdfium.PdfDocument:
        if self._error is not None:
            raise self._error
        if self._pdf is None:
            try:
                self._pdf = pdfium.PdfDocument(str(self.path))
            except Exception as error:  # noqa: BLE001 - remembered, re-raised per page
                self._error = error
                raise
        return self._pdf

    def close(self) -> None:
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None


def render_page(
    pdf: LazyPdf, page_number: int, dpi: int, grayscale: bool
) -> "Image.Image":
    """Render one whole page (1-based) at ``dpi``; mode ``L`` when grayscale.

    The PDF is opened on the first call for a document, not before.
    """
    page = pdf.get()[page_number - 1]
    try:
        bitmap = page.render(scale=dpi / POINTS_PER_INCH, grayscale=grayscale)
        try:
            # to_pil() may share the bitmap's buffer; copy before closing it.
            return bitmap.to_pil().copy()
        finally:
            bitmap.close()
    finally:
        page.close()


def _run_process(
    args: Sequence[str], timeout_seconds: float
) -> subprocess.CompletedProcess:
    """Start one process: the only place this module does. Tests replace this."""
    return subprocess.run(
        list(args), capture_output=True, timeout=timeout_seconds, check=False
    )


def _start(args: Sequence[str], timeout_seconds: float) -> subprocess.CompletedProcess:
    """:func:`_run_process`, with its failures turned into :class:`OcrError`."""
    try:
        return _run_process(args, timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise OcrError(
            f"Tesseract timed out after {timeout_seconds:g} s ({args[0]})"
        ) from error
    except OSError as error:
        raise OcrError(
            f"Tesseract could not be started ({args[0]}): "
            f"{type(error).__name__}: {error}"
        ) from error


def run_tesseract(
    image: "Image.Image",
    *,
    language: str,
    dpi: int,
    tesseract_cmd: str,
    timeout_seconds: float,
    work_dir: Optional[Path],
) -> TesseractOutput:
    """One Tesseract run on one image, writing plain text and TSV.

    Temporary files go to a fresh folder inside ``work_dir`` (the system temp
    folder when None) and are removed afterwards.
    """
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="ocr-page-", dir=work_dir))
    try:
        image_path = temp_dir / "page.png"
        image.save(image_path, format="PNG")
        output_base = temp_dir / "out"
        args = [
            tesseract_cmd,
            str(image_path),
            str(output_base),
            "-l",
            language,
            "--dpi",
            str(dpi),
            "txt",
            "tsv",
        ]
        completed = _start(args, timeout_seconds)
        if completed.returncode != 0:
            stderr = _decode(completed.stderr).strip()
            raise OcrError(
                f"Tesseract exited with code {completed.returncode}: "
                f"{_bounded(stderr, MAX_STDERR_CHARS)}"
            )
        text = output_base.with_suffix(".txt").read_text(encoding="utf-8")
        tsv = output_base.with_suffix(".tsv").read_text(encoding="utf-8")
        return TesseractOutput(text=text, tsv=tsv)
    finally:
        try:
            shutil.rmtree(temp_dir)
        except OSError as error:
            logger.warning(
                "Could not remove OCR temporary folder %s: %s: %s",
                temp_dir,
                type(error).__name__,
                error,
            )


def ocr_page(
    pdf: LazyPdf,
    page_number: int,
    *,
    language: str,
    dpi: int,
    grayscale: bool,
    tesseract_cmd: str,
    timeout_seconds: float,
    work_dir: Optional[Path],
) -> TesseractOutput:
    """The page seam: render one page and run Tesseract on it.

    Raises on any failure; the caller turns that into the page's OCR error.
    """
    try:
        image = render_page(pdf, page_number, dpi, grayscale)
    except Exception as error:  # noqa: BLE001 - any render failure is an OCR error
        raise OcrError(f"render failed: {type(error).__name__}: {error}") from error
    return run_tesseract(
        image,
        language=language,
        dpi=dpi,
        tesseract_cmd=tesseract_cmd,
        timeout_seconds=timeout_seconds,
        work_dir=work_dir,
    )


_VERSION_TOKEN = re.compile(r"v?\d+(?:\.\d+)+\S*")


def version_token(line: str) -> str:
    """Reduce ``<tesseract_cmd> --version``'s first line to its version token.

    ``"tesseract v5.5.3.20260724"`` gives ``"v5.5.3.20260724"``. Falls back to
    the raw line when no token after the program name looks like a version.
    """
    for token in line.split()[1:]:
        if _VERSION_TOKEN.fullmatch(token):
            return token
    return line.strip()


def query_tesseract_version(tesseract_cmd: str, timeout_seconds: float) -> str:
    """The version seam: run ``<tesseract_cmd> --version``. Raises on failure."""
    completed = _start([tesseract_cmd, "--version"], timeout_seconds)
    if completed.returncode != 0:
        stderr = _decode(completed.stderr).strip()
        raise OcrError(
            f"'{tesseract_cmd} --version' exited with code {completed.returncode}: "
            f"{_bounded(stderr, MAX_STDERR_CHARS)}"
        )
    output = _decode(completed.stdout).strip() or _decode(completed.stderr).strip()
    if not output:
        raise OcrError(f"'{tesseract_cmd} --version' printed nothing")
    return version_token(output.splitlines()[0])


# --- Cache ----------------------------------------------------------------------


def cache_key(
    sha256: str,
    page_number: int,
    dpi: int,
    grayscale: bool,
    language: str,
    tesseract_version: str,
) -> Dict[str, Any]:
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "sha256": sha256,
        "page_number": page_number,
        "dpi": dpi,
        "grayscale": grayscale,
        "language": language,
        "tesseract_version": tesseract_version,
    }


def cache_entry_path(cache_dir: Path, key: Mapping[str, Any]) -> Path:
    """One file per key, named after a hash of the whole key."""
    canonical = json.dumps(dict(key), sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def read_cache_entry(
    cache_dir: Path, key: Mapping[str, Any]
) -> Optional[TesseractOutput]:
    """The cached output for ``key``, or None on a miss.

    An unreadable or corrupt entry, or one holding another key, is a miss and
    is logged as a WARNING; the caller rewrites it after OCR.
    """
    path = cache_entry_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(entry, dict)
            or entry.get("key") != dict(key)
            or not isinstance(entry.get("text"), str)
            or not isinstance(entry.get("tsv"), str)
        ):
            raise ValueError("entry does not hold the requested key, text and TSV")
    except (OSError, ValueError) as error:
        logger.warning(
            "OCR cache entry %s is unreadable or corrupt (%s: %s); treated as a "
            "miss and rewritten",
            path,
            type(error).__name__,
            error,
        )
        return None
    return TesseractOutput(text=entry["text"], tsv=entry["tsv"])


def write_cache_entry(
    cache_dir: Path, key: Mapping[str, Any], output: TesseractOutput
) -> None:
    """Write one entry atomically: a temporary file, then a replace.

    A failed write is logged and changes nothing else.
    """
    path = cache_entry_path(cache_dir, key)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            prefix=f"{path.stem}.", suffix=".tmp", dir=cache_dir
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(
                    {"key": dict(key), "text": output.text, "tsv": output.tsv},
                    stream,
                    ensure_ascii=False,
                )
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
    except OSError as error:
        logger.warning(
            "Could not write OCR cache entry %s: %s: %s",
            path,
            type(error).__name__,
            error,
        )


# --- One run ----------------------------------------------------------------------


class OcrRun:
    """Run-level OCR state: the cache directory and the Tesseract version.

    One instance per run. The version is queried lazily, at the first routed
    page of the run, and never for a run without routed pages. If Tesseract
    cannot be started, one ERROR is logged for the run and every routed page
    gets an OCR error. ``cache_dir=None`` caches nothing, and temporary files
    then go to the system temp folder.
    """

    def __init__(self, cache_dir: Optional[PathLike] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._version_queried = False
        self._version: Optional[str] = None
        self._unavailable: Optional[str] = None

    def engine_version(self, config: OcrSettings) -> Optional[str]:
        if not self._version_queried:
            self._version_queried = True
            try:
                self._version = query_tesseract_version(
                    config.tesseract_cmd, config.timeout_seconds
                )
            except Exception as error:  # noqa: BLE001 - any failure: no Tesseract this run
                self._unavailable = (
                    str(error)
                    if isinstance(error, OcrError)
                    else f"{type(error).__name__}: {error}"
                )
                logger.error(
                    "Tesseract is unavailable (%s); every page routed to OCR in "
                    "this run gets an OCR error",
                    self._unavailable,
                )
        return self._version

    def ocr_document(
        self,
        pdf_path: PathLike,
        page_numbers: Sequence[int],
        *,
        sha256: Optional[str],
        language: str,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> DocumentOcr:
        """OCR the routed pages of one document, through the cache."""
        config = ocr_settings(settings)
        result = DocumentOcr(dpi=config.dpi, grayscale=config.grayscale)
        if not page_numbers:
            return result

        result.routed = True
        version = self.engine_version(config)
        result.engine_version = version
        use_cache = self.cache_dir is not None and sha256 is not None
        work_dir = self.cache_dir / TEMP_SUBDIR if self.cache_dir is not None else None
        from_cache = 0
        ocr_now = 0

        pdf = LazyPdf(pdf_path)
        try:
            for page_number in page_numbers:
                if version is None:
                    result.outcomes[page_number] = self._failed(
                        pdf_path,
                        page_number,
                        f"Tesseract unavailable: {self._unavailable}",
                    )
                    continue

                key = cache_key(
                    sha256 or "",
                    page_number,
                    config.dpi,
                    config.grayscale,
                    language,
                    version,
                )
                output = read_cache_entry(self.cache_dir, key) if use_cache else None
                if output is not None:
                    from_cache += 1
                else:
                    try:
                        output = ocr_page(
                            pdf,
                            page_number,
                            language=language,
                            dpi=config.dpi,
                            grayscale=config.grayscale,
                            tesseract_cmd=config.tesseract_cmd,
                            timeout_seconds=config.timeout_seconds,
                            work_dir=work_dir,
                        )
                    except Exception as error:  # noqa: BLE001 - caught per page (spec 003)
                        message = (
                            str(error)
                            if isinstance(error, OcrError)
                            else f"{type(error).__name__}: {error}"
                        )
                        result.outcomes[page_number] = self._failed(
                            pdf_path, page_number, message
                        )
                        continue
                    ocr_now += 1
                    if use_cache:
                        write_cache_entry(self.cache_dir, key, output)

                result.outcomes[page_number] = OcrOutcome(
                    text=ocr_text(output.text),
                    mean_confidence=mean_word_confidence(output.tsv),
                )
        finally:
            pdf.close()

        logger.info(
            "OCR for %s: %d page(s) routed, %d served from the cache, %d OCR'd now",
            pdf_path,
            len(page_numbers),
            from_cache,
            ocr_now,
        )
        return result

    @staticmethod
    def _failed(pdf_path: PathLike, page_number: int, message: str) -> OcrOutcome:
        message = _bounded(message)
        logger.warning("OCR failed for %s page %d: %s", pdf_path, page_number, message)
        return OcrOutcome(error=message)
