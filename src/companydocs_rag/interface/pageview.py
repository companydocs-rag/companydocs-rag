"""The hash to file map, the document types and the page view (spec 009,
D9, D11, R12-R14).

The map (R13): every PDF under the raw folder, found as the parse pass finds
them (``find_pdfs``, recursively), hashed once at startup with
``hashlib.file_digest``, which streams the file and gives what
``parse_pass.file_sha256`` gives. A file that cannot be read is left out
with one WARNING; when several files share a hash, the first in
``find_pdfs`` order is used, with one WARNING; no PDF at all is one WARNING.
These startup messages may name paths and file names. The map is not
updated after startup.

The document types (R14): spec 008's resolution, override file, then client
index, else none, resolved at startup for every mapped PDF from its path in
the map.

The page view (R12): a request names a hash and a 1-based page number,
never a path (:func:`parse_page_query`). :func:`render_png` looks the hash
up, re-hashes the file, checks the page against the page count pypdfium2
reports, renders that page in colour with ``ingestion.ocr.render_page`` and
encodes it as PNG in memory. Nothing is written to disk. A refusal raises
:class:`PageRefused` with its kind; the caller logs it without a path, a
file name or an exception message.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple
from urllib.parse import parse_qsl

from companydocs_rag.ingestion.cli import find_pdfs
from companydocs_rag.ingestion.language import language_settings
from companydocs_rag.ingestion.ocr import LazyPdf, render_page
from companydocs_rag.metadata.doctype import TypeOverrides, TypeResolver

logger = logging.getLogger(__name__)

PAGE_PATH = "/pagina"
PARAM_SHA256 = "sha256"
PARAM_PAGE = "nummer"
SHA256_PATTERN = re.compile("[0-9a-f]{64}")
PAGE_NUMBER_PATTERN = re.compile("[1-9][0-9]{0,5}")
PNG_FORMAT = "PNG"

# R12: the kinds of refusal a page request's WARNING names.
REFUSED_UNKNOWN = "unknown"
REFUSED_UNREADABLE = "unreadable"
REFUSED_CHANGED = "changed"
REFUSED_OUT_OF_RANGE = "out of range"
REFUSED_RENDER = "render"

_HASH_NAME = "sha256"


class PageRefused(Exception):
    """A page request that gets 404 (R12); ``kind`` is one of the
    ``REFUSED_*`` values."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


def page_url(file_sha256: str, page: int) -> str:
    """R12's URL of one page: ``/pagina?sha256=<hash>&nummer=<page>``."""
    return f"{PAGE_PATH}?{PARAM_SHA256}={file_sha256}&{PARAM_PAGE}={page}"


def is_page_number(value: object) -> bool:
    """Whether ``value`` is an integer that R12's page pattern accepts."""
    return type(value) is int and PAGE_NUMBER_PATTERN.fullmatch(str(value)) is not None


def parse_page_query(query: str) -> Optional[Tuple[str, int]]:
    """The hash and the page number of a page request's query, or None when
    it does not hold exactly ``sha256`` and ``nummer``, each once and nothing
    else, matching R12's patterns."""
    pairs = parse_qsl(query, keep_blank_values=True)
    if sorted(name for name, _ in pairs) != sorted((PARAM_SHA256, PARAM_PAGE)):
        return None
    values = dict(pairs)
    file_sha256, page = values[PARAM_SHA256], values[PARAM_PAGE]
    if SHA256_PATTERN.fullmatch(file_sha256) is None:
        return None
    if PAGE_NUMBER_PATTERN.fullmatch(page) is None:
        return None
    return file_sha256, int(page)


def file_digest(path: Path) -> str:
    """The hex sha256 of the file's bytes, streamed; raises ``OSError`` when
    the file cannot be read."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, _HASH_NAME).hexdigest()


def build_file_map(raw_dir: Path) -> Dict[str, Path]:
    """The hash to file map over every PDF under ``raw_dir`` (R13)."""
    pdfs = find_pdfs(raw_dir, recursive=True)
    if not pdfs:
        logger.warning("No PDF files under %s", raw_dir)
        return {}
    found: Dict[str, List[Path]] = {}
    for path in pdfs:
        try:
            digest = file_digest(path)
        except OSError as error:
            logger.warning(
                "Could not read %s to hash it; it is left out: %s: %s",
                path,
                type(error).__name__,
                error,
            )
            continue
        found.setdefault(digest, []).append(path)
    for digest, paths in found.items():
        if len(paths) > 1:
            logger.warning(
                "%d files share sha256 %s; the first is used", len(paths), digest
            )
    return {digest: paths[0] for digest, paths in found.items()}


def document_types(
    files: Mapping[str, Path], overrides: TypeOverrides
) -> Dict[str, Optional[str]]:
    """Each mapped document's type by its hash, from its path in the map
    (R14): the override file, then the client index, else None."""
    resolver = TypeResolver(overrides, language_settings()["index_filename"])
    return {
        digest: resolver.resolve(str(path)).document_type
        for digest, path in files.items()
    }


def render_png(
    files: Mapping[str, Path], file_sha256: str, page: int, dpi: int
) -> bytes:
    """The PNG of page ``page`` of the file whose hash is ``file_sha256``,
    rendered in colour at ``dpi``; raises :class:`PageRefused` (R12)."""
    path = files.get(file_sha256)
    if path is None:
        raise PageRefused(REFUSED_UNKNOWN)
    try:
        digest = file_digest(path)
    except OSError:
        raise PageRefused(REFUSED_UNREADABLE) from None
    if digest != file_sha256:
        raise PageRefused(REFUSED_CHANGED)
    pdf = LazyPdf(path)
    try:
        try:
            page_count = len(pdf.get())
        except Exception:  # noqa: BLE001 - any failure to open is R12's render
            raise PageRefused(REFUSED_RENDER) from None
        if page > page_count:
            raise PageRefused(REFUSED_OUT_OF_RANGE)
        try:
            image = render_page(pdf, page, dpi, grayscale=False)
            buffer = io.BytesIO()
            image.save(buffer, format=PNG_FORMAT)
        except Exception:  # noqa: BLE001 - any failure to render is R12's render
            raise PageRefused(REFUSED_RENDER) from None
    finally:
        pdf.close()
    return buffer.getvalue()
