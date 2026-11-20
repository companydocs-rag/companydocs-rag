"""The document library and the status block (spec 011).

Two read-only views the page fetches as JSON, each built once at startup
and never updated, like the hash to file map (R13):

- the library (``GET /documenten``): one entry per mapped PDF with its hash,
  its file name, the folders between the raw folder and the file, its
  document type (R14, else "type onbekend") and its page count as pypdfium2
  reports it (null when the file cannot be opened); sorted by folders, then
  name.
- the status (``GET /status``): the generator's model, whether its endpoint
  is on this machine, and the number of mapped documents.

Every text a person reads is built here from :mod:`.texts`; the page's
script sets each one with ``textContent``. Nothing is written to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlsplit

from companydocs_rag.ingestion.ocr import LazyPdf

from . import texts

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def page_count(path: Path) -> Optional[int]:
    """The number of pages pypdfium2 reports, or None when the file cannot
    be opened."""
    pdf = LazyPdf(path)
    try:
        return len(pdf.get())
    except Exception:  # noqa: BLE001 - any failure to open leaves the count out
        return None
    finally:
        pdf.close()


def _folders(raw_dir: Path, path: Path) -> List[str]:
    try:
        return list(path.relative_to(raw_dir).parent.parts)
    except ValueError:
        return []


def document_library(
    raw_dir: Path, files: Mapping[str, Path], types: Mapping[str, Optional[str]]
) -> List[Dict[str, Any]]:
    """The library: one entry per mapped PDF, sorted by folders, then name."""
    entries = []
    for digest, path in files.items():
        count = page_count(path)
        document_type = types.get(digest)
        entries.append(
            {
                "sha256": digest,
                "name": path.name,
                "folders": _folders(raw_dir, path),
                "type": texts.NO_DOCUMENT_TYPE
                if document_type is None
                else document_type,
                "pages": count,
                "pages_label": "" if count is None else texts.page_count(count),
            }
        )
    entries.sort(
        key=lambda entry: ([f.lower() for f in entry["folders"]], entry["name"].lower())
    )
    return entries


def status_block(base_url: str, model: str, document_count: int) -> Dict[str, str]:
    """The side bar's status: where the model runs, its name, the host of a
    remote endpoint and the number of mapped documents."""
    host = urlsplit(base_url).hostname or ""
    local = host in LOCAL_HOSTS
    return {
        "mode": texts.STATUS_LOCAL if local else texts.STATUS_API,
        "model": model,
        "host": "" if local else host,
        "documents": texts.document_count(document_count),
    }
