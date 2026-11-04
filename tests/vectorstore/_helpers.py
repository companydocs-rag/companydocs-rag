"""Shared helpers for the index tests (spec 006): fabricated chunk files.

Nothing under ``data/`` is read or written: chunk files are written in
``tmp_path``, most of them by ``chunk_document`` from fabricated normalized
records (``tests/chunking/_helpers.py``), and every run passes both folders.
Every Qdrant client opened here is closed again.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from qdrant_client import QdrantClient

from companydocs_rag.chunking.chunker import chunk_document
from companydocs_rag.vectorstore import cli
from companydocs_rag.vectorstore.store import COLLECTION_NAME
from tests.chunking._helpers import block, filler, normalized_record
from tests.embeddings._helpers import StubEmbedder, StubLoader

LOGGER = "companydocs_rag.vectorstore.cli"
LAST_ERROR = "Index not written; the previous index is unchanged"

# The payload's keys in D13's order.
PAYLOAD_KEYS = [
    "chunk_id",
    "chunk_index",
    "file_sha256",
    "file_path",
    "language",
    "language_source",
    "route",
    "text",
    "page_start",
    "page_end",
    "text_origins",
    "page_spans",
    "chunk_method",
    "clause_label",
    "clause_number",
    "part",
    "part_count",
]

STRUCTURED_SHA256 = "a1" * 32
GENERIC_SHA256 = "b2" * 32


def articles() -> str:
    """Three article sections of 500 characters: a structured document."""
    return (
        block("Artikel 1 Begrippen", 500)
        + block("Artikel 2 Huurprijs", 500)
        + block("Artikel 3 Duur", 500)
    )


def chunk_record(
    *pages: str,
    path: str = "invoer/fictief.pdf",
    sha256: Optional[str] = STRUCTURED_SHA256,
    status: str = "processed",
) -> Dict[str, Any]:
    """A chunk file's record, as ``companydocs-chunk`` writes it, from a
    fabricated normalized record."""
    return chunk_document(
        normalized_record(*pages, path=path, sha256=sha256, status=status)
    )


def structured_record() -> Dict[str, Any]:
    return chunk_record(articles(), path="invoer/a/fictief-huur.pdf")


def generic_record() -> Dict[str, Any]:
    return chunk_record(
        filler(3000), path="invoer/b/fictief-generiek.pdf", sha256=GENERIC_SHA256
    )


def failed_record() -> Dict[str, Any]:
    return chunk_record(
        path="invoer/c/fictief-mislukt.pdf", sha256=None, status="failed"
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hand_chunk(
    text: str,
    *,
    file_sha256: str,
    index: int = 0,
    label: Optional[str] = None,
    number: Optional[str] = None,
) -> Dict[str, Any]:
    """A chunk written by hand, in the shape ``companydocs-chunk`` writes,
    for fixtures ``chunk_document`` does not produce. Its id is a hash of
    the file hash, the index and the text, as unique as the chunker's."""
    labelled = label is not None
    return {
        "chunk_id": _sha256(f"{file_sha256}:{index}:{text}")[:20],
        "chunk_index": index,
        "file_sha256": file_sha256,
        "text": text,
        "text_sha256": _sha256(text),
        "char_start": 0,
        "char_end": len(text),
        "page_start": 1,
        "page_end": 1,
        "text_origins": ["native"],
        "page_spans": [
            {
                "page_number": 1,
                "char_start": 0,
                "char_end": len(text),
                "text_origin": "native",
                "tier": "Clean",
                "ocr_mean_confidence": None,
            }
        ],
        "chunk_method": "heading_section" if labelled else "generic",
        "clause_label": label,
        "clause_number": number,
        "numbering_run": 1 if labelled else None,
        "heading_family": "keyword" if labelled else None,
        "part": None,
        "part_count": None,
        "exceeds_max_chars": False,
    }


def hand_record(
    *texts: Any,
    sha256: Optional[str],
    path: str = "invoer/fictief.pdf",
    status: str = "processed",
) -> Dict[str, Any]:
    """A chunk file's record written by hand. Each text is a chunk; a
    (label, number, text) tuple is a labelled one."""
    chunks = []
    for index, item in enumerate(texts):
        label, number, text = item if isinstance(item, tuple) else (None, None, item)
        chunks.append(
            hand_chunk(
                text, file_sha256=sha256, index=index, label=label, number=number
            )
        )
    return {
        "schema_version": 1,
        "chunker_version": 1,
        "file": {"path": path, "sha256": sha256},
        "status": status,
        "normalized_hash": "00" * 32,
        "language": "nld",
        "language_source": "index",
        "structure_hint": "unknown",
        "route": "structured"
        if any(chunk["clause_label"] for chunk in chunks)
        else "generic",
        "empty_pages": [],
        "heading_evidence": {
            "family": None,
            "candidate_counts": {"keyword": 0, "decimal": 0},
            "runs": [],
        },
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def write_chunk_file(
    folder: Path, relative: str, record: Union[str, Mapping[str, Any]]
) -> Path:
    """Write one chunk file as ``companydocs-chunk`` does; a string is written
    as it is."""
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        record
        if isinstance(record, str)
        else json.dumps(record, indent=2, ensure_ascii=False)
    )
    path.write_text(text, encoding="utf-8")
    return path


def three_documents(folder: Path) -> Dict[str, Dict[str, Any]]:
    """A7's input in nested folders: structured, generic and failed. Returns
    the records by forward-slash relative path."""
    records = {
        "a/fictief-huur.json": structured_record(),
        "b/c/fictief-generiek.json": generic_record(),
        "b/fictief-mislukt.json": failed_record(),
    }
    for relative, record in records.items():
        write_chunk_file(folder, relative, record)
    return records


def run_index(
    chunks: Path,
    index: Path,
    *extra: str,
    loader: Optional[StubLoader] = None,
    embedder: Optional[StubEmbedder] = None,
) -> int:
    """``companydocs-index`` over ``chunks`` into ``index``, with the stub."""
    load = loader if loader is not None else StubLoader(embedder)
    return cli.main(
        ["--chunks-dir", str(chunks), "--index-dir", str(index), *extra],
        load_embedder=load,
    )


def tree(folder: Path) -> Dict[str, Optional[bytes]]:
    """Every file and folder under ``folder``, with the bytes of each file."""
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        if path.is_file()
        else None
        for path in sorted(folder.rglob("*"))
    }


def stored_points(index: Path) -> Dict[str, Any]:
    """Every point of the index by id, with payload and vectors; the client
    is closed again."""
    client = QdrantClient(path=str(index))
    try:
        points, _ = client.scroll(
            COLLECTION_NAME, limit=10_000, with_payload=True, with_vectors=True
        )
    finally:
        client.close()
    return {str(point.id): copy.deepcopy(point) for point in points}


def messages(caplog, level) -> list:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == level and record.name == LOGGER
    ]
