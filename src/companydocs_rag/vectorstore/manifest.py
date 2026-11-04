"""``manifest.json`` (spec 006, D16, D17): what an index was built from.

``companydocs-index`` writes it last, into the staging folder, once the Qdrant
client is closed (D14). A folder without it is not an index. It is UTF-8 JSON
with ``indent=2`` and ``ensure_ascii=False``, written as ``companydocs-chunk``
writes its files, and holds no timestamp, so two runs over the same input
write the same bytes. The keys and their order are :func:`build_manifest`'s.

Before a search, :func:`compare_manifest` checks eight keys against the
current settings and code constants (D17, R11); any difference makes the
index unusable. ``libraries`` holds the installed versions, read with
``importlib.metadata`` without importing the packages.
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from companydocs_rag.embeddings.dense import EMBEDDING_TEXT_VERSION
from companydocs_rag.embeddings.sparse import TERM_HASH, TOKENIZER_VERSION

from .store import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

MANIFEST_VERSION = 1
MANIFEST_FILE_NAME = "manifest.json"
STAGING_SUFFIX = ".building"
PREVIOUS_SUFFIX = ".previous"
REQUIRED_CHUNKS_SCHEMA_VERSION = 1

_DISTANCE = "Cosine"
# The manifest's library keys and the distributions they name.
_LIBRARIES = (
    ("qdrant_client", "qdrant-client"),
    ("sentence_transformers", "sentence-transformers"),
    ("transformers", "transformers"),
    ("torch", "torch"),
)
# D17: the compared keys, in order, with the object each sits in (R11).
_COMPARED: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("manifest_version", ()),
    ("model_name", ("embedding",)),
    ("model_revision", ("embedding",)),
    ("embedding_text_version", ("embedding",)),
    ("tokenizer_version", ("sparse",)),
    ("term_hash", ("sparse",)),
    ("bm25_k1", ("sparse",)),
    ("bm25_b", ("sparse",)),
)
_MISSING = object()


def library_versions() -> Dict[str, str]:
    """The installed versions of the four libraries, without importing them."""
    return {key: metadata.version(distribution) for key, distribution in _LIBRARIES}


def build_manifest(
    *,
    model_name: str,
    model_revision: str,
    dimension: int,
    max_tokens_limit: int,
    max_tokens_measured: Optional[int],
    max_tokens_chunk_id: Optional[str],
    bm25_k1: float,
    bm25_b: float,
    avgdl: Optional[float],
    chunk_count: int,
    vocabulary_size: int,
    term_id_collisions: int,
    documents: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """The manifest, keys in D16's order. ``documents`` are the chunk files,
    already in ``chunk_file`` order, each with ``chunk_file``,
    ``chunk_file_sha256``, ``file_sha256``, ``status`` and ``chunk_count``."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "embedding": {
            "model_name": model_name,
            "model_revision": model_revision,
            "dimension": dimension,
            "distance": _DISTANCE,
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
            "max_tokens_limit": max_tokens_limit,
            "max_tokens_measured": max_tokens_measured,
            "max_tokens_chunk_id": max_tokens_chunk_id,
        },
        "sparse": {
            "tokenizer_version": TOKENIZER_VERSION,
            "term_hash": TERM_HASH,
            "bm25_k1": float(bm25_k1),
            "bm25_b": float(bm25_b),
            "avgdl": avgdl,
            "chunk_count": chunk_count,
            "vocabulary_size": vocabulary_size,
            "term_id_collisions": term_id_collisions,
        },
        "collection": {
            "name": COLLECTION_NAME,
            "dense_vector": DENSE_VECTOR_NAME,
            "sparse_vector": SPARSE_VECTOR_NAME,
        },
        "inputs": {
            "chunks_schema_version": REQUIRED_CHUNKS_SCHEMA_VERSION,
            "documents": [
                {
                    "chunk_file": document["chunk_file"],
                    "chunk_file_sha256": document["chunk_file_sha256"],
                    "file_sha256": document["file_sha256"],
                    "status": document["status"],
                    "chunk_count": document["chunk_count"],
                }
                for document in documents
            ],
            "document_count": len(documents),
            "documents_without_chunks": sum(
                1 for document in documents if document["chunk_count"] == 0
            ),
        },
        "libraries": library_versions(),
    }


def write_manifest(folder: Path, manifest: Mapping[str, Any]) -> None:
    """Write ``manifest.json`` into ``folder``, as ``companydocs-chunk`` writes
    its files (``Path.write_text``, R10)."""
    (folder / MANIFEST_FILE_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_manifest(path: Path) -> Dict[str, Any]:
    """The manifest at ``path``. Raises ``OSError`` when it cannot be read and
    ``ValueError`` when it is not JSON or not a JSON object."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("the top level is not a JSON object")
    return manifest


def current_values(
    *, model_name: str, model_revision: str, bm25_k1: float, bm25_b: float
) -> Dict[str, Any]:
    """The values D17 compares, from the current settings and constants."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "model_name": model_name,
        "model_revision": model_revision,
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "term_hash": TERM_HASH,
        "bm25_k1": float(bm25_k1),
        "bm25_b": float(bm25_b),
    }


def _manifest_value(manifest: Mapping[str, Any], key: str, where: Sequence[str]) -> Any:
    node: Any = manifest
    for name in (*where, key):
        if not isinstance(node, Mapping) or name not in node:
            return _MISSING
        node = node[name]
    return node


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _same(found: Any, current: Any) -> bool:
    """Numbers compare by value; anything else must also have the same type."""
    if _is_number(found) and _is_number(current):
        return found == current
    return type(found) is type(current) and found == current


def _as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def compare_manifest(
    manifest: Mapping[str, Any], current: Mapping[str, Any]
) -> List[str]:
    """One ``Index does not match settings`` message per D17 key whose
    manifest value differs from ``current``, in D17's order (R11). A key that
    is absent reads as ``missing``; values are written as JSON."""
    messages: List[str] = []
    for key, where in _COMPARED:
        found = _manifest_value(manifest, key, where)
        if found is not _MISSING and _same(found, current[key]):
            continue
        shown = "missing" if found is _MISSING else _as_json(found)
        messages.append(
            f"Index does not match settings: {key} is {shown} in the index, "
            f"{_as_json(current[key])} now"
        )
    return messages
