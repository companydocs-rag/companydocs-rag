"""The ``companydocs-index`` command (spec 006): build the index from the chunks.

    companydocs-index [--chunks-dir DIR] [--index-dir DIR] [-q]

It reads every ``*.json`` file under ``--chunks-dir`` (default
``data/interim/chunks`` under the project root, where ``companydocs-chunk``
writes), recursively, skipping file names that start with ".", in the order
of their relative paths written with forward slashes (R10). Chunk files are
read, never written. Every run rebuilds the whole index (D14), in this order:

1. ``--chunks-dir`` must be a folder and the ``embeddings`` and ``retrieval``
   settings valid, else exit 2, before any file is read.
2. Every chunk file is read and checked (R6): one whose ``schema_version`` is
   missing or not the integer 1 is refused; one that cannot be read, is not
   JSON, not an object or not a valid chunk record fails. Then the chunk ids
   of all valid files are checked for duplicates (D12), and an existing
   ``--index-dir`` that is not empty and holds no ``manifest.json`` is
   refused (Q2). Every problem is logged; any problem ends the run, before
   the model is loaded (R7).
3. The embedder is loaded, which may download the model (D3), and every
   composed text is counted with the model's tokenizer against
   min(``embeddings.max_tokens``, the model's ``max_seq_length``) (D18).
4. Terms, ``avgdl``, BM25 values and term ids are computed (D5-D8).
5. In the staging folder ``<index-dir>.building``, removed first if present:
   the collection is created and one point per chunk uploaded in batches;
   the point count is checked (D12), the client closed and ``manifest.json``
   written last. Then the swap (R8): ``<index-dir>`` is renamed to
   ``<index-dir>.previous``, the staging folder to ``<index-dir>``, and the
   previous index removed. A failed second rename renames the previous index
   back. A failed removal is a WARNING; the next run removes the folder.
6. Two INFO summary lines.

Exit codes: 0 when the index is in place; 1 when it was not written, in which
case the last ERROR says so and the previous index is unchanged; 2 when the
command cannot start. One process at a time: Qdrant local mode locks the
index folder (D24).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from companydocs_rag.chunking.chunker import ROUTES
from companydocs_rag.chunking.cli import DEFAULT_CHUNKS_RELATIVE_PATH
from companydocs_rag.embeddings.dense import (
    Embedder,
    EmbeddingsSettings,
    SentenceTransformerEmbedder,
    SettingsError,
    compose_embedding_text,
    embeddings_settings,
)
from companydocs_rag.embeddings.sparse import (
    bm25_vector,
    sparse_terms,
    term_id_collisions,
)
from companydocs_rag.ingestion.parse_pass import STATUS_FAILED, STATUS_PROCESSED
from companydocs_rag.ingestion.settings import find_project_root
from companydocs_rag.retrieval.search import RetrievalSettings, retrieval_settings

from .manifest import (
    MANIFEST_FILE_NAME,
    PREVIOUS_SUFFIX,
    REQUIRED_CHUNKS_SCHEMA_VERSION,
    STAGING_SUFFIX,
    build_manifest,
    write_manifest,
)
from .store import (
    Point,
    count_points,
    create_collection,
    open_client,
    upload_points,
)

logger = logging.getLogger(__name__)

# Where the index lands unless --index-dir overrides it: git-ignored, like the
# chunk files it is built from.
DEFAULT_INDEX_RELATIVE_PATH = Path("data") / "processed" / "index"

EXIT_INDEX_WRITTEN = 0
EXIT_INDEX_NOT_WRITTEN = 1
EXIT_CANNOT_START = 2

_UPLOAD_BATCH_SIZE = 64
_CHUNK_ID = re.compile("[0-9a-f]{20}")

# Builds the embedder from the settings; replaced in tests.
EmbedderLoader = Callable[[EmbeddingsSettings], Embedder]


def _project_base() -> Path:
    root = find_project_root()
    return root if root is not None else Path.cwd()


def default_chunks_dir() -> Path:
    """The folder ``companydocs-chunk`` writes by default, under the project
    root (falling back to the working directory)."""
    return _project_base() / DEFAULT_CHUNKS_RELATIVE_PATH


def default_index_dir() -> Path:
    """The default index folder, resolved like :func:`default_chunks_dir`."""
    return _project_base() / DEFAULT_INDEX_RELATIVE_PATH


def find_chunk_files(folder: Path) -> List[Path]:
    """Every ``*.json`` file under ``folder``, recursively, except file names
    starting with ".", in the order of their forward-slash relative paths
    (R10)."""
    return sorted(
        (
            path
            for path in folder.rglob("*.json")
            if path.is_file() and not path.name.startswith(".")
        ),
        key=lambda path: path.relative_to(folder).as_posix(),
    )


def refused_schema_version(record: Mapping[str, Any]) -> Optional[str]:
    """How a refused record's ``schema_version`` is shown, or None when it is
    the integer 1: ``missing`` when absent, else the value as JSON."""
    if "schema_version" not in record:
        return "missing"
    value = record["schema_version"]
    if type(value) is int and value == REQUIRED_CHUNKS_SCHEMA_VERSION:
        return None
    return json.dumps(value)


# --- The chunk record (R6) ------------------------------------------------------------


class ChunkRecordError(ValueError):
    """A chunk file that is JSON but not a valid chunk record (R6)."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ChunkRecordError(message)


def _get(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ChunkRecordError(f"{where} lacks the key {key!r}")
    return mapping[key]


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_optional_str(value: Any) -> bool:
    return value is None or isinstance(value, str)


def validate_chunk_record(record: Any) -> None:
    """Raise :class:`ChunkRecordError` unless ``record`` holds what the index
    reads, with the types ``companydocs-chunk`` writes (R6). The schema version
    is checked before this, by the command."""
    _require(isinstance(record, Mapping), "the top level is not a JSON object")
    file_info = _get(record, "file", "the record")
    _require(isinstance(file_info, Mapping), "'file' is not an object")
    _require(
        isinstance(_get(file_info, "path", "'file'"), str),
        "'file.path' is not a string",
    )
    status = _get(record, "status", "the record")
    _require(
        status in (STATUS_PROCESSED, STATUS_FAILED),
        f"'status' is {json.dumps(status)}, not processed or failed",
    )
    sha256 = _get(file_info, "sha256", "'file'")
    if status == STATUS_FAILED:
        _require(_is_optional_str(sha256), "'file.sha256' is not a string or null")
    else:
        _require(
            isinstance(sha256, str),
            "'file.sha256' is not a string, and only a failed document may have none",
        )
    for key in ("language", "language_source"):
        _require(
            _is_optional_str(_get(record, key, "the record")),
            f"'{key}' is not a string or null",
        )
    route = _get(record, "route", "the record")
    _require(
        route in ROUTES, f"'route' is {json.dumps(route)}, not structured or generic"
    )
    chunks = _get(record, "chunks", "the record")
    _require(isinstance(chunks, list), "'chunks' is not a list")
    if status == STATUS_FAILED:
        _require(not chunks, "'chunks' is not empty, and a failed document has none")
    chunk_count = _get(record, "chunk_count", "the record")
    _require(_is_int(chunk_count), "'chunk_count' is not an integer")
    _require(
        chunk_count == len(chunks),
        f"'chunk_count' is {chunk_count}, not the number of chunks ({len(chunks)})",
    )
    for position, chunk in enumerate(chunks):
        _validate_chunk(chunk, f"chunks[{position}]", sha256)


def _validate_chunk(chunk: Any, where: str, file_sha256: Any) -> None:
    _require(isinstance(chunk, Mapping), f"{where} is not an object")
    chunk_id = _get(chunk, "chunk_id", where)
    _require(
        isinstance(chunk_id, str) and _CHUNK_ID.fullmatch(chunk_id) is not None,
        f"{where}.chunk_id is not 20 lowercase hex characters",
    )
    for key in ("chunk_index", "page_start", "page_end"):
        _require(_is_int(_get(chunk, key, where)), f"{where}.{key} is not an integer")
    _require(
        _get(chunk, "file_sha256", where) == file_sha256,
        f"{where}.file_sha256 is not the document's file.sha256",
    )
    text = _get(chunk, "text", where)
    _require(
        isinstance(text, str) and text != "", f"{where}.text is not a non-empty string"
    )
    origins = _get(chunk, "text_origins", where)
    _require(
        isinstance(origins, list) and all(isinstance(item, str) for item in origins),
        f"{where}.text_origins is not a list of strings",
    )
    _require(
        isinstance(_get(chunk, "page_spans", where), list),
        f"{where}.page_spans is not a list",
    )
    _require(
        isinstance(_get(chunk, "chunk_method", where), str),
        f"{where}.chunk_method is not a string",
    )
    for key in ("clause_label", "clause_number"):
        _require(
            _is_optional_str(_get(chunk, key, where)),
            f"{where}.{key} is not a string or null",
        )
    for key in ("part", "part_count"):
        value = _get(chunk, key, where)
        _require(
            value is None or _is_int(value), f"{where}.{key} is not an integer or null"
        )


# --- Reading and checking the input (step 2) ------------------------------------------


@dataclass(frozen=True)
class ChunkFile:
    """A valid chunk file."""

    relative: Path  # for messages, with the platform's separator
    chunk_file: str  # the forward-slash relative path (R10)
    sha256: str  # of the file's bytes
    record: Dict[str, Any]


@dataclass(frozen=True)
class IndexedChunk:
    """One chunk to index: its composed text and its payload (D4, D13)."""

    chunk_id: str
    composed: str
    payload: Dict[str, Any]


def _payload(record: Mapping[str, Any], chunk: Mapping[str, Any]) -> Dict[str, Any]:
    """The D13 payload: the chunk's own text, never the composed text."""
    return {
        "chunk_id": chunk["chunk_id"],
        "chunk_index": chunk["chunk_index"],
        "file_sha256": chunk["file_sha256"],
        "file_path": record["file"]["path"],
        "language": record["language"],
        "language_source": record["language_source"],
        "route": record["route"],
        "text": chunk["text"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "text_origins": chunk["text_origins"],
        "page_spans": chunk["page_spans"],
        "chunk_method": chunk["chunk_method"],
        "clause_label": chunk["clause_label"],
        "clause_number": chunk["clause_number"],
        "part": chunk["part"],
        "part_count": chunk["part_count"],
    }


def read_chunk_files(chunks_dir: Path) -> Tuple[List[ChunkFile], bool]:
    """Read and check every chunk file, logging each problem (R6, R7).
    Returns the valid files, in input order, and whether all passed."""
    paths = find_chunk_files(chunks_dir)
    passed = True
    if not paths:
        logger.error("No chunk files found in %s", chunks_dir)
        passed = False
    files: List[ChunkFile] = []
    for path in paths:
        relative = path.relative_to(chunks_dir)
        found: Optional[str] = None
        try:
            data = path.read_bytes()
            record = json.loads(data.decode("utf-8"))
            if not isinstance(record, dict):
                raise ChunkRecordError("the top level is not a JSON object")
            found = refused_schema_version(record)
            if found is None:
                validate_chunk_record(record)
        except (OSError, ValueError) as error:
            logger.error(
                "Could not read %s: %s: %s", relative, type(error).__name__, error
            )
            passed = False
            continue
        if found is not None:
            logger.error(
                "Refused %s: schema_version is %s, expected %d",
                relative,
                found,
                REQUIRED_CHUNKS_SCHEMA_VERSION,
            )
            passed = False
            continue
        files.append(
            ChunkFile(
                relative=relative,
                chunk_file=relative.as_posix(),
                sha256=hashlib.sha256(data).hexdigest(),
                record=record,
            )
        )
    return files, passed


def report_duplicates(files: Sequence[ChunkFile]) -> bool:
    """Log each later occurrence of a chunk id, naming the file of its first
    occurrence and its own (D12, R7). True when there is none."""
    first: Dict[str, Path] = {}
    unique = True
    for chunk_file in files:
        for chunk in chunk_file.record["chunks"]:
            chunk_id = chunk["chunk_id"]
            if chunk_id in first:
                logger.error(
                    "Duplicate chunk_id %s in %s and %s",
                    chunk_id,
                    first[chunk_id],
                    chunk_file.relative,
                )
                unique = False
            else:
                first[chunk_id] = chunk_file.relative
    return unique


def is_foreign_folder(index_dir: Path) -> bool:
    """Whether ``index_dir`` exists, is not empty and holds no
    ``manifest.json``: not an index, so it is not replaced (Q2)."""
    if not index_dir.exists() or (index_dir / MANIFEST_FILE_NAME).exists():
        return False
    if index_dir.is_dir():
        return any(index_dir.iterdir())
    return index_dir.stat().st_size > 0


# --- Building (steps 3-6) -------------------------------------------------------------


def _sibling(index_dir: Path, suffix: str) -> Path:
    return index_dir.with_name(index_dir.name + suffix)


def _rename(source: Path, target: Path) -> None:
    source.rename(target)


def _remove(path: Path) -> None:
    """Remove a folder with its contents, or a file; nothing if absent."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _swap(staging: Path, index_dir: Path, previous: Path) -> None:
    """Put the staging folder in place of ``index_dir`` (D14, R8)."""
    moved = False
    if index_dir.exists():
        _rename(index_dir, previous)
        moved = True
    try:
        _rename(staging, index_dir)
    except BaseException:
        if moved:
            _rename(previous, index_dir)
        raise
    if not moved:
        return
    try:
        _remove(previous)
    except OSError as error:
        logger.warning(
            "Could not remove the previous index %s: %s: %s",
            previous,
            type(error).__name__,
            error,
        )


@dataclass(frozen=True)
class IndexFigures:
    """What a written index holds, for the summary (R17)."""

    documents: int
    without_chunks: int
    chunks: int
    max_tokens: Optional[int]
    limit: int
    vocabulary: int
    collisions: int
    avgdl: Optional[float]


def log_summary(figures: IndexFigures) -> None:
    """The two INFO summary lines; ``avgdl`` and nulls as the manifest
    writes them (R17)."""
    logger.info(
        "Summary: documents=%d, without_chunks=%d, chunks=%d",
        figures.documents,
        figures.without_chunks,
        figures.chunks,
    )
    logger.info(
        "Summary: max_tokens=%s (limit %d), vocabulary=%d, term_id_collisions=%d, "
        "avgdl=%s",
        json.dumps(figures.max_tokens),
        figures.limit,
        figures.vocabulary,
        figures.collisions,
        json.dumps(figures.avgdl),
    )


def _chunks_of(files: Sequence[ChunkFile]) -> List[IndexedChunk]:
    return [
        IndexedChunk(
            chunk_id=chunk["chunk_id"],
            composed=compose_embedding_text(chunk),
            payload=_payload(chunk_file.record, chunk),
        )
        for chunk_file in files
        for chunk in chunk_file.record["chunks"]
    ]


def _count_tokens(
    chunks: Sequence[IndexedChunk], embedder: Embedder, limit: int
) -> Tuple[Optional[int], Optional[str], bool]:
    """D18: the largest count and its chunk (the first on a tie), and whether
    every composed text is within ``limit``; each one over it is logged."""
    largest: Optional[int] = None
    largest_id: Optional[str] = None
    within = True
    for chunk in chunks:
        count = embedder.count_tokens(chunk.composed)
        if largest is None or count > largest:
            largest, largest_id = count, chunk.chunk_id
        if count > limit:
            logger.error(
                "Embedding text of chunk %s has %d tokens, limit %d",
                chunk.chunk_id,
                count,
                limit,
            )
            within = False
    return largest, largest_id, within


def _build(
    files: Sequence[ChunkFile],
    index_dir: Path,
    embedder: Embedder,
    embeddings_config: EmbeddingsSettings,
    retrieval_config: RetrievalSettings,
) -> Optional[IndexFigures]:
    """Steps 3-5; the figures once the index is in place, else None."""
    chunks = _chunks_of(files)
    limit = min(embeddings_config.max_tokens, embedder.max_seq_length)
    largest, largest_id, within = _count_tokens(chunks, embedder, limit)
    if not within:
        return None

    terms = [sparse_terms(chunk.composed) for chunk in chunks]
    avgdl = sum(len(found) for found in terms) / len(terms) if terms else None
    k1, b = retrieval_config.bm25_k1, retrieval_config.bm25_b
    sparse = [bm25_vector(found, avgdl=avgdl, k1=k1, b=b) for found in terms]
    vocabulary = {term for found in terms for term in found}

    staging = _sibling(index_dir, STAGING_SUFFIX)
    previous = _sibling(index_dir, PREVIOUS_SUFFIX)
    _remove(staging)
    _remove(previous)  # left by a run whose final removal failed (Q1)
    index_dir.parent.mkdir(parents=True, exist_ok=True)

    client = open_client(staging)
    try:
        create_collection(client, embedder.dimension)
        dense = (
            embedder.encode_documents([chunk.composed for chunk in chunks])
            if chunks
            else []
        )
        if len(dense) != len(chunks):
            raise ValueError(
                f"the embedder returned {len(dense)} vectors for {len(chunks)} texts"
            )
        upload_points(
            client,
            (
                Point(
                    chunk_id=chunk.chunk_id,
                    dense=vector,
                    sparse_indices=values.indices,
                    sparse_values=values.values,
                    payload=chunk.payload,
                )
                for chunk, vector, values in zip(chunks, dense, sparse)
            ),
            _UPLOAD_BATCH_SIZE,
        )
        points = count_points(client)
    finally:
        client.close()
    if points != len(chunks):
        logger.error("Index has %d points, expected %d", points, len(chunks))
        return None

    documents = [
        {
            "chunk_file": chunk_file.chunk_file,
            "chunk_file_sha256": chunk_file.sha256,
            "file_sha256": chunk_file.record["file"]["sha256"],
            "status": chunk_file.record["status"],
            "chunk_count": chunk_file.record["chunk_count"],
        }
        for chunk_file in files
    ]
    collisions = term_id_collisions(vocabulary)
    manifest = build_manifest(
        model_name=embeddings_config.model_name,
        model_revision=embeddings_config.model_revision,
        dimension=embedder.dimension,
        max_tokens_limit=limit,
        max_tokens_measured=largest,
        max_tokens_chunk_id=largest_id,
        bm25_k1=k1,
        bm25_b=b,
        avgdl=avgdl,
        chunk_count=points,
        vocabulary_size=len(vocabulary),
        term_id_collisions=collisions,
        documents=documents,
    )
    write_manifest(staging, manifest)
    _swap(staging, index_dir, previous)
    return IndexFigures(
        documents=len(files),
        without_chunks=manifest["inputs"]["documents_without_chunks"],
        chunks=len(chunks),
        max_tokens=largest,
        limit=limit,
        vocabulary=len(vocabulary),
        collisions=collisions,
        avgdl=avgdl,
    )


def _load_embedder(config: EmbeddingsSettings) -> Embedder:
    """The index may download the model (D3)."""
    return SentenceTransformerEmbedder(config, local_files_only=False)


def build_index(
    chunks_dir: Path,
    index_dir: Path,
    embeddings_config: EmbeddingsSettings,
    retrieval_config: RetrievalSettings,
    load_embedder: EmbedderLoader = _load_embedder,
) -> bool:
    """Steps 2-6 of the command; True when the index is in place. Every
    problem is logged; the caller logs the run's last ERROR."""
    files, passed = read_chunk_files(chunks_dir)
    if not report_duplicates(files):
        passed = False
    if is_foreign_folder(index_dir):
        logger.error("Not an index, not replaced: %s", index_dir)
        passed = False
    if not passed:
        return False

    try:
        embedder = load_embedder(embeddings_config)
    except Exception as error:  # noqa: BLE001 - any load failure is Q3's
        logger.error(
            "Could not load embedding model %s@%s: %s: %s",
            embeddings_config.model_name,
            embeddings_config.model_revision,
            type(error).__name__,
            error,
        )
        return False

    try:
        figures = _build(
            files, index_dir, embedder, embeddings_config, retrieval_config
        )
    except Exception as error:  # noqa: BLE001 - any other error after the checks
        logger.error("Could not build the index: %s: %s", type(error).__name__, error)
        return False
    if figures is None:
        return False
    log_summary(figures)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companydocs-index",
        description=(
            "Build the search index from the chunk files: one Qdrant local "
            "collection with a dense and a BM25 sparse vector per chunk, and "
            "manifest.json. Every run rebuilds the whole index; the previous "
            "index is replaced only once the new one is complete."
        ),
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=default_chunks_dir(),
        help=(
            "Folder of the chunk files written by companydocs-chunk (default: "
            "data/interim/chunks under the project root -- git-ignored)"
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=default_index_dir(),
        help=(
            "Folder of the index (default: data/processed/index under the "
            "project root -- git-ignored)"
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only report warnings and errors",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    load_embedder: EmbedderLoader = _load_embedder,
) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.chunks_dir.is_dir():
        logger.error("Not a folder: %s", args.chunks_dir)
        return EXIT_CANNOT_START
    try:
        embeddings_config = embeddings_settings()
        retrieval_config = retrieval_settings()
    except SettingsError as error:
        logger.error("Invalid settings: %s", error)
        return EXIT_CANNOT_START

    if build_index(
        args.chunks_dir,
        args.index_dir,
        embeddings_config,
        retrieval_config,
        load_embedder,
    ):
        return EXIT_INDEX_WRITTEN
    logger.error("Index not written; the previous index is unchanged")
    return EXIT_INDEX_NOT_WRITTEN


if __name__ == "__main__":
    sys.exit(main())
