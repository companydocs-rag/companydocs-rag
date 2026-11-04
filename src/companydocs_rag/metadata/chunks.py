"""Reading the chunk files (spec 008, D6, R8, Q3).

Every chunk file under the input folder is found and ordered as
``companydocs-index`` finds them (:func:`.files.find_json_files`), then read
and checked before the first document is processed: UTF-8 JSON that is an
object; a ``schema_version`` that is the integer 1, else the file is refused
and its value shown as ``missing`` or as JSON; then the record, by the rules
of spec 006 R6 restated here (as ``companydocs-index`` checks a chunk file),
plus each ``page_spans`` entry being an object with an integer
``page_number`` and a string ``text_origin``. A file that cannot be read or
fails a check is refused too (Q3). Each refused file is logged as an ERROR.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from companydocs_rag.chunking.chunker import ROUTES
from companydocs_rag.ingestion.parse_pass import STATUS_FAILED, STATUS_PROCESSED

from .files import find_json_files, refused_schema_version

logger = logging.getLogger(__name__)

REQUIRED_CHUNKS_SCHEMA_VERSION = 1

_CHUNK_ID = re.compile("[0-9a-f]{20}")


class ChunkRecordError(ValueError):
    """A chunk file that is JSON but not a valid chunk record (R8)."""


@dataclass(frozen=True)
class ChunkFile:
    """A valid chunk file: its forward-slash path relative to the input
    folder, and its record."""

    relative: str
    record: Dict[str, Any]


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
    """Raise :class:`ChunkRecordError` unless ``record`` is a chunk record as
    ``companydocs-chunk`` writes it (R8). The schema version is checked before
    this."""
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
    spans = _get(chunk, "page_spans", where)
    _require(isinstance(spans, list), f"{where}.page_spans is not a list")
    for position, span in enumerate(spans):
        place = f"{where}.page_spans[{position}]"
        _require(isinstance(span, Mapping), f"{place} is not an object")
        _require(
            _is_int(_get(span, "page_number", place)),
            f"{place}.page_number is not an integer",
        )
        _require(
            isinstance(_get(span, "text_origin", place), str),
            f"{place}.text_origin is not a string",
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


def read_chunk_files(chunks_dir: Path) -> Tuple[List[ChunkFile], int]:
    """Read and check every chunk file, logging an ERROR per refused file
    (R8, Q3). Returns the valid files, in input order, and the number
    refused."""
    files: List[ChunkFile] = []
    refused = 0
    for path in find_json_files(chunks_dir):
        relative = path.relative_to(chunks_dir).as_posix()
        found = None
        try:
            record = json.loads(path.read_bytes().decode("utf-8"))
            if not isinstance(record, dict):
                raise ChunkRecordError("the top level is not a JSON object")
            found = refused_schema_version(record, REQUIRED_CHUNKS_SCHEMA_VERSION)
            if found is None:
                validate_chunk_record(record)
        except (OSError, ValueError) as error:
            logger.error(
                "Could not read %s: %s: %s", relative, type(error).__name__, error
            )
            refused += 1
            continue
        if found is not None:
            logger.error(
                "Refused %s: schema_version is %s, expected %d",
                relative,
                found,
                REQUIRED_CHUNKS_SCHEMA_VERSION,
            )
            refused += 1
            continue
        files.append(ChunkFile(relative=relative, record=record))
    return files, refused
