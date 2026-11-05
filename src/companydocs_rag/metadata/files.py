"""Files both subcommands read or write (spec 008, R6, R8, R16, R19).

JSON input folders are read as ``companydocs-index`` reads the chunk files
(spec 006 R10): every ``*.json`` file under the folder, recursively, file
names starting with "." skipped, in the order of their forward-slash relative
paths, which also name the files in messages. JSON output is written as
``companydocs-chunk`` writes its files: UTF-8, ``indent=2``,
``ensure_ascii=False``, folders created, an existing file replaced.

Every output file, the extraction files and the follow-up files, is written
to a new temporary file in its folder and then moved over the old one with
``os.replace`` (:func:`replace_file`; review F2), so it holds either its old
content or the new, never a part. A temporary file is named "." plus 16 hex
digits plus ".tmp": one that an interrupted process leaves behind starts
with "." and does not end in ".json", so :func:`find_json_files` never lists
it.

The user's CSV files and the client index are read as
:mod:`companydocs_rag.ingestion.language` reads its files: ``;``-separated,
UTF-8 with or without a byte order mark, a header row (line 1). Values are
stripped.

A document's file name is the last component of its ``file.path``, split on
"/" and "\\" on every platform (R6).
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

CSV_DELIMITER = ";"
CSV_ENCODING = "utf-8-sig"  # UTF-8 with BOM; also reads a file without one
TEMPORARY_PREFIX = "."
TEMPORARY_SUFFIX = ".tmp"

_SEPARATORS = re.compile(r"[\\/]")


def find_json_files(folder: Path) -> List[Path]:
    """Every ``*.json`` file under ``folder``, recursively, except file names
    starting with ".", in the order of their forward-slash relative paths."""
    return sorted(
        (
            path
            for path in folder.rglob("*.json")
            if path.is_file() and not path.name.startswith(".")
        ),
        key=lambda path: path.relative_to(folder).as_posix(),
    )


def file_name(path_value: str) -> str:
    """The last component of ``path_value``, split on "/" and "\\" (R6)."""
    return _SEPARATORS.split(path_value)[-1]


def path_components(value: str) -> List[str]:
    """``value`` split on "/" and "\\", empty and "." parts dropped (R6)."""
    return [part for part in _SEPARATORS.split(value) if part not in ("", ".")]


def refused_schema_version(record: Mapping[str, Any], expected: int) -> Optional[str]:
    """How a refused record's ``schema_version`` is shown, or None when it is
    the integer ``expected``: ``missing`` when absent, else the value as JSON
    (as spec 005 shows it)."""
    if "schema_version" not in record:
        return "missing"
    value = record["schema_version"]
    if type(value) is int and value == expected:
        return None
    return json.dumps(value)


def _new_temporary(folder: Path, *, binary: bool) -> Tuple[Path, IO[Any]]:
    """A temporary file in ``folder`` that did not exist, opened for writing:
    as bytes, or as text in UTF-8 with the platform's line ends."""
    while True:
        name = TEMPORARY_PREFIX + secrets.token_hex(8) + TEMPORARY_SUFFIX
        temporary = folder / name
        try:
            if binary:
                return temporary, temporary.open("xb")
            return temporary, temporary.open("x", encoding="utf-8")
        except FileExistsError:
            continue


def replace_file(path: Path, content: Union[str, bytes]) -> None:
    """Write ``content`` to ``path``, its folder created as needed: bytes as
    they are, a str as ``Path.write_text(content, encoding="utf-8")`` writes
    it. The content goes to a temporary file in the same folder first, which
    ``os.replace`` then moves over ``path``. When the write or the move
    fails, the temporary file is removed, the error raised, and ``path``
    keeps what it held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary, handle = _new_temporary(path.parent, binary=isinstance(content, bytes))
    try:
        with handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as ``companydocs-chunk`` writes its files (R16),
    through :func:`replace_file`."""
    replace_file(path, json.dumps(payload, indent=2, ensure_ascii=False))


class CsvFileError(Exception):
    """A CSV file that cannot be read, or lacks a required column."""


@dataclass(frozen=True)
class CsvRow:
    """One data row: its line in the file (the header is line 1) and its
    values, stripped, by column."""

    line: int
    values: Dict[str, str]


def read_csv_rows(path: Path, required_columns: Sequence[str]) -> List[CsvRow]:
    """The data rows of one ``;``-separated CSV file with a header row.
    Raises :class:`CsvFileError` when it cannot be read or decoded, or lacks
    one of ``required_columns``."""
    try:
        with path.open("r", encoding=CSV_ENCODING, newline="") as handle:
            reader = csv.DictReader(handle, delimiter=CSV_DELIMITER)
            fieldnames = list(reader.fieldnames or [])
            missing = [
                column for column in required_columns if column not in fieldnames
            ]
            if missing:
                raise CsvFileError("missing column(s) " + ", ".join(missing))
            rows = []
            for row in reader:
                # A row longer than the header keeps its extra values under the
                # key None; they belong to no column.
                values = {
                    key: (value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
                rows.append(CsvRow(line=reader.line_num, values=values))
            return rows
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CsvFileError(f"{type(error).__name__}: {error}") from error
