"""The document type (spec 008, D4, R6).

Order: the override file, then the client index, else none;
``document_type_source`` is ``override``, ``index`` or ``none``. The index
is looked up for every document, so ``index_type`` holds the index value
also when an override decides ``document_type``.

The override file (``Bestandsnaam;Documenttype``) is keyed by file name only.
A missing file is not an error (INFO). A row with an empty ``Bestandsnaam``
is ignored with a WARNING naming its line; a repeated name logs a WARNING
and the first row wins; an empty ``Documenttype`` is no entry, with a
WARNING, and resolution goes on to the index. A name that matches several
chunk files applies to each; one that matches none logs a WARNING.

The client index is the file named ``index_filename``
(``ingestion.parse_pass.language``, through
:func:`companydocs_rag.ingestion.language.language_settings`) in the first
folder above the PDF that holds one, walking up from the PDF's folder; a
relative ``file.path`` resolves against the project root. That file must hold
``Map``, ``Bestandsnaam`` and ``Documenttype``. If it cannot be read or lacks
one, no document under it has an index entry, one WARNING names the file, and
the walk does not go further up. A row matches on ``Map`` (split on "/" and
"\\", empty and "." parts dropped) plus ``Bestandsnaam``, against the PDF's
path relative to that folder; the first row per key wins. An empty
``Documenttype`` is no entry, with a WARNING. No such file in any ancestor:
a WARNING naming the document.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from .files import CsvFileError, file_name, path_components, read_csv_rows
from .settings import resolve_project_path

logger = logging.getLogger(__name__)

COLUMN_FOLDER = "Map"
COLUMN_FILENAME = "Bestandsnaam"
COLUMN_TYPE = "Documenttype"
OVERRIDE_COLUMNS = (COLUMN_FILENAME, COLUMN_TYPE)
INDEX_COLUMNS = (COLUMN_FOLDER, COLUMN_FILENAME, COLUMN_TYPE)

SOURCE_OVERRIDE = "override"
SOURCE_INDEX = "index"
SOURCE_NONE = "none"


class ManualFileError(Exception):
    """A user file that cannot be used: the run cannot start (exit 2).
    ``line`` is the offending row's line, when one row is the cause."""

    def __init__(self, path: Path, reason: str, *, line: Optional[int] = None) -> None:
        super().__init__(reason)
        self.path = path
        self.reason = reason
        self.line = line

    def describe(self) -> str:
        """``<path>: <reason>``, or ``<path>: line <n>: <reason>``."""
        if self.line is None:
            return f"{self.path}: {self.reason}"
        return f"{self.path}: line {self.line}: {self.reason}"


@dataclass(frozen=True)
class TypeOverrides:
    """The override file: file name -> the first row's ``Documenttype``,
    possibly empty. ``path`` is None when there is no file."""

    path: Optional[Path]
    entries: Dict[str, str]


def read_type_overrides(path: Path) -> TypeOverrides:
    """Read the override file at start. A missing file gives no entries
    (INFO); an unreadable one, or one that lacks a column, raises
    :class:`ManualFileError`."""
    if not path.exists():
        logger.info("No type overrides file at %s", path)
        return TypeOverrides(path=None, entries={})
    try:
        rows = read_csv_rows(path, OVERRIDE_COLUMNS)
    except CsvFileError as error:
        raise ManualFileError(path, str(error)) from error
    entries: Dict[str, str] = {}
    for row in rows:
        name = row.values[COLUMN_FILENAME]
        if name == "":
            logger.warning(
                "%s: line %d: empty %s; the row is ignored",
                path,
                row.line,
                COLUMN_FILENAME,
            )
            continue
        if name in entries:
            logger.warning(
                "%s lists %s more than once; the first row is used", path, name
            )
            continue
        entries[name] = row.values[COLUMN_TYPE]
    return TypeOverrides(path=path, entries=entries)


@dataclass(frozen=True)
class TypeResolution:
    """A document's ``index_type``, ``document_type`` and
    ``document_type_source`` (D4)."""

    index_type: Optional[str]
    document_type: Optional[str]
    source: str


def pdf_path(file_path: str) -> Path:
    """The PDF a ``file.path`` names: either separator read as a folder
    separator, a relative path under the project root, ``..`` resolved."""
    path = resolve_project_path(file_path.replace("\\", "/"))
    return Path(os.path.normpath(path))


class TypeResolver:
    """Resolves the type of each document of one run; each client index is
    read once, the first time a document under it is resolved."""

    def __init__(self, overrides: TypeOverrides, index_filename: str) -> None:
        self._overrides = overrides
        self._index_filename = index_filename
        self._indexes: Dict[Path, Optional[Dict[Tuple[str, ...], str]]] = {}

    def resolve(self, file_path: str) -> TypeResolution:
        """The type of the document whose ``file.path`` is ``file_path``."""
        name = file_name(file_path)
        index_type = self._index_type(file_path, name)
        override = self._overrides.entries.get(name)
        if override is not None:
            if override != "":
                return TypeResolution(index_type, override, SOURCE_OVERRIDE)
            logger.warning(
                "Empty %s for %s in %s", COLUMN_TYPE, name, self._overrides.path
            )
        if index_type is not None:
            return TypeResolution(index_type, index_type, SOURCE_INDEX)
        logger.warning("No document type for %s", file_path)
        return TypeResolution(None, None, SOURCE_NONE)

    def unmatched_overrides(self, names: Sequence[str]) -> None:
        """Log each override name that names none of ``names``."""
        present = set(names)
        for name in self._overrides.entries:
            if name not in present:
                logger.warning(
                    "%s: no chunk file is named %s", self._overrides.path, name
                )

    def _index_type(self, file_path: str, name: str) -> Optional[str]:
        pdf = pdf_path(file_path)
        for folder in pdf.parents:
            index_path = folder / self._index_filename
            if not index_path.is_file():
                continue
            entries = self._load_index(index_path)
            if entries is None:
                return None
            value = entries.get(pdf.relative_to(folder).parts)
            if value is None:
                return None
            if value == "":
                logger.warning("Empty %s for %s in %s", COLUMN_TYPE, name, index_path)
                return None
            return value
        logger.warning("No client index above %s", file_path)
        return None

    def _load_index(self, index_path: Path) -> Optional[Dict[Tuple[str, ...], str]]:
        if index_path in self._indexes:
            return self._indexes[index_path]
        entries: Optional[Dict[Tuple[str, ...], str]]
        try:
            rows = read_csv_rows(index_path, INDEX_COLUMNS)
        except CsvFileError as error:
            logger.warning("Client index %s is unreadable: %s", index_path, error)
            entries = None
        else:
            entries = {}
            for row in rows:
                key = (
                    *path_components(row.values[COLUMN_FOLDER]),
                    row.values[COLUMN_FILENAME],
                )
                entries.setdefault(key, row.values[COLUMN_TYPE])
        self._indexes[index_path] = entries
        return entries
