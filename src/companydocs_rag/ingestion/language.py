"""Document language for OCR: override file, client index, default (spec 003).

One language per document. There is no language detection. For each
processed document, the first source that yields a mapped code wins:

1. the override file (by default ``data/manual/language_overrides.csv``, see
   ``cli.py``), keyed by file name;
2. the client index (``index_filename``, ``00_INDEX.csv``) in the PDF's
   nearest ancestor folder, matched on the PDF's path relative to that folder;
3. ``default_language`` (``nld``), with a WARNING naming the document.

Both files follow the index's conventions: ``;``-separated, UTF-8 with BOM, a
header row. Their codes are the client's two-letter codes (``NL``, ``SK``,
...): stripped, uppercased and mapped through ``language_map`` to the
Tesseract code, which is what is recorded and what Tesseract gets. An empty or
unmapped code counts as no entry and logs a WARNING naming the code.

The override file's keys name client files, so it lives under ``data/``
outside ``data/raw/``, never in ``config/`` or any other tracked path.
"""

from __future__ import annotations

import csv
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .settings import parse_pass_section

logger = logging.getLogger(__name__)

# --- Settings defaults (ingestion.parse_pass.language) ----------------------
DEFAULT_LANGUAGE = "nld"
INDEX_FILENAME = "00_INDEX.csv"
# The client's two-letter codes -> Tesseract codes (ISO 639-2/T for these
# seven languages).
LANGUAGE_MAP: Dict[str, str] = {
    "NL": "nld",
    "EN": "eng",
    "DE": "deu",
    "FR": "fra",
    "PL": "pol",
    "SK": "slk",
    "CS": "ces",
}

# --- File conventions (Python constants, not settings) ----------------------
CSV_DELIMITER = ";"
CSV_ENCODING = "utf-8-sig"  # UTF-8 with BOM; also reads a file without one
COLUMN_FOLDER = "Map"  # index only: folder relative to the index folder
COLUMN_FILENAME = "Bestandsnaam"
COLUMN_LANGUAGE = "Taal"

INDEX_COLUMNS = (COLUMN_FOLDER, COLUMN_FILENAME, COLUMN_LANGUAGE)
OVERRIDE_COLUMNS = (COLUMN_FILENAME, COLUMN_LANGUAGE)

# --- Recorded values ---------------------------------------------------------
LANGUAGE_SOURCE_OVERRIDE = "override"
LANGUAGE_SOURCE_INDEX = "index"
LANGUAGE_SOURCE_DEFAULT = "default"

_FOLDER_SEPARATORS = re.compile(r"[\\/]")

PathLike = Union[str, Path]


def language_settings(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Resolve the language settings at call time, YAML overriding the defaults."""
    section = parse_pass_section("language", settings=settings)
    configured_map = section.get("language_map")
    mapping = configured_map if isinstance(configured_map, Mapping) else LANGUAGE_MAP
    return {
        "default_language": str(section.get("default_language", DEFAULT_LANGUAGE)),
        "index_filename": str(section.get("index_filename", INDEX_FILENAME)),
        "language_map": {
            str(key).strip().upper(): str(value) for key, value in mapping.items()
        },
    }


class UnreadableFileError(Exception):
    """An index or override file that cannot be read, or lacks a column."""


def _read_rows(path: Path, required_columns: Sequence[str]) -> List[Dict[str, str]]:
    """Read one ``;``-separated, UTF-8-with-BOM file with a header row."""
    try:
        with path.open("r", encoding=CSV_ENCODING, newline="") as handle:
            reader = csv.DictReader(handle, delimiter=CSV_DELIMITER)
            fieldnames = list(reader.fieldnames or [])
            missing = [
                column for column in required_columns if column not in fieldnames
            ]
            if missing:
                raise UnreadableFileError(f"missing column(s) {missing}")
            return [
                {key: (value or "") for key, value in row.items() if key is not None}
                for row in reader
            ]
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UnreadableFileError(f"{type(error).__name__}: {error}") from error


def _path_components(folder: str, filename: str) -> Tuple[str, ...]:
    """An index row's ``Map`` and ``Bestandsnaam`` as path components."""
    parts = [part for part in _FOLDER_SEPARATORS.split(folder) if part not in ("", ".")]
    return (*parts, filename)


def _map_code(raw: str, language_map: Mapping[str, str], what: str) -> Optional[str]:
    """Map a client code to a Tesseract code; None (with a WARNING) if it cannot."""
    code = raw.strip().upper()
    mapped = language_map.get(code) if code else None
    if mapped is None:
        logger.warning(
            "%s: language code %r is empty or not in language_map; ignored",
            what,
            raw,
        )
    return mapped


def _load_overrides(path: Path) -> Dict[str, str]:
    """Read the override file: file name -> raw language code."""
    if not path.exists():
        logger.info("No language override file at %s", path)
        return {}
    try:
        rows = _read_rows(path, OVERRIDE_COLUMNS)
    except UnreadableFileError as error:
        logger.error(
            "Language override file %s is unreadable (%s); no overrides applied",
            path,
            error,
        )
        return {}

    overrides: Dict[str, str] = {}
    for row in rows:
        filename = row.get(COLUMN_FILENAME, "")
        if filename in overrides:
            logger.warning(
                "Language override file %s lists %r more than once; the first "
                "row is used",
                path,
                filename,
            )
            continue
        overrides[filename] = row.get(COLUMN_LANGUAGE, "")
    logger.info("Read %d language override(s) from %s", len(overrides), path)
    return overrides


class LanguageResolver:
    """Run-level language resolution: the override file and a per-run index cache.

    One instance per run. The override file is read once, when the instance
    is created; each client index is read once, the first time a PDF under it
    is resolved. :meth:`report_end_of_run` logs the override entries that
    matched no processed PDF, and those that matched more than one.
    """

    def __init__(self, overrides_path: Optional[PathLike] = None) -> None:
        self._overrides: Dict[str, str] = {}
        if overrides_path is not None:
            self._overrides = _load_overrides(Path(overrides_path))
        self._override_matches: Dict[str, List[str]] = {
            filename: [] for filename in self._overrides
        }
        self._indexes: Dict[Path, Optional[Dict[Tuple[str, ...], str]]] = {}

    def resolve(
        self,
        pdf_path: PathLike,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[str, str]:
        """Return ``(language, language_source)`` for one processed PDF."""
        config = language_settings(settings)
        language_map = config["language_map"]
        path = Path(os.path.abspath(pdf_path))

        if path.name in self._overrides:
            self._override_matches[path.name].append(str(pdf_path))
            code = _map_code(
                self._overrides[path.name],
                language_map,
                f"language override for {path.name}",
            )
            if code is not None:
                return code, LANGUAGE_SOURCE_OVERRIDE

        raw = self._index_entry(path, config["index_filename"])
        if raw is not None:
            code = _map_code(raw, language_map, f"client index entry for {pdf_path}")
            if code is not None:
                return code, LANGUAGE_SOURCE_INDEX

        default = config["default_language"]
        logger.warning(
            "No language for %s from the override file or a client index; "
            "using the default %r",
            pdf_path,
            default,
        )
        return default, LANGUAGE_SOURCE_DEFAULT

    def _index_entry(self, path: Path, index_filename: str) -> Optional[str]:
        """The raw ``Taal`` value for this PDF in its nearest index, if any."""
        for folder in path.parents:
            index_path = folder / index_filename
            if not index_path.is_file():
                continue
            entries = self._load_index(index_path)
            if entries is None:
                # Unreadable: treated as absent, so the lookup carries on
                # upwards.
                continue
            return entries.get(path.relative_to(folder).parts)
        return None

    def _load_index(self, index_path: Path) -> Optional[Dict[Tuple[str, ...], str]]:
        if index_path in self._indexes:
            return self._indexes[index_path]
        entries: Optional[Dict[Tuple[str, ...], str]]
        try:
            rows = _read_rows(index_path, INDEX_COLUMNS)
        except UnreadableFileError as error:
            logger.error(
                "Client index %s is unreadable (%s); treated as absent",
                index_path,
                error,
            )
            entries = None
        else:
            entries = {}
            for row in rows:
                key = _path_components(
                    row.get(COLUMN_FOLDER, ""), row.get(COLUMN_FILENAME, "")
                )
                entries.setdefault(key, row.get(COLUMN_LANGUAGE, ""))
        self._indexes[index_path] = entries
        return entries

    def report_end_of_run(self) -> None:
        """Log override entries that matched no processed PDF, or several."""
        unmatched = [
            filename for filename, paths in self._override_matches.items() if not paths
        ]
        if unmatched:
            logger.warning(
                "Language override entries that matched no processed PDF: %s",
                ", ".join(repr(filename) for filename in unmatched),
            )
        for filename, paths in self._override_matches.items():
            if len(paths) > 1:
                logger.warning(
                    "Language override for %r matched %d PDFs: %s",
                    filename,
                    len(paths),
                    ", ".join(paths),
                )
