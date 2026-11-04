"""The ``companydocs-chunk`` command (spec 005): chunk the normalized files.

    companydocs-chunk [--normalized-dir DIR] [--chunks-dir DIR] [-q]

It reads every ``*.json`` file under ``--normalized-dir`` (default
``data/interim/normalized`` under the project root, where
``companydocs-normalize`` writes), recursively and sorted, skipping file names
that start with ".". Apart from the settings, it reads nothing else.

Per file: a normalized file whose ``schema_version`` is missing or not the
integer 1 is refused; one that cannot be read or parsed, is not a valid
normalized record, or fails while it is chunked or written, fails. Either way
nothing is written for it, it is logged as an ERROR, and the run goes on.
Every other file is chunked (:func:`.chunker.chunk_document`, with the
structure hint ``unknown``) and written under ``--chunks-dir`` (default
``data/interim/chunks``) at the same relative path, as normalize writes its
files: UTF-8, ``indent=2``, ``ensure_ascii=False``. A failed normalized
document is written too, without chunks. There is no cache: every run
recomputes everything, and a second run over the same input writes
byte-identical files.

Exit codes: 0 when every file was written; 1 when any file was refused or
failed; 2 when ``--normalized-dir`` is not a folder or the ``chunking``
settings are invalid, checked in that order, before any file is read. The run
ends with four INFO summary lines, which count the written documents.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from companydocs_rag.ingestion.normalize_cli import DEFAULT_NORMALIZED_RELATIVE_PATH
from companydocs_rag.ingestion.settings import find_project_root

from .chunker import (
    CHUNK_METHODS,
    REJECT_REASONS,
    REQUIRED_NORMALIZED_SCHEMA_VERSION,
    ROUTES,
    STRUCTURE_HINT_UNKNOWN,
    ChunkingConfigError,
    ChunkingSettings,
    NormalizedRecordError,
    chunk_document,
    chunking_settings,
)

logger = logging.getLogger(__name__)

# Where the chunk files land unless --chunks-dir overrides it: git-ignored,
# like the normalized files they are derived from.
DEFAULT_CHUNKS_RELATIVE_PATH = Path("data") / "interim" / "chunks"

EXIT_ALL_WRITTEN = 0
EXIT_FILES_NOT_WRITTEN = 1
EXIT_CANNOT_START = 2


def _project_base() -> Path:
    root = find_project_root()
    return root if root is not None else Path.cwd()


def default_normalized_dir() -> Path:
    """The folder ``companydocs-normalize`` writes by default, under the
    project root (falling back to the working directory)."""
    return _project_base() / DEFAULT_NORMALIZED_RELATIVE_PATH


def default_chunks_dir() -> Path:
    """The default output folder, resolved like :func:`default_normalized_dir`."""
    return _project_base() / DEFAULT_CHUNKS_RELATIVE_PATH


def find_normalized_files(folder: Path) -> List[Path]:
    """Every ``*.json`` file under ``folder``, recursively and sorted, except
    file names starting with ".", as normalize skips them."""
    return sorted(
        path
        for path in folder.rglob("*.json")
        if path.is_file() and not path.name.startswith(".")
    )


def refused_schema_version(record: Mapping[str, Any]) -> Optional[str]:
    """How a refused record's ``schema_version`` is shown, or None when it is
    the integer 1: ``missing`` when absent, else the value as JSON."""
    if "schema_version" not in record:
        return "missing"
    value = record["schema_version"]
    if type(value) is int and value == REQUIRED_NORMALIZED_SCHEMA_VERSION:
        return None
    return json.dumps(value)


def _write_json(path: Path, payload: Any) -> None:
    """As normalize writes its files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class RunTotals:
    """What one run did, for its summary; all but the first three counts
    cover the written documents only."""

    written: int = 0
    refused: int = 0
    failed: int = 0
    routes: Dict[str, int] = field(
        default_factory=lambda: {route: 0 for route in ROUTES}
    )
    runs: Dict[str, int] = field(
        default_factory=lambda: {
            "accepted": 0,
            **{reason: 0 for reason in REJECT_REASONS},
        }
    )
    chunks: Dict[str, int] = field(
        default_factory=lambda: {
            "total": 0,
            **{method: 0 for method in CHUNK_METHODS},
        }
    )
    exceeds_max_chars: int = 0

    def add(self, document: Mapping[str, Any]) -> None:
        """Count one written document."""
        self.written += 1
        self.routes[document["route"]] += 1
        for run in document["heading_evidence"]["runs"]:
            self.runs["accepted" if run["accepted"] else run["reject_reason"]] += 1
        for chunk in document["chunks"]:
            self.chunks["total"] += 1
            self.chunks[chunk["chunk_method"]] += 1
            if chunk["exceeds_max_chars"]:
                self.exceeds_max_chars += 1


def _pairs(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name}={value}" for name, value in counts.items())


def log_summary(totals: RunTotals) -> None:
    """The four INFO summary lines (R8)."""
    logger.info(
        "Summary: documents written=%d, refused=%d, failed=%d",
        totals.written,
        totals.refused,
        totals.failed,
    )
    logger.info("Summary: routes %s", _pairs(totals.routes))
    logger.info("Summary: runs %s", _pairs(totals.runs))
    logger.info(
        "Summary: chunks %s; exceeds_max_chars=%d",
        _pairs(totals.chunks),
        totals.exceeds_max_chars,
    )


def _log_failed(relative: Path, error: BaseException) -> None:
    logger.error("Could not chunk %s: %s: %s", relative, type(error).__name__, error)


def chunk_folder(
    normalized_dir: Path, chunks_dir: Path, config: ChunkingSettings
) -> RunTotals:
    """Chunk every normalized file under ``normalized_dir`` into
    ``chunks_dir``, log the end of the run and return its totals."""
    totals = RunTotals()
    files = find_normalized_files(normalized_dir)
    if not files:
        logger.warning("No normalized files found in %s", normalized_dir)

    for path in files:
        relative = path.relative_to(normalized_dir)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise NormalizedRecordError("the top level is not a JSON object")
        except (OSError, ValueError) as error:
            _log_failed(relative, error)
            totals.failed += 1
            continue
        found = refused_schema_version(record)
        if found is not None:
            logger.error(
                "Refused %s: schema_version is %s, expected %d",
                relative,
                found,
                REQUIRED_NORMALIZED_SCHEMA_VERSION,
            )
            totals.refused += 1
            continue
        try:
            document = chunk_document(
                record, config=config, structure_hint=STRUCTURE_HINT_UNKNOWN
            )
            _write_json(chunks_dir / relative, document)
        except Exception as error:  # noqa: BLE001 - one bad file must not stop the run
            _log_failed(relative, error)
            totals.failed += 1
            continue
        totals.add(document)

    if totals.refused:
        logger.error(
            "Refused %d normalized file(s) whose schema_version is not %d; "
            "rerun companydocs-normalize",
            totals.refused,
            REQUIRED_NORMALIZED_SCHEMA_VERSION,
        )
    if totals.failed:
        logger.error("%d normalized file(s) could not be chunked", totals.failed)
    log_summary(totals)
    return totals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companydocs-chunk",
        description=(
            "Chunk the normalized files: per document, its chunks with their "
            "page provenance, cut at the document's own article and clause "
            "numbering where an accepted numbering run is found and at line "
            "boundaries elsewhere. The text is never changed."
        ),
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=default_normalized_dir(),
        help=(
            "Folder of the normalized files written by companydocs-normalize "
            "(default: data/interim/normalized under the project root -- "
            "git-ignored)"
        ),
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=default_chunks_dir(),
        help=(
            "Folder to write one JSON file per normalized file, at the same "
            "relative path (default: data/interim/chunks under the project "
            "root -- git-ignored)"
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only report warnings and errors",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.normalized_dir.is_dir():
        logger.error("Not a folder: %s", args.normalized_dir)
        return EXIT_CANNOT_START
    try:
        config = chunking_settings()
    except ChunkingConfigError as error:
        logger.error("Invalid settings: %s", error)
        return EXIT_CANNOT_START

    totals = chunk_folder(args.normalized_dir, args.chunks_dir, config)
    if totals.refused or totals.failed:
        return EXIT_FILES_NOT_WRITTEN
    return EXIT_ALL_WRITTEN


if __name__ == "__main__":
    sys.exit(main())
