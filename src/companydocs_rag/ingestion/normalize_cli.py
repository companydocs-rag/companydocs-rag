"""The ``companydocs-normalize`` command (spec 004): normalize the page text store.

    companydocs-normalize [--page-text-dir DIR] [--normalized-dir DIR] [-q]

It reads every ``*.json`` file under ``--page-text-dir`` (default
``data/interim/page_text`` under the project root, where the parse pass writes
its page text store), recursively and sorted, skipping file names that start
with ".". Apart from the settings, it reads nothing else.

Per file: a store file whose ``schema_version`` is missing or not the integer
2 is refused; one that cannot be read or parsed, or is not a valid store
record, fails. Either way nothing is written for it, it is logged as an ERROR,
and the run goes on. Every other file is normalized
(:func:`.normalization.normalize_document`) and written under
``--normalized-dir`` (default ``data/interim/normalized``) at the same relative
path, as the store is written: UTF-8, ``indent=2``, ``ensure_ascii=False``. A
failed store document is written without pages. There is no cache: every run
recomputes everything, and a second run over the same store writes
byte-identical files.

Exit codes: 0 when every file was written; 1 when any file was refused or
failed; 2 when ``--page-text-dir`` is not a folder, or when
``ingestion.normalization`` lacks the label families or the conjunction lists
or holds a family regex that does not compile. Both are checked in that
order, before any file is read. The run ends with five INFO summary lines.
Hard normalization is never called here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableSet, Optional, Sequence

from .cli import DEFAULT_PAGE_TEXT_RELATIVE_PATH
from .normalization import (
    DEHYPHENATION_COUNTERS,
    OPERATION_COUNTERS,
    REASON_PAGE_LABEL,
    REASON_PAGE_NUMBER_RUN,
    REQUIRED_PAGE_TEXT_SCHEMA_VERSION,
    TEXT_ORIGINS,
    NormalizationConfigError,
    NormalizationSettings,
    StoreRecordError,
    normalization_settings,
    normalize_document,
)
from .settings import find_project_root

logger = logging.getLogger(__name__)

# Where the normalized files land unless --normalized-dir overrides it:
# git-ignored, like the page text store it is derived from.
DEFAULT_NORMALIZED_RELATIVE_PATH = Path("data") / "interim" / "normalized"

EXIT_ALL_WRITTEN = 0
EXIT_FILES_NOT_WRITTEN = 1
EXIT_CANNOT_START = 2


def _project_base() -> Path:
    root = find_project_root()
    return root if root is not None else Path.cwd()


def default_page_text_dir() -> Path:
    """The page text store the parse pass writes by default, under the
    project root (falling back to the working directory)."""
    return _project_base() / DEFAULT_PAGE_TEXT_RELATIVE_PATH


def default_normalized_dir() -> Path:
    """The default output folder, resolved like :func:`default_page_text_dir`."""
    return _project_base() / DEFAULT_NORMALIZED_RELATIVE_PATH


def find_store_files(folder: Path) -> List[Path]:
    """Every ``*.json`` file under ``folder``, recursively and sorted, except
    file names starting with ".", as the parse pass skips them."""
    return sorted(
        path
        for path in folder.rglob("*.json")
        if path.is_file() and not path.name.startswith(".")
    )


def refused_schema_version(record: Mapping[str, Any]) -> Optional[str]:
    """How a refused record's ``schema_version`` is shown, or None when it is
    the integer 2: ``missing`` when absent, else the value as JSON."""
    if "schema_version" not in record:
        return "missing"
    value = record["schema_version"]
    if type(value) is int and value == REQUIRED_PAGE_TEXT_SCHEMA_VERSION:
        return None
    return json.dumps(value)


def _write_json(path: Path, payload: Any) -> None:
    """As the parse pass writes the page text store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class RunTotals:
    """What one run did, for its summary."""

    families: Sequence[str]
    written: int = 0
    refused: int = 0
    failed: int = 0
    pages: Dict[str, int] = field(
        default_factory=lambda: {"total": 0, **{origin: 0 for origin in TEXT_ORIGINS}}
    )
    removed_by_reason: Dict[str, int] = field(
        default_factory=lambda: {REASON_PAGE_LABEL: 0, REASON_PAGE_NUMBER_RUN: 0}
    )
    removed_by_family: Dict[str, int] = field(default_factory=dict)
    dehyphenation: Dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in DEHYPHENATION_COUNTERS}
    )
    operations: Dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in OPERATION_COUNTERS}
    )

    def __post_init__(self) -> None:
        self.removed_by_family = {name: 0 for name in self.families}

    def add(self, document: Mapping[str, Any]) -> None:
        """Count one written document."""
        self.written += 1
        for page in document["pages"]:
            self.pages["total"] += 1
            self.pages[page["text_origin"]] += 1
            for removed in page["removed_lines"]:
                self.removed_by_reason[removed["reason"]] += 1
                if removed["family"] is not None:
                    self.removed_by_family[removed["family"]] += 1
            counts = page["counts"]
            for name in DEHYPHENATION_COUNTERS:
                self.dehyphenation[name] += counts["dehyphenation"][name]
            for name in OPERATION_COUNTERS:
                self.operations[name] += int(counts[name])


def _pairs(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name}={value}" for name, value in counts.items())


def log_summary(totals: RunTotals) -> None:
    """The five INFO summary lines."""
    logger.info(
        "Summary: documents written=%d, refused=%d, failed=%d",
        totals.written,
        totals.refused,
        totals.failed,
    )
    logger.info("Summary: pages %s", _pairs(totals.pages))
    logger.info(
        "Summary: removed lines %s; by family %s",
        _pairs(totals.removed_by_reason),
        _pairs(totals.removed_by_family),
    )
    logger.info("Summary: dehyphenation %s", _pairs(totals.dehyphenation))
    logger.info("Summary: operations %s", _pairs(totals.operations))


def _log_failed(relative: Path, error: BaseException) -> None:
    logger.error(
        "Could not normalize %s: %s: %s", relative, type(error).__name__, error
    )


def normalize_folder(
    page_text_dir: Path, normalized_dir: Path, config: NormalizationSettings
) -> RunTotals:
    """Normalize every store file under ``page_text_dir`` into
    ``normalized_dir``, log the end of the run and return its totals."""
    totals = RunTotals(families=[name for name, _ in config.families])
    files = find_store_files(page_text_dir)
    if not files:
        logger.warning("No page text files found in %s", page_text_dir)
    warned_languages: MutableSet[Optional[str]] = set()

    for path in files:
        relative = path.relative_to(page_text_dir)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise StoreRecordError("the top level is not a JSON object")
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
                REQUIRED_PAGE_TEXT_SCHEMA_VERSION,
            )
            totals.refused += 1
            continue
        try:
            document = normalize_document(
                record, config=config, warned_languages=warned_languages
            )
            _write_json(normalized_dir / relative, document)
        except Exception as error:  # noqa: BLE001 - one bad file must not stop the run
            _log_failed(relative, error)
            totals.failed += 1
            continue
        totals.add(document)

    if totals.refused:
        logger.error(
            "Refused %d page text file(s) whose schema_version is not %d; "
            "rerun companydocs-parse-pass",
            totals.refused,
            REQUIRED_PAGE_TEXT_SCHEMA_VERSION,
        )
    if totals.failed:
        logger.error("%d page text file(s) could not be normalized", totals.failed)
    log_summary(totals)
    return totals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companydocs-normalize",
        description=(
            "Normalize the page text store: per document, the soft-normalized "
            "text of every page, with every removed line and every join "
            "recorded."
        ),
    )
    parser.add_argument(
        "--page-text-dir",
        type=Path,
        default=default_page_text_dir(),
        help=(
            "Folder of the page text store written by companydocs-parse-pass "
            "(default: data/interim/page_text under the project root -- "
            "git-ignored)"
        ),
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=default_normalized_dir(),
        help=(
            "Folder to write one JSON file per store file, at the same relative "
            "path (default: data/interim/normalized under the project root -- "
            "git-ignored)"
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

    if not args.page_text_dir.is_dir():
        logger.error("Not a folder: %s", args.page_text_dir)
        return EXIT_CANNOT_START
    try:
        config = normalization_settings()
    except NormalizationConfigError as error:
        logger.error("Invalid settings: %s", error)
        return EXIT_CANNOT_START

    totals = normalize_folder(args.page_text_dir, args.normalized_dir, config)
    if totals.refused or totals.failed:
        return EXIT_FILES_NOT_WRITTEN
    return EXIT_ALL_WRITTEN


if __name__ == "__main__":
    sys.exit(main())
