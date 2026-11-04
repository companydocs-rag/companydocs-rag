"""The ``companydocs-metadata`` command (spec 008, D1, D11, D12, R2).

    companydocs-metadata extract [--chunks-dir DIR] [--extracted-dir DIR]
                                [--type-overrides PATH] [--relations PATH]
    companydocs-metadata report --reference-date YYYY-MM-DD
                               [--extracted-dir DIR] [--report-dir DIR]

``extract`` (:mod:`.extract`) reads the chunk files (default
``data/interim/chunks`` under the project root, where ``companydocs-chunk``
writes), assigns each document its type from the override file (default
``data/manual/document_type_overrides.csv``) and the client index, reads the
relations file (default ``data/manual/relations.csv``), asks the local LLM
(Ollama by default, which has to run) for the date and term fields of each
document of the lease family, and writes one extraction file per chunk file
under ``--extracted-dir`` (default ``data/processed/metadata/extracted``) and
one JSONL log per run in the ``log_dir`` setting. Exit codes: 0 done, no
document ``partial`` or ``error``; 1 the chunk folder is missing or a chunk
file was refused; 2 the command cannot start; 3 the generator is
unreachable; 4 done with at least one ``partial`` or ``error`` document.

``report`` (:mod:`.report`) reads the extraction files and writes
``followup.md``, ``followup.json`` and ``followup.csv`` under
``--report-dir`` (default ``data/processed/metadata``) for the reference date
given; it reads no clock. Exit codes: 0 done; 1 the extraction folder is
missing or empty, or a file was refused; 2 bad arguments or settings.

Logging goes to stderr as ``LEVEL message``, at INFO; nothing goes to
stdout. ``extract``'s modules are imported only when it runs: they import
spec 007's generator client, which ``report`` does not need.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import httpx

from companydocs_rag.chunking.cli import DEFAULT_CHUNKS_RELATIVE_PATH

from .report import run_report
from .settings import resolve_project_path

logger = logging.getLogger(__name__)

# Where the command reads and writes unless a flag says otherwise: the user's
# files in data/manual/, derived files in data/processed/ (both git-ignored).
DEFAULT_EXTRACTED_RELATIVE_PATH = Path("data") / "processed" / "metadata" / "extracted"
DEFAULT_REPORT_RELATIVE_PATH = Path("data") / "processed" / "metadata"
DEFAULT_TYPE_OVERRIDES_RELATIVE_PATH = (
    Path("data") / "manual" / "document_type_overrides.csv"
)
DEFAULT_RELATIONS_RELATIVE_PATH = Path("data") / "manual" / "relations.csv"

_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def reference_date(text: str) -> date:
    """``--reference-date``: ``YYYY-MM-DD`` and a real date (R19)."""
    if _ISO_DATE.fullmatch(text) is None:
        raise argparse.ArgumentTypeError(f"not a date of the form YYYY-MM-DD: {text}")
    try:
        return date(int(text[0:4]), int(text[5:7]), int(text[8:10]))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a real date: {text}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companydocs-metadata",
        description=(
            "The date and term fields of the lease family: extract them with a "
            "local LLM, then report the follow-up dates for a reference date."
        ),
    )
    commands = parser.add_subparsers(
        dest="command", required=True, metavar="{extract,report}"
    )
    extract = commands.add_parser(
        "extract",
        help="Extract the fields of every chunked document of the lease family",
        description=(
            "Assign a document type to every chunk file, ask the local LLM "
            "(Ollama by default) for the date and term fields of each document "
            "of the lease family, check every quote against its chunk, and write "
            "one extraction file per chunk file and one JSONL log per run."
        ),
    )
    extract.add_argument(
        "--chunks-dir",
        type=Path,
        default=resolve_project_path(DEFAULT_CHUNKS_RELATIVE_PATH),
        help=(
            "Folder of the chunk files written by companydocs-chunk (default: "
            "data/interim/chunks under the project root -- git-ignored)"
        ),
    )
    extract.add_argument(
        "--extracted-dir",
        type=Path,
        default=resolve_project_path(DEFAULT_EXTRACTED_RELATIVE_PATH),
        help=(
            "Folder to write one extraction file per chunk file, at the same "
            "relative path (default: data/processed/metadata/extracted under "
            "the project root -- git-ignored)"
        ),
    )
    extract.add_argument(
        "--type-overrides",
        type=Path,
        default=resolve_project_path(DEFAULT_TYPE_OVERRIDES_RELATIVE_PATH),
        help=(
            "CSV file Bestandsnaam;Documenttype that overrides the client "
            "index (default: data/manual/document_type_overrides.csv; a missing "
            "file is not an error)"
        ),
    )
    extract.add_argument(
        "--relations",
        type=Path,
        default=resolve_project_path(DEFAULT_RELATIONS_RELATIVE_PATH),
        help=(
            "CSV file Bestandsnaam;Relatie;Doelbestand;Detail (default: "
            "data/manual/relations.csv; a missing file is not an error)"
        ),
    )
    report = commands.add_parser(
        "report",
        help="Write the follow-up overview for a reference date",
        description=(
            "Compute the expiry date, the next renewal date and the latest "
            "notice date of every extracted document for the reference date, "
            "and write followup.md, followup.json and followup.csv."
        ),
    )
    report.add_argument(
        "--reference-date",
        type=reference_date,
        required=True,
        metavar="YYYY-MM-DD",
        help="The date the follow-up dates are computed for",
    )
    report.add_argument(
        "--extracted-dir",
        type=Path,
        default=resolve_project_path(DEFAULT_EXTRACTED_RELATIVE_PATH),
        help=(
            "Folder of the extraction files written by extract (default: "
            "data/processed/metadata/extracted under the project root)"
        ),
    )
    report.add_argument(
        "--report-dir",
        type=Path,
        default=resolve_project_path(DEFAULT_REPORT_RELATIVE_PATH),
        help=(
            "Folder to write the three follow-up files (default: "
            "data/processed/metadata under the project root -- git-ignored)"
        ),
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    transport: Optional[httpx.BaseTransport] = None,
) -> int:
    """The command; ``transport`` goes to the httpx client (tests pass an
    ``httpx.MockTransport``)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # httpx logs every request at INFO; R2's one line per request is this
    # command's own.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.command == "extract":
        # Imported here: the extraction modules import spec 007's generator
        # client, and with it qdrant_client, which report never needs (Q1).
        from .extract import run_extract

        return run_extract(
            args.chunks_dir,
            args.extracted_dir,
            args.type_overrides,
            args.relations,
            transport=transport,
        )
    return run_report(args.extracted_dir, args.report_dir, args.reference_date)


if __name__ == "__main__":
    sys.exit(main())
