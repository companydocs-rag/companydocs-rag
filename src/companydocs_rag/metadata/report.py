"""The ``report`` subcommand (spec 008, D12, D13, R19-R25).

In this order (R19): the ``metadata`` settings (exit 2); ``--extracted-dir``,
which must be a folder holding at least one extraction file, each a JSON
object with ``schema_version`` 1, D10's keys and the types the report reads
(else an ERROR per file, exit 1, nothing written); then, per document whose
``extraction_status`` is not ``not_in_scope``, ordered by ``file.path`` and
then by the extraction file's relative path, the chosen values, E0, the three
derived dates (:mod:`.dates`) and the texts (:mod:`.texts`); then
``followup.md``, ``followup.json`` and ``followup.csv`` in ``--report-dir``,
created as needed, each replaced through a temporary file
(:func:`.files.replace_file`).

Date arithmetic that leaves years 1-9999 ends nothing (I1): the entry it
affects is ``not_computable``, the report goes on, and each range failure is
one WARNING ``<relative path>: <entry>: date out of range; not computed``
(:func:`.dates.range_failures`); the exit code is unaffected.

The report reads no clock: its only date is ``--reference-date``, and two
runs over the same input with the same date write byte-identical files.
``amendment_types`` is read from today's settings, not from the extract run.

Pages (R21): a field's pages are the distinct page pairs of its counted
candidates. Provisions (R23): the verified items, deduplicated by their
hard-normalized quote, each with the pages of it and its duplicates. The
notes of a document, in order: the term-end mismatch, the relations (R24),
the incomplete extraction, the scan (R25: every page span of the counted
candidates and of the verified provision items has ``text_origin`` ``ocr``).
"""

from __future__ import annotations

import copy
import csv
import io
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from companydocs_rag.embeddings.dense import SettingsError
from companydocs_rag.ingestion.normalization import hard_normalize
from companydocs_rag.ingestion.quality import TEXT_ORIGIN_OCR

from .dates import FieldView, TermEnd, derive, effective_term_end, range_failures
from .fields import (
    BATCH_ERROR,
    BATCH_STATUSES,
    DURATION_KEYS,
    DURATION_KINDS,
    EXTRACTION_ERROR,
    EXTRACTION_NOT_IN_SCOPE,
    EXTRACTION_PARTIAL,
    EXTRACTION_STATUSES,
    FIELD_STATUSES,
    FIELDS,
    INITIAL_TERM_END,
    PARTIES,
    PROVISIONS,
    STATUS_FOUND,
    STATUS_NOT_FOUND,
    UNITS,
    counted_candidates,
    distinct_pairs,
    page_pair,
    value_fits,
)
from .files import (
    file_name,
    find_json_files,
    refused_schema_version,
    replace_file,
    write_json,
)
from .relations import (
    RELATION_AMENDS,
    RELATION_DUPLICATE_OF,
    RELATIONS,
    STATUS_RESOLVED,
    STATUS_TARGET_NOT_FOUND,
)
from .settings import metadata_settings
from .texts import (
    COLUMNS,
    DERIVED,
    DERIVED_COMPUTED,
    DERIVED_LABELS,
    FIELD_LABELS,
    PARTY_NAMES,
    PROVISIONS_HEADING,
    SCAN_NOTE,
    amended_by_note,
    amends_note,
    derived_cell,
    duplicate_note,
    field_cell,
    footer,
    incomplete_note,
    mismatch_note,
    pages_text,
    value_text,
)

logger = logging.getLogger(__name__)

EXIT_DONE = 0
EXIT_INPUT_REFUSED = 1
EXIT_CANNOT_START = 2

REQUIRED_EXTRACTION_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
MARKDOWN_NAME = "followup.md"
JSON_NAME = "followup.json"
CSV_NAME = "followup.csv"  # [CUT-1]
CSV_DELIMITER = ";"
CSV_ENCODING = "utf-8-sig"

# D10: the extraction file's top-level keys.
EXTRACTION_KEYS = (
    "schema_version",
    "extractor_version",
    "file",
    "language",
    "language_source",
    "index_type",
    "document_type",
    "document_type_source",
    "relations",
    "extraction_status",
    "model",
    "reasoning_effort",
    "batches",
    "fields",
    PROVISIONS,
)
_RELATION_STATUSES = (STATUS_RESOLVED, STATUS_TARGET_NOT_FOUND)
_PIPE = "|"
_ESCAPED_PIPE = "\\|"
_LINE_FEED = "\n"


# --- Reading the extraction files (R19) -----------------------------------------------


class ExtractionRecordError(ValueError):
    """An extraction file that is JSON but not one the report can read."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExtractionRecordError(message)


def _is_optional_str(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _check_spans(spans: Any, where: str) -> None:
    _require(isinstance(spans, list), f"{where}.page_spans is not a list")
    for position, span in enumerate(spans):
        _require(
            isinstance(span, Mapping)
            and type(span.get("page_number")) is int
            and isinstance(span.get("text_origin"), str),
            f"{where}.page_spans[{position}] is not an object with an integer "
            "page_number and a string text_origin",
        )


def _check_object(item: Any, keys: Sequence[str], where: str) -> None:
    _require(isinstance(item, Mapping), f"{where} is not an object")
    for key in keys:
        _require(key in item, f"{where} lacks the key {key!r}")


def _stored_value_fits(field: str, kind: str, value: Any) -> bool:
    """A counted candidate's value as ``extract`` stores it (R12)."""
    if not value_fits(field, kind, value):
        return False
    if kind in DURATION_KINDS:
        return (
            set(value) == set(DURATION_KEYS)
            and type(value["amount"]) is int
            and value["unit"] in UNITS
        )
    return True


def _check_fields(fields: Any) -> None:
    _check_object(fields, FIELDS, "'fields'")
    for field in FIELDS:
        where = f"fields.{field}"
        entry = fields[field]
        _check_object(entry, ("status", "candidates"), where)
        _require(
            entry["status"] in FIELD_STATUSES,
            f"{where}.status is not one of " + ", ".join(FIELD_STATUSES),
        )
        candidates = entry["candidates"]
        _require(isinstance(candidates, list), f"{where}.candidates is not a list")
        for position, candidate in enumerate(candidates):
            place = f"{where}.candidates[{position}]"
            _check_object(
                candidate,
                ("value", "value_kind", "quote", "counted", "page_spans"),
                place,
            )
            _require(
                isinstance(candidate["value_kind"], str),
                f"{place}.value_kind is not a string",
            )
            _require(
                isinstance(candidate["quote"], str), f"{place}.quote is not a string"
            )
            _require(
                isinstance(candidate["counted"], bool), f"{place}.counted is not a bool"
            )
            _check_spans(candidate["page_spans"], place)
            if candidate["counted"]:
                _require(
                    _stored_value_fits(
                        field, candidate["value_kind"], candidate["value"]
                    ),
                    f"{place} is counted but its value does not fit its value_kind",
                )
        if entry["status"] != STATUS_NOT_FOUND:
            _require(
                bool(counted_candidates(candidates)),
                f"{where}.status is {entry['status']} without a counted candidate",
            )


def validate_extraction_record(record: Any) -> None:
    """Raise :class:`ExtractionRecordError` unless ``record`` holds D10's keys
    with the types the report reads (R19). The schema version is checked
    before this."""
    _check_object(record, EXTRACTION_KEYS, "the record")
    file_info = record["file"]
    _check_object(file_info, ("path", "sha256"), "'file'")
    _require(isinstance(file_info["path"], str), "'file.path' is not a string")
    _require(
        _is_optional_str(file_info["sha256"]), "'file.sha256' is not a string or null"
    )
    for key in ("language", "document_type"):
        _require(_is_optional_str(record[key]), f"'{key}' is not a string or null")
    relations = record["relations"]
    _require(isinstance(relations, list), "'relations' is not a list")
    for position, relation in enumerate(relations):
        where = f"relations[{position}]"
        _check_object(
            relation, ("relation", "target_file_sha256", "detail", "status"), where
        )
        _require(
            relation["relation"] in RELATIONS,
            f"{where}.relation is not one of " + ", ".join(RELATIONS),
        )
        _require(
            _is_optional_str(relation["target_file_sha256"]),
            f"{where}.target_file_sha256 is not a string or null",
        )
        _require(isinstance(relation["detail"], str), f"{where}.detail is not a string")
        _require(
            relation["status"] in _RELATION_STATUSES,
            f"{where}.status is not one of " + ", ".join(_RELATION_STATUSES),
        )
    _require(
        record["extraction_status"] in EXTRACTION_STATUSES,
        "'extraction_status' is not one of " + ", ".join(EXTRACTION_STATUSES),
    )
    batches = record["batches"]
    _require(isinstance(batches, list), "'batches' is not a list")
    for position, batch in enumerate(batches):
        _check_object(batch, ("status",), f"batches[{position}]")
        _require(
            batch["status"] in BATCH_STATUSES,
            f"batches[{position}].status is not one of " + ", ".join(BATCH_STATUSES),
        )
    _check_fields(record["fields"])
    provisions = record[PROVISIONS]
    _require(isinstance(provisions, list), f"'{PROVISIONS}' is not a list")
    for position, item in enumerate(provisions):
        where = f"{PROVISIONS}[{position}]"
        _check_object(
            item,
            ("party", "as_written", "quote", "quote_verified", "page_spans"),
            where,
        )
        _require(
            item["party"] in PARTIES,
            f"{where}.party is not one of " + ", ".join(PARTIES),
        )
        for key in ("as_written", "quote"):
            _require(isinstance(item[key], str), f"{where}.{key} is not a string")
        _require(
            isinstance(item["quote_verified"], bool),
            f"{where}.quote_verified is not a bool",
        )
        _check_spans(item["page_spans"], where)


@dataclass(frozen=True)
class ExtractionFile:
    """A valid extraction file: its forward-slash path relative to the input
    folder, and its record."""

    relative: str
    record: Dict[str, Any]


def read_extraction_files(folder: Path) -> Tuple[List[ExtractionFile], int]:
    """Read and check every extraction file, logging an ERROR per refused
    file. Returns the valid files, in R19's order, and the number refused."""
    files: List[ExtractionFile] = []
    refused = 0
    for path in find_json_files(folder):
        relative = path.relative_to(folder).as_posix()
        found = None
        try:
            record = json.loads(path.read_bytes().decode("utf-8"))
            if not isinstance(record, dict):
                raise ExtractionRecordError("the top level is not a JSON object")
            found = refused_schema_version(record, REQUIRED_EXTRACTION_SCHEMA_VERSION)
            if found is None:
                validate_extraction_record(record)
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
                REQUIRED_EXTRACTION_SCHEMA_VERSION,
            )
            refused += 1
            continue
        files.append(ExtractionFile(relative=relative, record=record))
    files.sort(key=lambda item: (item.record["file"]["path"], item.relative))
    return files, refused


# --- One document ---------------------------------------------------------------------


def field_views(record: Mapping[str, Any]) -> Dict[str, FieldView]:
    """Each field's status, chosen value (the first counted candidate's
    stored value, R13) and pages (R21)."""
    views: Dict[str, FieldView] = {}
    for field in FIELDS:
        entry = record["fields"][field]
        counted = counted_candidates(entry["candidates"])
        found = entry["status"] == STATUS_FOUND
        views[field] = FieldView(
            name=field,
            status=entry["status"],
            kind=counted[0]["value_kind"] if found else None,
            value=copy.deepcopy(counted[0]["value"]) if found else None,
            pages=distinct_pairs(counted),
        )
    return views


def shown_provisions(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The verified provision items, deduplicated by hard-normalized quote;
    each with the pages of it and its duplicates (R21, R23)."""
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for item in record[PROVISIONS]:
        if item["quote_verified"] is True:
            groups.setdefault(hard_normalize(item["quote"]), []).append(item)
    return [
        {
            "party": items[0]["party"],
            "as_written": items[0]["as_written"],
            "quote": items[0]["quote"],
            "pages": distinct_pairs(items),
        }
        for items in groups.values()
    ]


def _read_from_scan(record: Mapping[str, Any]) -> bool:
    """R25: at least one evidence page, and every one OCR."""
    spans = [
        span
        for field in FIELDS
        for candidate in counted_candidates(record["fields"][field]["candidates"])
        for span in candidate["page_spans"]
    ]
    spans += [
        span
        for item in record[PROVISIONS]
        if item["quote_verified"] is True
        for span in item["page_spans"]
    ]
    return bool(spans) and all(span["text_origin"] == TEXT_ORIGIN_OCR for span in spans)


@dataclass(frozen=True)
class DocumentReport:
    """One document of the report: its ``followup.json`` entry, its table
    cells and its section of ``followup.md``."""

    entry: Dict[str, Any]
    cells: List[str]
    section: List[str]


def _one_line(text: str) -> str:
    """R23: a line feed in a quote becomes a space."""
    return text.replace(_LINE_FEED, " ")


def _pages_suffix(pairs: Sequence[Sequence[int]]) -> str:
    if not pairs:
        return ""
    return f" (p. {', '.join(pages_text(pair) for pair in pairs)})"


def _section(
    name: str,
    cells: Sequence[str],
    derived: Mapping[str, Mapping[str, Any]],
    provisions: Sequence[Mapping[str, Any]],
    notes: Sequence[str],
    record: Mapping[str, Any],
) -> List[str]:
    """R23: the field lines with their quotes, the derived lines with their
    calculations, the provisions, then the notes."""
    lines = [f"## {name}", ""]
    for position, field in enumerate(FIELDS):
        lines.append(f"- {FIELD_LABELS[field]}: {cells[2 + position]}")
        seen: List[Tuple[str, Any]] = []
        for candidate in counted_candidates(record["fields"][field]["candidates"]):
            pair = page_pair(candidate["page_spans"])
            key = (hard_normalize(candidate["quote"]), pair)
            if key in seen:
                continue
            seen.append(key)
            text = value_text(field, candidate["value_kind"], candidate["value"])
            pages = "" if pair is None else f" (p. {pages_text(pair)})"
            lines.append(f'  - {text}: "{_one_line(candidate["quote"])}"{pages}')
    for name_of_date in DERIVED:
        entry = derived[name_of_date]
        lines.append(f"- {DERIVED_LABELS[name_of_date]}: {derived_cell(entry)}")
        if entry["status"] == DERIVED_COMPUTED:
            lines.append(f"  - {entry['calculation']}")
    if provisions:
        lines += ["", f"### {PROVISIONS_HEADING}", ""]
        for item in provisions:
            lines.append(
                f'- {PARTY_NAMES[item["party"]]}: "{_one_line(item["quote"])}"'
                + _pages_suffix(item["pages"])
            )
    for note in notes:
        lines += ["", note]
    return lines


def _notes(
    item: ExtractionFile,
    term_end: TermEnd,
    views: Mapping[str, FieldView],
    files: Sequence[ExtractionFile],
    first_names: Mapping[str, str],
) -> List[str]:
    """D13's notes, in R23's order."""
    record = item.record
    notes: List[str] = []
    if term_end.mismatch:
        notes.append(
            mismatch_note(
                term_end.value, views[INITIAL_TERM_END].pages, term_end.computed
            )
        )
    for relation in record["relations"]:
        target = first_names.get(relation["target_file_sha256"])
        if relation["status"] != STATUS_RESOLVED or target is None:
            continue
        if relation["relation"] == RELATION_AMENDS:
            notes.append(amends_note(target, relation["detail"]))
        elif relation["relation"] == RELATION_DUPLICATE_OF:  # [CUT-2]
            notes.append(duplicate_note(target))
    own_sha256 = record["file"]["sha256"]
    if own_sha256 is not None:
        for source in files:
            for relation in source.record["relations"]:
                if (
                    relation["relation"] == RELATION_AMENDS
                    and relation["status"] == STATUS_RESOLVED
                    and relation["target_file_sha256"] == own_sha256
                ):
                    notes.append(
                        amended_by_note(
                            file_name(source.record["file"]["path"]),
                            relation["detail"],
                        )
                    )
    if record["extraction_status"] in (EXTRACTION_PARTIAL, EXTRACTION_ERROR):
        batches = record["batches"]
        failed = sum(1 for batch in batches if batch["status"] == BATCH_ERROR)
        notes.append(incomplete_note(failed, len(batches)))
    if _read_from_scan(record):
        notes.append(SCAN_NOTE)
    return notes


def document_report(
    item: ExtractionFile,
    reference: date,
    amendment_types: Sequence[str],
    files: Sequence[ExtractionFile],
    first_names: Mapping[str, str],
) -> DocumentReport:
    """One document's entry, cells and section (D12, D13); a WARNING per
    range failure (I1)."""
    record = item.record
    name = file_name(record["file"]["path"])
    views = field_views(record)
    term_end = effective_term_end(views)
    derived = derive(
        views,
        term_end,
        reference,
        amendment=record["document_type"] in amendment_types,
    )
    for failed in range_failures(term_end, derived):
        logger.warning("%s: %s: date out of range; not computed", item.relative, failed)
    provisions = shown_provisions(record)
    document_type = record["document_type"]
    cells = [
        name,
        "" if document_type is None else document_type,
        *[
            field_cell(
                field,
                views[field].status,
                views[field].kind,
                views[field].value,
                views[field].pages,
            )
            for field in FIELDS
        ],
        *[derived_cell(derived[name_of_date]) for name_of_date in DERIVED],
    ]
    notes = _notes(item, term_end, views, files, first_names)
    entry = {
        "file": {"path": record["file"]["path"], "sha256": record["file"]["sha256"]},
        "file_name": name,
        "document_type": document_type,
        "language": record["language"],
        "extraction_status": record["extraction_status"],
        "relations": copy.deepcopy(record["relations"]),
        "fields": {
            field: {
                "status": views[field].status,
                "value": copy.deepcopy(views[field].value),
                "value_kind": views[field].kind,
                "pages": [list(pair) for pair in views[field].pages],
            }
            for field in FIELDS
        },
        "effective_initial_term_end": term_end.to_dict(),
        "derived": derived,
        PROVISIONS: provisions,
    }
    section = _section(name, cells, derived, provisions, notes, record)
    return DocumentReport(entry=entry, cells=cells, section=section)


def build_report(
    files: Sequence[ExtractionFile], reference: date, amendment_types: Sequence[str]
) -> List[DocumentReport]:
    """Every document whose ``extraction_status`` is not ``not_in_scope``,
    in R19's order."""
    first_names: Dict[str, str] = {}
    for item in files:
        sha256 = item.record["file"]["sha256"]
        if sha256 is not None:
            first_names.setdefault(sha256, file_name(item.record["file"]["path"]))
    return [
        document_report(item, reference, amendment_types, files, first_names)
        for item in files
        if item.record["extraction_status"] != EXTRACTION_NOT_IN_SCOPE
    ]


# --- The three files (R23) ------------------------------------------------------------


def _table_row(cells: Sequence[str]) -> str:
    return (
        "| " + " | ".join(cell.replace(_PIPE, _ESCAPED_PIPE) for cell in cells) + " |"
    )


def render_markdown(documents: Sequence[DocumentReport], reference: date) -> str:
    """``followup.md``: the table, a section per document, the footer."""
    lines = [
        _table_row(COLUMNS),
        "|" + "|".join("---" for _ in COLUMNS) + "|",
        *[_table_row(document.cells) for document in documents],
    ]
    for document in documents:
        lines += ["", *document.section]
    lines += ["", footer(reference)]
    return _LINE_FEED.join(lines) + _LINE_FEED


def render_csv(documents: Sequence[DocumentReport]) -> bytes:
    """``followup.csv`` [CUT-1]: the column labels and the cells, ";", CRLF,
    UTF-8 with a byte order mark."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=CSV_DELIMITER)
    writer.writerow(COLUMNS)
    writer.writerows(document.cells for document in documents)
    return buffer.getvalue().encode(CSV_ENCODING)


def report_json(documents: Sequence[DocumentReport], reference: date) -> Dict[str, Any]:
    """``followup.json``'s content (D12)."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "reference_date": reference.isoformat(),
        "documents": [document.entry for document in documents],
    }


def run_report(extracted_dir: Path, report_dir: Path, reference: date) -> int:
    """``companydocs-metadata report``; returns the exit code (D11)."""
    try:
        config = metadata_settings()
    except SettingsError as error:
        logger.error("Invalid settings: %s", error)
        return EXIT_CANNOT_START
    if not extracted_dir.is_dir():
        logger.error("Not a folder: %s", extracted_dir)
        return EXIT_INPUT_REFUSED
    if not find_json_files(extracted_dir):
        logger.error("No extraction files found in %s", extracted_dir)
        return EXIT_INPUT_REFUSED
    files, refused = read_extraction_files(extracted_dir)
    if refused:
        return EXIT_INPUT_REFUSED

    documents = build_report(files, reference, config.amendment_types)
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = report_dir / MARKDOWN_NAME
    replace_file(markdown, render_markdown(documents, reference).encode("utf-8"))
    logger.info("Wrote %s", markdown)
    summary = report_dir / JSON_NAME
    write_json(summary, report_json(documents, reference))
    logger.info("Wrote %s", summary)
    table = report_dir / CSV_NAME
    replace_file(table, render_csv(documents))
    logger.info("Wrote %s", table)
    return EXIT_DONE
