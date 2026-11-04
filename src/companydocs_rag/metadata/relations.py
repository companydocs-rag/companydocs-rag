"""Relations between documents (spec 008, D5, R7).

The relations file (``Bestandsnaam;Relatie;Doelbestand;Detail``) is read at
start with the rules of the override file (:mod:`.doctype`): a missing file
is not an error (INFO); a missing column, an unreadable file, or a
``Relatie`` other than ``amends`` and ``duplicate_of`` makes it malformed
(exit 2). A row with an empty ``Bestandsnaam`` is ignored with a WARNING
naming its line, before its ``Relatie`` is looked at; a repeated
``Bestandsnaam`` logs a WARNING and the first row wins, so a document has
zero or one relation.

The relation belongs to every chunk file whose file name is ``Bestandsnaam``.
Its target is resolved when exactly one chunk file of the run, in scope or
not, has the file name ``Doelbestand`` and a ``file.sha256``; else it is
``target_not_found``, with a WARNING saying why.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .doctype import COLUMN_FILENAME, ManualFileError
from .files import CsvFileError, file_name, read_csv_rows

logger = logging.getLogger(__name__)

COLUMN_RELATION = "Relatie"
COLUMN_TARGET = "Doelbestand"
COLUMN_DETAIL = "Detail"
RELATIONS_COLUMNS = (COLUMN_FILENAME, COLUMN_RELATION, COLUMN_TARGET, COLUMN_DETAIL)

RELATION_AMENDS = "amends"
RELATION_DUPLICATE_OF = "duplicate_of"  # [CUT-2]
RELATIONS = (RELATION_AMENDS, RELATION_DUPLICATE_OF)
STATUS_RESOLVED = "resolved"
STATUS_TARGET_NOT_FOUND = "target_not_found"


@dataclass(frozen=True)
class RelationRow:
    """One row of the relations file that is used."""

    name: str
    relation: str
    target: str
    detail: str


@dataclass(frozen=True)
class RelationsFile:
    """The relations file: the first row per ``Bestandsnaam``, in file
    order. ``path`` is None when there is no file."""

    path: Optional[Path]
    rows: Dict[str, RelationRow]


def read_relations(path: Path) -> RelationsFile:
    """Read the relations file at start. A missing file gives no rows
    (INFO); a malformed one raises :class:`ManualFileError`."""
    if not path.exists():
        logger.info("No relations file at %s", path)
        return RelationsFile(path=None, rows={})
    try:
        rows = read_csv_rows(path, RELATIONS_COLUMNS)
    except CsvFileError as error:
        raise ManualFileError(path, str(error)) from error
    used: Dict[str, RelationRow] = {}
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
        relation = row.values[COLUMN_RELATION]
        if relation not in RELATIONS:
            raise ManualFileError(
                path,
                f"{COLUMN_RELATION} is {relation!r}, not " + " or ".join(RELATIONS),
                line=row.line,
            )
        if name in used:
            logger.warning(
                "%s lists %s more than once; the first row is used", path, name
            )
            continue
        used[name] = RelationRow(
            name=name,
            relation=relation,
            target=row.values[COLUMN_TARGET],
            detail=row.values[COLUMN_DETAIL],
        )
    return RelationsFile(path=path, rows=used)


def _entry(row: RelationRow, target_sha256: Optional[str]) -> Dict[str, Any]:
    """R7's entry, keys in order."""
    return {
        "relation": row.relation,
        "target_file_sha256": target_sha256,
        "detail": row.detail,
        "status": STATUS_TARGET_NOT_FOUND if target_sha256 is None else STATUS_RESOLVED,
    }


def _target_sha256(
    row: RelationRow, by_name: Mapping[str, Sequence[Any]]
) -> Optional[str]:
    """The target's ``file.sha256``, or None with a WARNING saying why."""
    found = by_name.get(row.target, []) if row.target != "" else []
    with_sha256 = [
        record["file"]["sha256"]
        for record in found
        if record["file"]["sha256"] is not None
    ]
    if len(with_sha256) == 1:
        return with_sha256[0]
    if not found:
        reason = "not found"
    elif not with_sha256:
        reason = "has no file sha256"
    else:
        reason = f"matches {len(found)} chunk files"
    logger.warning("Relation of %s: target %s %s", row.name, row.target, reason)
    return None


def resolve_relations(
    relations: RelationsFile, records: Sequence[Mapping[str, Any]]
) -> List[List[Dict[str, Any]]]:
    """Each chunk record's ``relations``, in the order of ``records``. A row
    whose name matches no chunk file logs a WARNING."""
    by_name: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        by_name.setdefault(file_name(record["file"]["path"]), []).append(record)
    entries: Dict[str, Dict[str, Any]] = {}
    for name, row in relations.rows.items():
        if name not in by_name:
            logger.warning("%s: no chunk file is named %s", relations.path, name)
            continue
        entries[name] = _entry(row, _target_sha256(row, by_name))
    return [
        [dict(entries[name])] if name in entries else []
        for name in (file_name(record["file"]["path"]) for record in records)
    ]
