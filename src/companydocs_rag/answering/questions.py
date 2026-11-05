"""The questions file of ``companydocs-answer --questions`` (spec 007, D11, R17).

A CSV file in UTF-8, with or without a byte order mark, read with Python's
csv defaults (double quotes included) except the delimiter: a comma or a
semicolon, the one that splits the header line into cells that include both
``question_id`` and ``question``. The header must hold those two names
exactly; other columns are ignored and never copied. Lines are numbered from
1, the header being line 1, and a row is named by the line it starts on.
Blank lines are skipped. A row with fewer cells than the header has empty
cells for the missing columns. ``question_id``s are compared as exact
strings, and an empty one is allowed once. Questions keep their text as
given and their file order.

:func:`read_questions` raises :class:`QuestionsFileError` with the message
the command logs: ``Invalid questions file <path>: line <n>: <reason>``,
where the reason is ``empty question`` (also a whitespace-only one),
``duplicate question_id <id>, first on line <m>``, ``missing column <name>``,
``not UTF-8`` or ``no header``; ``Invalid questions file <path>: no
questions`` for a header without rows; or ``Could not read <path>:
<exception type>: <message>`` for a path that cannot be read.
"""

from __future__ import annotations

import codecs
import csv
import io
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

COLUMN_QUESTION_ID = "question_id"
COLUMN_QUESTION = "question"
_COLUMNS = (COLUMN_QUESTION_ID, COLUMN_QUESTION)
_DELIMITERS = (",", ";")


class Question(NamedTuple):
    """One row: its ``question_id``, its question and the line it starts on.
    A single question from the command line has no ``question_id``."""

    question_id: Optional[str]
    question: str
    line: int


class QuestionsFileError(ValueError):
    """The questions file cannot be used; the text is the ERROR message."""


def _invalid(path: Path, line: Optional[int], reason: str) -> QuestionsFileError:
    where = "" if line is None else f"line {line}: "
    return QuestionsFileError(f"Invalid questions file {path}: {where}{reason}")


def _is_blank(row: List[str]) -> bool:
    return not row or (len(row) == 1 and not row[0].strip())


def _cells(line: str, delimiter: str) -> List[str]:
    return next(csv.reader([line], delimiter=delimiter), [])


def _delimiter(path: Path, header: str, line: int) -> str:
    """The delimiter that splits ``header`` into cells holding both column
    names; else the first missing column, from the split that finds most."""
    best: List[str] = []
    for delimiter in _DELIMITERS:
        cells = _cells(header, delimiter)
        if all(column in cells for column in _COLUMNS):
            return delimiter
        found = [column for column in _COLUMNS if column in cells]
        if len(found) > len(best):
            best = found
    missing = next(column for column in _COLUMNS if column not in best)
    raise _invalid(path, line, f"missing column {missing}")


def read_questions(path: Path) -> List[Question]:
    """The questions of the file at ``path``, in file order (R17)."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise QuestionsFileError(
            f"Could not read {path}: {type(error).__name__}: {error}"
        ) from None
    # The byte order mark is dropped before decoding, so the error's offset
    # counts in the same bytes as the line feeds before it.
    body = raw[len(codecs.BOM_UTF8) :] if raw.startswith(codecs.BOM_UTF8) else raw
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        line = body[: error.start].count(b"\n") + 1
        raise _invalid(path, line, "not UTF-8") from None

    lines = io.StringIO(text, newline="").readlines()
    header_line = next(
        (number for number, line in enumerate(lines, start=1) if line.strip()), None
    )
    if header_line is None:
        raise _invalid(path, 1, "no header")
    delimiter = _delimiter(path, lines[header_line - 1].rstrip("\r\n"), header_line)

    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    header: Optional[List[str]] = None
    positions: Dict[str, int] = {}
    seen: Dict[str, int] = {}
    questions: List[Question] = []
    previous = 0
    for row in reader:
        start, previous = previous + 1, reader.line_num
        if _is_blank(row):
            continue
        if header is None:
            header = row
            positions = {column: header.index(column) for column in _COLUMNS}
            continue
        cells = {
            column: row[index] if index < len(row) else ""
            for column, index in positions.items()
        }
        question_id, question = cells[COLUMN_QUESTION_ID], cells[COLUMN_QUESTION]
        if not question.strip():
            raise _invalid(path, start, "empty question")
        if question_id in seen:
            raise _invalid(
                path,
                start,
                f"duplicate question_id {question_id}, "
                f"first on line {seen[question_id]}",
            )
        seen[question_id] = start
        questions.append(Question(question_id, question, start))
    if not questions:
        raise _invalid(path, None, "no questions")
    return questions
