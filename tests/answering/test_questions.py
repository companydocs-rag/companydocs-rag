"""The questions file (spec 007, D11, R17, Q9, A11).

Files are written as bytes in ``tmp_path``, so the byte order mark, the line
ends and invalid UTF-8 are exactly what the test says.
"""

from __future__ import annotations

import logging

import pytest

from companydocs_rag.answering import cli
from companydocs_rag.answering.questions import Question, read_questions
from tests.answering._helpers import (
    FakeRetriever,
    Generator,
    log_records,
    simple_result,
    write_manifest,
)

BOM = b"\xef\xbb\xbf"
LOGGER = "companydocs_rag.answering.cli"


def _file(tmp_path, raw, name="vragen.csv"):
    path = tmp_path / name
    path.write_bytes(raw)
    return path


def _run(tmp_path, path):
    write_manifest(tmp_path / "index")
    retriever, generator = FakeRetriever(simple_result()), Generator()
    code = cli.main(
        ["--questions", str(path), "--index-dir", str(tmp_path / "index")],
        retriever=retriever,
        transport=generator.transport,
    )
    return code, retriever, generator


def test_a_file_with_a_bom_an_extra_column_and_a_blank_line(tmp_path, capsys):
    path = _file(
        tmp_path,
        BOM
        + b"question_id,question,toelichting\r\n"
        + b"v1,Wat is de fictieve huurprijs?,EXTRA-WAARDE-EEN\r\n"
        + b"\r\n"
        + b"v2,Wie is de fictieve verhuurder?,EXTRA-WAARDE-TWEE\r\n",
    )

    code, retriever, _ = _run(tmp_path, path)

    assert code == 0
    records = log_records(tmp_path / "logs")
    assert [(r["question_id"], r["question"]) for r in records] == [
        ("v1", "Wat is de fictieve huurprijs?"),
        ("v2", "Wie is de fictieve verhuurder?"),
    ]
    assert [call[0] for call in retriever.calls] == [
        "Wat is de fictieve huurprijs?",
        "Wie is de fictieve verhuurder?",
    ]
    out = capsys.readouterr().out
    log_text = next((tmp_path / "logs").iterdir()).read_text(encoding="utf-8")
    for extra in ("EXTRA-WAARDE-EEN", "EXTRA-WAARDE-TWEE", "toelichting"):
        assert extra not in out
        assert extra not in log_text


def test_the_line_numbers_name_the_line_a_row_starts_on(tmp_path):
    path = _file(
        tmp_path,
        b'question_id,question\n\nv1,"Een vraag\nover twee regels?"\nv2,Twee?\n',
    )

    assert read_questions(path) == [
        Question("v1", "Een vraag\nover twee regels?", 3),
        Question("v2", "Twee?", 5),
    ]


# (name, file bytes, the expected message after "Invalid questions file <path>: ")
INVALID = [
    (
        "empty-question",
        b"question_id,question\nv1,Een?\n\nv2,\nv3,Drie?\n",
        "line 4: empty question",
    ),
    (
        "blank-question",
        b"question_id,question\nv1,Een?\nv2,   \n",
        "line 3: empty question",
    ),
    (
        "short-row",
        b"question_id,question\nv1\n",
        "line 2: empty question",
    ),
    (
        "duplicate-id",
        b"question_id,question\n\nv1,Een?\nv2,Twee?\nv1,Nog een?\n",
        "line 5: duplicate question_id v1, first on line 3",
    ),
    (
        "second-empty-id",
        b"question_id,question\n,Een?\n,Twee?\n",
        "line 3: duplicate question_id , first on line 2",
    ),
    (
        "missing-question_id",
        b"id,question\nv1,Een?\n",
        "line 1: missing column question_id",
    ),
    (
        "missing-question",
        b"question_id;vraag\nv1;Een?\n",
        "line 1: missing column question",
    ),
    (
        "not-utf8",
        b"question_id,question\nv1,Een?\nv2,Caf\xe9?\n",
        "line 3: not UTF-8",
    ),
    # The bad byte at byte 2 of line 3 without a byte order mark; with one, at
    # byte 0, 2 or 5 of line 3, or at byte 0 of line 4 after a blank line
    # (review F3).
    (
        "not-utf8-at-byte-2",
        b"question_id,question\nv1,Een?\nv2\xe9,Twee?\n",
        "line 3: not UTF-8",
    ),
    (
        "bom-not-utf8-at-byte-0",
        BOM + b"question_id,question\nv1,Een?\n\xe9v2,Twee?\n",
        "line 3: not UTF-8",
    ),
    (
        "bom-not-utf8-at-byte-2",
        BOM + b"question_id,question\nv1,Een?\nv2\xe9,Twee?\n",
        "line 3: not UTF-8",
    ),
    (
        "bom-not-utf8-at-byte-5",
        BOM + b"question_id,question\nv1,Een?\nv2,Tw\xe9e?\n",
        "line 3: not UTF-8",
    ),
    (
        "bom-not-utf8-after-a-blank-line",
        BOM + b"question_id,question\nv1,Een?\n\n\xe9v3,Drie?\n",
        "line 4: not UTF-8",
    ),
    ("header-only", b"question_id,question\r\n", "no questions"),
    ("empty-file", b"", "line 1: no header"),
    ("bom-only", BOM, "line 1: no header"),
]


@pytest.mark.parametrize(
    ("raw", "message"),
    [case[1:] for case in INVALID],
    ids=[case[0] for case in INVALID],
)
def test_an_invalid_file_exits_2_before_anything_is_opened(
    tmp_path, caplog, capsys, raw, message
):
    path = _file(tmp_path, raw)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code, retriever, generator = _run(tmp_path, path)

    assert code == 2
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [f"Invalid questions file {path}: {message}"]
    assert capsys.readouterr().out == ""
    assert (retriever.calls, retriever.closed, generator.requests) == ([], 0, [])
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("kind", ["folder", "missing"])
def test_a_path_that_is_not_a_file_exits_2(tmp_path, caplog, capsys, kind):
    path = tmp_path / "vragen"
    if kind == "folder":
        path.mkdir()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code, retriever, generator = _run(tmp_path, path)

    assert code == 2
    (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert error.startswith(f"Could not read {path}: ")
    assert capsys.readouterr().out == ""
    assert (retriever.calls, generator.requests) == ([], [])


def test_a_semicolon_file_gives_the_same_records_as_a_comma_file(tmp_path):
    rows = [
        ("v1", "Wat staat er in artikel 3, lid 2?"),
        ("v2", "Wie is de fictieve huurder; en wie de verhuurder?"),
    ]
    comma = _file(
        tmp_path,
        (
            "question_id,question\n"
            'v1,"Wat staat er in artikel 3, lid 2?"\n'
            "v2,Wie is de fictieve huurder; en wie de verhuurder?\n"
        ).encode("utf-8"),
        "komma.csv",
    )
    semicolon = _file(
        tmp_path,
        (
            "question_id;question\n"
            "v1;Wat staat er in artikel 3, lid 2?\n"
            'v2;"Wie is de fictieve huurder; en wie de verhuurder?"\n'
        ).encode("utf-8"),
        "puntkomma.csv",
    )

    for path in (comma, semicolon):
        found = [(q.question_id, q.question) for q in read_questions(path)]
        assert found == rows


def test_the_columns_may_come_in_any_order(tmp_path):
    path = _file(tmp_path, b"extra;question;question_id\nx;Een?;v1\n")

    assert read_questions(path) == [Question("v1", "Een?", 2)]


def test_the_records_follow_the_file_order(tmp_path):
    path = _file(tmp_path, b"question_id,question\nv3,Drie?\nv1,Een?\nv2,Twee?\n")

    code, _, _ = _run(tmp_path, path)

    assert code == 0
    assert [r["question_id"] for r in log_records(tmp_path / "logs")] == [
        "v3",
        "v1",
        "v2",
    ]
