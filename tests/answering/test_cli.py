"""The ``companydocs-answer`` command (spec 007, D11, R16, R18, R19, Q4, Q7, A12).

The fakes are injected through ``main``'s keyword-only seams; the settings
come from a file under ``tmp_path`` through ``COMPANYDOCS_RAG_SETTINGS``.
"""

from __future__ import annotations

import io
import json
import logging
import sys

import httpx
import pytest

from companydocs_rag.answering import cli
from companydocs_rag.ingestion.settings import find_project_root
from tests.answering._helpers import (
    FakeRetriever,
    Generator,
    chunk,
    log_records,
    reply,
    search_result,
    settings_for,
    simple_result,
    use_settings,
    write_manifest,
)
from tests.embeddings._helpers import install_fake_sentence_transformers

LOGGER = "companydocs_rag.answering.cli"


@pytest.fixture
def index(tmp_path):
    write_manifest(tmp_path / "index")
    return tmp_path / "index"


def _run(index, *args, retriever=None, generator=None):
    return cli.main(
        [*args, "--index-dir", str(index)],
        retriever=retriever
        if retriever is not None
        else FakeRetriever(simple_result()),
        transport=(generator if generator is not None else Generator()).transport,
    )


def _messages(caplog, level):
    return [r.getMessage() for r in caplog.records if r.levelno == level]


def _errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def _questions(tmp_path, *rows):
    path = tmp_path / "vragen.csv"
    lines = ["question_id,question", *[f"{qid},{question}" for qid, question in rows]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- Output (R18) ---------------------------------------------------------------------


def test_one_question_prints_the_status_the_answer_and_the_sources(index, capsys):
    first = chunk(
        "c01",
        "Fictieve tekst een.",
        file_path="invoer/fictief-huur.pdf",
        page_start=2,
        page_end=3,
        clause_label="Artikel 2 Fictieve huurprijs",
    )
    second = chunk(
        "c02", "Fictieve tekst twee.", file_path="invoer/b.pdf", page_start=5
    )
    result = search_result(dense=[(first, 0.9), (second, 0.8)], sparse=[(first, 3.0)])

    code = _run(
        index,
        "Wat is de fictieve huurprijs?",
        retriever=FakeRetriever(result),
        generator=Generator(reply("  De huurprijs is fictief [2][1].  \n")),
    )

    assert code == 0
    assert capsys.readouterr().out == (
        "Status: answered\n"
        "De huurprijs is fictief [2][1].\n"
        "\n"
        "[2] invoer/b.pdf pagina 5 (native)\n"
        "[1] invoer/fictief-huur.pdf pagina 2-3 (native)"
        " kop: Artikel 2 Fictieve huurprijs\n"
    )


def test_a_score_gate_record_prints_the_status_line_only(
    index, capsys, tmp_path, monkeypatch
):
    use_settings(tmp_path, monkeypatch, settings_for(tmp_path, min_dense_score=0.99))

    code = _run(index, "Fictieve vraag?")

    assert code == 0
    assert capsys.readouterr().out == "Status: insufficient_evidence (score_gate)\n"


def test_an_abstention_by_the_model_prints_its_source_and_text(index, capsys):
    generator = Generator(reply("ONVOLDOENDE BEWIJS, geen fictieve bron."))

    assert _run(index, "Fictieve vraag?", generator=generator) == 0

    assert capsys.readouterr().out == (
        "Status: insufficient_evidence (prompt)\n"
        "ONVOLDOENDE BEWIJS, geen fictieve bron.\n"
    )


def test_json_prints_the_record(index, capsys, tmp_path):
    assert _run(index, "Fictieve vraag?", "--json") == 0

    out = capsys.readouterr().out
    printed = json.loads(out)
    assert out == json.dumps(printed, indent=2, ensure_ascii=False) + "\n"
    (record,) = log_records(tmp_path / "logs")
    assert printed == record


def test_a_questions_file_prints_the_counts_and_the_log_path(index, capsys, tmp_path):
    path = _questions(tmp_path, ("v1", "Een?"), ("v2", "Twee?"), ("v3", "Drie?"))
    generator = Generator(
        reply("Fictief [1]."),
        reply("ONVOLDOENDE BEWIJS."),
        reply("Fictief [2]."),
    )

    code = _run(index, "--questions", str(path), generator=generator)

    assert code == 0
    (log_path,) = (tmp_path / "logs").iterdir()
    assert capsys.readouterr().out == (
        "answered: 2\n"
        "insufficient_evidence: 1\n"
        "uncited: 0\n"
        "error: 0\n"
        f"log: {log_path}\n"
    )


def test_non_ascii_through_a_cp1252_stdout_arrives_as_utf8(index, monkeypatch):
    text = "Fictief: " + chr(0x0159) + "e" + chr(0x0159) + " o" + chr(0x0142) + "a [1]."
    raw = io.BytesIO()
    stdout = io.TextIOWrapper(raw, encoding="cp1252", newline="\n")
    monkeypatch.setattr(sys, "stdout", stdout)

    assert _run(index, "Fictieve vraag?", generator=Generator(reply(text))) == 0

    stdout.flush()
    assert text in raw.getvalue().decode("utf-8")


# --- Exit 0, 1 and 4 ------------------------------------------------------------------


def test_answered_and_abstained_records_exit_0(index, tmp_path, caplog):
    path = _questions(tmp_path, ("v1", "Een?"), ("v2", "Twee?"), ("v3", "Drie?"))
    generator = Generator(
        reply("Fictief [1]."), reply("Onvoldoende bewijs."), reply("Geen bron.")
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _run(index, "--questions", str(path), generator=generator)

    assert code == 0
    assert [r["status"] for r in log_records(tmp_path / "logs")] == [
        "answered",
        "insufficient_evidence",
        "uncited",
    ]
    assert _messages(caplog, logging.WARNING) == []


def test_a_folder_without_manifest_exits_1_through_the_real_retriever(
    tmp_path, monkeypatch, caplog, capsys
):
    fake = install_fake_sentence_transformers(monkeypatch)
    folder = tmp_path / "geen-index"
    folder.mkdir()
    generator = Generator()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = cli.main(
            ["Fictieve vraag?", "--index-dir", str(folder)],
            transport=generator.transport,
        )

    assert code == 1
    assert _errors(caplog) == [f"No index at {folder}: manifest.json missing"]
    assert capsys.readouterr().out == ""
    assert fake.attempts == []
    assert generator.requests == []


def test_an_error_record_exits_4(index, tmp_path, caplog):
    path = _questions(tmp_path, ("v1", "Een?"), ("v2", "Twee?"), ("v3", "Drie?"))
    generator = Generator(
        reply("Fictief [1]."),
        httpx.Response(500, text="fictieve serverfout"),
        reply("Fictief [2]."),
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _run(index, "--questions", str(path), generator=generator)

    assert code == 4
    records = log_records(tmp_path / "logs")
    assert [(r["question_id"], r["status"]) for r in records] == [
        ("v1", "answered"),
        ("v2", "error"),
        ("v3", "answered"),
    ]
    assert records[1]["error"]["type"] == "http_error"
    assert _messages(caplog, logging.WARNING) == [
        "Question v2: http_error: HTTP 500: fictieve serverfout"
    ]
    assert _errors(caplog) == []


# --- Exit 3 (R3, R18, Q4) -------------------------------------------------------------


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (
            lambda: httpx.ConnectError("fictief: geweigerd"),
            "Generator unavailable at http://127.0.0.1:11434/v1: "
            "ConnectError: fictief: geweigerd",
        ),
        (
            lambda: httpx.Response(404, text='{"error":"model not found"}'),
            "Generator unavailable at http://127.0.0.1:11434/v1: "
            'HTTP 404: {"error":"model not found"}',
        ),
        (
            lambda: httpx.Response(500, text="fictieve fout"),
            "Generator failed at warm-up: HTTP 500: fictieve fout",
        ),
    ],
    ids=["connect-error", "http-404", "generator-error"],
)
def test_a_failed_warm_up_exits_3(
    index, tmp_path, monkeypatch, caplog, capsys, failure, error
):
    use_settings(tmp_path, monkeypatch, settings_for(tmp_path, warmup=True))
    retriever, generator = FakeRetriever(simple_result()), Generator(failure())

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _run(index, "Fictieve vraag?", retriever=retriever, generator=generator)

    assert code == 3
    assert _errors(caplog) == [error]
    assert capsys.readouterr().out == ""
    assert len(generator.requests) == 1
    assert (retriever.calls, retriever.closed) == ([], 1)
    assert not (tmp_path / "logs").exists()


def test_an_unavailable_generator_stops_the_run(index, tmp_path, caplog, capsys):
    path = _questions(tmp_path, ("v1", "Een?"), ("v2", "Twee?"), ("v3", "Drie?"))
    generator = Generator(
        reply("Fictief [1]."), httpx.ConnectError("fictief: weg"), reply("Fictief [1].")
    )
    retriever = FakeRetriever(simple_result())

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _run(
            index, "--questions", str(path), retriever=retriever, generator=generator
        )

    assert code == 3
    records = log_records(tmp_path / "logs")
    assert [(r["question_id"], r["status"]) for r in records] == [
        ("v1", "answered"),
        ("v2", "error"),
    ]
    assert records[1]["error"]["type"] == "generator_unavailable"
    assert len(generator.requests) == 2
    assert [call[0] for call in retriever.calls] == ["Een?", "Twee?"]
    assert retriever.closed == 1
    assert _errors(caplog) == [
        "Generator unavailable at http://127.0.0.1:11434/v1: ConnectError: fictief: weg"
    ]
    assert _messages(caplog, logging.WARNING) == [
        "Question v2: generator_unavailable: ConnectError: fictief: weg"
    ]
    (log_path,) = (tmp_path / "logs").iterdir()
    assert capsys.readouterr().out == (
        "answered: 1\n"
        "insufficient_evidence: 0\n"
        "uncited: 0\n"
        "error: 1\n"
        f"log: {log_path}\n"
    )


def test_a_single_question_with_an_unavailable_generator_exits_3(
    index, tmp_path, capsys
):
    generator = Generator(httpx.Response(404, text="onbekend model"))

    code = _run(index, "Fictieve vraag?", generator=generator)

    assert code == 3
    assert capsys.readouterr().out == "Status: error\n"
    (record,) = log_records(tmp_path / "logs")
    assert record["error"] == {
        "type": "generator_unavailable",
        "message": "HTTP 404: onbekend model",
    }


# --- Exit 2 (R16) ---------------------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\t"], ids=["empty", "spaces", "tab"])
def test_an_empty_question_exits_2_first(tmp_path, caplog, capsys, question):
    retriever, generator = FakeRetriever(simple_result()), Generator()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _run(tmp_path, question, retriever=retriever, generator=generator) == 2
        assert (
            _run(
                tmp_path / "bestaat-niet",
                question,
                retriever=retriever,
                generator=generator,
            )
            == 2
        )

    assert _errors(caplog) == ["Empty question", "Empty question"]
    assert capsys.readouterr().out == ""
    assert (retriever.calls, generator.requests) == ([], [])


def test_a_missing_index_folder_exits_2_before_the_settings(
    tmp_path, monkeypatch, caplog, capsys
):
    use_settings(tmp_path, monkeypatch, {"answering": {"max_tokens": 0}})
    missing = tmp_path / "bestaat-niet"
    a_file = tmp_path / "bestand.json"
    a_file.write_text("{}", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _run(missing, "Fictieve vraag?") == 2
        assert _run(a_file, "Fictieve vraag?") == 2

    assert _errors(caplog) == [f"Not a folder: {missing}", f"Not a folder: {a_file}"]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("settings", "start"),
    [
        (
            {"embeddings": {"batch_size": 0}},
            "Invalid settings: embeddings.batch_size: ",
        ),
        ({"retrieval": {"rrf_k": 0}}, "Invalid settings: retrieval.rrf_k: "),
        ({"answering": {"max_tokens": 0}}, "Invalid settings: answering.max_tokens: "),
        (
            {"embeddings": {"batch_size": 0}, "answering": {"max_tokens": 0}},
            "Invalid settings: embeddings.batch_size: ",
        ),
        (
            {"retrieval": {"rrf_k": 0}, "answering": {"max_tokens": 0}},
            "Invalid settings: retrieval.rrf_k: ",
        ),
    ],
    ids=["embeddings", "retrieval", "answering", "embeddings-first", "retrieval-first"],
)
def test_invalid_settings_exit_2_in_section_order(
    index, tmp_path, monkeypatch, caplog, capsys, settings, start
):
    use_settings(tmp_path, monkeypatch, settings)
    retriever, generator = FakeRetriever(simple_result()), Generator()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _run(index, "Fictieve vraag?", retriever=retriever, generator=generator)

    assert code == 2
    (error,) = _errors(caplog)
    assert error.startswith(start)
    assert capsys.readouterr().out == ""
    assert (retriever.calls, generator.requests) == ([], [])


def test_the_prompt_file_is_checked_before_the_api_key(
    index, tmp_path, monkeypatch, caplog
):
    monkeypatch.delenv("COMPANYDOCS_FICTIEF_NIET_GEZET", raising=False)
    use_settings(
        tmp_path,
        monkeypatch,
        settings_for(
            tmp_path,
            prompt_file=str(tmp_path / "geen-prompt.txt"),
            api_key_env="COMPANYDOCS_FICTIEF_NIET_GEZET",
        ),
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _run(index, "Fictieve vraag?") == 2

    (error,) = _errors(caplog)
    assert error.startswith("Invalid settings: answering.prompt_file: ")


@pytest.mark.parametrize(
    "argv",
    [
        ["Een?", "--questions", "vragen.csv"],
        [],
        ["--questions", "vragen.csv", "--json"],
        ["Een?", "--selection-mode", "hybride"],
    ],
    ids=["both-forms", "neither", "json-with-questions", "unknown-mode"],
)
def test_usage_errors_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(
            argv,
            retriever=FakeRetriever(simple_result()),
            transport=Generator().transport,
        )

    assert raised.value.code == 2
    assert capsys.readouterr().out == ""


# --- Selection mode, help, defaults ---------------------------------------------------


def test_selection_mode_per_path_holds_for_every_record(index, tmp_path):
    path = _questions(tmp_path, ("v1", "Een?"), ("v2", "Twee?"))
    retriever = FakeRetriever(simple_result())

    code = _run(
        index,
        "--questions",
        str(path),
        "--selection-mode",
        "per_path",
        retriever=retriever,
    )

    assert code == 0
    records = log_records(tmp_path / "logs")
    assert all(record["path_hits"] is not None for record in records)
    assert {r["settings"]["answering"]["selection_mode"] for r in records} == {
        "per_path"
    }
    assert [call[1] for call in retriever.calls] == [True, True]


def test_help_lists_the_arguments_and_nothing_else():
    parser = cli.build_parser()

    options = sorted(
        option for action in parser._actions for option in action.option_strings
    )
    positionals = [action for action in parser._actions if not action.option_strings]
    assert options == [
        "--help",
        "--index-dir",
        "--json",
        "--questions",
        "--selection-mode",
        "-h",
    ]
    assert [(action.dest, action.metavar) for action in positionals] == [
        ("question", "QUESTION")
    ]
    text = parser.format_help()
    for name in (
        "QUESTION",
        "--questions",
        "--selection-mode",
        "--json",
        "--index-dir",
    ):
        assert name in text


def test_the_default_index_folder_and_the_exit_codes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    args = cli.build_parser().parse_args(["Een?"])

    assert args.index_dir == find_project_root() / "data" / "processed" / "index"
    assert (args.question, args.questions, args.json, args.selection_mode) == (
        "Een?",
        None,
        False,
        None,
    )
    assert (
        cli.EXIT_ANSWER_DONE,
        cli.EXIT_INDEX_UNUSABLE,
        cli.EXIT_CANNOT_START,
        cli.EXIT_GENERATOR_UNAVAILABLE,
        cli.EXIT_QUESTION_ERRORS,
    ) == (0, 1, 2, 3, 4)
