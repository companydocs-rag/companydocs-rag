"""Startup of ``companydocs-demo`` (spec 009, D6, D13, R2-R4; Q2, Q3; A2).

``main`` runs with its seams: a raw folder and an override file under
``tmp_path``, spec 007's fake Retriever or the real one, and a
MockTransport. ``spy_on_main`` records every attempt to open an
``Answerer``, every server and every accepted connection.
"""

from __future__ import annotations

import http.client
import logging
import socket
import socketserver
import sys

import httpx
import pytest
from qdrant_client import QdrantClient

from companydocs_rag.interface import cli
from companydocs_rag.interface.server import DemoServer
from tests.answering._helpers import (
    FakeRetriever,
    Generator,
    settings_for,
    simple_result,
    use_settings,
    write_manifest,
)
from tests.embeddings._helpers import install_fake_sentence_transformers, write_settings
from tests.interface._helpers import (
    HOST,
    MainRun,
    Recorder,
    ask,
    free_port,
    request,
    spy_on_main,
    write_csv,
    write_pdf,
)
from tests.vectorstore._helpers import run_index, three_documents

LOGGER = "companydocs_rag.interface"


@pytest.fixture
def folders(tmp_path):
    """An index with a fabricated manifest, a raw folder with one PDF, and
    the path of an override file that does not exist."""
    write_manifest(tmp_path / "index")
    raw = tmp_path / "raw"
    write_pdf(raw / "fictief-huur.pdf")
    return tmp_path / "index", raw, tmp_path / "handmatig" / "typen.csv"


def _errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def _main(argv, raw, overrides, *, retriever="fake", generator=None):
    return cli.main(
        argv,
        raw_dir=raw,
        type_overrides=overrides,
        retriever=FakeRetriever(simple_result()) if retriever == "fake" else retriever,
        transport=(generator if generator is not None else Generator()).transport,
    )


# --- Exit 2 ---------------------------------------------------------------------------


def _assert_cannot_start(code, caplog, capsys, spies, start):
    assert code == 2
    (error,) = _errors(caplog)
    assert error.startswith(start), error
    assert capsys.readouterr().out == ""
    assert spies.attempts == 0
    assert spies.accepted == 0


def test_a_usage_error_exits_2(folders, monkeypatch, capsys):
    index, raw, overrides = folders
    spies = spy_on_main(monkeypatch)

    with pytest.raises(SystemExit) as raised:
        _main(["--index-dir", str(index), "--unknown"], raw, overrides)

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("error:") == 1
    assert (spies.attempts, spies.servers) == (0, [])


def test_an_index_dir_that_is_not_a_folder_exits_2(
    folders, tmp_path, monkeypatch, caplog, capsys
):
    _, raw, overrides = folders
    missing = tmp_path / "bestaat-niet"
    spies = spy_on_main(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _main(["--index-dir", str(missing)], raw, overrides)

    _assert_cannot_start(code, caplog, capsys, spies, f"Not a folder: {missing}")
    assert spies.servers == []


@pytest.mark.parametrize(
    ("settings", "start"),
    [
        (
            {"embeddings": {"batch_size": 0}},
            "Invalid settings: embeddings.batch_size: ",
        ),
        ({"retrieval": {"rrf_k": 0}}, "Invalid settings: retrieval.rrf_k: "),
        ({"answering": {"max_tokens": 0}}, "Invalid settings: answering.max_tokens: "),
        ({"interface": {"render_dpi": 0}}, "Invalid settings: interface.render_dpi: "),
        (
            {"answering": {"max_tokens": 0}, "interface": {"render_dpi": 0}},
            "Invalid settings: answering.max_tokens: ",
        ),
    ],
    ids=["embeddings", "retrieval", "answering", "interface", "answering-first"],
)
def test_invalid_settings_exit_2_in_section_order(
    folders, tmp_path, monkeypatch, caplog, capsys, settings, start
):
    index, raw, overrides = folders
    write_settings(tmp_path, monkeypatch, settings)
    spies = spy_on_main(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _main(["--index-dir", str(index)], raw, overrides)

    _assert_cannot_start(code, caplog, capsys, spies, start)
    assert spies.servers == []


def test_a_missing_prompt_file_exits_2(folders, tmp_path, monkeypatch, caplog, capsys):
    index, raw, overrides = folders
    use_settings(
        tmp_path,
        monkeypatch,
        settings_for(tmp_path, prompt_file=str(tmp_path / "geen-prompt.txt")),
    )
    spies = spy_on_main(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _main(["--index-dir", str(index)], raw, overrides)

    _assert_cannot_start(
        code, caplog, capsys, spies, "Invalid settings: answering.prompt_file: "
    )
    assert spies.servers == []


def test_an_unset_api_key_variable_exits_2(
    folders, tmp_path, monkeypatch, caplog, capsys
):
    index, raw, overrides = folders
    monkeypatch.delenv("COMPANYDOCS_FICTIEF_NIET_GEZET", raising=False)
    use_settings(
        tmp_path,
        monkeypatch,
        settings_for(tmp_path, api_key_env="COMPANYDOCS_FICTIEF_NIET_GEZET"),
    )
    spies = spy_on_main(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _main(["--index-dir", str(index)], raw, overrides)

    _assert_cannot_start(
        code, caplog, capsys, spies, "Invalid settings: answering.api_key_env: "
    )
    assert spies.servers == []


def test_a_missing_raw_folder_exits_2(folders, tmp_path, monkeypatch, caplog, capsys):
    index, _, overrides = folders
    missing = tmp_path / "geen-raw"
    spies = spy_on_main(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _main(["--index-dir", str(index)], missing, overrides)

    _assert_cannot_start(code, caplog, capsys, spies, f"Not a folder: {missing}")
    assert spies.servers == []


def test_a_malformed_override_file_exits_2(folders, monkeypatch, caplog, capsys):
    index, raw, overrides = folders
    write_csv(overrides, [("Bestandsnaam", "Soort"), ("fictief-huur.pdf", "Huur")])
    spies = spy_on_main(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _main(["--index-dir", str(index)], raw, overrides)

    _assert_cannot_start(
        code,
        caplog,
        capsys,
        spies,
        f"Invalid type overrides file {overrides}: missing column(s) Documenttype",
    )
    assert spies.servers == []


@pytest.mark.parametrize("holder_reuses_address", [False, True], ids=["plain", "reuse"])
def test_a_port_in_use_exits_2_naming_the_port(
    folders, monkeypatch, caplog, capsys, holder_reuses_address
):
    """The holder with SO_REUSEADDR set stands for a second server built as
    http.server builds one: without the exclusive bind it would start."""
    index, raw, overrides = folders
    spies = spy_on_main(monkeypatch)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        if holder_reuses_address:
            holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind((HOST, 0))
        holder.listen()
        port = holder.getsockname()[1]

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            code = _main(
                ["--index-dir", str(index), "--port", str(port)], raw, overrides
            )

    _assert_cannot_start(
        code,
        caplog,
        capsys,
        spies,
        f"Port {port} on 127.0.0.1 is not available: ",
    )
    assert spies.servers == []


def test_a_second_server_on_the_same_port_cannot_bind():
    port = free_port()
    first = DemoServer(port)
    try:
        with pytest.raises(OSError):
            DemoServer(port)
    finally:
        first.server_close()


# --- Exit 1 ---------------------------------------------------------------------------


def test_an_index_dir_without_manifest_exits_1_through_the_real_retriever(
    folders, tmp_path, monkeypatch, caplog, capsys
):
    _, raw, overrides = folders
    fake = install_fake_sentence_transformers(monkeypatch)
    folder = tmp_path / "geen-index"
    folder.mkdir()
    generator = Generator()
    spies = spy_on_main(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _main(
            ["--index-dir", str(folder)],
            raw,
            overrides,
            retriever=None,
            generator=generator,
        )

    assert code == 1
    assert _errors(caplog) == [f"No index at {folder}: manifest.json missing"]
    assert capsys.readouterr().out == ""
    assert fake.attempts == []
    assert generator.requests == []
    assert spies.answerers == []
    assert spies.accepted == 0


def test_an_index_in_use_exits_1(folders, tmp_path, monkeypatch, caplog, capsys):
    _, raw, overrides = folders
    chunks, index = tmp_path / "chunks", tmp_path / "echte-index"
    three_documents(chunks)
    assert run_index(chunks, index) == 0
    fake = install_fake_sentence_transformers(monkeypatch)
    generator = Generator()
    spies = spy_on_main(monkeypatch)

    holder = QdrantClient(path=str(index))
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            code = _main(
                ["--index-dir", str(index)],
                raw,
                overrides,
                retriever=None,
                generator=generator,
            )
    finally:
        holder.close()

    assert code == 1
    assert _errors(caplog) == [f"Index is in use by another process: {index}"]
    assert capsys.readouterr().out == ""
    assert fake.attempts == []
    assert generator.requests == []
    assert spies.answerers == []
    assert spies.accepted == 0
    QdrantClient(path=str(index)).close()  # the index is free again


# --- Exit 3 (Q2) ----------------------------------------------------------------------


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
    ids=["connect-error", "http-404", "http-500"],
)
def test_a_failed_warm_up_exits_3(
    folders, tmp_path, monkeypatch, caplog, capsys, failure, error
):
    index, raw, overrides = folders
    use_settings(tmp_path, monkeypatch, settings_for(tmp_path, warmup=True))
    retriever, generator = FakeRetriever(simple_result()), Generator(failure())
    spies = spy_on_main(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _main(
            ["--index-dir", str(index)],
            raw,
            overrides,
            retriever=retriever,
            generator=generator,
        )

    assert code == 3
    assert _errors(caplog) == [error]
    assert capsys.readouterr().out == ""
    assert len(generator.requests) == 1
    assert (retriever.calls, retriever.closed) == ([], 1)
    assert spies.answerers == []
    assert spies.accepted == 0
    assert not (tmp_path / "logs").exists()


# --- A valid start --------------------------------------------------------------------


def test_a_valid_start_prints_the_url_after_the_answerer_opened(
    folders, monkeypatch, caplog
):
    index, raw, overrides = folders
    port = free_port()
    stdout = Recorder()
    monkeypatch.setattr(sys, "stdout", stdout)
    spies = spy_on_main(monkeypatch, stdout)
    retriever = FakeRetriever(simple_result())

    run = MainRun(
        ["--index-dir", str(index), "--port", str(port)],
        raw_dir=raw,
        type_overrides=overrides,
        retriever=retriever,
        transport=Generator().transport,
    ).start()
    try:
        run.wait_for_url(stdout)
        response = request(port, "GET", "/")
    finally:
        code = run.finish()

    assert code == 0
    assert response.status == 200
    assert stdout.getvalue() == f"http://127.0.0.1:{port}/\n"
    (answerer,) = spies.answerers
    assert answerer.stdout_at_open == ""
    assert answerer.close_calls == 1
    assert retriever.closed == 1
    (server,) = spies.servers
    assert server["address"] == (HOST, port)
    assert server["reuse_address"] == 0
    if server["exclusive"] is not None:
        assert server["exclusive"] != 0
    assert spies.accepted == 1
    assert _errors(caplog) == []


def test_ctrl_c_while_serving_closes_the_socket_and_the_answerer_and_exits_0(
    folders, monkeypatch, capsys
):
    index, raw, overrides = folders
    spies = spy_on_main(monkeypatch)

    def interrupted(server, stop):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "serve", interrupted)
    retriever = FakeRetriever(simple_result())

    code = _main(
        ["--index-dir", str(index), "--port", str(free_port())],
        raw,
        overrides,
        retriever=retriever,
    )

    assert code == 0
    (answerer,) = spies.answerers
    assert answerer.close_calls == 1
    assert retriever.closed == 1
    (server,) = spies.servers
    assert server["server"].socket.fileno() == -1
    assert capsys.readouterr().out.count("\n") == 1


def test_ctrl_c_during_an_answer_is_no_d6_exception_and_exits_0(
    folders, monkeypatch, caplog
):
    """R2: KeyboardInterrupt is never caught as one of D6's other
    exceptions; it ends the run as Ctrl+C while serving does."""
    index, raw, overrides = folders
    port = free_port()
    stdout = Recorder()
    monkeypatch.setattr(sys, "stdout", stdout)
    spies = spy_on_main(monkeypatch)

    def interrupt(query):
        raise KeyboardInterrupt

    retriever = FakeRetriever(interrupt)
    run = MainRun(
        ["--index-dir", str(index), "--port", str(port)],
        raw_dir=raw,
        type_overrides=overrides,
        retriever=retriever,
        transport=Generator().transport,
    ).start()
    run.wait_for_url(stdout)
    with caplog.at_level(logging.WARNING):
        with pytest.raises((http.client.HTTPException, ConnectionError)):
            ask(port, "Fictieve vraag?")
        code = run.finish()

    assert code == 0
    assert [call[0] for call in retriever.calls] == ["Fictieve vraag?"]
    (answerer,) = spies.answerers
    assert answerer.close_calls == 1
    assert retriever.closed == 1
    assert [r for r in caplog.records if r.exc_info] == []
    (server,) = spies.servers
    assert server["server"].socket.fileno() == -1


def test_the_server_class_has_no_threading_mixin():
    assert not issubclass(DemoServer, socketserver.ThreadingMixIn)
    assert not any("Mixin" in base.__name__ for base in DemoServer.__mro__)
    assert DemoServer.allow_reuse_address is False


def test_help_lists_port_index_dir_and_h_only():
    parser = cli.build_parser()

    options = sorted(
        option for action in parser._actions for option in action.option_strings
    )
    assert options == ["--help", "--index-dir", "--port", "-h"]
    assert [action for action in parser._actions if not action.option_strings] == []
    text = parser.format_help()
    for name in ("--port", "--index-dir", "-h"):
        assert name in text


def test_the_defaults_and_the_exit_codes(monkeypatch, tmp_path):
    from companydocs_rag.ingestion.settings import find_project_root

    monkeypatch.chdir(tmp_path)
    args = cli.build_parser().parse_args([])

    assert args.index_dir == find_project_root() / "data" / "processed" / "index"
    assert args.port is None
    assert cli.DEFAULT_RAW_RELATIVE_PATH.as_posix() == "data/raw"
    assert cli.DEFAULT_TYPE_OVERRIDES_RELATIVE_PATH.as_posix() == (
        "data/manual/document_type_overrides.csv"
    )
    assert (
        cli.EXIT_STOPPED,
        cli.EXIT_INDEX_UNUSABLE,
        cli.EXIT_CANNOT_START,
        cli.EXIT_GENERATOR_UNAVAILABLE,
    ) == (0, 1, 2, 3)
