"""The ``interface`` settings (spec 009, D14, R15; Q6, Q7; A1)."""

from __future__ import annotations

import logging
import sys

import pytest
import yaml

from companydocs_rag.embeddings.dense import SettingsError
from companydocs_rag.ingestion.settings import find_project_root
from companydocs_rag.interface import cli, settings
from companydocs_rag.interface.settings import InterfaceSettings, interface_settings
from tests.answering._helpers import (
    FakeRetriever,
    Generator,
    simple_result,
    write_manifest,
)
from tests.embeddings._helpers import write_settings
from tests.interface._helpers import (
    MainRun,
    Recorder,
    free_port,
    request,
    spy_on_main,
)

LOGGER = "companydocs_rag.interface.cli"
KEYS = [
    ("port", "PORT"),
    ("render_dpi", "RENDER_DPI"),
    ("max_question_chars", "MAX_QUESTION_CHARS"),
]
Q7_DEFAULTS = {"port": 8765, "render_dpi": 150, "max_question_chars": 1000}


def _yaml_text():
    return (find_project_root() / "config" / "settings.yaml").read_text(
        encoding="utf-8"
    )


def _settings(tmp_path, **interface):
    return {
        "answering": {"warmup": False, "log_dir": str(tmp_path / "logs")},
        "interface": interface,
    }


# --- The defaults and the YAML section ------------------------------------------------


def test_the_constants_are_q7s_values():
    assert list(InterfaceSettings.__dataclass_fields__) == [key for key, _ in KEYS]
    for key, constant in KEYS:
        assert getattr(settings, constant) == Q7_DEFAULTS[key], key
    config = interface_settings({})
    for key, _ in KEYS:
        assert getattr(config, key) == Q7_DEFAULTS[key], key


def test_the_yaml_section_holds_the_python_defaults_in_order():
    section = yaml.safe_load(_yaml_text())["interface"]

    assert list(section) == [key for key, _ in KEYS]
    for key, constant in KEYS:
        assert section[key] == getattr(settings, constant), key
        assert type(section[key]) is int, key


def test_the_section_sits_between_embeddings_and_answering():
    sections = list(yaml.safe_load(_yaml_text()))

    position = sections.index("interface")
    assert sections[position - 1 : position + 2] == [
        "embeddings",
        "interface",
        "answering",
    ]


def _comments():
    """The comment lines above each key of the YAML section."""
    text = _yaml_text()
    lines = text[text.index("\ninterface:") + 1 :].splitlines()[1:]
    comments, found = [], {}
    for line in lines:
        if not line.strip():
            continue
        if not line.startswith("  "):
            break
        stripped = line.strip()
        if stripped.startswith("#"):
            comments.append(stripped.lstrip("#").strip())
        elif line[2] != " " and ":" in stripped:
            found[stripped.split(":")[0]] = comments
            comments = []
    return found


def test_every_key_has_a_comment_with_r15s_figures():
    comments = _comments()

    assert list(comments) == [key for key, _ in KEYS]
    assert all(comments[key] for key, _ in KEYS)
    assert len(comments["port"]) == 1
    render = " ".join(comments["render_dpi"])
    assert "A4" in render and "1240 x 1754" in render and "8.7 MB" in render
    budget = " ".join(comments["max_question_chars"])
    for figure in ("15,900", "6,450", "2.47", "800", "7,250", "8,192", "940", "2,300"):
        assert figure in budget, figure


def test_the_section_header_names_the_spec_the_command_and_the_package():
    text = _yaml_text()
    header = text[: text.index("\ninterface:")].rsplit("\n\n", 1)[-1]

    for part in ("spec 009", "companydocs-demo", "src/companydocs_rag/interface/"):
        assert part in header, part


# --- Invalid and valid values (R15) ---------------------------------------------------

INVALID = [
    *[("port", value) for value in ("8765", 8765.0, True, None, 0, 65536)],
    *[("render_dpi", value) for value in ("150", 150.0, True, None, 49, 301)],
    *[("max_question_chars", value) for value in ("1000", 2.5, True, None, 0)],
]


def _case_id(case):
    key, value = case
    return f"{key}={value!r}"


@pytest.mark.parametrize(("key", "value"), INVALID, ids=[_case_id(c) for c in INVALID])
def test_an_invalid_value_is_a_settings_error(key, value):
    with pytest.raises(SettingsError) as raised:
        interface_settings({"interface": {key: value}})

    assert (raised.value.section, raised.value.key) == ("interface", key)
    assert str(raised.value).startswith(f"interface.{key}: ")


@pytest.mark.parametrize(("key", "value"), INVALID, ids=[_case_id(c) for c in INVALID])
def test_an_invalid_value_makes_the_command_exit_2(
    tmp_path, monkeypatch, caplog, capsys, key, value
):
    write_settings(tmp_path, monkeypatch, _settings(tmp_path, **{key: value}))
    write_manifest(tmp_path / "index")
    (tmp_path / "raw").mkdir()
    spies = spy_on_main(monkeypatch)
    retriever, generator = FakeRetriever(simple_result()), Generator()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = cli.main(
            ["--index-dir", str(tmp_path / "index")],
            raw_dir=tmp_path / "raw",
            type_overrides=tmp_path / "geen-typen.csv",
            retriever=retriever,
            transport=generator.transport,
        )

    assert code == 2
    (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert error.startswith(f"Invalid settings: interface.{key}: ")
    assert capsys.readouterr().out == ""
    assert (spies.attempts, spies.servers) == (0, [])
    assert (retriever.calls, retriever.closed, generator.requests) == ([], 0, [])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("port", 1),
        ("port", 65535),
        ("render_dpi", 50),
        ("render_dpi", 300),
        ("max_question_chars", 1),
        ("max_question_chars", 1000000),
    ],
)
def test_both_ends_of_each_range_are_valid(key, value):
    found = getattr(interface_settings({"interface": {key: value}}), key)

    assert found == value
    assert type(found) is int


def test_an_absent_key_takes_its_default():
    config = interface_settings({"interface": {"render_dpi": 72}})

    assert config == InterfaceSettings(
        port=8765, render_dpi=72, max_question_chars=1000
    )


def test_the_settings_are_read_at_call_time(tmp_path, monkeypatch):
    found = []
    for maximum in (11, 22):
        write_settings(
            tmp_path, monkeypatch, _settings(tmp_path, max_question_chars=maximum)
        )
        found.append(interface_settings().max_question_chars)

    assert found == [11, 22]


def test_the_first_invalid_key_is_reported():
    with pytest.raises(SettingsError) as raised:
        interface_settings(
            {"interface": {"max_question_chars": 0, "render_dpi": 10, "port": 0}}
        )

    assert raised.value.key == "port"


# --- --port (R15) ---------------------------------------------------------------------


def test_port_replaces_the_setting_for_the_run(tmp_path, monkeypatch):
    configured, given = free_port(), free_port()
    write_settings(tmp_path, monkeypatch, _settings(tmp_path, port=configured))
    write_manifest(tmp_path / "index")
    (tmp_path / "raw").mkdir()
    stdout = Recorder()
    monkeypatch.setattr(sys, "stdout", stdout)
    spies = spy_on_main(monkeypatch)

    run = MainRun(
        ["--index-dir", str(tmp_path / "index"), "--port", str(given)],
        raw_dir=tmp_path / "raw",
        type_overrides=tmp_path / "geen-typen.csv",
        retriever=FakeRetriever(simple_result()),
        transport=Generator().transport,
    ).start()
    try:
        run.wait_for_url(stdout)
        assert stdout.getvalue() == f"http://127.0.0.1:{given}/\n"
        assert request(given, "GET", "/").status == 200
    finally:
        code = run.finish()

    assert code == 0
    assert [server["address"][1] for server in spies.servers] == [given]


@pytest.mark.parametrize("port", ["0", "65536", "abc", "-1", ""])
def test_a_port_outside_the_range_is_a_usage_error(port, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(["--port", port])

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("error:") == 1
    assert "--port" in captured.err
