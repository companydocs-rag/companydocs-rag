"""``report``'s exit codes and the clock (spec 008, D11, D12, R19; Q2, A14)."""

from __future__ import annotations

import ast
import json
import logging

import pytest

from companydocs_rag.ingestion.settings import find_project_root
from tests.metadata._helpers import (
    LOGGER,
    SHA_A,
    SHA_B,
    extraction_record,
    found,
    report,
    use_settings,
    write_json_file,
)

PACKAGE = find_project_root() / "src" / "companydocs_rag" / "metadata"
# The modules ``report`` uses; test_hygiene checks the list against a run.
REPORT_MODULES = [
    "__init__.py",
    "cli.py",
    "dates.py",
    "doctype.py",
    "fields.py",
    "files.py",
    "relations.py",
    "report.py",
    "settings.py",
    "texts.py",
]
# The extraction modules that may read a clock (Q2).
CLOCK_MODULES = ["extract.py"]


def _errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def _valid(tmp_path):
    write_json_file(
        tmp_path / "extracted",
        "a.json",
        extraction_record(
            "bron/fictief-a.pdf",
            SHA_A,
            fields={"signing_date": found("2031-01-01", "date")},
        ),
    )


# --- 0 --------------------------------------------------------------------------------


def test_a_valid_folder_exits_0_and_writes_the_three_files(tmp_path, caplog):
    _valid(tmp_path)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert report(tmp_path) == 0

    folder = tmp_path / "rapport"
    assert sorted(path.name for path in folder.iterdir()) == [
        "followup.csv",
        "followup.json",
        "followup.md",
    ]
    wrote = [m for m in caplog.messages if m.startswith("Wrote ")]
    assert wrote == [
        f"Wrote {folder / 'followup.md'}",
        f"Wrote {folder / 'followup.json'}",
        f"Wrote {folder / 'followup.csv'}",
    ]


def test_nothing_in_scope_is_a_table_without_rows(tmp_path):
    write_json_file(
        tmp_path / "extracted",
        "statuten.json",
        extraction_record(
            "bron/fictief-statuten.pdf",
            SHA_B,
            document_type="Statuten",
            status="not_in_scope",
            batches=[],
        ),
    )

    assert report(tmp_path) == 0

    markdown = (tmp_path / "rapport" / "followup.md").read_text(encoding="utf-8")
    lines = markdown.split("\n")
    assert lines[1].startswith("|---|")
    assert lines[2] == ""
    assert lines[3].startswith("Referentiedatum: 01-09-2031.")
    summary = json.loads((tmp_path / "rapport" / "followup.json").read_text("utf-8"))
    assert summary["documents"] == []


# --- 1 --------------------------------------------------------------------------------


def test_a_missing_folder_exits_1(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert report(tmp_path) == 1

    assert _errors(caplog) == [f"Not a folder: {tmp_path / 'extracted'}"]
    assert not (tmp_path / "rapport").exists()


def test_a_folder_without_files_exits_1(tmp_path, caplog):
    (tmp_path / "extracted").mkdir()
    (tmp_path / "extracted" / ".verborgen.json").write_text("{}", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert report(tmp_path) == 1

    assert _errors(caplog) == [f"No extraction files found in {tmp_path / 'extracted'}"]
    assert not (tmp_path / "rapport").exists()


def _without(key):
    record = extraction_record("bron/fictief-b.pdf", SHA_B)
    del record[key]
    return record


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            {**extraction_record("bron/fictief-b.pdf", SHA_B), "schema_version": 2},
            "Refused b.json: schema_version is 2, expected 1",
        ),
        (
            _without("schema_version"),
            "Refused b.json: schema_version is missing, expected 1",
        ),
        ("{geen json", "Could not read b.json: "),
        ("[]", "Could not read b.json: "),
        (_without("batches"), "Could not read b.json: "),
        (_without("model"), "Could not read b.json: "),
    ],
    ids=[
        "version-2",
        "version-missing",
        "not-json",
        "not-an-object",
        "no-batches",
        "no-model",
    ],
)
def test_a_refused_extraction_file_exits_1_and_writes_nothing(
    tmp_path, caplog, content, message
):
    _valid(tmp_path)
    write_json_file(tmp_path / "extracted", "b.json", content)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert report(tmp_path) == 1

    (error,) = _errors(caplog)
    assert error.startswith(message)
    assert not (tmp_path / "rapport").exists()


# --- 2 --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [[], ["--reference-date", "2031-02-30"], ["--reference-date", "30-09-2031"]],
    ids=["missing", "not-a-real-date", "wrong-form"],
)
def test_a_bad_reference_date_exits_2(tmp_path, arguments):
    from companydocs_rag.metadata import cli

    _valid(tmp_path)

    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "report",
                *arguments,
                "--extracted-dir",
                str(tmp_path / "extracted"),
                "--report-dir",
                str(tmp_path / "rapport"),
            ]
        )

    assert raised.value.code == 2
    assert not (tmp_path / "rapport").exists()


def test_invalid_settings_exit_2(tmp_path, monkeypatch, caplog):
    use_settings(tmp_path, monkeypatch, amendment_types=[""])
    _valid(tmp_path)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert report(tmp_path) == 2

    (error,) = _errors(caplog)
    assert error.startswith("Invalid settings: metadata.amendment_types: ")
    assert not (tmp_path / "rapport").exists()


def test_amendment_types_are_read_when_report_runs(tmp_path, monkeypatch):
    _valid(tmp_path)
    use_settings(tmp_path, monkeypatch, amendment_types=["Huurovereenkomst"])

    assert report(tmp_path) == 0

    summary = json.loads((tmp_path / "rapport" / "followup.json").read_text("utf-8"))
    (document,) = summary["documents"]
    assert document["derived"]["expiry_date"]["reason"] == "amendment"


# --- The clock (Q2) -------------------------------------------------------------------

_CLOCK_CALLS = {
    "datetime": {"now", "utcnow", "today"},
    "date": {"today"},
    "time": {"time", "time_ns", "localtime", "gmtime", "monotonic", "perf_counter"},
}


def _terminal_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def clock_reads(source):
    """The clock reads in ``source``: calls such as ``datetime.now()`` or
    ``time.monotonic()``, and clock functions imported from ``time``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = _terminal_name(node.func.value)
            if node.func.attr in _CLOCK_CALLS.get(owner, set()):
                found.append(f"{owner}.{node.func.attr}")
        elif isinstance(node, ast.ImportFrom) and node.module == "time":
            found += [
                f"time.{alias.name}"
                for alias in node.names
                if alias.name in _CLOCK_CALLS["time"]
            ]
    return found


def test_the_scan_finds_every_kind_of_clock_read():
    source = (
        "import time\nimport datetime as dt\nfrom datetime import date, datetime\n"
        "from time import perf_counter\n"
        "a = datetime.now()\nb = dt.datetime.utcnow()\nc = date.today()\n"
        "d = time.monotonic()\ne = time.gmtime()\n"
    )

    assert sorted(clock_reads(source)) == [
        "date.today",
        "datetime.now",
        "datetime.utcnow",
        "time.gmtime",
        "time.monotonic",
        "time.perf_counter",
    ]


def test_only_the_extraction_modules_read_a_clock():
    reading = {
        path.name
        for path in PACKAGE.glob("*.py")
        if clock_reads(path.read_text(encoding="utf-8"))
    }

    assert reading == set(CLOCK_MODULES)
    assert not reading & set(REPORT_MODULES)
