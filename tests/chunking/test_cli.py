"""The companydocs-chunk command (spec 005): K1, K14 and the output file.

Normalized files are fabricated in ``tmp_path`` (``_helpers.normalized_record``);
two are written by ``normalize_document`` from fabricated store records. Every
run passes both folders, so nothing under ``data/`` is read or written.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from companydocs_rag.chunking import cli
from companydocs_rag.chunking.chunker import CHUNK_MAX_CHARS
from companydocs_rag.ingestion import normalize_cli
from companydocs_rag.ingestion.normalization import normalize_document
from companydocs_rag.ingestion.settings import find_project_root, load_settings
from tests.chunking._helpers import (
    block,
    body,
    check_chunks,
    filler,
    lines,
    normalized_page,
    normalized_record,
)

LOGGER = "companydocs_rag.chunking.cli"


def _write(folder: Path, relative: str, record) -> Path:
    """Write one normalized file; a string is written as it is."""
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        record
        if isinstance(record, str)
        else json.dumps(record, indent=2, ensure_ascii=False)
    )
    path.write_text(text, encoding="utf-8")
    return path


def _run(normalized: Path, out: Path, *extra: str) -> int:
    return cli.main(
        ["--normalized-dir", str(normalized), "--chunks-dir", str(out), *extra]
    )


def _messages(caplog, level):
    return [r.getMessage() for r in caplog.records if r.levelno == level]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _tree(folder: Path):
    """Every file and folder under ``folder``, with the bytes of each file."""
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        if path.is_file()
        else None
        for path in sorted(folder.rglob("*"))
    }


def _written(out: Path):
    return sorted(path.relative_to(out).as_posix() for path in out.rglob("*.json"))


def _articles():
    return (
        block("Artikel 1 Begrippen", 500)
        + block("Artikel 2 Huurprijs", 500)
        + block("Artikel 3 Duur", 500)
    )


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """Point the settings at a YAML file written here, for one test."""

    def use(settings):
        path = tmp_path / "instellingen" / "instellingen.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            yaml.safe_dump(settings, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        monkeypatch.setenv("COMPANYDOCS_RAG_SETTINGS", str(path))
        load_settings.cache_clear()

    yield use
    load_settings.cache_clear()


# --- The command and its options (A1) -------------------------------------------------


def test_pyproject_registers_the_command():
    with (find_project_root() / "pyproject.toml").open("rb") as handle:
        scripts = tomllib.load(handle)["project"]["scripts"]

    assert scripts["companydocs-chunk"] == "companydocs_rag.chunking.cli:main"


def test_the_parser_has_the_two_folders_quiet_and_help_and_nothing_else():
    parser = cli.build_parser()

    options = sorted(
        option for action in parser._actions for option in action.option_strings
    )
    assert options == [
        "--chunks-dir",
        "--help",
        "--normalized-dir",
        "--quiet",
        "-h",
        "-q",
    ]
    assert [action for action in parser._actions if not action.option_strings] == []


def test_the_default_folders_are_under_data_interim_of_the_project_root(
    monkeypatch, tmp_path
):
    root = find_project_root()
    monkeypatch.chdir(tmp_path)

    args = cli.build_parser().parse_args([])

    assert args.normalized_dir == root / "data" / "interim" / "normalized"
    assert args.chunks_dir == root / "data" / "interim" / "chunks"
    assert args.normalized_dir == normalize_cli.default_normalized_dir()
    assert cli.DEFAULT_CHUNKS_RELATIVE_PATH == Path("data", "interim", "chunks")
    assert (args.quiet, cli.EXIT_ALL_WRITTEN) == (False, 0)
    assert (cli.EXIT_FILES_NOT_WRITTEN, cli.EXIT_CANNOT_START) == (1, 2)


# --- Normalized files in, chunk files out (A2) ----------------------------------------


def test_valid_files_mirror_the_input_and_the_others_are_refused_or_fail(
    tmp_path, caplog
):
    source, out = tmp_path / "normalized", tmp_path / "chunks"
    _write(source, "a/fictief-1.json", normalized_record(_articles()))
    _write(source, "a/b/fictief-2.json", normalized_record("Fictieve tekst."))
    _write(source, "fictief-3.json", normalized_record(status="failed", sha256=None))
    newer = normalized_record("Tekst.")
    newer["schema_version"] = 2
    _write(source, "oud/v2.json", newer)
    unversioned = normalized_record("Tekst.")
    del unversioned["schema_version"]
    _write(source, "oud/zonder.json", unversioned)
    _write(source, "kapot.json", "{geen geldige json")
    without_pages = normalized_record("Tekst.")
    del without_pages["pages"]
    _write(source, "ongeldig.json", without_pages)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = _run(source, out)

    assert code == 1
    assert _written(out) == ["a/b/fictief-2.json", "a/fictief-1.json", "fictief-3.json"]
    errors = _messages(caplog, logging.ERROR)
    assert (
        f"Refused {Path('oud', 'v2.json')}: schema_version is 2, expected 1" in errors
    )
    assert (
        f"Refused {Path('oud', 'zonder.json')}: schema_version is missing, expected 1"
        in errors
    )
    failed = [message for message in errors if message.startswith("Could not")]
    assert len(failed) == 2
    assert failed[0].startswith("Could not chunk kapot.json: JSONDecodeError: ")
    assert failed[1] == (
        "Could not chunk ongeldig.json: NormalizedRecordError: "
        "the record lacks the key 'pages'"
    )
    assert errors[-2:] == [
        "Refused 2 normalized file(s) whose schema_version is not 1; "
        "rerun companydocs-normalize",
        "2 normalized file(s) could not be chunked",
    ]
    assert len(errors) == 6
    for name in ("a/fictief-1.json", "a/b/fictief-2.json", "fictief-3.json"):
        check_chunks(_read(source / name), _read(out / name), max_chars=CHUNK_MAX_CHARS)


@pytest.mark.parametrize(
    ("value", "shown"),
    [(2, "2"), (1.0, "1.0"), ("1", '"1"'), (True, "true"), (None, "null")],
    ids=["two", "float", "string", "true", "null"],
)
def test_only_the_integer_1_is_accepted(tmp_path, caplog, value, shown):
    source, out = tmp_path / "normalized", tmp_path / "chunks"
    record = normalized_record("Tekst.")
    record["schema_version"] = value
    _write(source, "fictief.json", record)

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        code = _run(source, out)

    assert code == 1
    assert not out.exists()
    assert _messages(caplog, logging.ERROR) == [
        f"Refused fictief.json: schema_version is {shown}, expected 1",
        "Refused 1 normalized file(s) whose schema_version is not 1; "
        "rerun companydocs-normalize",
    ]


def _json_without(key):
    record = normalized_record("Tekst.")
    del record[key]
    return json.dumps(record)


def _json_with_page(key, value):
    record = normalized_record("Tekst.")
    record["pages"][0][key] = value
    return json.dumps(record)


def _json_with_file_hash(value):
    return json.dumps(normalized_record("Tekst.", sha256=value))


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("{geen geldige json", "JSONDecodeError: "),
        ("[1]", "NormalizedRecordError: the top level is not a JSON object"),
        (
            _json_without("normalized_sha256"),
            "NormalizedRecordError: the record lacks the key 'normalized_sha256'",
        ),
        (
            _json_with_file_hash(None),
            "NormalizedRecordError: 'file.sha256' is not a string, and only a "
            "failed document may have none",
        ),
        (
            _json_with_page("text_origin", "scan"),
            'NormalizedRecordError: pages[0].text_origin is "scan", not native, '
            "ocr or none",
        ),
    ],
    ids=[
        "not-json",
        "not-an-object",
        "no-hash",
        "processed-without-file-hash",
        "origin",
    ],
)
def test_a_file_that_is_not_a_valid_record_fails_and_the_rest_is_written(
    tmp_path, caplog, content, error
):
    source, out = tmp_path / "normalized", tmp_path / "chunks"
    _write(source, "fictief.json", content)
    _write(source, "goed.json", normalized_record("Tekst."))

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        code = _run(source, out)

    assert code == 1
    assert _written(out) == ["goed.json"]
    first, last = _messages(caplog, logging.ERROR)
    assert first.startswith(f"Could not chunk fictief.json: {error}")
    assert last == "1 normalized file(s) could not be chunked"


def test_an_empty_input_folder_is_a_warning_and_exits_0(tmp_path, caplog):
    source = tmp_path / "normalized"
    source.mkdir()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert _run(source, tmp_path / "chunks") == 0

    assert _messages(caplog, logging.WARNING) == [
        f"No normalized files found in {source}"
    ]
    assert _messages(caplog, logging.INFO)[0] == (
        "Summary: documents written=0, refused=0, failed=0"
    )
    assert not (tmp_path / "chunks").exists()


def test_file_names_starting_with_a_dot_are_skipped_but_dot_folders_are_read(
    tmp_path,
):
    source, out = tmp_path / "normalized", tmp_path / "chunks"
    _write(source, "fictief.json", normalized_record("Tekst."))
    _write(source, ".verborgen.json", normalized_record("Tekst."))
    _write(source, ".map/fictief.json", normalized_record("Tekst."))

    assert _run(source, out) == 0
    assert _written(out) == [".map/fictief.json", "fictief.json"]


def test_a_missing_normalized_folder_exits_2_before_anything_is_written(
    tmp_path, caplog
):
    not_a_folder = tmp_path / "bestand.json"
    not_a_folder.write_text("{}", encoding="utf-8")
    missing = tmp_path / "bestaat-niet"

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert _run(missing, tmp_path / "chunks") == 2
        assert _run(not_a_folder, tmp_path / "chunks") == 2

    assert _messages(caplog, logging.ERROR) == [
        f"Not a folder: {missing}",
        f"Not a folder: {not_a_folder}",
    ]
    assert _messages(caplog, logging.INFO) == []
    assert not (tmp_path / "chunks").exists()


def test_the_normalized_folder_is_checked_before_the_settings(
    tmp_path, caplog, settings_file
):
    settings_file({"chunking": {"chunk_target_chars": 0}})
    missing = tmp_path / "bestaat-niet"

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert _run(missing, tmp_path / "chunks") == 2

    assert _messages(caplog, logging.ERROR) == [f"Not a folder: {missing}"]


# Each R6 case: a key and a value that makes the settings invalid.
INVALID_SETTINGS = [
    ("heading_keywords", "Artikel"),
    ("heading_keywords", []),
    ("heading_keywords", [""]),
    ("heading_keywords", ["Artikel", 3]),
    ("heading_keywords", None),
    *[
        (key, value)
        for key, minimum in [
            ("heading_max_gap", 0),
            ("min_heading_run_length", 1),
            ("min_run_median_section_chars", 0),
            ("max_label_inherit_chars", 1),
            ("chunk_target_chars", 1),
            ("chunk_max_chars", 1),
            ("chunk_overlap_chars", 0),
            ("clause_label_max_chars", 1),
        ]
        for value in (minimum - 1, True, 2.5, "3", None)
    ],
]


@pytest.mark.parametrize(("key", "value"), INVALID_SETTINGS)
def test_invalid_settings_stop_the_command_before_it_writes_anything(
    tmp_path, caplog, settings_file, key, value
):
    settings_file({"chunking": {key: value}})
    source, out = tmp_path / "normalized", tmp_path / "chunks"
    _write(source, "fictief.json", normalized_record(_articles()))

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = _run(source, out)

    assert code == 2
    assert not out.exists()
    (error,) = _messages(caplog, logging.ERROR)
    assert error.startswith(f"Invalid settings: chunking.{key} ")
    assert _messages(caplog, logging.INFO) == []


def test_the_settings_file_is_read_by_the_command(tmp_path, settings_file):
    settings_file({"chunking": {"min_run_median_section_chars": 501}})
    source, out = tmp_path / "normalized", tmp_path / "chunks"
    _write(source, "fictief.json", normalized_record(_articles()))

    assert _run(source, out) == 0

    (run,) = _read(out / "fictief.json")["heading_evidence"]["runs"]
    assert run["reject_reason"] == "sections_too_short"


# --- The end of the run (R8) ----------------------------------------------------------


def test_the_run_ends_with_exactly_the_four_summary_lines(tmp_path, caplog):
    """Lines of 60 characters: the first section, 140 lines, is cut into
    pieces of 20 lines that start 18 lines apart; six start before 6000
    characters, two after."""
    long_section = block("Artikel 1 " + body(49), 8400, line_length=60)
    source, out = tmp_path / "normalized", tmp_path / "chunks"
    _write(
        source,
        "a.json",
        normalized_record(
            long_section
            + block("Artikel 2 Huurprijs", 500)
            + block("Artikel 3 Duur", 500)
        ),
    )
    _write(source, "b.json", normalized_record(lines("1 Een", "2 Twee", "3 Drie")))
    _write(
        source,
        "c.json",
        normalized_record(
            block("Artikel 1 Begrippen", 600) + block("Artikel 2 Huurprijs", 600)
        ),
    )
    _write(source, "d.json", normalized_record(lines(body(100), body(2500), body(100))))
    _write(source, "e.json", normalized_record(status="failed", sha256=None))
    refused = normalized_record("Tekst.")
    refused["schema_version"] = 2
    _write(source, "f.json", refused)
    _write(source, "g.json", "niet eens json")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert _run(source, out) == 1

    assert _messages(caplog, logging.INFO) == [
        "Summary: documents written=5, refused=1, failed=1",
        "Summary: routes structured=1, generic=4",
        "Summary: runs accepted=1, run_too_short=1, sections_too_short=1",
        "Summary: chunks total=15, heading_section=2, heading_section_part=6, "
        "heading_section_overflow=2, generic=5; exceeds_max_chars=1",
    ]


def test_quiet_leaves_only_warnings_and_errors(tmp_path):
    """Run as a process, where the logging set-up is the command's own."""
    source, out = tmp_path / "normalized", tmp_path / "chunks"
    _write(source, "fictief.json", normalized_record(_articles()))
    command = [
        sys.executable,
        "-m",
        "companydocs_rag.chunking.cli",
        "--normalized-dir",
        str(source),
        "--chunks-dir",
        str(out),
    ]

    loud = subprocess.run(command, capture_output=True, text=True, timeout=120)
    quiet = subprocess.run(
        [*command, "-q"], capture_output=True, text=True, timeout=120
    )

    assert (loud.returncode, quiet.returncode) == (0, 0)
    assert loud.stderr.splitlines() == [
        "INFO Summary: documents written=1, refused=0, failed=0",
        "INFO Summary: routes structured=1, generic=0",
        "INFO Summary: runs accepted=1, run_too_short=0, sections_too_short=0",
        "INFO Summary: chunks total=3, heading_section=3, heading_section_part=0, "
        "heading_section_overflow=0, generic=0; exceeds_max_chars=0",
    ]
    assert quiet.stderr == ""


# --- The output file (A3) -------------------------------------------------------------


def test_the_output_is_written_as_normalize_writes_its_files(tmp_path):
    section_sign = chr(0x00A7)  # U+00A7 SECTION SIGN
    record = normalized_record(
        block(f"{section_sign} 1 Begrippen", 500)
        + block(f"{section_sign} 2 Huurprijs", 500)
        + block(f"{section_sign} 3 Duur", 500)
    )
    source, out = tmp_path / "normalized", tmp_path / "chunks"
    _write(source, "fictief.json", record)

    assert _run(source, out) == 0

    written = (out / "fictief.json").read_text(encoding="utf-8")
    document = json.loads(written)
    assert written == json.dumps(document, indent=2, ensure_ascii=False)
    assert f'"clause_label": "{section_sign} 3 Duur"' in written
    normalize_cli._write_json(tmp_path / "zo-schrijft-normalize.json", document)
    assert (out / "fictief.json").read_bytes() == (
        tmp_path / "zo-schrijft-normalize.json"
    ).read_bytes()
    check_chunks(record, document, max_chars=CHUNK_MAX_CHARS)


def _store_record(status="processed", pages=()):
    """A fabricated page text store record (schema_version 2), as the parse
    pass writes it."""
    processed = status == "processed"
    return {
        "schema_version": 2,
        "file": {
            "path": "invoer/fictief.pdf",
            "filename": "fictief.pdf",
            "sha256": "ef" * 32 if processed else None,
        },
        "status": status,
        "page_count": len(pages),
        "language": "nld" if processed else None,
        "language_source": "index" if processed else None,
        "ocr_engine": None,
        "pages": list(pages),
    }


def _store_page(page_number, text, text_origin):
    native = text_origin == "native"
    return {
        "page_number": page_number,
        "text": text,
        "source": "native" if native else "no_text_layer",
        "tier": "Clean" if native else None,
        "text_origin": text_origin,
        "language": "nld",
        "language_source": "index",
        "ocr_mean_confidence": 90.0 if text_origin == "ocr" else None,
        "no_text_reason": "ocr_empty" if text_origin == "none" else None,
    }


def test_documents_as_normalize_writes_them_are_chunked(tmp_path):
    """A failed document, written without pages, gets an output like any
    other (Q1); a processed one is chunked over its native and OCR pages."""
    failed = normalize_document(_store_record(status="failed"))
    processed = normalize_document(
        _store_record(
            pages=[
                _store_page(1, _articles().rstrip("\n"), "native"),
                _store_page(2, "", "none"),
                _store_page(3, filler(300).rstrip("\n"), "ocr"),
            ]
        )
    )
    source, out = tmp_path / "normalized", tmp_path / "chunks"
    _write(source, "mislukt.json", failed)
    _write(source, "verwerkt.json", processed)

    assert _run(source, out) == 0

    document = _read(out / "mislukt.json")
    assert failed["pages"] == []
    assert document["file"] == failed["file"]
    assert (document["status"], document["route"]) == ("failed", "generic")
    assert (document["chunk_count"], document["chunks"]) == (0, [])
    chunked = _read(out / "verwerkt.json")
    check_chunks(processed, chunked, max_chars=CHUNK_MAX_CHARS)
    assert chunked["normalized_hash"] == processed["normalized_sha256"]
    assert chunked["empty_pages"] == [2]
    assert chunked["route"] == "structured"
    assert chunked["chunks"][-1]["text_origins"] == ["native", "ocr"]


# --- Reruns and the input (A12) -------------------------------------------------------


def _mixed_input(source: Path) -> None:
    _write(
        source,
        "a/fictief.json",
        normalized_record(
            _articles(),
            normalized_page(2, ""),
            normalized_page(3, filler(2600), text_origin="ocr"),
        ),
    )
    _write(source, "b/generiek.json", normalized_record(filler(3000)))
    _write(source, "b/mislukt.json", normalized_record(status="failed", sha256=None))


def test_a_second_run_writes_byte_identical_files(tmp_path):
    source, out = tmp_path / "normalized", tmp_path / "chunks"
    _mixed_input(source)

    assert _run(source, out) == 0
    first = _tree(out)
    assert _run(source, out) == 0

    assert _tree(out) == first
    assert sorted(name for name, data in first.items() if data is not None) == [
        "a/fictief.json",
        "b/generiek.json",
        "b/mislukt.json",
    ]


def test_the_run_changes_no_input_and_writes_nothing_outside_the_chunks_folder(
    tmp_path,
):
    source = tmp_path / "normalized"
    out = tmp_path / "uit" / "chunks"
    _mixed_input(source)
    before = _tree(tmp_path)

    assert _run(source, out) == 0

    after = _tree(tmp_path)
    outside = {
        name: data
        for name, data in after.items()
        if name not in ("uit", "uit/chunks") and not name.startswith("uit/chunks/")
    }
    assert outside == before
    for name in ("a/fictief.json", "b/generiek.json", "b/mislukt.json"):
        check_chunks(_read(source / name), _read(out / name), max_chars=CHUNK_MAX_CHARS)
