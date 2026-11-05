"""``extract``'s exit codes (spec 008, D11, R8, R15, R18; Q1, Q3, Q6, A10)."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from tests.metadata._helpers import (
    LEASE,
    LOGGER,
    OTHER_TYPE,
    SHA_A,
    SHA_B,
    SHA_C,
    Generator,
    answer,
    candidate,
    chunk_record,
    extract,
    extraction_files,
    lease_text,
    log_records,
    messages,
    months,
    reply,
    use_settings,
    write_json_file,
    write_overrides,
)

GOOD = answer(
    notice_period=[candidate(months(3), "duration", "termijn van drie maanden")]
)


def _errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def _documents(tmp_path, *names, in_scope=True):
    """Documents named ``fictief-<name>.pdf``; their type through the
    override file."""
    for position, name in enumerate(names):
        sha = f"{position + 1:02d}" * 32
        write_json_file(
            tmp_path / "chunks",
            f"{name}.json",
            chunk_record(tmp_path / "los" / f"fictief-{name}.pdf", sha, lease_text()),
        )
    return write_overrides(
        tmp_path / "handmatig" / "types.csv",
        [(f"fictief-{name}.pdf", LEASE if in_scope else OTHER_TYPE) for name in names],
    )


# --- 0 --------------------------------------------------------------------------------


def test_extracted_and_not_in_scope_documents_exit_0(tmp_path):
    write_json_file(
        tmp_path / "chunks",
        "huur.json",
        chunk_record(tmp_path / "los" / "fictief-huur.pdf", SHA_A, lease_text()),
    )
    write_json_file(
        tmp_path / "chunks",
        "statuten.json",
        chunk_record(tmp_path / "los" / "fictief-statuten.pdf", SHA_B, lease_text()),
    )
    overrides = write_overrides(
        tmp_path / "handmatig" / "types.csv",
        [("fictief-huur.pdf", LEASE), ("fictief-statuten.pdf", OTHER_TYPE)],
    )

    assert extract(tmp_path, overrides=overrides, generator=Generator(reply(GOOD))) == 0


def test_an_in_scope_document_without_a_scoring_chunk_exits_0(tmp_path):
    write_json_file(
        tmp_path / "chunks",
        "leeg.json",
        chunk_record(tmp_path / "los" / "fictief-leeg.pdf", SHA_A, "Fictief."),
    )
    overrides = write_overrides(
        tmp_path / "handmatig" / "types.csv", [("fictief-leeg.pdf", LEASE)]
    )

    assert extract(tmp_path, overrides=overrides) == 0


def test_an_empty_chunks_folder_warns_and_exits_0(tmp_path, caplog):
    (tmp_path / "chunks").mkdir()
    generator = Generator()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert extract(tmp_path, generator=generator) == 0

    assert f"No chunk files found in {tmp_path / 'chunks'}" in messages(
        caplog, logging.WARNING
    )
    assert generator.requests == []
    assert not (tmp_path / "extracted").exists()


# --- 1 --------------------------------------------------------------------------------


def test_a_missing_chunks_folder_exits_1(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert extract(tmp_path) == 1

    assert _errors(caplog) == [f"Not a folder: {tmp_path / 'chunks'}"]


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (2, "2"),
        (1.0, "1.0"),
        ("1", '"1"'),
        (True, "true"),
        (None, "null"),
        ("missing", "missing"),
    ],
    ids=["2", "1.0", "string", "true", "null", "missing"],
)
def test_a_refused_schema_version_exits_1_before_any_request(
    tmp_path, monkeypatch, caplog, value, shown
):
    use_settings(tmp_path, monkeypatch, warmup=True)
    overrides = _documents(tmp_path, "a-huur", "c-huur")
    refused = chunk_record(tmp_path / "los" / "fictief-b-huur.pdf", SHA_C, lease_text())
    if value == "missing":
        del refused["schema_version"]
    else:
        refused["schema_version"] = value
    write_json_file(tmp_path / "chunks", "b-huur.json", refused)
    generator = Generator()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(tmp_path, overrides=overrides, generator=generator)

    assert code == 1
    assert _errors(caplog) == [
        f"Refused b-huur.json: schema_version is {shown}, expected 1",
        "Refused 1 chunk file(s); rerun companydocs-chunk",
    ]
    assert generator.requests == []
    assert not (tmp_path / "extracted").exists()
    assert not (tmp_path / "logs").exists()


def _invalid_record(tmp_path):
    record = chunk_record(tmp_path / "los" / "fictief-b-huur.pdf", SHA_C, lease_text())
    record["chunks"][0]["page_spans"][0]["page_number"] = "1"
    return json.dumps(record)


INVALID_FILES = {
    "not-utf-8": lambda tmp_path: b'{"schema_version": 1, "x": "\xff"}',
    "not-json": lambda tmp_path: b"{geen json",
    "not-an-object": lambda tmp_path: b"[1, 2]",
    "record-check": lambda tmp_path: _invalid_record(tmp_path).encode("utf-8"),
    "chunks-not-a-list": lambda tmp_path: json.dumps(
        {
            **chunk_record(tmp_path / "los" / "fictief-b-huur.pdf", SHA_C),
            "chunks": {},
        }
    ).encode("utf-8"),
}


@pytest.mark.parametrize("name", list(INVALID_FILES))
def test_an_unreadable_or_invalid_chunk_file_exits_1(
    tmp_path, monkeypatch, caplog, name
):
    use_settings(tmp_path, monkeypatch, warmup=True)
    overrides = _documents(tmp_path, "a-huur", "c-huur")
    (tmp_path / "chunks" / "b-huur.json").write_bytes(INVALID_FILES[name](tmp_path))
    generator = Generator()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(tmp_path, overrides=overrides, generator=generator)

    assert code == 1
    first, last = _errors(caplog)
    assert first.startswith("Could not read b-huur.json: ")
    assert last == "Refused 1 chunk file(s); rerun companydocs-chunk"
    assert generator.requests == []
    assert not (tmp_path / "extracted").exists()
    assert not (tmp_path / "logs").exists()


# --- 2 --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [[], ["extract", "--onbekend"], ["onbekend"], ["report"]],
    ids=["no-subcommand", "unknown-flag", "unknown-subcommand", "no-reference-date"],
)
def test_argparse_errors_exit_2(argv):
    from companydocs_rag.metadata import cli

    with pytest.raises(SystemExit) as raised:
        cli.main(argv)

    assert raised.value.code == 2


def test_invalid_settings_win_over_a_missing_chunks_folder(
    tmp_path, monkeypatch, caplog
):
    use_settings(tmp_path, monkeypatch, seed=-1)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert extract(tmp_path) == 2

    assert _errors(caplog) == [
        "Invalid settings: metadata.seed: must be an integer of at least 0, not -1"
    ]


def test_a_malformed_relations_file_wins_over_a_refused_chunk_file(tmp_path, caplog):
    (tmp_path / "chunks").mkdir()
    (tmp_path / "chunks" / "kapot.json").write_bytes(b"{geen json")
    relations = tmp_path / "handmatig" / "relaties.csv"
    relations.parent.mkdir()
    relations.write_text("Bestandsnaam;Relatie\n", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert extract(tmp_path, relations=relations) == 2

    (error,) = _errors(caplog)
    assert error.startswith(f"Invalid relations file {relations}: ")


# --- 3 --------------------------------------------------------------------------------


def _connect_error(request):
    raise httpx.ConnectError("fictief geweigerd", request=request)


def _connect_timeout(request):
    raise httpx.ConnectTimeout("fictief verlopen", request=request)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (_connect_error, "Generator unavailable at http://127.0.0.1:11434/v1: "),
        (_connect_timeout, "Generator unavailable at http://127.0.0.1:11434/v1: "),
        (
            httpx.Response(404, content=b"model niet gevonden"),
            "Generator unavailable at http://127.0.0.1:11434/v1: HTTP 404",
        ),
        (
            httpx.Response(500, content=b"fictieve fout"),
            "Generator failed at warm-up: HTTP 500",
        ),
    ],
    ids=["connect-error", "connect-timeout", "http-404", "http-500"],
)
def test_a_failed_warm_up_exits_3_without_an_extraction_request(
    tmp_path, monkeypatch, caplog, failure, message
):
    use_settings(tmp_path, monkeypatch, warmup=True)
    overrides = _documents(tmp_path, "huur")
    generator = Generator(failure)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(tmp_path, overrides=overrides, generator=generator)

    assert code == 3
    (body,) = generator.bodies
    assert body["max_tokens"] == 1
    (error,) = _errors(caplog)
    assert error.startswith(message)
    assert not (tmp_path / "extracted").exists()
    assert not (tmp_path / "logs").exists()
    info = messages(caplog, logging.INFO)
    assert any(line.startswith("Summary: documents ") for line in info)


def test_an_unreachable_generator_mid_run_stops_with_exit_3(tmp_path, caplog):
    overrides = _documents(tmp_path, "a-huur", "b-huur", "c-huur")
    generator = Generator(reply(GOOD), _connect_error, reply(GOOD))

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(tmp_path, overrides=overrides, generator=generator)

    assert code == 3
    assert len(generator.requests) == 2
    assert sorted(extraction_files(tmp_path / "extracted")) == ["a-huur.json"]
    first, failed = log_records(tmp_path / "logs")
    assert first["status"] == "ok"
    assert failed["file"]["path"].endswith("fictief-b-huur.pdf")
    assert failed["status"] == "error"
    assert failed["error"]["type"] == "generator_unavailable"
    assert failed["response"] is None
    (error,) = _errors(caplog)
    assert error.startswith("Generator unavailable at http://127.0.0.1:11434/v1: ")
    info = messages(caplog, logging.INFO)
    assert "Summary: documents extracted=1, partial=0, error=0, not_in_scope=0" in info
    assert "Summary: requests=2, batches ok=1, error=0" in info


def test_3_wins_over_4(tmp_path):
    overrides = _documents(tmp_path, "a-huur", "b-huur")
    generator = Generator(httpx.Response(500, content=b"fout"), _connect_error)

    assert extract(tmp_path, overrides=overrides, generator=generator) == 3
    files = extraction_files(tmp_path / "extracted")
    assert files["a-huur.json"]["extraction_status"] == "error"
    assert "b-huur.json" not in files


# --- 4 --------------------------------------------------------------------------------


def test_a_partial_document_exits_4(tmp_path, monkeypatch):
    use_settings(tmp_path, monkeypatch, selection_budget_chars=10)
    write_json_file(
        tmp_path / "chunks",
        "huur.json",
        chunk_record(
            tmp_path / "los" / "fictief-huur.pdf", SHA_A, lease_text(1), lease_text(2)
        ),
    )
    overrides = write_overrides(
        tmp_path / "handmatig" / "types.csv", [("fictief-huur.pdf", LEASE)]
    )
    generator = Generator(reply(GOOD), reply("geen json"))

    assert extract(tmp_path, overrides=overrides, generator=generator) == 4
    record = extraction_files(tmp_path / "extracted")["huur.json"]
    assert record["extraction_status"] == "partial"


def test_an_error_document_exits_4(tmp_path, caplog):
    overrides = _documents(tmp_path, "huur")
    generator = Generator(reply("geen json"))

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert extract(tmp_path, overrides=overrides, generator=generator) == 4

    assert "[1/1] huur.json: Huurovereenkomst (error)" in messages(caplog, logging.INFO)


def test_an_unexpected_exception_is_an_error_and_the_run_goes_on(
    tmp_path, monkeypatch, caplog
):
    from companydocs_rag.metadata import extract as extract_module

    overrides = _documents(tmp_path, "a-huur", "b-huur")
    original = extract_module.ExtractRun._document
    calls = []

    def flaky(self, label, chunk_file, resolution, relations):
        calls.append(chunk_file.relative)
        if chunk_file.relative == "a-huur.json":
            raise RuntimeError("fictieve storing")
        return original(self, label, chunk_file, resolution, relations)

    monkeypatch.setattr(extract_module.ExtractRun, "_document", flaky)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(tmp_path, overrides=overrides, generator=Generator(reply(GOOD)))

    assert code == 4
    assert calls == ["a-huur.json", "b-huur.json"]
    assert _errors(caplog) == [
        "Could not extract a-huur.json: RuntimeError: fictieve storing"
    ]
    assert sorted(extraction_files(tmp_path / "extracted")) == ["b-huur.json"]
