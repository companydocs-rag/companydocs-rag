"""The extraction file and the log (spec 008, D10, R16, R17; Q2, A11).

The clock is fixed by replacing ``extract._utc_now``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from companydocs_rag.metadata import extract as extract_module
from tests.metadata._helpers import (
    LEASE,
    SHA_A,
    SHA_B,
    Generator,
    answer,
    candidate,
    chunk,
    chunk_record,
    extract,
    log_records,
    months,
    provision,
    reply,
    use_settings,
    write_json_file,
    write_overrides,
)

FIXED = datetime(2031, 2, 3, 4, 5, 6, 789012, tzinfo=timezone.utc)
E_DIAERESIS = chr(0x00EB)  # U+00EB LATIN SMALL LETTER E WITH DIAERESIS

TOP_KEYS = [
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
    "other_termination_provisions",
]
BATCH_KEYS = [
    "batch_index",
    "chunk_ids",
    "chars",
    "status",
    "error_type",
    "prompt_tokens",
    "completion_tokens",
    "finish_reason",
    "duration_s",
]
FIELD_NAMES = [
    "signing_date",
    "commencement_date",
    "initial_term",
    "initial_term_end",
    "renewal",
    "notice_period",
]
CANDIDATE_KEYS = [
    "value",
    "value_kind",
    "as_written",
    "quote",
    "quote_verified",
    "value_valid",
    "counted",
    "batch_index",
    "chunk_id",
    "page_spans",
]
PROVISION_KEYS = [
    "party",
    "as_written",
    "quote",
    "quote_verified",
    "batch_index",
    "chunk_id",
    "page_spans",
]
LOG_KEYS = [
    "schema_version",
    "timestamp_utc",
    "file",
    "batch_index",
    "batch_count",
    "chunk_ids",
    "prompt",
    "request",
    "response",
    "chars_per_prompt_token",
    "context_nearly_full",
    "duration_s",
    "status",
    "error",
]
TEXT = (
    f"De fictieve huur gaat in op 1 april 2030.\nOpzegging na be{E_DIAERESIS}indiging "
    "met een termijn van drie maanden."
)
GOOD = answer(
    notice_period=[
        candidate(months(3), "duration", "termijn van drie maanden"),
        candidate(months(2), "duration", "elders"),
    ],
    other_termination_provisions=[
        provision("tenant", f"opzegging na be{E_DIAERESIS}indiging"),
        provision("landlord", "elders"),
    ],
)


@pytest.fixture
def fixed_clock(monkeypatch):
    monkeypatch.setattr(extract_module, "_utc_now", lambda: FIXED)


def _lease(tmp_path, relative="a/b/huur.json"):
    record = chunk_record(
        tmp_path / "los" / f"fictief-{E_DIAERESIS}-huur.pdf",
        SHA_A,
        chunk(SHA_A, 0, TEXT, page_start=2, page_end=3),
        language="nld",
        language_source="override",
    )
    write_json_file(tmp_path / "chunks", relative, record)
    overrides = write_overrides(
        tmp_path / "handmatig" / "types.csv",
        [(f"fictief-{E_DIAERESIS}-huur.pdf", LEASE)],
    )
    return record, overrides


# --- The extraction file (R16) --------------------------------------------------------


def test_the_extraction_file_has_d10s_keys_in_order(tmp_path, fixed_clock):
    record, overrides = _lease(tmp_path)

    assert extract(tmp_path, overrides=overrides, generator=Generator(reply(GOOD))) == 0

    path = tmp_path / "extracted" / "a" / "b" / "huur.json"
    text = path.read_text(encoding="utf-8")
    written = json.loads(text)
    assert text == json.dumps(written, indent=2, ensure_ascii=False)
    assert f"fictief-{E_DIAERESIS}-huur.pdf" in text
    assert list(written) == TOP_KEYS
    assert (written["schema_version"], written["extractor_version"]) == (1, 1)
    assert written["file"] == record["file"]
    assert (written["language"], written["language_source"]) == ("nld", "override")
    assert (written["model"], written["reasoning_effort"]) == (
        "companydocs-gemma4-e4b-8k",
        "high",
    )
    (batch,) = written["batches"]
    assert list(batch) == BATCH_KEYS
    assert batch["chunk_ids"] == [record["chunks"][0]["chunk_id"]]
    assert batch["chars"] == len(TEXT)
    assert isinstance(batch["duration_s"], float)
    assert list(written["fields"]) == FIELD_NAMES
    for name in FIELD_NAMES:
        assert list(written["fields"][name]) == ["status", "candidates"]
    found = written["fields"]["notice_period"]["candidates"]
    assert [list(item) for item in found] == [CANDIDATE_KEYS, CANDIDATE_KEYS]
    assert found[0]["page_spans"] == record["chunks"][0]["page_spans"]
    assert (found[0]["counted"], found[1]["counted"]) == (True, False)
    provisions = written["other_termination_provisions"]
    assert [list(item) for item in provisions] == [PROVISION_KEYS, PROVISION_KEYS]
    assert [item["quote_verified"] for item in provisions] == [True, False]


def test_an_existing_extraction_file_is_replaced(tmp_path):
    _, overrides = _lease(tmp_path, relative="huur.json")
    write_json_file(tmp_path / "extracted", "huur.json", {"oud": True})

    assert extract(tmp_path, overrides=overrides) == 0

    written = json.loads((tmp_path / "extracted" / "huur.json").read_text("utf-8"))
    assert list(written) == TOP_KEYS


# --- The log (R17) --------------------------------------------------------------------


def test_one_compact_line_per_extraction_request(tmp_path, monkeypatch, fixed_clock):
    use_settings(tmp_path, monkeypatch, warmup=True)
    record, overrides = _lease(tmp_path)
    generator = Generator(
        reply("warm"), reply(GOOD, reasoning="Fictieve redenering.", prompt_tokens=1000)
    )

    assert extract(tmp_path, overrides=overrides, generator=generator) == 0

    (path,) = (tmp_path / "logs").iterdir()
    assert path.name == "extract-20310203T040506Z.jsonl"
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    (line,) = raw.decode("utf-8").splitlines()
    entry = json.loads(line)
    assert line == json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    assert list(entry) == LOG_KEYS
    assert entry["schema_version"] == 1
    assert entry["timestamp_utc"] == "2031-02-03T04:05:06.789Z"
    assert entry["file"] == record["file"]
    assert (entry["batch_index"], entry["batch_count"]) == (1, 1)
    assert entry["chunk_ids"] == [record["chunks"][0]["chunk_id"]]
    assert (
        entry["prompt"]["system_file"] == "config/prompts/metadata_extract_system.txt"
    )
    assert len(entry["prompt"]["system_sha256"]) == 64
    assert entry["request"] == generator.bodies[1]
    assert entry["response"] == {
        "content": GOOD,
        "reasoning": "Fictieve redenering.",
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 12,
            "cached_tokens": None,
        },
    }
    characters = sum(len(m["content"]) for m in generator.bodies[1]["messages"])
    assert entry["chars_per_prompt_token"] == characters / 1000
    assert entry["context_nearly_full"] is False
    assert (entry["status"], entry["error"]) == ("ok", None)


def test_context_nearly_full_is_logged_only(tmp_path, monkeypatch):
    use_settings(tmp_path, monkeypatch, context_window=4999)
    _, overrides = _lease(tmp_path, relative="huur.json")

    assert (
        extract(
            tmp_path,
            overrides=overrides,
            generator=Generator(reply(GOOD, prompt_tokens=1000)),
        )
        == 0
    )

    (entry,) = log_records(tmp_path / "logs")
    assert entry["context_nearly_full"] is True
    assert entry["status"] == "ok"


def test_a_failed_request_is_logged_without_a_response(tmp_path):
    import httpx

    _, overrides = _lease(tmp_path, relative="huur.json")
    generator = Generator(httpx.Response(500, content=b"fictieve fout"))

    assert extract(tmp_path, overrides=overrides, generator=generator) == 4

    (entry,) = log_records(tmp_path / "logs")
    assert entry["response"] is None
    assert (entry["chars_per_prompt_token"], entry["context_nearly_full"]) == (
        None,
        None,
    )
    assert entry["status"] == "error"
    assert entry["error"] == {
        "type": "http_error",
        "message": "HTTP 500: fictieve fout",
    }
    assert entry["request"] == generator.bodies[0]


def test_a_second_run_in_the_same_second_writes_its_own_log(tmp_path, fixed_clock):
    _, overrides = _lease(tmp_path, relative="huur.json")

    for _ in range(2):
        assert extract(tmp_path, overrides=overrides) == 0

    assert sorted(path.name for path in (tmp_path / "logs").iterdir()) == [
        "extract-20310203T040506Z-2.jsonl",
        "extract-20310203T040506Z.jsonl",
    ]


def test_no_log_without_an_extraction_request(tmp_path, monkeypatch):
    use_settings(tmp_path, monkeypatch, warmup=True)
    write_json_file(
        tmp_path / "chunks",
        "leeg.json",
        chunk_record(tmp_path / "los" / "fictief-leeg.pdf", SHA_B, "Fictief."),
    )
    overrides = write_overrides(
        tmp_path / "handmatig" / "types.csv", [("fictief-leeg.pdf", LEASE)]
    )
    generator = Generator()

    assert extract(tmp_path, overrides=overrides, generator=generator) == 0

    assert generator.requests == []
    assert not (tmp_path / "logs").exists()
