"""Batch errors and ``extraction_status`` (spec 008, D9, R12, R14, Q6, A9;
review F3)."""

from __future__ import annotations

import json
import logging
import math

import httpx
import pytest

from companydocs_rag.ingestion.settings import find_project_root
from companydocs_rag.metadata import checks
from companydocs_rag.metadata.request import user_message
from tests.metadata._helpers import (
    LEASE,
    LOGGER,
    SHA_A,
    SHA_B,
    Generator,
    answer,
    candidate,
    chunk,
    chunk_record,
    extract,
    extraction_files,
    lease_text,
    log_records,
    messages,
    months,
    provision,
    reply,
    use_settings,
    write_json_file,
    write_overrides,
)

PROMPT = (
    find_project_root() / "config" / "prompts" / "metadata_extract_system.txt"
).read_text(encoding="utf-8")
TEXT = lease_text()
QUOTE = "termijn van drie maanden"
GOOD = answer(notice_period=[candidate(months(3), "duration", QUOTE)])


def _lease(tmp_path, *documents):
    """In-scope documents (name, record) through the override file."""
    for name, record in documents:
        write_json_file(tmp_path / "chunks", f"{name}.json", record)
    return write_overrides(
        tmp_path / "handmatig" / "types.csv",
        [(f"fictief-{name}.pdf", LEASE) for name, _ in documents],
    )


def _document(tmp_path, name="huur", sha=SHA_A, *texts, **kwargs):
    return name, chunk_record(
        tmp_path / "los" / f"fictief-{name}.pdf", sha, *(texts or (TEXT,)), **kwargs
    )


def _run(tmp_path, *replies):
    overrides = _lease(tmp_path, _document(tmp_path))
    generator = Generator(*replies)
    code = extract(tmp_path, overrides=overrides, generator=generator)
    return code, extraction_files(tmp_path / "extracted")["huur.json"], generator


def _characters():
    return len(PROMPT) + len(user_message([chunk(SHA_A, 0, TEXT)]))


# --- Batch error types (D9, R12) ------------------------------------------------------


def _status(status, body=b"fictieve fout"):
    return httpx.Response(status, content=body)


def _request_error(kind):
    def respond(request):
        raise kind("fictief", request=request)

    return respond


FAILURES = {
    "http-400": (_status(400), "http_error"),
    "http-500": (_status(500), "http_error"),
    "read-timeout": (_request_error(httpx.ReadTimeout), "read_timeout"),
    "transport": (_request_error(httpx.RemoteProtocolError), "transport_error"),
    "malformed": (httpx.Response(200, content=b"geen json {"), "malformed_response"),
}


@pytest.mark.parametrize("name", list(FAILURES))
def test_a_failed_request_is_a_batch_error_without_a_response(tmp_path, caplog, name):
    response, error_type = FAILURES[name]

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code, record, _ = _run(tmp_path, response)

    assert code == 4
    assert record["extraction_status"] == "error"
    (batch,) = record["batches"]
    assert (batch["status"], batch["error_type"]) == ("error", error_type)
    assert (
        batch["prompt_tokens"],
        batch["completion_tokens"],
        batch["finish_reason"],
    ) == (None, None, None)
    assert all(not field["candidates"] for field in record["fields"].values())
    warnings = messages(caplog, logging.WARNING)
    assert any(w.startswith(f"huur.json: batch 1/1: {error_type}: ") for w in warnings)


def _without(key):
    content = json.loads(answer())
    del content[key]
    return json.dumps(content)


SCHEMA_MISMATCHES = {
    "field-key-missing": _without("renewal"),
    "extra-top-level-key": json.dumps({**json.loads(answer()), "fictief": []}),
    "field-not-a-list": json.dumps({**json.loads(answer()), "renewal": {}}),
    "candidate-without-quote": json.dumps(
        {
            **json.loads(answer()),
            "notice_period": [
                {
                    key: value
                    for key, value in candidate(months(3), "duration", QUOTE).items()
                    if key != "quote"
                }
            ],
        }
    ),
    "candidate-extra-key": json.dumps(
        {
            **json.loads(answer()),
            "notice_period": [{**candidate(months(3), "duration", QUOTE), "pagina": 1}],
        }
    ),
    "as-written-not-a-string": json.dumps(
        {
            **json.loads(answer()),
            "notice_period": [
                {**candidate(months(3), "duration", QUOTE), "as_written": 3}
            ],
        }
    ),
    "party-not-one-of-four": answer(
        other_termination_provisions=[provision("tenant and landlord", QUOTE)]
    ),
    "not-an-object": json.dumps([]),
}


@pytest.mark.parametrize("name", list(SCHEMA_MISMATCHES))
def test_a_content_without_the_schemas_structure_is_a_schema_mismatch(tmp_path, name):
    code, record, _ = _run(tmp_path, reply(SCHEMA_MISMATCHES[name]))

    assert code == 4
    (batch,) = record["batches"]
    assert (batch["status"], batch["error_type"]) == ("error", "schema_mismatch")
    assert all(not field["candidates"] for field in record["fields"].values())
    assert record["other_termination_provisions"] == []


@pytest.mark.parametrize(
    ("respond", "error_type"),
    [
        (reply(GOOD, finish_reason="length"), "generation_truncated"),
        (reply("geen json"), "invalid_json"),
        (reply(empty=True), "invalid_json"),
        (reply(" \n\t "), "invalid_json"),
    ],
    ids=["length", "not-json", "null", "whitespace"],
)
def test_an_unusable_response_is_a_batch_error_that_keeps_its_usage(
    tmp_path, respond, error_type
):
    code, record, _ = _run(tmp_path, respond)

    assert code == 4
    (batch,) = record["batches"]
    assert (batch["status"], batch["error_type"]) == ("error", error_type)
    assert batch["prompt_tokens"] == math.ceil(_characters() / 3)
    assert batch["completion_tokens"] == 12
    assert batch["finish_reason"] in ("stop", "length")
    assert record["fields"]["notice_period"] == {
        "status": "not_found",
        "candidates": [],
    }


NESTED = "[" * 100_000 + "]" * 100_000


def test_a_content_nested_too_deeply_to_parse_is_invalid_content():
    with pytest.raises(checks.InvalidContent, match="^the content is not JSON: "):
        checks.parse_content(NESTED)


def test_any_exception_while_the_content_is_parsed_is_invalid_content(monkeypatch):
    def broken(text):
        raise RuntimeError("fictieve parserfout")

    monkeypatch.setattr(checks.json, "loads", broken)

    with pytest.raises(
        checks.InvalidContent, match="^the content is not JSON: fictieve parserfout$"
    ):
        checks.parse_content("{}")


def test_a_content_nested_too_deeply_is_an_invalid_json_batch_with_its_record(
    tmp_path, caplog
):
    overrides = _lease(
        tmp_path,
        _document(tmp_path, "a-huur", SHA_A),
        _document(tmp_path, "b-huur", SHA_B),
    )
    generator = Generator(reply(NESTED), reply(GOOD))

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(tmp_path, overrides=overrides, generator=generator)

    assert code == 4
    files = extraction_files(tmp_path / "extracted")
    assert files["a-huur.json"]["extraction_status"] == "error"
    (batch,) = files["a-huur.json"]["batches"]
    assert (batch["status"], batch["error_type"]) == ("error", "invalid_json")
    assert batch["prompt_tokens"] == math.ceil(
        (len(PROMPT) + len(user_message([chunk(SHA_A, 0, TEXT)]))) / 3
    )
    assert files["b-huur.json"]["extraction_status"] == "extracted"
    failed, ok = log_records(tmp_path / "logs")
    assert failed["file"]["path"].endswith("fictief-a-huur.pdf")
    assert (failed["status"], failed["error"]["type"]) == ("error", "invalid_json")
    assert failed["error"]["message"].startswith("the content is not JSON: ")
    assert failed["response"]["content"] == NESTED
    assert ok["status"] == "ok"
    assert messages(caplog, logging.ERROR) == []
    warnings = messages(caplog, logging.WARNING)
    assert any(
        w.startswith("a-huur.json: batch 1/1: invalid_json: the content is not JSON: ")
        for w in warnings
    )
    info = messages(caplog, logging.INFO)
    assert any(
        line.startswith("[1/2] a-huur.json: batch 1/1: error, ") for line in info
    )
    assert "Summary: documents extracted=1, partial=0, error=1, not_in_scope=0" in info
    assert "Summary: requests=2, batches ok=1, error=1" in info


@pytest.mark.parametrize(
    ("content", "above", "error_type"),
    [
        (GOOD, False, None),
        (GOOD, True, "prompt_truncation_suspected"),
        ("geen json", True, "prompt_truncation_suspected"),
    ],
    ids=["equal", "just-above", "just-above-not-json"],
)
def test_the_tripwire(tmp_path, monkeypatch, content, above, error_type):
    limit = _characters() / 100
    if above:
        limit = math.nextafter(limit, 0.0)
    use_settings(tmp_path, monkeypatch, max_chars_per_prompt_token=limit)

    code, record, _ = _run(tmp_path, reply(content, prompt_tokens=100))

    (batch,) = record["batches"]
    assert batch["error_type"] == error_type
    assert batch["prompt_tokens"] == 100
    assert code == (4 if above else 0)
    if not above:
        assert record["fields"]["notice_period"]["status"] == "found"


# --- extraction_status (D9, R14, Q6) --------------------------------------------------


def test_every_batch_ok_is_extracted(tmp_path):
    code, record, _ = _run(tmp_path, reply(GOOD))

    assert code == 0
    assert record["extraction_status"] == "extracted"
    assert record["fields"]["notice_period"]["status"] == "found"
    (batch,) = record["batches"]
    assert (batch["status"], batch["error_type"]) == ("ok", None)


def test_one_of_two_batches_failing_is_partial(tmp_path, monkeypatch):
    texts = [TEXT, "De looptijd en de opzegging, fictief, ingang in 2031."]
    use_settings(tmp_path, monkeypatch, selection_budget_chars=len(TEXT))
    overrides = _lease(tmp_path, _document(tmp_path, "huur", SHA_A, *texts))
    generator = Generator(reply(GOOD), _status(500))

    assert extract(tmp_path, overrides=overrides, generator=generator) == 4

    record = extraction_files(tmp_path / "extracted")["huur.json"]
    assert record["extraction_status"] == "partial"
    assert [batch["status"] for batch in record["batches"]] == ["ok", "error"]
    assert record["fields"]["notice_period"]["status"] == "found"
    assert record["fields"]["notice_period"]["candidates"][0]["value"] == months(3)


def test_all_batches_failing_is_error(tmp_path, monkeypatch):
    texts = [TEXT, "De looptijd en de opzegging, fictief, ingang in 2031."]
    use_settings(tmp_path, monkeypatch, selection_budget_chars=len(TEXT))
    overrides = _lease(tmp_path, _document(tmp_path, "huur", SHA_A, *texts))
    generator = Generator(reply("geen json"), _status(500))

    assert extract(tmp_path, overrides=overrides, generator=generator) == 4

    record = extraction_files(tmp_path / "extracted")["huur.json"]
    assert record["extraction_status"] == "error"
    assert [batch["error_type"] for batch in record["batches"]] == [
        "invalid_json",
        "http_error",
    ]


def test_a_failed_chunk_record_in_scope_is_error_without_a_request(tmp_path):
    overrides = _lease(
        tmp_path,
        (
            "mislukt",
            chunk_record(
                tmp_path / "los" / "fictief-mislukt.pdf", None, status="failed"
            ),
        ),
    )
    generator = Generator()

    assert extract(tmp_path, overrides=overrides, generator=generator) == 4

    record = extraction_files(tmp_path / "extracted")["mislukt.json"]
    assert record["extraction_status"] == "error"
    assert record["batches"] == []
    assert generator.requests == []


@pytest.mark.parametrize(
    "texts", [("Fictief zonder iets.",), ()], ids=["score-0", "no-chunks"]
)
def test_nothing_to_send_is_extracted_with_one_warning(tmp_path, caplog, texts):
    record = chunk_record(tmp_path / "los" / "fictief-leeg.pdf", SHA_A, *texts)
    overrides = _lease(tmp_path, ("leeg", record))
    generator = Generator()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(tmp_path, overrides=overrides, generator=generator)

    assert code == 0
    written = extraction_files(tmp_path / "extracted")["leeg.json"]
    assert written["extraction_status"] == "extracted"
    assert written["batches"] == []
    assert {field["status"] for field in written["fields"].values()} == {"not_found"}
    assert generator.requests == []
    scores = [w for w in messages(caplog, logging.WARNING) if "scores" in w]
    assert scores == ["leeg.json: no chunk scores above 0"]


def test_the_run_goes_on_after_an_error_document(tmp_path):
    overrides = _lease(
        tmp_path,
        _document(tmp_path, "a-huur", SHA_A),
        _document(tmp_path, "b-huur", SHA_B),
    )
    generator = Generator(_status(500), reply(GOOD))

    assert extract(tmp_path, overrides=overrides, generator=generator) == 4

    files = extraction_files(tmp_path / "extracted")
    assert files["a-huur.json"]["extraction_status"] == "error"
    assert files["b-huur.json"]["extraction_status"] == "extracted"
