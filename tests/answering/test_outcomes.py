"""One fabricated case per outcome (spec 007, D6-D8, R11, Q1-Q4; A6, A7, A8).

Each case checks ``status``, ``abstention_source``, ``error`` and the number
of requests the recording MockTransport saw.
"""

from __future__ import annotations

import json
import logging
import math

import httpx
import pytest

from companydocs_rag.answering import cli
from companydocs_rag.answering.answerer import Answerer, GeneratorUnavailableError
from companydocs_rag.embeddings.dense import SettingsError
from companydocs_rag.ingestion.settings import find_project_root
from tests.answering._helpers import (
    MALFORMED,
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

NO_CITATIONS = {"valid": [], "invalid": []}
KEY_VARIABLE = "COMPANYDOCS_FICTIEVE_SLEUTEL_ONTBREEKT"
SYSTEM = (
    (find_project_root() / "config" / "prompts" / "answer_system_nl.txt")
    .read_bytes()
    .decode("utf-8")
)


def _answer(tmp_path, result, *replies, selection_mode=None, **answering):
    index = tmp_path / "index"
    write_manifest(index)
    generator = Generator(*replies)
    retriever = FakeRetriever(result)
    with Answerer(
        index,
        settings=settings_for(tmp_path, **answering),
        retriever=retriever,
        transport=generator.transport,
    ) as answerer:
        record = answerer.answer("Fictieve vraag?", selection_mode=selection_mode)
    return record, generator, retriever


def _outcome(record):
    return (record.status, record.abstention_source, record.error)


# --- Outcomes with a response (A6) ----------------------------------------------------


@pytest.mark.parametrize(
    ("content", "status", "source"),
    [
        ("De fictieve huur staat in [2].", "answered", None),
        ("ONVOLDOENDE BEWIJS. Geen bron noemt het.", "insufficient_evidence", "prompt"),
        ("Een fictief antwoord zonder bron.", "uncited", None),
    ],
)
def test_the_three_statuses_of_an_answer(tmp_path, content, status, source):
    record, generator, _ = _answer(tmp_path, simple_result(), reply(content))

    assert _outcome(record) == (status, source, None)
    assert len(generator.requests) == 1
    assert record.answer_text == content


def test_an_answered_record_has_its_sources(tmp_path):
    record, _, _ = _answer(tmp_path, simple_result(), reply("Zie [3] en [1] en [3]."))

    assert [source["prompt_number"] for source in record.sources] == [3, 1]
    assert [source["chunk_id"] for source in record.sources] == ["c03", "c01"]
    assert record.citations == {"valid": [3, 1], "invalid": []}


def test_the_marker_with_a_citation_has_no_sources(tmp_path):
    """A5, D7: the marker plus [2] is insufficient_evidence from the prompt,
    with sources [] and the citation still logged (review F6)."""
    content = "ONVOLDOENDE BEWIJS, alleen [2] raakt het onderwerp."
    record, generator, _ = _answer(tmp_path, simple_result(), reply(content))

    assert _outcome(record) == ("insufficient_evidence", "prompt", None)
    assert len(generator.requests) == 1
    assert record.answer_text == content
    assert record.sources == []
    assert record.citations == {"valid": [2], "invalid": []}
    (logged,) = log_records(tmp_path / "logs")
    assert logged == record.to_dict()


@pytest.mark.parametrize(
    "content", ["", None, "  \n\t  "], ids=["empty", "null", "blank"]
)
def test_empty_content_is_an_error(tmp_path, content):
    record, generator, _ = _answer(tmp_path, simple_result(), reply(content))

    assert (record.status, record.abstention_source) == ("error", None)
    assert record.error["type"] == "empty_answer"
    assert len(generator.requests) == 1
    assert record.answer_text == (None if content is None else "")
    assert record.citations == NO_CITATIONS
    assert record.sources == []


@pytest.mark.parametrize(
    ("finish_reason", "truncated"), [("length", True), ("stop", False)]
)
def test_finish_reason_length_sets_truncated(tmp_path, finish_reason, truncated):
    record, _, _ = _answer(
        tmp_path, simple_result(), reply("Fictief [1].", finish_reason=finish_reason)
    )

    assert record.status == "answered"
    assert record.generation["truncated"] is truncated
    assert record.generation["finish_reason"] == finish_reason


# --- Failed requests (A6, R11, Q4) ----------------------------------------------------


def _gzip_garbage():
    """A body that claims gzip but is not: httpx raises DecodingError, "a
    malformed encoding", when the client reads it."""
    return httpx.Response(
        200,
        headers={"Content-Encoding": "gzip"},
        stream=httpx.ByteStream(b"geen gzip"),
    )


# (name, the reply, the error type, the start of the error message)
FAILURES = [
    (
        "http-400",
        lambda: httpx.Response(400, json={"error": "fictief"}),
        "http_error",
        "HTTP 400: ",
    ),
    (
        "http-500",
        lambda: httpx.Response(500, text="fictieve fout"),
        "http_error",
        "HTTP 500: fictieve fout",
    ),
    (
        "read-timeout",
        lambda: httpx.ReadTimeout("fictief"),
        "read_timeout",
        "ReadTimeout",
    ),
    ("read-error", lambda: httpx.ReadError("fictief"), "transport_error", "ReadError"),
    (
        "remote-protocol",
        lambda: httpx.RemoteProtocolError("fictief"),
        "transport_error",
        "RemoteProtocolError",
    ),
    (
        "write-timeout",
        lambda: httpx.WriteTimeout("fictief"),
        "transport_error",
        "WriteTimeout",
    ),
    (
        "not-json",
        lambda: httpx.Response(200, content=b"{geen"),
        "malformed_response",
        "malformed response: ",
    ),
    ("not-decodable", _gzip_garbage, "malformed_response", "malformed response: "),
]


@pytest.mark.parametrize(
    ("make", "error_type", "message"),
    [case[1:] for case in FAILURES],
    ids=[case[0] for case in FAILURES],
)
def test_a_failed_request_is_an_error_record(tmp_path, make, error_type, message):
    record, generator, _ = _answer(tmp_path, simple_result(), make())

    assert (record.status, record.abstention_source) == ("error", None)
    assert record.error["type"] == error_type
    assert len(generator.requests) == 1
    assert record.answer_text is None
    assert record.citations == NO_CITATIONS
    assert record.sources == []
    sent = json.loads(generator.requests[0].content)
    assert record.prompt["user_message"] == sent["messages"][1]["content"]
    assert record.generation["model"] == "companydocs-gemma4-e4b-8k"
    assert record.generation["base_url"] == "http://127.0.0.1:11434/v1"
    assert isinstance(record.generation["generation_ms"], float)
    assert {
        key: value
        for key, value in record.generation.items()
        if key not in ("model", "base_url", "generation_ms")
    } == dict.fromkeys(
        [
            "finish_reason",
            "truncated",
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "context_nearly_full",
            "reasoning",
        ]
    )
    assert record.error["message"].startswith(message)


def test_the_malformed_responses_of_r4_are_error_records(tmp_path):
    for name, make in MALFORMED.items():
        folder = tmp_path / name
        folder.mkdir()
        record, generator, _ = _answer(folder, simple_result(), make())
        assert record.error["type"] == "malformed_response", name
        assert len(generator.requests) == 1


def test_an_unavailable_generator_writes_the_record_then_raises(tmp_path):
    write_manifest(tmp_path / "index")
    generator = Generator(httpx.ConnectError("fictief: geweigerd"))
    with Answerer(
        tmp_path / "index",
        settings=settings_for(tmp_path),
        retriever=FakeRetriever(simple_result()),
        transport=generator.transport,
    ) as answerer:
        with pytest.raises(GeneratorUnavailableError) as raised:
            answerer.answer("Fictieve vraag?", question_id="v7")

    (record,) = log_records(tmp_path / "logs")
    assert record["question_id"] == "v7"
    assert record["status"] == "error"
    assert record["error"]["type"] == "generator_unavailable"
    assert record["error"]["message"] == "ConnectError: fictief: geweigerd"
    assert record == raised.value.result.to_dict()
    assert record["prompt"]["user_message"].endswith("Vraag: Fictieve vraag?")
    assert record["generation"]["prompt_tokens"] is None


# --- Records without a request (A6, R11, Q2, Q3) --------------------------------------


def _assert_without_request(record):
    assert record.context == []
    assert record.prompt is None
    assert record.generation is None
    assert record.answer_text is None
    assert record.sources == []
    assert record.citations == NO_CITATIONS


def test_zero_candidates(tmp_path):
    for gate in (None, 0.5):
        folder = tmp_path / f"gate-{gate}"
        folder.mkdir()
        record, generator, _ = _answer(folder, search_result(), min_dense_score=gate)

        assert _outcome(record) == ("insufficient_evidence", "no_candidates", None)
        assert generator.requests == []
        assert record.dropped_for_budget == []
        assert record.candidates == []
        _assert_without_request(record)


def test_the_score_gate(tmp_path):
    record, generator, _ = _answer(tmp_path, simple_result(), min_dense_score=0.95)

    assert _outcome(record) == ("insufficient_evidence", "score_gate", None)
    assert generator.requests == []
    assert record.dropped_for_budget == []
    assert len(record.candidates) == 3
    _assert_without_request(record)


def test_nothing_fits_the_budget(tmp_path):
    record, generator, _ = _answer(tmp_path, simple_result(4), max_context_chars=20)

    assert (record.status, record.abstention_source) == ("error", None)
    assert record.error["type"] == "context_budget_exceeded"
    assert generator.requests == []
    assert record.dropped_for_budget == ["c01", "c02", "c03", "c04"]
    _assert_without_request(record)


def test_a_command_run_with_a_budget_error_exits_4(
    tmp_path, monkeypatch, caplog, capsys
):
    use_settings(tmp_path, monkeypatch, settings_for(tmp_path, max_context_chars=20))
    write_manifest(tmp_path / "index")
    generator = Generator()

    with caplog.at_level(logging.WARNING, logger="companydocs_rag.answering.cli"):
        code = cli.main(
            ["Fictieve vraag?", "--index-dir", str(tmp_path / "index")],
            retriever=FakeRetriever(simple_result()),
            transport=generator.transport,
        )

    assert code == 4
    assert generator.requests == []
    (warning,) = [
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert warning.startswith("Question -: context_budget_exceeded: ")
    assert capsys.readouterr().out.splitlines()[0] == "Status: error"


# --- The tripwire and context_nearly_full (A7, D6) ------------------------------------


def _characters(tmp_path):
    """The characters of the system and the user message of the fixture's
    prompt, from a run with the default settings."""
    folder = tmp_path / "dry-run"
    folder.mkdir()
    record, generator, _ = _answer(folder, simple_result(), reply("Fictief [1]."))
    body = json.loads(generator.requests[0].content)
    characters = sum(len(message["content"]) for message in body["messages"])
    assert characters == len(SYSTEM) + len(record.prompt["user_message"])
    return characters


def test_a_ratio_just_above_the_threshold_trips(tmp_path):
    characters = _characters(tmp_path)
    ratio = characters / 100
    record, generator, _ = _answer(
        tmp_path,
        simple_result(),
        reply("De fictieve huur staat in [1].", prompt_tokens=100),
        max_chars_per_prompt_token=math.nextafter(ratio, 0.0),
    )

    assert (record.status, record.abstention_source) == ("error", None)
    assert record.error["type"] == "prompt_truncation_suspected"
    assert len(generator.requests) == 1
    assert record.answer_text == "De fictieve huur staat in [1]."
    assert record.citations == {"valid": [1], "invalid": []}
    assert record.sources == []


def test_a_ratio_equal_to_the_threshold_does_not_trip(tmp_path):
    characters = _characters(tmp_path)
    record, _, _ = _answer(
        tmp_path,
        simple_result(),
        reply("De fictieve huur staat in [1].", prompt_tokens=100),
        max_chars_per_prompt_token=characters / 100,
    )

    assert _outcome(record) == ("answered", None, None)


def test_the_tripwire_comes_before_the_empty_content_check(tmp_path):
    record, _, _ = _answer(tmp_path, simple_result(), reply("", prompt_tokens=1))

    assert record.error["type"] == "prompt_truncation_suspected"


@pytest.mark.parametrize(
    ("prompt_tokens", "nearly_full"), [(7393, True), (7392, False)]
)
def test_context_nearly_full(tmp_path, prompt_tokens, nearly_full):
    """max_tokens 800 and context_window 8192: 7393 + 800 is above 8192, and
    7392 + 800 equals it."""
    record, _, _ = _answer(
        tmp_path,
        simple_result(),
        reply("De fictieve huur staat in [2].", prompt_tokens=prompt_tokens),
        max_tokens=800,
        context_window=8192,
    )

    assert record.generation["context_nearly_full"] is nearly_full
    assert record.generation["prompt_tokens"] == prompt_tokens
    assert _outcome(record) == ("answered", None, None)


# --- The score gate (A8, D8, Q1) ------------------------------------------------------


def test_without_a_gate_low_scores_do_not_gate(tmp_path):
    low = chunk("c01", "Fictieve bepaling.")
    result = search_result(dense=[(low, 0.01)], sparse=[(low, 0.5)])

    record, generator, retriever = _answer(tmp_path, result, reply("Fictief [1]."))

    assert record.status == "answered"
    assert len(generator.requests) == 1
    assert retriever.calls == [("Fictieve vraag?", False)]


@pytest.mark.parametrize("mode", ["fused", "per_path"])
@pytest.mark.parametrize(
    ("gate", "gated"), [(0.9, False), (0.9000001, True)], ids=["equal", "above"]
)
def test_the_gate_compares_the_dense_rank_1_score(tmp_path, mode, gate, gated):
    first, second = chunk("c01", "Fictief een."), chunk("c02", "Fictief twee.")
    result = search_result(
        dense=[(first, 0.9), (second, 0.8)], sparse=[(second, 3.0), (first, 2.0)]
    )

    record, generator, retriever = _answer(
        tmp_path,
        result,
        reply("Fictief [1]."),
        selection_mode=mode,
        min_dense_score=gate,
    )

    assert retriever.calls == [("Fictieve vraag?", True)]
    if gated:
        assert (record.status, record.abstention_source) == (
            "insufficient_evidence",
            "score_gate",
        )
        assert generator.requests == []
    else:
        assert record.status == "answered"
        assert len(generator.requests) == 1
    assert (record.path_hits is None) is (mode == "fused")


@pytest.mark.parametrize("mode", ["fused", "per_path"])
def test_the_dense_rank_1_chunk_need_not_be_a_candidate(tmp_path, mode):
    """X is dense rank 1 at the gate's value, but not among the two
    candidates: Y (dense 0.4) and W (sparse only)."""
    x, y, w = (chunk(f"chunk-{name}", f"Fictief {name}.") for name in "xyw")
    result = search_result(
        dense=[(x, 0.5), (y, 0.4)], sparse=[(w, 9.0), (y, 8.0)], candidate_count=2
    )
    candidates = {c["chunk_id"]: c["dense_score"] for c in result.candidates}
    assert candidates == {"chunk-y": 0.4, "chunk-w": None}

    record, generator, _ = _answer(
        tmp_path,
        result,
        reply("Fictief [1]."),
        selection_mode=mode,
        min_dense_score=0.5,
    )

    assert record.abstention_source is None
    assert record.status == "answered"
    assert len(generator.requests) == 1


# --- The API (R2, R3) -----------------------------------------------------------------


class _ClosingTransport(httpx.MockTransport):
    """A MockTransport that records whether its client closed it."""

    closed = False

    def close(self):
        self.closed = True


def test_leaving_the_with_block_closes_the_retriever_and_the_client(tmp_path):
    write_manifest(tmp_path / "index")
    retriever = FakeRetriever(simple_result())
    transport = _ClosingTransport(Generator())

    with Answerer(
        tmp_path / "index",
        settings=settings_for(tmp_path),
        retriever=retriever,
        transport=transport,
    ) as answerer:
        answerer.answer("Fictieve vraag?")
        assert (retriever.closed, transport.closed) == (0, False)

    assert (retriever.closed, transport.closed) == (1, True)
    answerer.close()
    assert retriever.closed == 1


@pytest.mark.parametrize(
    "key",
    ["max_tokens", "prompt_file", "api_key_env", None],
    ids=["setting", "prompt-file", "api-key", "manifest"],
)
def test_a_failure_while_opening_closes_an_injected_retriever(
    tmp_path, monkeypatch, key
):
    """The settings, the prompt file and the API key come before the
    Retriever in the opening (R2); an injected one belongs to the answerer
    from the start and is closed all the same (review F5). A missing
    ``manifest.json``, after the Retriever, closes it too."""
    index = tmp_path / "index"
    write_manifest(index)
    monkeypatch.delenv(KEY_VARIABLE, raising=False)
    invalid = {
        "max_tokens": 0,
        "prompt_file": str(tmp_path / "ontbreekt.txt"),
        "api_key_env": KEY_VARIABLE,
    }
    answering = {} if key is None else {key: invalid[key]}
    if key is None:
        (index / "manifest.json").unlink()
    retriever, generator = FakeRetriever(simple_result()), Generator()

    with pytest.raises(FileNotFoundError if key is None else SettingsError) as raised:
        Answerer(
            index,
            settings=settings_for(tmp_path, **answering),
            retriever=retriever,
            transport=generator.transport,
        )

    if key is not None:
        assert (raised.value.section, raised.value.key) == ("answering", key)
    assert retriever.closed == 1
    assert (retriever.calls, generator.requests) == ([], [])
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("mode", ["hybride", "", "FUSED"])
def test_an_unknown_selection_mode_is_a_value_error(tmp_path, mode):
    write_manifest(tmp_path / "index")
    retriever, generator = FakeRetriever(simple_result()), Generator()

    with Answerer(
        tmp_path / "index",
        settings=settings_for(tmp_path),
        retriever=retriever,
        transport=generator.transport,
    ) as answerer:
        with pytest.raises(ValueError, match="selection_mode"):
            answerer.answer("Fictieve vraag?", selection_mode=mode)
        with pytest.raises(ValueError, match="Empty question"):
            answerer.answer(" \n ")

    assert (retriever.calls, generator.requests) == ([], [])
    assert not (tmp_path / "logs").exists()
