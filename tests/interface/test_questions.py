"""Questions and their failures (spec 009, D3-D6, R6, R7; Q5; A10, A11)."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from companydocs_rag.answering.answerer import Answerer, GeneratorUnavailableError
from companydocs_rag.ingestion.parse_pass import file_sha256
from companydocs_rag.interface import texts
from companydocs_rag.interface.server import MAX_BODY_BYTES
from tests.answering._helpers import (
    FakeRetriever,
    Generator,
    chunk,
    log_records,
    reply,
    search_result,
    span,
    write_manifest,
)
from tests.embeddings._helpers import write_settings
from tests.interface._helpers import (
    FakeAnswerer,
    answer_result,
    ask,
    parse_html,
    raw_request,
    request,
    running_server,
    source,
    split_response,
    write_pdf,
)

MAXIMUM = 60
OUTSIDE_THE_BMP = chr(0x1D11E)  # U+1D11E MUSICAL SYMBOL G CLEF


@pytest.fixture
def raw(tmp_path, monkeypatch):
    write_settings(
        tmp_path,
        monkeypatch,
        {
            "answering": {"warmup": False, "log_dir": str(tmp_path / "logs")},
            "interface": {"max_question_chars": MAXIMUM},
        },
    )
    folder = tmp_path / "raw"
    write_pdf(folder / "fictief-huur.pdf", (200, 300))
    return folder


def _message(response):
    return parse_html(response.text).find(cls="melding").text()


def _post(port, body, content_type="application/json"):
    return request(
        port, "POST", "/vraag", body=body, headers={"Content-Type": content_type}
    )


# --- A10: the answerer's calls --------------------------------------------------------


def test_each_question_is_one_call_with_the_question_as_sent(raw):
    first = "  Wat is de fictieve huurprijs?\nEn de duur?  "
    second = "Tweede fictieve vraag?"
    answerer = FakeAnswerer()

    with running_server(answerer, raw) as server:
        responses = [ask(server.port, first), ask(server.port, second)]

    assert [response.status for response in responses] == [200, 200]
    assert answerer.calls == [((first,), {}), ((second,), {})]
    assert parse_html(responses[0].text).find(cls="vraag").text() == first
    assert parse_html(responses[1].text).find(cls="vraag").text() == second


@pytest.mark.parametrize(
    "question", ["", "   ", " \t\n "], ids=["empty", "spaces", "mixed"]
)
def test_an_empty_question_is_not_answered(raw, question):
    answerer = FakeAnswerer()

    with running_server(answerer, raw) as server:
        response = ask(server.port, question)

    assert response.status == 200
    assert _message(response) == texts.EMPTY_QUESTION
    assert answerer.calls == []


def test_a_question_one_character_too_long_is_not_answered(raw):
    too_long = OUTSIDE_THE_BMP + "x" * MAXIMUM
    just_right = OUTSIDE_THE_BMP + "x" * (MAXIMUM - 1)
    assert len(too_long) == MAXIMUM + 1
    assert len(just_right.encode("utf-16-le")) // 2 == MAXIMUM + 1
    answerer = FakeAnswerer()

    with running_server(answerer, raw) as server:
        refused = ask(server.port, too_long)
        answered = ask(server.port, just_right)

    assert refused.status == 200
    assert _message(refused) == f"De vraag is te lang (maximaal {MAXIMUM} tekens)."
    assert parse_html(refused.text).find(cls="vraag").text() == too_long
    assert answered.status == 200
    assert answerer.calls == [((just_right,), {})]


MALFORMED_BODIES = {
    "not-json": b"{geen json",
    "not-utf8": b'{"vraag": "\xff"}',
    "a-list": b'["Fictieve vraag?"]',
    "a-string": b'"Fictieve vraag?"',
    "no-vraag": b'{"vraagje": "Fictieve vraag?"}',
    "empty-object": b"{}",
    "extra-key": b'{"vraag": "Fictieve vraag?", "modus": "fused"}',
    "repeated-key": b'{"vraag": "Een?", "vraag": "Twee?"}',
    "vraag-a-number": b'{"vraag": 5}',
    "vraag-null": b'{"vraag": null}',
    "vraag-a-list": b'{"vraag": ["Fictieve vraag?"]}',
    "empty-body": b"",
}


@pytest.mark.parametrize("case", sorted(MALFORMED_BODIES))
def test_a_malformed_body_is_400_without_a_call(raw, case):
    answerer = FakeAnswerer()

    with running_server(answerer, raw) as server:
        response = _post(server.port, MALFORMED_BODIES[case])

    assert response.status == 400
    assert response.headers["Content-Type"] == "text/plain; charset=utf-8"
    assert response.text == texts.MALFORMED_REQUEST
    assert answerer.calls == []


@pytest.mark.parametrize(
    "content_type",
    ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data"],
)
def test_another_content_type_is_400_without_a_call(raw, content_type):
    answerer = FakeAnswerer()
    body = json.dumps({"vraag": "Fictieve vraag?"}).encode("utf-8")

    with running_server(answerer, raw) as server:
        response = _post(server.port, body, content_type)

    assert response.status == 400
    assert response.text == texts.MALFORMED_REQUEST
    assert answerer.calls == []


def test_json_with_a_charset_parameter_is_accepted(raw):
    answerer = FakeAnswerer()
    body = json.dumps({"vraag": "Fictieve vraag?"}).encode("utf-8")

    with running_server(answerer, raw) as server:
        response = _post(server.port, body, "application/json; charset=utf-8")

    assert response.status == 200
    assert answerer.calls == [(("Fictieve vraag?",), {})]


def test_a_body_over_1_mib_is_400_and_not_read(raw):
    assert MAX_BODY_BYTES == 1024 * 1024
    answerer = FakeAnswerer()
    head = (
        "POST /vraag HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {MAX_BODY_BYTES + 1}\r\n\r\n"
    ).encode("ascii")

    with running_server(answerer, raw) as server:
        status_line, _, body = split_response(raw_request(server.port, head))

    assert status_line.split()[1] == "400"
    assert body.decode("utf-8") == texts.MALFORMED_REQUEST
    assert answerer.calls == []


def test_a_get_of_the_question_path_is_404_without_a_call(raw):
    answerer = FakeAnswerer()

    with running_server(answerer, raw) as server:
        response = request(server.port, "GET", "/vraag?vraag=Fictieve%20vraag%3F")

    assert response.status == 404
    assert response.text == texts.PAGE_NOT_FOUND
    assert answerer.calls == []


def test_an_answered_fragment_holds_the_question(raw):
    answerer = FakeAnswerer(answer_result(sources=[source(1)]))

    with running_server(answerer, raw) as server:
        response = ask(server.port, "Waar staat de fictieve huurprijs?")

    tree = parse_html(response.text)
    assert tree.find(cls="vraag").text() == "Waar staat de fictieve huurprijs?"
    assert tree.find(cls="antwoordtekst").text() == "Fictief antwoord [1]."


# --- A11: failures per question (D6, Q5) ----------------------------------------------


def test_an_unavailable_generator_gives_its_text_and_the_next_question_is_answered(raw):
    answerer = FakeAnswerer(GeneratorUnavailableError("ConnectError: fictief"))

    with running_server(answerer, raw) as server:
        failed = ask(server.port, "Een?")
        answered = ask(server.port, "Twee?")

    assert failed.status == 200
    assert _message(failed) == texts.GENERATOR_UNAVAILABLE
    assert answered.status == 200
    assert parse_html(answered.text).find(cls="antwoordtekst").text() == (
        "Fictief antwoord [1]."
    )


def test_another_exception_gives_500_with_a_traceback_and_the_server_goes_on(
    raw, caplog
):
    answerer = FakeAnswerer(RuntimeError("fictieve fout"))

    with caplog.at_level(logging.WARNING):
        with running_server(answerer, raw) as server:
            caplog.clear()
            failed = ask(server.port, "Een?")
            answered = ask(server.port, "Twee?")

    assert failed.status == 500
    assert failed.headers["Content-Type"] == "text/html; charset=utf-8"
    assert _message(failed) == texts.SERVER_ERROR
    assert parse_html(failed.text).find(cls="vraag").text() == "Een?"
    (record,) = [r for r in caplog.records if r.exc_info]
    assert record.levelname == "ERROR"
    assert "Traceback (most recent call last)" in caplog.text
    assert "RuntimeError: fictieve fout" in caplog.text
    assert answered.status == 200


def test_through_the_real_answerer_failures_and_an_answer_are_logged(
    raw, tmp_path, caplog
):
    digest = file_sha256(raw / "fictief-huur.pdf")
    found = chunk(
        "c01",
        "Fictieve huurprijs van 100 euro.",
        file_sha256=digest,
        file_path="raw/fictief-huur.pdf",
        page_spans=[span(1)],
    )
    write_manifest(tmp_path / "index")
    generator = Generator(
        httpx.ConnectError("fictief: geweigerd"),
        httpx.Response(500, text="fictieve serverfout"),
        reply("De huurprijs is fictief [1]."),
    )
    answerer = Answerer(
        tmp_path / "index",
        retriever=FakeRetriever(
            search_result(dense=[(found, 0.9)], sparse=[(found, 3.0)])
        ),
        transport=generator.transport,
    )

    with answerer, running_server(answerer, raw) as server:
        responses = [
            ask(server.port, question) for question in ("Een?", "Twee?", "Drie?")
        ]

    assert [response.status for response in responses] == [200, 200, 200]
    assert _message(responses[0]) == texts.GENERATOR_UNAVAILABLE
    assert _message(responses[1]) == texts.GENERATOR_ERROR
    tree = parse_html(responses[2].text)
    assert tree.find(cls="antwoordtekst").text() == "De huurprijs is fictief [1]."
    assert tree.find(cls="bron-bestand").text() == "fictief-huur.pdf"
    records = log_records(tmp_path / "logs")
    assert [(r["question"], r["status"]) for r in records] == [
        ("Een?", "error"),
        ("Twee?", "error"),
        ("Drie?", "answered"),
    ]
    assert [r["error"]["type"] for r in records[:2]] == [
        "generator_unavailable",
        "http_error",
    ]
    assert len(generator.requests) == 3
