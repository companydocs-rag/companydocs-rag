"""The generator request (spec 007, D1, D5, R2, R4, R10, I2, A4).

Every request goes to a recording ``httpx.MockTransport``. Failing requests
are raised by the handler (httpx's connect, connect-timeout and read-timeout
errors) or answered with a status or a malformed body. The proxy test sends
nothing: it only looks up which transport httpx mounts for the URL.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request

import httpx
import pytest

from companydocs_rag.answering import cli
from companydocs_rag.answering.answerer import (
    Answerer,
    GeneratorError,
    GeneratorUnavailableError,
)
from companydocs_rag.answering.generator import chat_completions_url
from companydocs_rag.answering.settings import (
    BASE_URL,
    REQUEST_FIELDS,
    answering_settings,
)
from companydocs_rag.embeddings.dense import SettingsError
from tests.answering._helpers import (
    MALFORMED,
    FakeRetriever,
    Generator,
    completion_body,
    json_response,
    log_records,
    reply,
    settings_for,
    simple_result,
    use_settings,
    write_manifest,
)

KEY_VARIABLE = "COMPANYDOCS_FICTIEVE_SLEUTEL"
KEY_VALUE = "fictief-geheim-7Q2xZ"


def _open(tmp_path, generator, retriever=None, **answering):
    write_manifest(tmp_path / "index")
    return Answerer(
        tmp_path / "index",
        settings=settings_for(tmp_path, **answering),
        retriever=retriever
        if retriever is not None
        else FakeRetriever(simple_result()),
        transport=generator.transport,
    )


def _run(tmp_path, *args, retriever=None, generator=None):
    return cli.main(
        [*args, "--index-dir", str(tmp_path / "index")],
        retriever=retriever
        if retriever is not None
        else FakeRetriever(simple_result()),
        transport=(generator if generator is not None else Generator()).transport,
    )


# --- The request (D5, R10) ------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    ["http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1/"],
    ids=["plain", "trailing-slash"],
)
def test_one_post_with_the_body_in_order(tmp_path, base_url):
    generator = Generator()
    with _open(
        tmp_path,
        generator,
        base_url=base_url,
        model="fictief-model",
        temperature=0.25,
        max_tokens=321,
        seed=7,
        reasoning_effort="low",
        extra_body={"keep_alive": "5m", "fictieve_optie": [1, 2]},
    ) as answerer:
        answerer.answer("Fictieve vraag?")

    (request,) = generator.requests
    assert request.method == "POST"
    assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
    (body,) = generator.bodies
    assert list(body) == [
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "seed",
        "stream",
        "reasoning_effort",
        "keep_alive",
        "fictieve_optie",
    ]
    assert body["model"] == "fictief-model"
    assert body["temperature"] == 0.25
    assert body["max_tokens"] == 321
    assert body["seed"] == 7
    assert body["stream"] is False
    assert body["reasoning_effort"] == "low"
    assert (body["keep_alive"], body["fictieve_optie"]) == ("5m", [1, 2])
    assert [message["role"] for message in body["messages"]] == ["system", "user"]


def test_the_default_body(tmp_path):
    generator = Generator()
    with _open(tmp_path, generator) as answerer:
        answerer.answer("Fictieve vraag?")

    (body,) = generator.bodies
    assert {key: value for key, value in body.items() if key != "messages"} == {
        "model": "companydocs-gemma4-e4b-8k",
        "temperature": 0.0,
        "max_tokens": 800,
        "seed": 0,
        "stream": False,
        "reasoning_effort": "none",
        "keep_alive": "60m",
    }


def test_a_null_reasoning_effort_is_left_out(tmp_path):
    generator = Generator()
    with _open(tmp_path, generator, reasoning_effort=None, extra_body={}) as answerer:
        answerer.answer("Fictieve vraag?")

    (body,) = generator.bodies
    assert list(body) == [
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "seed",
        "stream",
    ]


def test_a_null_temperature_is_left_out(tmp_path):
    # Hosted models that reject a temperature get a body without the key.
    generator = Generator()
    with _open(
        tmp_path, generator, temperature=None, reasoning_effort=None, extra_body={}
    ) as answerer:
        answerer.answer("Fictieve vraag?")

    (body,) = generator.bodies
    assert list(body) == ["model", "messages", "max_tokens", "seed", "stream"]


@pytest.mark.parametrize("field", REQUEST_FIELDS)
def test_a_request_field_in_extra_body_is_refused(
    tmp_path, monkeypatch, caplog, capsys, field
):
    assert len(REQUEST_FIELDS) == 7
    for reasoning_effort in ("none", None):
        with pytest.raises(SettingsError) as raised:
            answering_settings(
                {
                    "answering": {
                        "reasoning_effort": reasoning_effort,
                        "extra_body": {"keep_alive": "5m", field: "x"},
                    }
                }
            )
        assert (raised.value.section, raised.value.key) == ("answering", "extra_body")

    use_settings(tmp_path, monkeypatch, settings_for(tmp_path, extra_body={field: "x"}))
    write_manifest(tmp_path / "index")
    retriever, generator = FakeRetriever(simple_result()), Generator()
    with caplog.at_level(logging.WARNING):
        code = _run(
            tmp_path, "Fictieve vraag?", retriever=retriever, generator=generator
        )

    assert code == 2
    (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert error.startswith("Invalid settings: answering.extra_body: ")
    assert capsys.readouterr().out == ""
    assert (generator.requests, retriever.calls) == ([], [])


# --- The API key (R10) ----------------------------------------------------------------


def test_without_api_key_env_no_authorization_header(tmp_path):
    generator = Generator()
    with _open(tmp_path, generator, warmup=True) as answerer:
        answerer.answer("Fictieve vraag?")

    assert len(generator.requests) == 2
    assert all("authorization" not in request.headers for request in generator.requests)


def test_a_set_key_variable_is_sent_and_never_shown(
    tmp_path, monkeypatch, caplog, capsys
):
    monkeypatch.setenv(KEY_VARIABLE, KEY_VALUE)
    use_settings(
        tmp_path,
        monkeypatch,
        settings_for(tmp_path, api_key_env=KEY_VARIABLE, warmup=True),
    )
    write_manifest(tmp_path / "index")
    generator = Generator(reply("Fictief"), reply("Geen bron, alleen tekst."))

    with caplog.at_level(logging.DEBUG):
        code = _run(tmp_path, "Fictieve vraag?", "--json", generator=generator)

    assert code == 0
    assert [request.headers["authorization"] for request in generator.requests] == [
        f"Bearer {KEY_VALUE}"
    ] * 2
    out = capsys.readouterr()
    assert KEY_VALUE not in out.out and KEY_VALUE not in out.err
    assert KEY_VALUE not in caplog.text
    (record,) = log_records(tmp_path / "logs")
    assert KEY_VALUE not in json.dumps(record, ensure_ascii=False)
    assert record["settings"]["answering"]["api_key_env"] == KEY_VARIABLE


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_an_unset_or_empty_key_variable_cannot_start(
    tmp_path, monkeypatch, caplog, capsys, value
):
    if value is None:
        monkeypatch.delenv(KEY_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(KEY_VARIABLE, value)
    write_manifest(tmp_path / "index")
    retriever, generator = FakeRetriever(simple_result()), Generator()

    with pytest.raises(SettingsError) as raised:
        Answerer(
            tmp_path / "index",
            settings=settings_for(tmp_path, api_key_env=KEY_VARIABLE),
            retriever=retriever,
            transport=generator.transport,
        )
    assert (raised.value.section, raised.value.key) == ("answering", "api_key_env")

    use_settings(
        tmp_path, monkeypatch, settings_for(tmp_path, api_key_env=KEY_VARIABLE)
    )
    with caplog.at_level(logging.WARNING):
        code = _run(
            tmp_path, "Fictieve vraag?", retriever=retriever, generator=generator
        )

    assert code == 2
    (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert error.startswith("Invalid settings: answering.api_key_env: ")
    assert KEY_VARIABLE in error
    assert capsys.readouterr().out == ""
    assert (generator.requests, retriever.calls) == ([], [])
    assert not (tmp_path / "logs").exists()


# --- The warm-up (D5, R10) ------------------------------------------------------------


def test_the_warm_up_is_one_request_before_any_answer(tmp_path):
    generator = Generator()
    with _open(tmp_path, generator, warmup=True) as answerer:
        assert len(generator.requests) == 1
        assert not (tmp_path / "logs").exists()
        (warm_up,) = generator.bodies
        assert warm_up["messages"] == [{"role": "user", "content": "ok"}]
        assert warm_up["max_tokens"] == 1
        assert list(warm_up) == [
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "seed",
            "stream",
            "reasoning_effort",
            "keep_alive",
        ]
        answerer.answer("Fictieve vraag?")

    assert len(generator.requests) == 2
    assert len(log_records(tmp_path / "logs")) == 1


def test_without_warm_up_opening_sends_nothing(tmp_path):
    generator = Generator()
    with _open(tmp_path, generator, warmup=False):
        assert generator.requests == []


# --- Failures (R4) --------------------------------------------------------------------


UNAVAILABLE = {
    "connect-error": lambda: httpx.ConnectError("fictief: verbinding geweigerd"),
    "connect-timeout": lambda: httpx.ConnectTimeout("fictief: geen verbinding"),
    "http-404": lambda: httpx.Response(404, json={"error": "model 'x' not found"}),
}
FAILED = {
    "http-400": lambda: httpx.Response(400, json={"error": "fictief"}),
    "http-500": lambda: httpx.Response(500, text="fictieve serverfout"),
    "read-timeout": lambda: httpx.ReadTimeout("fictief: te traag"),
    **{f"malformed-{name}": make for name, make in MALFORMED.items()},
}


@pytest.mark.parametrize("case", list(UNAVAILABLE))
def test_an_unavailable_generator_at_warm_up(tmp_path, case):
    generator = Generator(UNAVAILABLE[case]())
    retriever = FakeRetriever(simple_result())

    with pytest.raises(GeneratorUnavailableError):
        _open(tmp_path, generator, retriever=retriever, warmup=True)

    assert len(generator.requests) == 1
    assert retriever.closed == 1
    assert retriever.calls == []
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("case", list(FAILED))
def test_a_failed_warm_up_is_a_generator_error(tmp_path, case):
    generator = Generator(FAILED[case]())
    retriever = FakeRetriever(simple_result())

    with pytest.raises(GeneratorError) as raised:
        _open(tmp_path, generator, retriever=retriever, warmup=True)

    assert not isinstance(raised.value, GeneratorUnavailableError)
    assert len(generator.requests) == 1
    assert retriever.closed == 1
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("case", list(UNAVAILABLE))
def test_an_unavailable_generator_during_an_answer_raises(tmp_path, case):
    generator = Generator(UNAVAILABLE[case]())
    with _open(tmp_path, generator) as answerer:
        with pytest.raises(GeneratorUnavailableError) as raised:
            answerer.answer("Fictieve vraag?", question_id="v1")

    assert len(generator.requests) == 1
    assert raised.value.result.status == "error"


def test_the_two_errors_are_separate_classes():
    assert issubclass(GeneratorError, Exception)
    assert issubclass(GeneratorUnavailableError, Exception)
    assert not issubclass(GeneratorError, GeneratorUnavailableError)
    assert not issubclass(GeneratorUnavailableError, GeneratorError)


def test_absent_optional_fields_read_as_null(tmp_path):
    body = {
        "choices": [{"message": {"role": "assistant", "content": "Fictief [1]."}}],
        "usage": {"prompt_tokens": 400},
    }
    generator = Generator(json_response(body)())
    with _open(tmp_path, generator) as answerer:
        record = answerer.answer("Fictieve vraag?")

    assert record.status == "answered"
    generation = record.generation
    assert (
        generation["finish_reason"],
        generation["completion_tokens"],
        generation["cached_tokens"],
        generation["reasoning"],
        generation["truncated"],
    ) == (None, None, None, None, False)


def test_reasoning_and_cached_tokens_are_logged(tmp_path):
    body = completion_body(
        "Fictief [1].",
        prompt_tokens=400,
        cached_tokens=128,
        reasoning="Fictieve redenering.",
        completion_tokens=40,
    )
    generator = Generator(json_response(body)())
    with _open(tmp_path, generator) as answerer:
        record = answerer.answer("Fictieve vraag?")

    assert record.generation["reasoning"] == "Fictieve redenering."
    assert record.generation["cached_tokens"] == 128
    assert record.generation["completion_tokens"] == 40
    assert record.answer_text == "Fictief [1]."


def test_the_timeouts_come_from_the_settings(tmp_path):
    timeouts = []

    def respond(request):
        timeouts.append(request.extensions["timeout"])
        return reply("Fictief [1].")(request)

    generator = Generator(respond)
    with _open(
        tmp_path, generator, connect_timeout_s=2.5, read_timeout_s=99.0
    ) as answerer:
        answerer.answer("Fictieve vraag?")

    assert timeouts == [{"connect": 2.5, "read": 99.0, "write": 99.0, "pool": 99.0}]


# --- No proxy (I2, review F2) ---------------------------------------------------------

PROXY = "http://proxy.voorbeeld.invalid:3128"


def _without_proxy_variables(monkeypatch):
    """Remove every variable urllib reads as a proxy setting: each name that
    ends in ``_proxy``, in any case, ``NO_PROXY`` included."""
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)


def _pool_for(client, url):
    """The class of the connection pool httpx picks for ``url``:
    ``ConnectionPool`` connects directly, ``HTTPProxy`` through a proxy. It
    only looks up the transport httpx mounts for the URL; nothing is sent."""
    return type(client._transport_for_url(httpx.URL(url))._pool).__name__


@pytest.mark.parametrize(
    "proxy_source",
    [
        "environment",
        pytest.param(
            "windows-user-settings",
            marks=pytest.mark.skipif(
                sys.platform != "win32",
                reason="the Windows user proxy exists only on Windows",
            ),
        ),
    ],
)
def test_the_client_goes_to_base_url_directly_whatever_proxy_is_set(
    tmp_path, monkeypatch, proxy_source
):
    _without_proxy_variables(monkeypatch)
    if proxy_source == "environment":
        monkeypatch.setenv("HTTP_PROXY", PROXY)
    else:
        # What urllib reads from Windows' Internet Settings for a user proxy
        # (ProxyEnable 1, ProxyServer proxy.voorbeeld.invalid:3128). On
        # Windows, urllib.request.getproxies looks this function up at every
        # call, so the reference httpx bound at import sees it too.
        monkeypatch.setattr(
            urllib.request,
            "getproxies_registry",
            lambda: {"http": PROXY, "https": PROXY, "ftp": PROXY},
        )
    url = chat_completions_url(BASE_URL)

    # The control: in the same environment httpx's default client does go
    # through the proxy, so the proxy is in effect.
    with httpx.Client() as default:
        assert _pool_for(default, url) == "HTTPProxy"

    write_manifest(tmp_path / "index")
    retriever = FakeRetriever(simple_result())
    with Answerer(
        tmp_path / "index", settings=settings_for(tmp_path), retriever=retriever
    ) as answerer:
        assert _pool_for(answerer._client._client, url) == "ConnectionPool"

    assert retriever.calls == []
