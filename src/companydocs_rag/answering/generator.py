"""The generator request (spec 007, D5, R4, R10).

One ``POST {base_url}/chat/completions`` per request through httpx, a
trailing "/" of ``base_url`` dropped first, without retries. The body holds
``model``, ``messages``, ``temperature``, ``max_tokens``, ``seed``,
``stream`` (false) and ``reasoning_effort`` (left out when null), in that
order, then the keys of ``extra_body`` in their configured order. The
``Authorization: Bearer <value>`` header is sent only when ``api_key_env``
names a variable; its value is read at open and never written anywhere else.
Timeouts: ``connect_timeout_s`` for connecting, ``read_timeout_s`` for
reading, writing and the pool. The client is created with ``trust_env=False``
(spec 007 I2): it ignores the proxy environment variables and Windows' proxy
settings and sends every request directly to ``base_url``, since a proxy
would otherwise receive the prompts, contract text included, also for
127.0.0.1. In httpx 0.28.1 it then ignores ``SSL_CERT_FILE`` and
``SSL_CERT_DIR`` too.

A failed request raises :class:`RequestFailure`, which the answerer turns
into its public errors (R4): ``unavailable`` for a refused or unresolvable
connection, a connect timeout and HTTP 404; otherwise ``error_type`` is
``http_error`` (another status outside 2xx), ``read_timeout``,
``transport_error`` (any other httpx transport error) or
``malformed_response``. A body that httpx cannot decode
(``httpx.DecodingError``, "a malformed encoding") is a malformed response.

Malformed (R4): the body is not a JSON object; ``choices`` is not a
non-empty list; ``choices[0].message`` is not an object; its ``content`` is
neither a string nor null (an absent ``content`` reads as null); ``usage``
is not an object; ``usage.prompt_tokens`` is not an integer of at least 1.
An absent ``finish_reason``, ``completion_tokens``,
``prompt_tokens_details.cached_tokens`` or ``reasoning`` reads as None;
``reasoning`` is kept as returned, never parsed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import httpx

from companydocs_rag.embeddings.dense import SettingsError

from .settings import AnsweringSettings

CHAT_COMPLETIONS_PATH = "/chat/completions"
ERROR_GENERATOR_UNAVAILABLE = "generator_unavailable"
ERROR_HTTP = "http_error"
ERROR_READ_TIMEOUT = "read_timeout"
ERROR_TRANSPORT = "transport_error"
ERROR_MALFORMED = "malformed_response"

_NOT_FOUND = 404


class RequestFailure(Exception):
    """A request that did not give a usable response. ``cause`` is the text
    the messages and the record show; ``unavailable`` marks a generator that
    cannot be reached (R4)."""

    def __init__(self, error_type: str, cause: str, *, unavailable: bool) -> None:
        super().__init__(cause)
        self.error_type = error_type
        self.cause = cause
        self.unavailable = unavailable


@dataclass(frozen=True)
class Completion:
    """What a well-formed response holds (R4)."""

    content: Optional[str]
    finish_reason: Any
    prompt_tokens: int
    completion_tokens: Any
    cached_tokens: Any
    reasoning: Any


def read_api_key(config: AnsweringSettings) -> Optional[str]:
    """The API key: the value of the variable ``api_key_env`` names, or None
    when it names none. An unset or empty variable raises ``SettingsError``
    for ``answering.api_key_env``, naming the variable only (R10)."""
    name = config.api_key_env
    if name is None:
        return None
    value = os.environ.get(name)
    if value is None:
        raise SettingsError(
            "answering", "api_key_env", f"the environment variable {name} is not set"
        )
    if value == "":
        raise SettingsError(
            "answering", "api_key_env", f"the environment variable {name} is empty"
        )
    return value


def chat_completions_url(base_url: str) -> str:
    """``base_url`` without a trailing "/", then ``/chat/completions``."""
    base = base_url[:-1] if base_url.endswith("/") else base_url
    return base + CHAT_COMPLETIONS_PATH


def request_body(
    config: AnsweringSettings, messages: List[Dict[str, str]], max_tokens: int
) -> Dict[str, Any]:
    """The JSON body, keys in R10's order."""
    body: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": max_tokens,
        "seed": config.seed,
        "stream": False,
    }
    if config.reasoning_effort is not None:
        body["reasoning_effort"] = config.reasoning_effort
    body.update(config.extra_body)
    return body


def _cause(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _malformed(reason: str) -> RequestFailure:
    return RequestFailure(
        ERROR_MALFORMED, f"malformed response: {reason}", unavailable=False
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_completion(content: bytes) -> Completion:
    """The fields of a 2xx response's body; raises :class:`RequestFailure`
    with ``malformed_response`` per R4."""
    try:
        body = json.loads(content)
    except ValueError:
        raise _malformed("the body is not JSON") from None
    if not isinstance(body, Mapping):
        raise _malformed("the body is not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _malformed("choices is not a non-empty list")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    if not isinstance(message, Mapping):
        raise _malformed("choices[0].message is not an object")
    text = message.get("content")
    if text is not None and not isinstance(text, str):
        raise _malformed("choices[0].message.content is neither a string nor null")
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        raise _malformed("usage is not an object")
    prompt_tokens = usage.get("prompt_tokens")
    if not _is_int(prompt_tokens) or prompt_tokens < 1:
        raise _malformed("usage.prompt_tokens is not an integer of at least 1")
    details = usage.get("prompt_tokens_details")
    return Completion(
        content=text,
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=prompt_tokens,
        completion_tokens=usage.get("completion_tokens"),
        cached_tokens=(
            details.get("cached_tokens") if isinstance(details, Mapping) else None
        ),
        reasoning=message.get("reasoning"),
    )


class GeneratorClient:
    """The httpx client for one answerer: no retries, the settings' timeouts,
    no proxy (``trust_env=False``, I2), an injectable transport (tests use
    ``httpx.MockTransport``)."""

    def __init__(
        self,
        config: AnsweringSettings,
        *,
        api_key: Optional[str],
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self._config = config
        self._url = chat_completions_url(config.base_url)
        headers = {} if api_key is None else {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.Client(
            transport=transport,
            headers=headers,
            timeout=httpx.Timeout(
                config.read_timeout_s, connect=config.connect_timeout_s
            ),
            trust_env=False,
        )

    def complete(
        self, messages: List[Dict[str, str]], *, max_tokens: int
    ) -> Completion:
        """Send one request and return its well-formed response; raises
        :class:`RequestFailure` otherwise (R4)."""
        body = request_body(self._config, messages, max_tokens)
        try:
            response = self._client.post(self._url, json=body)
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            raise RequestFailure(
                ERROR_GENERATOR_UNAVAILABLE, _cause(error), unavailable=True
            ) from error
        except httpx.ReadTimeout as error:
            raise RequestFailure(
                ERROR_READ_TIMEOUT, _cause(error), unavailable=False
            ) from error
        except httpx.TransportError as error:
            raise RequestFailure(
                ERROR_TRANSPORT, _cause(error), unavailable=False
            ) from error
        except httpx.DecodingError as error:
            raise _malformed(_cause(error)) from error
        status = response.status_code
        if status == _NOT_FOUND:
            raise RequestFailure(
                ERROR_GENERATOR_UNAVAILABLE,
                f"HTTP {status}: {response.text.strip()}",
                unavailable=True,
            )
        if not 200 <= status < 300:
            raise RequestFailure(
                ERROR_HTTP, f"HTTP {status}: {response.text.strip()}", unavailable=False
            )
        return parse_completion(response.content)

    def close(self) -> None:
        self._client.close()
