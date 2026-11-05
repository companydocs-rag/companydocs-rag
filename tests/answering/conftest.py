"""Fixtures for every test in ``tests/answering/`` (spec 007, A15).

``no_server``: any request that does not go through an injected
``httpx.MockTransport`` fails its test. httpx's real transports and every
socket connection are replaced by a function that calls ``pytest.fail``,
which no ``except Exception`` in the code under test catches. Proxy
variables are removed, so httpx mounts no proxy transport either.

``safe_settings``: ``COMPANYDOCS_RAG_SETTINGS`` points at a file under
``tmp_path`` with the log there and no warm-up, so a test that uses the
settings file never writes under ``data/``. A test that needs other settings
writes its own file over it.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from companydocs_rag.ingestion.settings import load_settings
from tests.embeddings._helpers import write_settings

_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def refuse_request(*args, **kwargs):
    pytest.fail("a request went past the injected httpx.MockTransport")


@pytest.fixture(autouse=True)
def no_server(monkeypatch):
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse_request)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", refuse_request
    )
    monkeypatch.setattr(socket.socket, "connect", refuse_request)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse_request)


@pytest.fixture(autouse=True)
def safe_settings(tmp_path, monkeypatch):
    write_settings(
        tmp_path,
        monkeypatch,
        {"answering": {"warmup": False, "log_dir": str(tmp_path / "logs")}},
    )
    yield
    load_settings.cache_clear()
