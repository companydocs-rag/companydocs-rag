"""Fixtures for every test in ``tests/interface/`` (spec 009, R19, A16).

``no_server``: any request through a real httpx transport fails its test, as
in ``tests/answering/``: httpx's real transports are replaced by a function
that calls ``pytest.fail``, which no ``except Exception`` in the code under
test catches, and the proxy variables are removed. Sockets are left to the
server under test only: a connection to 127.0.0.1 on a port that a test
registered for its server (``_helpers.free_port``) goes through; any other
connection, Ollama's port included, fails the test.

``safe_settings``: ``COMPANYDOCS_RAG_SETTINGS`` points at a file under
``tmp_path`` with no warm-up and the answer log there, so no test writes
under ``data/``. A test that needs other settings writes its own file over
it.

``no_interim``: ``main``'s default chunk folder and OCR cache for the
highlighter (spec 011) are folders under ``tmp_path`` that do not exist, so
no test reads ``data/interim/``. A test that needs them passes
``chunks_dir`` and ``ocr_cache_dir``.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from companydocs_rag.ingestion.settings import load_settings
from companydocs_rag.interface import cli
from tests.embeddings._helpers import write_settings
from tests.interface._helpers import HOST, SERVER_PORTS

_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex


def refuse_request(*args, **kwargs):
    pytest.fail("a request went past the injected httpx.MockTransport")


def _to_the_server_under_test(address) -> bool:
    return (
        isinstance(address, tuple)
        and len(address) >= 2
        and address[0] == HOST
        and address[1] in SERVER_PORTS
    )


def _guarded_connect(self, address):
    if not _to_the_server_under_test(address):
        refuse_request()
    return _REAL_CONNECT(self, address)


def _guarded_connect_ex(self, address):
    if not _to_the_server_under_test(address):
        refuse_request()
    return _REAL_CONNECT_EX(self, address)


@pytest.fixture(autouse=True)
def no_server(monkeypatch):
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse_request)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", refuse_request
    )
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    yield
    SERVER_PORTS.clear()


@pytest.fixture(autouse=True)
def no_interim(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli, "DEFAULT_HIGHLIGHT_CHUNKS_RELATIVE_PATH", tmp_path / "geen-chunks"
    )
    monkeypatch.setattr(
        cli, "DEFAULT_HIGHLIGHT_OCR_CACHE_RELATIVE_PATH", tmp_path / "geen-ocr"
    )


@pytest.fixture(autouse=True)
def safe_settings(tmp_path, monkeypatch):
    write_settings(
        tmp_path,
        monkeypatch,
        {"answering": {"warmup": False, "log_dir": str(tmp_path / "logs")}},
    )
    yield
    load_settings.cache_clear()
