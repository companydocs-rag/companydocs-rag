"""Connections (spec 009, D2, R17; Q8; A3).

Each request goes over a raw socket as HTTP/1.1 with ``Connection:
keep-alive``; the server answers with ``Connection: close`` and closes the
connection while the client keeps its end open.
"""

from __future__ import annotations

import json
import socket
import time

import pytest

from companydocs_rag.ingestion.parse_pass import file_sha256
from companydocs_rag.interface.server import IDLE_TIMEOUT_S
from tests.interface._helpers import (
    HOST,
    FakeAnswerer,
    get_request,
    request,
    running_server,
    split_response,
    write_pdf,
)

CLOSE_WITHIN_S = 2.0


@pytest.fixture
def raw(tmp_path):
    folder = tmp_path / "raw"
    write_pdf(folder / "fictief-huur.pdf", (200, 300))
    return folder


def _question_request() -> bytes:
    body = json.dumps({"vraag": "Fictieve vraag?"}).encode("utf-8")
    head = (
        f"POST /vraag HTTP/1.1\r\nHost: {HOST}\r\nConnection: keep-alive\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode("ascii")
    return head + body


def _read_response(connection: socket.socket) -> bytes:
    """Read one response: the head, then ``Content-Length`` bytes."""
    received = b""
    while b"\r\n\r\n" not in received:
        chunk = connection.recv(65536)
        assert chunk, "the connection closed before the response head"
        received += chunk
    _, headers, body = split_response(received)
    length = int(headers["content-length"])
    while len(body) < length:
        chunk = connection.recv(65536)
        assert chunk, "the connection closed before the body"
        body += chunk
    return received


def _cases(raw):
    digest = file_sha256(raw / "fictief-huur.pdf")
    return {
        "page": (get_request("/"), 200),
        "question": (_question_request(), 200),
        "page-view": (get_request(f"/pagina?sha256={digest}&nummer=1"), 200),
        "400": (get_request("/pagina?sha256=fictief&nummer=1"), 400),
        "404": (get_request("/onbekend"), 404),
    }


@pytest.mark.parametrize("case", ["page", "question", "page-view", "400", "404"])
def test_a_keep_alive_request_gets_connection_close_and_a_closed_connection(raw, case):
    data, status = _cases(raw)[case]
    with running_server(FakeAnswerer(), raw) as server:
        connection = socket.create_connection((HOST, server.port), timeout=10)
        try:
            connection.sendall(data)
            status_line, headers, _ = split_response(_read_response(connection))

            assert status_line.split()[1] == str(status)
            assert headers["connection"] == "close"
            connection.settimeout(CLOSE_WITHIN_S)
            assert connection.recv(1) == b""

            # The client still holds its end; a second connection is served.
            assert request(server.port, "GET", "/").status == 200
        finally:
            connection.close()


def test_an_idle_connection_holds_the_server_no_longer_than_the_idle_timeout(raw):
    assert IDLE_TIMEOUT_S == 2.0
    with running_server(FakeAnswerer(), raw) as server:
        idle = socket.create_connection((HOST, server.port), timeout=10)
        try:
            started = time.monotonic()
            time.sleep(0.2)  # the server accepts the idle connection first
            response = request(server.port, "GET", "/")
            elapsed = time.monotonic() - started

            assert response.status == 200
            assert elapsed <= IDLE_TIMEOUT_S + 1.0
            idle.settimeout(CLOSE_WITHIN_S)
            assert idle.recv(1) == b""  # cut off, without a response
        finally:
            idle.close()
