"""The HTTP server (spec 009, D1-D6, D11, D12, R5-R7, R12, R16, R17; Q8).

:class:`DemoServer` is ``http.server.HTTPServer`` without a threading mixin,
bound to 127.0.0.1 only and exclusively: without ``SO_REUSEADDR``, and with
``SO_EXCLUSIVEADDRUSE`` where the platform has it (Windows), so a second
server on the same port fails instead of starting. It handles one request
at a time; each accepted connection's socket gets a timeout of
:data:`IDLE_TIMEOUT_S` (Q8), which does not apply to the time spent
answering. Every response closes its connection (D2) and carries
``Content-Length``, ``Connection: close``, ``Cache-Control: no-store`` and
``X-Content-Type-Options: nosniff``; the page also carries its
Content-Security-Policy (R17).

Routes (R5): ``GET /`` the page; ``POST /vraag`` a question (R6, R7);
``GET /pagina?sha256=<hash>&nummer=<n>`` a page (R12); and spec 011's two
read-only views as ``application/json``, built at startup
(:mod:`.library`): ``GET /documenten`` the library and ``GET /status`` the
status block; and
``GET /markering?sha256=<hash>&nummer=<n>&chunk=<id>`` the highlight of a
cited passage on its page (:mod:`.highlight`), ``{"boxes": [...]}``, empty
when there is none, 400 for a malformed query. Any other ``GET``
path is 404 with "Pagina niet gevonden."; any other ``POST`` path is 400
with "Ongeldig verzoek."; any other method, and every error ``http.server``
raises before routing, is its own status with "Ongeldig verzoek.". Every
error body is a D16 text as ``text/plain; charset=utf-8``.

:class:`DemoApp` holds what a request needs: the answerer (any object with
``answer(question)`` and ``close()``), the hash to file map, the document
types, the settings and the page. It calls ``answer(question)`` and nothing
else (D3).

Logging (R16): one line per request on stderr, ``<method> <path> <status>
<ms> ms``, where the path of a page request carries its query (the hash and
the page number) once the query is valid, and every other path is shown
without its query; never a request body, a question or an answer. The
interface's own messages go through ``logging``.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlsplit

from companydocs_rag.answering.answerer import GeneratorUnavailableError

from . import texts
from .fragments import message_fragment, result_fragment
from .highlight import CHUNK_ID_PATTERN
from .htmlpage import Page
from .pageview import (
    PAGE_NUMBER_PATTERN,
    PAGE_PATH,
    PARAM_PAGE,
    PARAM_SHA256,
    SHA256_PATTERN,
    PageRefused,
    page_url,
    parse_page_query,
    render_png,
)
from .settings import InterfaceSettings

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
ROOT_PATH = "/"
QUESTION_PATH = "/vraag"
LIBRARY_PATH = "/documenten"
STATUS_PATH = "/status"
HIGHLIGHT_PATH = "/markering"
PARAM_CHUNK = "chunk"
QUESTION_KEY = "vraag"
# Q8: an idle connection holds the server at most this long.
IDLE_TIMEOUT_S = 2.0
# How often the serving loop looks at its stop event.
POLL_INTERVAL_S = 0.5
# R6: a question's body above this many bytes is refused without being read.
MAX_BODY_BYTES = 1024 * 1024

CONTENT_TYPE_HTML = "text/html; charset=utf-8"
CONTENT_TYPE_TEXT = "text/plain; charset=utf-8"
CONTENT_TYPE_PNG = "image/png"
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_JSON_RESPONSE = "application/json; charset=utf-8"

STATUS_OK = 200
STATUS_BAD_REQUEST = 400
STATUS_NOT_FOUND = 404
STATUS_SERVER_ERROR = 500

_DIGITS = re.compile("[0-9]+")
_NO_VALUE = "-"
_ENCODING = "utf-8"
_NS_PER_MS = 1_000_000


class _Pairs(list):
    """A JSON object read as its list of (key, value) pairs, so that a
    repeated key is seen."""


def parse_question(body: bytes) -> Optional[str]:
    """The question of a request body (R6), or None when the body is not
    UTF-8 JSON holding an object with exactly one key, ``vraag``, whose
    value is a string."""
    try:
        text = body.decode(_ENCODING)
        value = json.loads(text, object_pairs_hook=_Pairs)
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, _Pairs) or len(value) != 1:
        return None
    key, question = value[0]
    if key != QUESTION_KEY or not isinstance(question, str):
        return None
    return question


class DemoApp:
    """What the requests need (D3, R6, R7, R12; spec 011).

    The answer counter, the ``<a>`` of the card ids (R10), starts at the
    time the app is built, in whole milliseconds since the Unix epoch, and
    goes up by one per answer the answerer returned, so that ids from two
    server runs do not coincide in a page left open across a restart (M1).

    ``library`` and ``status`` are spec 011's views (:mod:`.library`),
    encoded once here; left out, the library and the status are empty.
    ``highlighter`` gives the highlights (:mod:`.highlight`); left out,
    there are none.
    """

    def __init__(
        self,
        answerer: Any,
        files: Mapping[str, Path],
        types: Mapping[str, Optional[str]],
        settings: InterfaceSettings,
        page: Page,
        *,
        library: Sequence[Mapping[str, Any]] = (),
        status: Optional[Mapping[str, Any]] = None,
        highlighter: Any = None,
    ) -> None:
        self.answerer = answerer
        self.highlighter = highlighter
        self.files = files
        self.types = types
        self.settings = settings
        self.page = page
        self.views = {
            LIBRARY_PATH: _json({"documents": list(library)}),
            STATUS_PATH: _json(status if status is not None else {}),
        }
        self._next_answer = time.time_ns() // _NS_PER_MS

    def question(self, question: str) -> Tuple[int, bytes]:
        """The status and the fragment of one question (R6, R7). The
        answerer is called only for a question that is neither empty nor too
        long, with the question as received."""
        maximum = self.settings.max_question_chars
        if not question.strip():
            return STATUS_OK, _encoded(message_fragment(question, texts.EMPTY_QUESTION))
        if len(question) > maximum:
            message = texts.question_too_long(maximum)
            return STATUS_OK, _encoded(message_fragment(question, message))
        try:
            result = self.answerer.answer(question)
        except GeneratorUnavailableError:
            message = texts.GENERATOR_UNAVAILABLE
            return STATUS_OK, _encoded(message_fragment(question, message))
        except Exception:
            logger.exception("Exception while answering a question")
            return STATUS_SERVER_ERROR, _server_error(question)
        answer_number = self._next_answer
        self._next_answer += 1
        try:
            fragment = result_fragment(
                question, result, answer_number, self.files, self.types
            )
        except Exception:
            logger.exception("Exception while showing an answer")
            return STATUS_SERVER_ERROR, _server_error(question)
        return STATUS_OK, _encoded(fragment)

    def page_png(self, file_sha256: str, page: int) -> bytes:
        """The PNG of one page; raises :class:`PageRefused` (R12)."""
        return render_png(self.files, file_sha256, page, self.settings.render_dpi)

    def highlight(self, file_sha256: str, page: int, chunk_id: str) -> bytes:
        """The highlight of one chunk on one page as JSON; no boxes without
        a highlighter or when computing them fails."""
        boxes: Any = []
        if self.highlighter is not None:
            try:
                boxes = self.highlighter.boxes(file_sha256, page, chunk_id)
            except Exception:
                logger.exception("Exception while computing a highlight")
                boxes = []
        return _json({"boxes": boxes})


def parse_highlight_query(query: str) -> Optional[Tuple[str, int, str]]:
    """The hash, page number and chunk id of a highlight request, or None
    when it does not hold exactly ``sha256``, ``nummer`` and ``chunk``, each
    once, matching their patterns."""
    pairs = parse_qsl(query, keep_blank_values=True)
    names = sorted(name for name, _ in pairs)
    if names != sorted((PARAM_SHA256, PARAM_PAGE, PARAM_CHUNK)):
        return None
    values = dict(pairs)
    file_sha256, page, chunk_id = (
        values[PARAM_SHA256],
        values[PARAM_PAGE],
        values[PARAM_CHUNK],
    )
    if (
        SHA256_PATTERN.fullmatch(file_sha256) is None
        or PAGE_NUMBER_PATTERN.fullmatch(page) is None
        or CHUNK_ID_PATTERN.fullmatch(chunk_id) is None
    ):
        return None
    return file_sha256, int(page), chunk_id


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode(_ENCODING, errors="replace")


def _encoded(fragment: str) -> bytes:
    # A lone surrogate, which JSON can carry, cannot be encoded as UTF-8.
    return fragment.encode(_ENCODING, errors="replace")


def _server_error(question: str) -> bytes:
    return _encoded(message_fragment(question, texts.SERVER_ERROR))


class DemoHandler(BaseHTTPRequestHandler):
    """One request per connection (D2); see the module docstring."""

    # StreamRequestHandler sets this timeout on the connection's socket (Q8).
    timeout = IDLE_TIMEOUT_S
    server: "DemoServer"

    def setup(self) -> None:
        super().setup()
        self._started = time.perf_counter()
        self._logged_path: Optional[str] = None

    def parse_request(self) -> bool:
        self._started = time.perf_counter()
        return super().parse_request()

    # http.server's own log lines are replaced by R16's request line.
    def log_message(self, format: str, *args: Any) -> None:
        return

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        return

    def send_error(
        self, code: int, message: Optional[str] = None, explain: Optional[str] = None
    ) -> None:
        """Every error ``http.server`` raises itself, with a D16 body (R5)."""
        self._text(code, texts.MALFORMED_REQUEST)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        self._logged_path = url.path
        if url.path == ROOT_PATH:
            page = self.server.app.page
            self._respond(
                STATUS_OK,
                CONTENT_TYPE_HTML,
                page.html,
                [("Content-Security-Policy", page.content_security_policy)],
            )
        elif url.path == PAGE_PATH:
            self._page(url.query)
        elif url.path == HIGHLIGHT_PATH:
            parsed = parse_highlight_query(url.query)
            if parsed is None:
                self._text(STATUS_BAD_REQUEST, texts.MALFORMED_REQUEST)
            else:
                self._respond(
                    STATUS_OK,
                    CONTENT_TYPE_JSON_RESPONSE,
                    self.server.app.highlight(*parsed),
                )
        elif url.path in self.server.app.views:
            self._respond(
                STATUS_OK, CONTENT_TYPE_JSON_RESPONSE, self.server.app.views[url.path]
            )
        else:
            self._text(STATUS_NOT_FOUND, texts.PAGE_NOT_FOUND)

    def do_POST(self) -> None:
        url = urlsplit(self.path)
        self._logged_path = url.path
        body = self._read_body()
        if url.path != QUESTION_PATH:
            self._text(STATUS_BAD_REQUEST, texts.MALFORMED_REQUEST)
            return
        question = None
        if body is not None and self.headers.get_content_type() == CONTENT_TYPE_JSON:
            question = parse_question(body)
        if question is None:
            self._text(STATUS_BAD_REQUEST, texts.MALFORMED_REQUEST)
            return
        status, fragment = self.server.app.question(question)
        self._respond(status, CONTENT_TYPE_HTML, fragment)

    def _read_body(self) -> Optional[bytes]:
        """The body, read when one ``Content-Length`` of at most
        :data:`MAX_BODY_BYTES` announces it; else None, and nothing read."""
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or _DIGITS.fullmatch(lengths[0].strip()) is None:
            return None
        length = int(lengths[0].strip())
        if length > MAX_BODY_BYTES:
            return None
        try:
            body = self.rfile.read(length)
        except OSError:
            return None
        return body if len(body) == length else None

    def _page(self, query: str) -> None:
        parsed = parse_page_query(query)
        if parsed is None:
            self._text(STATUS_BAD_REQUEST, texts.MALFORMED_REQUEST)
            return
        file_sha256, page = parsed
        self._logged_path = page_url(file_sha256, page)
        try:
            png = self.server.app.page_png(file_sha256, page)
        except PageRefused as refused:
            logger.warning(
                "Page request %s page %d: %s", file_sha256, page, refused.kind
            )
            self._text(STATUS_NOT_FOUND, texts.PAGE_NOT_FOUND)
            return
        self._respond(STATUS_OK, CONTENT_TYPE_PNG, png)

    def _text(self, status: int, text: str) -> None:
        self._respond(status, CONTENT_TYPE_TEXT, text.encode(_ENCODING))

    def _respond(
        self,
        status: int,
        content_type: str,
        body: bytes,
        headers: Iterable[Tuple[str, str]] = (),
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)
        finally:
            self.close_connection = True
            self._request_line(status)

    def _request_line(self, status: int) -> None:
        """R16's line: method, path, status and elapsed milliseconds."""
        path = self._logged_path
        if path is None:
            path = _NO_VALUE
        elapsed_ms = (time.perf_counter() - self._started) * 1000.0
        method = self.command or _NO_VALUE
        sys.stderr.write(f"{method} {path} {status} {elapsed_ms:.0f} ms\n")
        sys.stderr.flush()


class DemoServer(HTTPServer):
    """The single-threaded server on 127.0.0.1 (D1), bound exclusively."""

    allow_reuse_address = False
    allow_reuse_port = False

    def __init__(self, port: int) -> None:
        self.app: Optional[DemoApp] = None
        super().__init__((HOST, port), DemoHandler)

    def server_bind(self) -> None:
        """Bind without ``SO_REUSEADDR`` and, where the platform has it, with
        ``SO_EXCLUSIVEADDRUSE``; no name lookup of the host."""
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


def serve(server: DemoServer, stop: threading.Event) -> None:
    """Handle requests, one at a time, until ``stop`` is set; Ctrl+C raises
    ``KeyboardInterrupt`` out of here."""
    server.timeout = POLL_INTERVAL_S
    while not stop.is_set():
        server.handle_request()
