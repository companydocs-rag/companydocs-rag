"""Shared helpers for the spec 009 tests: the fake answerer, fabricated
``AnswerResult`` objects, fabricated PDFs, a server runner, a runner for
``main``, HTTP clients and a small HTML tree.

Every path a test writes is under ``tmp_path``; the PDFs are generated in
the test from Pillow images, with fabricated names. Text is fabricated,
never taken from a contract. :class:`FakeAnswerer` has ``answer(question)``
and ``close()`` and records its calls. :func:`running_server` runs the
server in a thread of the test process on 127.0.0.1 with any such answerer;
:class:`MainRun` runs ``main`` in a thread until its stop seam is set. A port
comes from :func:`free_port`, which registers it, so the request guard in
``conftest.py`` lets connections to it through.
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from PIL import Image

from companydocs_rag.answering.answerer import AnswerResult
from companydocs_rag.answering.provenance import page_provenance
from companydocs_rag.interface import cli
from companydocs_rag.interface.htmlpage import build_page
from companydocs_rag.interface.pageview import build_file_map, document_types
from companydocs_rag.interface.server import DemoApp, DemoServer, serve
from companydocs_rag.interface.settings import interface_settings
from companydocs_rag.metadata.doctype import read_type_overrides

HOST = "127.0.0.1"
# Ports of servers under test; conftest.py's guard lets connections to them
# through and clears the set after each test.
SERVER_PORTS: Set[int] = set()
REQUEST_TIMEOUT_S = 60.0
SHA_A = "a1" * 32
SHA_B = "b2" * 32
UTF8_BOM = b"\xef\xbb\xbf"


def free_port() -> int:
    """A port on 127.0.0.1 that nothing listens on now, registered for the
    server under test."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        port = probe.getsockname()[1]
    SERVER_PORTS.add(port)
    return port


def wait_until(predicate: Callable[[], bool], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("the condition did not hold in time")
        time.sleep(0.02)


# --- Fabricated PDFs and folders ------------------------------------------------------


def write_pdf(
    path: Path,
    *sizes: Tuple[int, int],
    colour: Tuple[int, int, int] = (200, 30, 30),
) -> Path:
    """A PDF with one page per size, in points (72 dpi), each page filled
    with ``colour``."""
    pages = sizes or ((200, 300),)
    images = [Image.new("RGB", size, colour) for size in pages]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, "PDF", resolution=72.0, save_all=True, append_images=images[1:]
    )
    return path


def write_csv(path: Path, rows: Sequence[Sequence[str]]) -> Path:
    """A ``;``-separated file, UTF-8 with BOM, as the user's files are."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(";".join(row) + "\n" for row in rows)
    path.write_bytes(UTF8_BOM + text.encode("utf-8"))
    return path


def write_overrides(path: Path, rows: Sequence[Sequence[str]]) -> Path:
    return write_csv(path, [("Bestandsnaam", "Documenttype"), *rows])


def write_index(folder: Path, rows: Sequence[Sequence[str]]) -> Path:
    """A client index ``00_INDEX.csv`` in ``folder``: (Map, Bestandsnaam,
    Documenttype) rows."""
    return write_csv(
        folder / "00_INDEX.csv", [("Map", "Bestandsnaam", "Documenttype"), *rows]
    )


def snapshot(folder: Path) -> Dict[str, Tuple[bytes, int]]:
    """Every file under ``folder`` with its bytes and modification time."""
    return {
        path.relative_to(folder).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


# --- Fabricated answers ---------------------------------------------------------------


def page_entry(
    page: int,
    *,
    origin: Any = "native",
    tier: Any = "Clean",
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """One entry of a source's ``pages``, as spec 007 writes it."""
    return {
        "page": page,
        "text_origin": origin,
        "tier": tier,
        "ocr_mean_confidence": confidence,
        "provenance": page_provenance(origin, tier),
    }


def source(
    number: int,
    *,
    file_sha256: str = SHA_A,
    file_path: str = "bron/fictief-huur.pdf",
    clause_label: Optional[str] = None,
    pages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """One entry of an ``AnswerResult``'s ``sources``, keys in spec 007's
    order."""
    entries = pages if pages is not None else [page_entry(1)]
    return {
        "prompt_number": number,
        "chunk_id": f"c{number:02d}fictief",
        "file_sha256": file_sha256,
        "file_path": file_path,
        "page_start": entries[0]["page"] if entries else 1,
        "page_end": entries[-1]["page"] if entries else 1,
        "clause_label": clause_label,
        "pages": entries,
    }


def answer_result(
    status: str = "answered",
    *,
    answer_text: Optional[str] = "Fictief antwoord [1].",
    sources: Optional[List[Dict[str, Any]]] = None,
    abstention_source: Optional[str] = None,
    error: Optional[Dict[str, str]] = None,
    question: str = "Fictieve vraag?",
    user_message: str = "Fictieve bronnen.",
    scores: Sequence[float] = (0.7311, 0.6522),
) -> AnswerResult:
    """A fabricated ``AnswerResult``: every field filled, with fabricated
    candidates whose scores are ``scores``."""
    found = sources if sources is not None else []
    candidates = [
        {
            "rank": rank,
            "chunk_id": f"c{rank:02d}fictief",
            "rrf_score": score,
            "dense_rank": rank,
            "dense_score": score,
            "sparse_rank": rank,
            "sparse_score": score * 10,
            "file_sha256": SHA_A,
            "page_start": 1,
            "page_end": 1,
        }
        for rank, score in enumerate(scores, start=1)
    ]
    return AnswerResult(
        schema_version=1,
        timestamp_utc="2031-01-02T03:04:05.000Z",
        question_id=None,
        question=question,
        status=status,
        abstention_source=abstention_source,
        answer_text=answer_text,
        sources=found,
        citations={"valid": [s["prompt_number"] for s in found], "invalid": []},
        context=[],
        dropped_for_budget=[],
        candidates=candidates,
        path_hits=None,
        prompt={
            "system_file": "config/prompts/answer_system_nl.txt",
            "system_sha256": "0" * 64,
            "user_message": user_message,
        },
        generation=None,
        retrieval_ms=12.5,
        settings={},
        index={},
        error=error,
    )


class FakeAnswerer:
    """An object with ``answer(question)`` and ``close()``. Each call takes
    the next outcome: an ``AnswerResult``, an exception to raise, or a
    function of the question; when they run out, ``default`` answers. It
    records every call's arguments and every ``close``."""

    def __init__(self, *outcomes: Any, default: Any = None) -> None:
        self.outcomes = list(outcomes)
        self.default = default if default is not None else answer_result()
        self.calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        self.closed = 0

    def answer(self, *args: Any, **kwargs: Any) -> AnswerResult:
        self.calls.append((args, kwargs))
        outcome = self.outcomes.pop(0) if self.outcomes else self.default
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(*args)
        return outcome

    def close(self) -> None:
        self.closed += 1

    @property
    def questions(self) -> List[Any]:
        return [args[0] if args else None for args, _ in self.calls]


# --- The server under test ------------------------------------------------------------


@dataclass
class Running:
    port: int
    server: DemoServer
    files: Dict[str, Path]
    types: Dict[str, Optional[str]]


@contextlib.contextmanager
def running_server(
    answerer: Any, raw_dir: Path, *, type_overrides: Optional[Path] = None
) -> Iterator[Running]:
    """The server on 127.0.0.1 in a thread, built as ``main`` builds it, with
    ``answerer``; the settings are read from the test's settings file."""
    overrides_path = (
        type_overrides
        if type_overrides is not None
        else raw_dir.parent / "geen-typen.csv"
    )
    files = build_file_map(raw_dir)
    types = document_types(files, read_type_overrides(overrides_path))
    port = free_port()
    server = DemoServer(port)
    server.app = DemoApp(answerer, files, types, interface_settings(), build_page())
    stop = threading.Event()
    errors: List[BaseException] = []

    def run() -> None:
        try:
            serve(server, stop)
        except BaseException as error:  # noqa: BLE001 - reported to the test
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield Running(port=port, server=server, files=files, types=types)
    finally:
        stop.set()
        thread.join(timeout=30)
        server.server_close()
    assert not thread.is_alive()
    if errors:
        raise errors[0]


@dataclass
class Spies:
    """What ``main`` built: each attempt to open an ``Answerer``, each opened
    one, each server, and the connections the servers accepted."""

    attempts: int = 0
    answerers: List[Any] = field(default_factory=list)
    servers: List[Dict[str, Any]] = field(default_factory=list)
    accepted: int = 0


def spy_on_main(monkeypatch: Any, stdout: Optional["Recorder"] = None) -> Spies:
    """Replace ``cli.Answerer`` and ``cli.DemoServer`` with recording
    subclasses. An opened answerer records what stdout held when it opened
    and how often it was closed; a server records its socket's address and
    options when it was bound."""
    spies = Spies()

    class RecordingAnswerer(cli.Answerer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            spies.attempts += 1
            super().__init__(*args, **kwargs)
            self.stdout_at_open = None if stdout is None else stdout.getvalue()
            self.close_calls = getattr(self, "close_calls", 0)
            spies.answerers.append(self)

        def close(self) -> None:
            self.close_calls = getattr(self, "close_calls", 0) + 1
            super().close()

    class RecordingServer(cli.DemoServer):
        def __init__(self, port: int) -> None:
            super().__init__(port)
            exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            spies.servers.append(
                {
                    "server": self,
                    "address": self.socket.getsockname(),
                    "reuse_address": self.socket.getsockopt(
                        socket.SOL_SOCKET, socket.SO_REUSEADDR
                    ),
                    "exclusive": None
                    if exclusive is None
                    else self.socket.getsockopt(socket.SOL_SOCKET, exclusive),
                }
            )

        def process_request(self, request: Any, client_address: Any) -> None:
            spies.accepted += 1
            super().process_request(request, client_address)

    monkeypatch.setattr(cli, "Answerer", RecordingAnswerer)
    monkeypatch.setattr(cli, "DemoServer", RecordingServer)
    return spies


class Recorder(io.StringIO):
    """A stdout or stderr that threads can write to while a test reads it."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            return super().write(text)

    def getvalue(self) -> str:
        with self._lock:
            return super().getvalue()


class MainRun:
    """``cli.main`` in a thread until its stop seam is set."""

    def __init__(self, argv: Sequence[str], **seams: Any) -> None:
        self.stop = threading.Event()
        self.code: Optional[int] = None
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run, args=(list(argv), seams), daemon=True
        )

    def _run(self, argv: List[str], seams: Dict[str, Any]) -> None:
        try:
            self.code = cli.main(argv, stop=self.stop, **seams)
        except BaseException as error:  # noqa: BLE001 - reported to the test
            self.error = error

    def start(self) -> "MainRun":
        self._thread.start()
        return self

    def wait_for_url(self, stdout: Recorder) -> None:
        wait_until(lambda: stdout.getvalue() != "" or not self._thread.is_alive())

    def finish(self, timeout: float = 30.0) -> Optional[int]:
        self.stop.set()
        self._thread.join(timeout)
        assert not self._thread.is_alive()
        if self.error is not None:
            raise self.error
        return self.code


# --- HTTP -----------------------------------------------------------------------------


@dataclass
class Response:
    status: int
    headers: Any
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")


def request(
    port: int,
    method: str,
    target: str,
    *,
    body: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Response:
    connection = http.client.HTTPConnection(HOST, port, timeout=REQUEST_TIMEOUT_S)
    try:
        connection.request(method, target, body=body, headers=headers or {})
        response = connection.getresponse()
        return Response(response.status, response.headers, response.read())
    finally:
        connection.close()


def ask(
    port: int, question: Any, *, content_type: str = "application/json"
) -> Response:
    """``POST /vraag`` with ``{"vraag": question}`` as UTF-8 JSON."""
    body = json.dumps({"vraag": question}, ensure_ascii=False).encode("utf-8")
    return request(
        port, "POST", "/vraag", body=body, headers={"Content-Type": content_type}
    )


def raw_request(port: int, data: bytes, *, timeout: float = 10.0) -> bytes:
    """Send ``data`` over a new connection and read until the server closes
    it."""
    with socket.create_connection((HOST, port), timeout=timeout) as connection:
        connection.sendall(data)
        chunks = []
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def split_response(raw: bytes) -> Tuple[str, Dict[str, str], bytes]:
    """The status line, the headers (names lowercased) and the body."""
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("iso-8859-1").split("\r\n")
    headers: Dict[str, str] = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return lines[0], headers, body


def get_request(target: str, *, extra: str = "") -> bytes:
    return (
        f"GET {target} HTTP/1.1\r\nHost: {HOST}\r\nConnection: keep-alive\r\n"
        f"{extra}\r\n"
    ).encode("ascii")


# --- A small HTML tree ----------------------------------------------------------------

VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}


@dataclass
class Node:
    tag: str
    attrs: Dict[str, Optional[str]]
    children: List[Any] = field(default_factory=list)

    @property
    def classes(self) -> List[str]:
        return (self.attrs.get("class") or "").split()

    def text(self) -> str:
        return "".join(
            child if isinstance(child, str) else child.text() for child in self.children
        )

    def iter(self) -> Iterator["Node"]:
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.iter()

    def find_all(
        self, tag: Optional[str] = None, cls: Optional[str] = None
    ) -> List["Node"]:
        return [
            node
            for node in self.iter()
            if (tag is None or node.tag == tag) and (cls is None or cls in node.classes)
        ]

    def find(self, tag: Optional[str] = None, cls: Optional[str] = None) -> "Node":
        (node,) = self.find_all(tag, cls)
        return node


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", {})
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, dict(attrs))
        self.stack[-1].children.append(node)
        if tag not in VOID_ELEMENTS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Node(tag, dict(attrs)))

    def handle_endtag(self, tag):
        for position in range(len(self.stack) - 1, 0, -1):
            if self.stack[position].tag == tag:
                del self.stack[position:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def parse_html(text: str) -> Node:
    builder = _TreeBuilder()
    builder.feed(text)
    builder.close()
    return builder.root
