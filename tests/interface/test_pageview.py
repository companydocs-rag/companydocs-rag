"""The page view and the hash to file map (spec 009, D11, R12, R13; Q3, Q4;
A8, A9)."""

from __future__ import annotations

import io
import logging
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

from companydocs_rag.ingestion.parse_pass import file_sha256
from companydocs_rag.interface import texts
from companydocs_rag.interface.pageview import build_file_map
from tests.answering._helpers import (
    FakeRetriever,
    Generator,
    simple_result,
    write_manifest,
)
from tests.embeddings._helpers import write_settings
from tests.interface._helpers import (
    FakeAnswerer,
    MainRun,
    Recorder,
    free_port,
    raw_request,
    request,
    running_server,
    snapshot,
    split_response,
    write_pdf,
)

SERVER_LOGGER = "companydocs_rag.interface.server"
MAP_LOGGER = "companydocs_rag.interface.pageview"
DPI = 100
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
FOLDER_NAMES = ("fictiefmap-boven", "fictiefmap-onder", "fictiefmap-los")
FILE_NAMES = (
    "fictief-twee-paginas.pdf",
    "fictief-een-pagina.pdf",
    "fictief-kapot.pdf",
)
TWO_PAGE_SIZES = ((200, 300), (333, 151))


@pytest.fixture
def raw(tmp_path, monkeypatch):
    """Fabricated PDFs in nested folders, one with two pages of different
    sizes, one file that is not a PDF; ``render_dpi`` is 100."""
    write_settings(
        tmp_path,
        monkeypatch,
        {
            "answering": {"warmup": False, "log_dir": str(tmp_path / "logs")},
            "interface": {"render_dpi": DPI},
        },
    )
    folder = tmp_path / "raw"
    write_pdf(
        folder / "fictiefmap-boven" / "fictiefmap-onder" / "fictief-twee-paginas.pdf",
        *TWO_PAGE_SIZES,
    )
    write_pdf(folder / "fictiefmap-los" / "fictief-een-pagina.pdf", (240, 160))
    broken = folder / "fictiefmap-los" / "fictief-kapot.pdf"
    broken.write_bytes(b"geen pdf, fictieve inhoud")
    return folder


def _hash(raw, name):
    (path,) = raw.rglob(name)
    return file_sha256(path)


def _page(port, query):
    return request(port, "GET", f"/pagina?{query}")


# --- A valid request ------------------------------------------------------------------


def test_a_valid_request_gives_the_page_as_a_colour_png(raw):
    digest = _hash(raw, "fictief-twee-paginas.pdf")

    with running_server(FakeAnswerer(), raw) as server:
        responses = [_page(server.port, f"sha256={digest}&nummer={n}") for n in (1, 2)]

    for response, (width, height) in zip(responses, TWO_PAGE_SIZES):
        assert response.status == 200
        assert response.headers["Content-Type"] == "image/png"
        assert response.body.startswith(PNG_SIGNATURE)
        image = Image.open(io.BytesIO(response.body))
        assert image.mode == "RGB"
        red, green, blue = image.getpixel((image.width // 2, image.height // 2))[:3]
        assert red > green + 100 and red > blue + 100
        assert abs(image.width - width * DPI / 72) <= 1
        assert abs(image.height - height * DPI / 72) <= 1


# --- 400 and 404 ----------------------------------------------------------------------


def _malformed(digest):
    other = "&nummer=1"
    return [
        "sha256=..%2F..%2Fx" + other,
        "sha256=C%3A%5Cx" + other,
        "sha256=%2Fetc%2Fpasswd" + other,
        "sha256=a%2F.." + other,
        "sha256=fictief-twee-paginas.pdf" + other,
        "sha256=fictiefmap-boven%2Ffictief-twee-paginas.pdf" + other,
        f"sha256={digest.upper()}" + other,
        f"sha256={digest[:63]}" + other,
        f"sha256={digest}0" + other,
        f"sha256={digest[:63]}g" + other,
        "sha256=" + other,
        "nummer=1",
        f"sha256={digest}",
        f"sha256={digest}&sha256={digest}&nummer=1",
        f"sha256={digest}&nummer=1&nummer=1",
        f"sha256={digest}&nummer=1&extra=1",
        "",
        *[
            f"sha256={digest}&nummer={number}"
            for number in (
                "0",
                "-1",
                "%2B1",
                "+1",
                "01",
                "1.5",
                "abc",
                "%201",
                "1234567",
                "",
            )
        ],
    ]


def _not_found(raw):
    return [
        ("0" * 64, 1, "unknown"),
        (_hash(raw, "fictief-twee-paginas.pdf"), 3, "out of range"),
        (_hash(raw, "fictief-een-pagina.pdf"), 2, "out of range"),
        (_hash(raw, "fictief-kapot.pdf"), 1, "render"),
    ]


def _forbidden_strings(raw):
    return [
        str(raw),
        raw.as_posix(),
        str(raw.parent),
        *FOLDER_NAMES,
        *FILE_NAMES,
        *(name[: -len(".pdf")] for name in FILE_NAMES),
    ]


def test_refused_requests_give_400_or_404_and_name_no_path(raw, caplog, capsys):
    digest = _hash(raw, "fictief-twee-paginas.pdf")
    before = snapshot(raw)
    capsys.readouterr()

    with caplog.at_level(logging.WARNING):
        with running_server(FakeAnswerer(), raw) as server:
            capsys.readouterr()  # the startup lines may name files
            caplog.clear()
            malformed = [
                raw_request(
                    server.port,
                    f"GET /pagina?{query} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode(
                        "ascii"
                    ),
                )
                for query in _malformed(digest)
            ]
            not_found = [
                raw_request(
                    server.port,
                    f"GET /pagina?sha256={sha}&nummer={n} HTTP/1.1\r\n"
                    "Host: 127.0.0.1\r\n\r\n".encode("ascii"),
                )
                for sha, n, _ in _not_found(raw)
            ]
            stderr = capsys.readouterr().err
            warnings = [
                r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
            ]

    for raw_response in malformed:
        status_line, _, body = split_response(raw_response)
        assert status_line.split()[1] == "400", status_line
        assert body.decode("utf-8") == texts.MALFORMED_REQUEST
    for raw_response in not_found:
        status_line, _, body = split_response(raw_response)
        assert status_line.split()[1] == "404", status_line
        assert body.decode("utf-8") == texts.PAGE_NOT_FOUND
    assert warnings == [
        f"Page request {sha} page {n}: {kind}" for sha, n, kind in _not_found(raw)
    ]
    request_lines = stderr.strip().splitlines()
    assert len(request_lines) == len(malformed) + len(not_found)
    for forbidden in _forbidden_strings(raw):
        for raw_response in malformed + not_found:
            assert forbidden.encode("utf-8") not in raw_response, forbidden
        assert forbidden not in stderr, forbidden
        assert not any(forbidden in message for message in warnings), forbidden
    for sha, n, _ in _not_found(raw):
        assert f"GET /pagina?sha256={sha}&nummer={n} 404 " in stderr

    assert snapshot(raw) == before


def test_a_file_changed_after_startup_gives_404(raw, caplog):
    path = next(raw.rglob("fictief-een-pagina.pdf"))
    digest = file_sha256(path)

    with caplog.at_level(logging.WARNING, logger=SERVER_LOGGER):
        with running_server(FakeAnswerer(), raw) as server:
            assert _page(server.port, f"sha256={digest}&nummer=1").status == 200
            write_pdf(path, (240, 160), colour=(30, 200, 30))
            assert file_sha256(path) != digest
            response = _page(server.port, f"sha256={digest}&nummer=1")

    assert response.status == 404
    assert response.text == texts.PAGE_NOT_FOUND
    assert [r.getMessage() for r in caplog.records if r.name == SERVER_LOGGER] == [
        f"Page request {digest} page 1: changed"
    ]


def test_a_file_deleted_after_startup_gives_404(raw, caplog):
    path = next(raw.rglob("fictief-een-pagina.pdf"))
    digest = file_sha256(path)

    with caplog.at_level(logging.WARNING, logger=SERVER_LOGGER):
        with running_server(FakeAnswerer(), raw) as server:
            assert _page(server.port, f"sha256={digest}&nummer=1").status == 200
            path.unlink()
            response = _page(server.port, f"sha256={digest}&nummer=1")

    assert response.status == 404
    assert response.text == texts.PAGE_NOT_FOUND
    assert [r.getMessage() for r in caplog.records if r.name == SERVER_LOGGER] == [
        f"Page request {digest} page 1: unreadable"
    ]


def test_a_page_added_after_startup_is_unknown(raw):
    with running_server(FakeAnswerer(), raw) as server:
        late = write_pdf(raw / "fictief-later.pdf", (100, 100))
        response = _page(server.port, f"sha256={file_sha256(late)}&nummer=1")

    assert response.status == 404


# --- A9: the map (R13) ----------------------------------------------------------------


def test_the_map_holds_every_pdf_under_the_folder_by_its_sha256(tmp_path, caplog):
    raw = tmp_path / "raw"
    top = write_pdf(raw / "fictief-boven.pdf", (100, 100))
    nested = write_pdf(raw / "fictiefmap" / "diep" / "fictief-diep.pdf", (120, 100))
    write_pdf(raw / ".fictief-verborgen.pdf", (140, 100))
    (raw / "fictief-notitie.txt").write_text("geen pdf", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=MAP_LOGGER):
        files = build_file_map(raw)

    assert files == {file_sha256(top): top, file_sha256(nested): nested}
    assert caplog.records == []


def test_identical_files_give_one_entry_the_first_and_one_warning(tmp_path, caplog):
    raw = tmp_path / "raw"
    first = write_pdf(raw / "a-fictief.pdf", (100, 100))
    second = raw / "b-fictief.pdf"
    shutil.copyfile(first, second)
    digest = file_sha256(first)

    with caplog.at_level(logging.WARNING, logger=MAP_LOGGER):
        files = build_file_map(raw)

    assert files == {digest: first}
    assert [r.getMessage() for r in caplog.records] == [
        f"2 files share sha256 {digest}; the first is used"
    ]


def test_a_file_that_cannot_be_read_is_left_out_with_a_warning(
    tmp_path, caplog, monkeypatch
):
    raw = tmp_path / "raw"
    readable = write_pdf(raw / "fictief-leesbaar.pdf", (100, 100))
    locked = write_pdf(raw / "fictief-slot.pdf", (120, 100))
    real_open = Path.open

    def open_unless_locked(self, *args, **kwargs):
        if self.name == locked.name:
            raise PermissionError(13, "fictief: geen toegang", str(self))
        return real_open(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_unless_locked)
        with caplog.at_level(logging.WARNING, logger=MAP_LOGGER):
            files = build_file_map(raw)

    assert files == {file_sha256(readable): readable}
    (message,) = [r.getMessage() for r in caplog.records]
    assert message.startswith(f"Could not read {locked}")
    assert "PermissionError" in message


def test_an_empty_raw_folder_gives_one_warning_and_the_server_starts(
    tmp_path, monkeypatch, caplog
):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_manifest(tmp_path / "index")
    port = free_port()
    stdout = Recorder()
    monkeypatch.setattr(sys, "stdout", stdout)

    with caplog.at_level(logging.WARNING, logger=MAP_LOGGER):
        run = MainRun(
            ["--index-dir", str(tmp_path / "index"), "--port", str(port)],
            raw_dir=raw,
            type_overrides=tmp_path / "geen-typen.csv",
            retriever=FakeRetriever(simple_result()),
            transport=Generator().transport,
        ).start()
        try:
            run.wait_for_url(stdout)
            status = request(port, "GET", "/").status
        finally:
            code = run.finish()

    assert (code, status) == (0, 200)
    assert [r.getMessage() for r in caplog.records if r.name == MAP_LOGGER] == [
        f"No PDF files under {raw}"
    ]
