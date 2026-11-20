"""The library and the status block (spec 011).

The documents are fictitious PDFs under ``tmp_path``.
"""

from __future__ import annotations

import json

import pytest

from companydocs_rag.ingestion.parse_pass import file_sha256
from companydocs_rag.interface import texts
from companydocs_rag.interface.library import (
    document_library,
    page_count,
    status_block,
)
from companydocs_rag.interface.pageview import build_file_map
from tests.interface._helpers import (
    FakeAnswerer,
    request,
    running_server,
    write_pdf,
)


@pytest.fixture
def raw(tmp_path):
    folder = tmp_path / "raw"
    write_pdf(
        folder / "levering-b" / "02_Huur" / "fictief-huur.pdf", (200, 300), (200, 300)
    )
    write_pdf(folder / "levering-a" / "fictief-statuten.pdf", (200, 300))
    return folder


# --- The library ------------------------------------------------------------------


def test_the_library_lists_every_mapped_pdf_with_folders_type_and_pages(raw):
    files = build_file_map(raw)
    huur = file_sha256(raw / "levering-b" / "02_Huur" / "fictief-huur.pdf")
    types = {digest: None for digest in files}
    types[huur] = "Huurovereenkomst"

    library = document_library(raw, files, types)

    assert [entry["name"] for entry in library] == [
        "fictief-statuten.pdf",
        "fictief-huur.pdf",
    ]
    first, second = library
    assert first["folders"] == ["levering-a"]
    assert first["type"] == texts.NO_DOCUMENT_TYPE
    assert (first["pages"], first["pages_label"]) == (1, "1 pagina")
    assert second == {
        "sha256": huur,
        "name": "fictief-huur.pdf",
        "folders": ["levering-b", "02_Huur"],
        "type": "Huurovereenkomst",
        "pages": 2,
        "pages_label": "2 pagina's",
    }


def test_a_pdf_that_cannot_be_opened_has_no_page_count(tmp_path):
    broken = tmp_path / "kapot.pdf"
    broken.write_bytes(b"%PDF-1.4 geen echte inhoud")

    assert page_count(broken) is None


# --- The status block -------------------------------------------------------------


def test_the_status_tells_a_local_endpoint_from_a_remote_one():
    local = status_block("http://127.0.0.1:11434/v1", "lokaal-model", 53)
    remote = status_block("https://api.example.com/v1", "extern-model", 1)

    assert local == {
        "mode": texts.STATUS_LOCAL,
        "model": "lokaal-model",
        "host": "",
        "documents": "53 documenten",
    }
    assert remote["mode"] == texts.STATUS_API
    assert remote["host"] == "api.example.com"


# --- The routes -------------------------------------------------------------------


def test_the_two_views_are_served_as_json(raw):
    files = build_file_map(raw)
    library = document_library(raw, files, {digest: None for digest in files})
    status = status_block("http://127.0.0.1:11434/v1", "lokaal-model", len(files))

    with running_server(FakeAnswerer(), raw, library=library, status=status) as server:
        responses = {
            path: request(server.port, "GET", path)
            for path in ("/documenten", "/status")
        }
        follow_up = request(server.port, "GET", "/opvolging")

    for path, response in responses.items():
        assert response.status == 200, path
        assert response.headers["Content-Type"] == "application/json; charset=utf-8"
        assert response.headers["Cache-Control"] == "no-store"
    assert json.loads(responses["/documenten"].body) == {"documents": library}
    assert json.loads(responses["/status"].body) == status
    # The follow-up view was removed from the interface.
    assert follow_up.status == 404


def test_without_views_the_routes_answer_empty(raw):
    with running_server(FakeAnswerer(), raw) as server:
        library = request(server.port, "GET", "/documenten")
        status = request(server.port, "GET", "/status")

    assert json.loads(library.body) == {"documents": []}
    assert json.loads(status.body) == {}
