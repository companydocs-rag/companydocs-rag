"""The highlight of a cited passage on its page (spec 011).

Every document, chunk file and OCR cache entry here is fictitious and
written under ``tmp_path``: an image PDF with an OCR cache entry for the
OCR route, and a small hand-written PDF with a text layer for the native
route.
"""

from __future__ import annotations

import json
import os

import pytest

from companydocs_rag.ingestion.parse_pass import file_sha256
from companydocs_rag.interface.fragments import card_html
from companydocs_rag.interface.highlight import (
    Highlighter,
    Word,
    index_ocr_cache,
    line_boxes,
    matched_run,
    passages,
    word_key,
)
from companydocs_rag.interface.pageview import build_file_map
from tests.interface._helpers import (
    FakeAnswerer,
    parse_html,
    request,
    running_server,
    source,
    write_pdf,
)

CHUNK_ID = "0123456789abcdef0123"
TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num"
    "\tleft\ttop\twidth\theight\tconf\ttext"
)
LINES = [
    ["De", "huurder", "betaalt", "maandelijks"],
    ["een", "vast", "bedrag", "vooraf"],
    ["Artikel", "zes", "gaat", "over", "iets", "anders"],
]


def _tsv(lines, width=1000, height=2000):
    rows = [TSV_HEADER, f"1\t1\t0\t0\t0\t0\t0\t0\t{width}\t{height}\t-1\t"]
    for number, words in enumerate(lines, start=1):
        for position, text in enumerate(words, start=1):
            left, top = 100 + 120 * (position - 1), 100 * number
            rows.append(
                f"5\t1\t1\t1\t{number}\t{position}\t{left}\t{top}\t100\t40\t95\t{text}"
            )
    return "\n".join(rows) + "\n"


def _ocr_entry(folder, digest, page, tsv, name="entry"):
    folder.mkdir(parents=True, exist_ok=True)
    key = {"cache_format_version": 1, "sha256": digest, "page_number": page}
    path = folder / f"{name}.json"
    path.write_text(json.dumps({"key": key, "text": "", "tsv": tsv}), encoding="utf-8")
    return path


def _chunk_file(folder, digest, chunks):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "fictief.json"
    path.write_text(json.dumps({"chunks": chunks}), encoding="utf-8")
    return path


def _chunk(digest, text, spans, chunk_id=CHUNK_ID):
    return {
        "chunk_id": chunk_id,
        "file_sha256": digest,
        "text": text,
        "page_spans": spans,
    }


def _span(page, start, end, origin="ocr"):
    return {
        "page_number": page,
        "char_start": start,
        "char_end": end,
        "text_origin": origin,
    }


def _text_pdf(path, lines, rotate=0):
    """A one-page PDF with a text layer: one line of Helvetica per entry."""
    content = (
        "BT /F1 12 Tf 14 TL 72 700 Td "
        + " ".join(f"({line}) Tj T*" for line in lines)
        + " ET"
    )
    stream = content.encode("ascii")
    page = (
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R"
        + (f" /Rotate {rotate}" if rotate else "")
        + " >>"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        page.encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def scan(tmp_path):
    raw = tmp_path / "raw"
    pdf = write_pdf(raw / "fictief-scan.pdf", (200, 300))
    return raw, file_sha256(pdf)


# --- Passages and matching ------------------------------------------------------------


def test_the_spans_split_the_chunk_text_at_one_separator():
    text = "eerste pagina tekst" + "\n" + "tweede"
    chunk = _chunk("a" * 64, text, [_span(1, 40, 59), _span(2, 0, 6)])

    found = passages(chunk)

    assert found[1].text == "eerste pagina tekst"
    assert found[2].text == "tweede"
    assert found[1].text_origin == "ocr"


def test_lengths_that_do_not_add_up_give_the_whole_text_per_page():
    chunk = _chunk("a" * 64, "abc def ghi", [_span(1, 0, 3), _span(2, 0, 20)])

    found = passages(chunk)

    assert found[1].text == found[2].text == "abc def ghi"


def test_words_are_compared_case_folded_and_without_punctuation():
    assert word_key("Huurder,") == "huurder"
    assert word_key("7:230a") == "7230a"
    assert word_key("--") == ""


def test_an_exact_passage_matches_its_run():
    page = ["a", "b", "c", "d", "e", "f", "g", "h"]

    assert matched_run(page, ["c", "d", "e", "f", "g", "h"]) == (2, 7)


def test_a_weak_or_short_alignment_gives_no_run():
    page = ["a", "b", "c", "d", "e", "f", "g", "h"]

    assert matched_run(page, ["x", "y", "z", "a", "q", "r"]) is None
    assert matched_run(page, ["c", "d"]) is None
    assert matched_run([], ["a", "b", "c"]) is None


def test_one_box_per_line():
    words = [
        Word("a", (1,), 0.1, 0.1, 0.2, 0.12),
        Word("b", (1,), 0.25, 0.1, 0.3, 0.13),
        Word("c", (2,), 0.1, 0.2, 0.15, 0.22),
    ]

    assert line_boxes(words) == [[0.1, 0.1, 0.2, 0.03], [0.1, 0.2, 0.05, 0.02]]


# --- The OCR route --------------------------------------------------------------------


def test_an_ocr_passage_is_highlighted_line_by_line(tmp_path, scan):
    raw, digest = scan
    text = "De huurder betaalt maandelijks een vast bedrag vooraf"
    _chunk_file(
        tmp_path / "chunks", digest, [_chunk(digest, text, [_span(1, 0, len(text))])]
    )
    _ocr_entry(tmp_path / "ocr", digest, 1, _tsv(LINES))
    highlighter = Highlighter(
        build_file_map(raw), tmp_path / "chunks", tmp_path / "ocr"
    )

    boxes = highlighter.boxes(digest, 1, CHUNK_ID)

    assert boxes == [[0.1, 0.05, 0.46, 0.02], [0.1, 0.1, 0.46, 0.02]]


def test_no_highlight_for_another_page_file_or_chunk(tmp_path, scan):
    raw, digest = scan
    text = "De huurder betaalt maandelijks een vast bedrag vooraf"
    _chunk_file(
        tmp_path / "chunks", digest, [_chunk(digest, text, [_span(1, 0, len(text))])]
    )
    _ocr_entry(tmp_path / "ocr", digest, 1, _tsv(LINES))
    highlighter = Highlighter(
        build_file_map(raw), tmp_path / "chunks", tmp_path / "ocr"
    )

    assert highlighter.boxes(digest, 2, CHUNK_ID) == []
    assert highlighter.boxes("0" * 64, 1, CHUNK_ID) == []
    assert highlighter.boxes(digest, 1, "f" * 20) == []


def test_a_passage_the_page_does_not_hold_is_not_highlighted(tmp_path, scan):
    raw, digest = scan
    text = "Een heel andere tekst die nergens op deze pagina staat"
    _chunk_file(
        tmp_path / "chunks", digest, [_chunk(digest, text, [_span(1, 0, len(text))])]
    )
    _ocr_entry(tmp_path / "ocr", digest, 1, _tsv(LINES))
    highlighter = Highlighter(
        build_file_map(raw), tmp_path / "chunks", tmp_path / "ocr"
    )

    assert highlighter.boxes(digest, 1, CHUNK_ID) == []


def test_without_a_cache_entry_there_is_no_highlight(tmp_path, scan):
    raw, digest = scan
    text = "De huurder betaalt maandelijks een vast bedrag vooraf"
    _chunk_file(
        tmp_path / "chunks", digest, [_chunk(digest, text, [_span(1, 0, len(text))])]
    )
    highlighter = Highlighter(
        build_file_map(raw), tmp_path / "chunks", tmp_path / "geen"
    )

    assert highlighter.boxes(digest, 1, CHUNK_ID) == []


def test_the_newest_of_two_cache_entries_for_a_page_is_used(tmp_path):
    digest = "b" * 64
    old = _ocr_entry(tmp_path, digest, 3, "", name="oud")
    new = _ocr_entry(tmp_path, digest, 3, "", name="nieuw")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    assert index_ocr_cache(tmp_path) == {(digest, 3): new}


# --- The native route -----------------------------------------------------------------


def test_a_text_layer_passage_is_highlighted(tmp_path):
    raw = tmp_path / "raw"
    pdf = _text_pdf(
        raw / "fictief-tekst.pdf",
        ["De huurder betaalt maandelijks", "een vast bedrag vooraf", "Artikel zes"],
    )
    digest = file_sha256(pdf)
    text = "De huurder betaalt maandelijks een vast bedrag vooraf"
    span = _span(1, 0, len(text), origin="native")
    _chunk_file(tmp_path / "chunks", digest, [_chunk(digest, text, [span])])
    highlighter = Highlighter(build_file_map(raw), tmp_path / "chunks", None)

    boxes = highlighter.boxes(digest, 1, CHUNK_ID)

    assert len(boxes) == 2
    for left, top, width, height in boxes:
        assert 0 < left < 1 and 0 < top < 1 and width > 0 and height > 0
    assert boxes[0][1] < boxes[1][1]


def test_a_rotated_text_layer_page_is_not_highlighted(tmp_path):
    raw = tmp_path / "raw"
    pdf = _text_pdf(
        raw / "fictief-gedraaid.pdf", ["De huurder betaalt maandelijks"], rotate=90
    )
    digest = file_sha256(pdf)
    text = "De huurder betaalt maandelijks"
    span = _span(1, 0, len(text), origin="native")
    _chunk_file(tmp_path / "chunks", digest, [_chunk(digest, text, [span])])
    highlighter = Highlighter(build_file_map(raw), tmp_path / "chunks", None)

    assert highlighter.boxes(digest, 1, CHUNK_ID) == []


# --- The card and the route -----------------------------------------------------------


def test_a_card_carries_a_valid_chunk_id_only(scan):
    raw, digest = scan
    files = build_file_map(raw)
    valid = {**source(1, file_sha256=digest), "chunk_id": CHUNK_ID}
    invalid = {**source(2, file_sha256=digest), "chunk_id": 'x" onclick="y'}

    (card,) = parse_html(card_html(valid, 1, files, {})).find_all("div", "bron")
    (other,) = parse_html(card_html(invalid, 1, files, {})).find_all("div", "bron")

    assert card.attrs["data-chunk"] == CHUNK_ID
    assert "data-chunk" not in other.attrs


def test_the_route_answers_boxes_or_400(tmp_path, scan):
    raw, digest = scan
    text = "De huurder betaalt maandelijks een vast bedrag vooraf"
    _chunk_file(
        tmp_path / "chunks", digest, [_chunk(digest, text, [_span(1, 0, len(text))])]
    )
    _ocr_entry(tmp_path / "ocr", digest, 1, _tsv(LINES))
    highlighter = Highlighter(
        build_file_map(raw), tmp_path / "chunks", tmp_path / "ocr"
    )
    query = f"/markering?sha256={digest}&nummer=1&chunk={CHUNK_ID}"

    with running_server(FakeAnswerer(), raw, highlighter=highlighter) as server:
        found = request(server.port, "GET", query)
        malformed = [
            request(server.port, "GET", path)
            for path in (
                f"/markering?sha256={digest}&nummer=1",
                f"/markering?sha256={digest}&nummer=0&chunk={CHUNK_ID}",
                f"/markering?sha256={digest}&nummer=1&chunk=XYZ",
                f"{query}&chunk={CHUNK_ID}",
            )
        ]

    assert found.status == 200
    assert found.headers["Content-Type"] == "application/json; charset=utf-8"
    assert len(json.loads(found.body)["boxes"]) == 2
    assert [response.status for response in malformed] == [400, 400, 400, 400]


def test_without_a_highlighter_the_route_answers_no_boxes(scan):
    raw, digest = scan

    with running_server(FakeAnswerer(), raw) as server:
        response = request(
            server.port, "GET", f"/markering?sha256={digest}&nummer=1&chunk={CHUNK_ID}"
        )

    assert response.status == 200
    assert json.loads(response.body) == {"boxes": []}
