"""Source cards and provenance labels (spec 009, D9, D10, R10, R11; Q1, Q3,
Q9; A7)."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from companydocs_rag.ingestion.parse_pass import file_sha256
from companydocs_rag.interface import texts
from companydocs_rag.interface.fragments import provenance_label, span_label
from tests.interface._helpers import (
    FakeAnswerer,
    answer_result,
    ask,
    page_entry,
    parse_html,
    request,
    running_server,
    source,
    write_index,
    write_overrides,
    write_pdf,
)

N = 12


# --- D10's mapping (R11) --------------------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "tier", "label"),
    [
        ("native", "Clean", texts.LABEL_TEXT_LAYER),
        ("native", "Decent", texts.LABEL_TEXT_LAYER),
        ("native", "Garbage", texts.label_low_quality(N)),
        ("ocr", None, texts.label_scan(N)),
        ("ocr", "Clean", texts.label_scan(N)),
        ("ocr", "Garbage", texts.label_scan(N)),
        ("none", None, texts.LABEL_NO_TEXT),
        ("none", "Clean", texts.LABEL_NO_TEXT),
        ("native", None, texts.label_unknown(N)),
        ("native", "clean", texts.label_unknown(N)),
        ("native", "Other", texts.label_unknown(N)),
        ("native", 5, texts.label_unknown(N)),
        ("Native", "Clean", texts.label_unknown(N)),
        ("unknown", "Clean", texts.label_unknown(N)),
        (None, "Clean", texts.label_unknown(N)),
        (5, "Clean", texts.label_unknown(N)),
    ],
)
def test_d10s_mapping(origin, tier, label):
    assert provenance_label(origin, tier, N) == label
    assert span_label({"page": N, "text_origin": origin, "tier": tier}) == label


def test_a_span_without_the_two_keys_is_unknown():
    assert span_label({"page": N}) == texts.label_unknown(N)


def test_the_labels_fill_in_the_page_number():
    assert provenance_label("native", "Garbage", 3) == (
        "tekstlaag van slechte kwaliteit, controleer op pagina 3"
    )
    assert provenance_label("ocr", None, 3) == (
        "gelezen van een scan, controleer cijfers op pagina 3"
    )
    assert provenance_label("x", None, 3) == "herkomst onbekend, controleer op pagina 3"


# --- Cards through the server (D9, R10, Q3 (a), Q9) -----------------------------------


@pytest.fixture
def three_documents(tmp_path):
    """Three fabricated PDFs typed at startup from their paths: by the
    override file, by a client index in a parent folder, and not at all."""
    raw = tmp_path / "raw"
    by_override = write_pdf(raw / "los" / "fictief-huur.pdf", (200, 300), (300, 200))
    by_index = write_pdf(raw / "klant" / "dossier" / "fictief-statuten.pdf", (250, 250))
    untyped = write_pdf(raw / "los" / "fictief-onbekend.pdf", (180, 260), (260, 180))
    write_index(raw / "klant", [("dossier", "fictief-statuten.pdf", "Statuten")])
    overrides = write_overrides(
        tmp_path / "handmatig" / "typen.csv", [("fictief-huur.pdf", "Huurovereenkomst")]
    )
    return raw, overrides, [file_sha256(p) for p in (by_override, by_index, untyped)]


def _card(tree, number):
    (card,) = [
        node
        for node in tree.find_all(cls="bron")
        if node.attrs["id"].endswith(f"-{number}")
    ]
    return card


def test_the_cards_show_number_type_name_hint_and_pages_with_links(three_documents):
    raw, overrides, (huur, statuten, onbekend) = three_documents
    sources = [
        source(
            3,
            file_sha256=onbekend,
            pages=[page_entry(2, origin="native", tier="Decent")],
        ),
        source(
            1,
            file_sha256=huur,
            clause_label="Artikel 1 Fictieve huurprijs",
            pages=[
                page_entry(1),
                page_entry(2, origin="ocr", tier=None, confidence=87.654),
            ],
        ),
        source(
            2,
            file_sha256=statuten,
            pages=[page_entry(1, origin="native", tier="Garbage")],
        ),
    ]
    result = answer_result(
        answer_text="Fictief [3], [1] en [2].",
        sources=sources,
        user_message="Fictieve bron, zie pagina 4321.",
        scores=(0.73114, 0.65227),
    )

    with running_server(FakeAnswerer(result), raw, type_overrides=overrides) as server:
        response = ask(server.port, "Fictieve vraag?")
        tree = parse_html(response.text)
        links = {
            (a.attrs["href"]): a
            for a in tree.find_all("a")
            if a.attrs["href"].startswith("/pagina")
        }
        images = {href: request(server.port, "GET", href) for href in links}

    assert response.status == 200
    (sources_block,) = tree.find_all(cls="bronnen")
    assert sources_block.find("h3").text() == "Bronnen"
    cards = sources_block.find_all(cls="bron")
    ids = [card.attrs["id"] for card in cards]
    answer_number = ids[0].split("-")[1]
    assert ids == [f"bron-{answer_number}-{n}" for n in (1, 2, 3)]

    first, second, third = (_card(tree, n) for n in (1, 2, 3))
    assert first.find(cls="bron-nummer").text() == "[1]"
    assert first.find(cls="bron-type").text() == "Huurovereenkomst"
    assert first.find(cls="bron-bestand").text() == "fictief-huur.pdf"
    assert (
        first.find(cls="bron-hint").text() == "kop (hint): Artikel 1 Fictieve huurprijs"
    )
    assert second.find(cls="bron-type").text() == "Statuten"
    assert second.find(cls="bron-bestand").text() == "fictief-statuten.pdf"
    assert second.find_all(cls="bron-hint") == []
    assert third.find(cls="bron-type").text() == "type onbekend"
    assert third.find(cls="bron-bestand").text() == "fictief-onbekend.pdf"

    expected_lines = {
        1: [
            ("1", texts.LABEL_TEXT_LAYER, huur),
            ("2", texts.label_scan(2), huur),
        ],
        2: [("1", texts.label_low_quality(1), statuten)],
        3: [("2", texts.LABEL_TEXT_LAYER, onbekend)],
    }
    for number, card in ((1, first), (2, second), (3, third)):
        lines = card.find(cls="bron-paginas").find_all("li")
        assert len(lines) == len(expected_lines[number])
        for line, (page, label, digest) in zip(lines, expected_lines[number]):
            assert line.find(cls="paginanummer").text() == page
            assert line.find(cls="herkomst").text() == label
            (link,) = line.find_all("a")
            assert link.text() == texts.page_link(page)
            assert link.attrs["target"] == "_blank"
            assert link.attrs["rel"] == "noopener noreferrer"
            assert link.attrs["href"] == f"/pagina?sha256={digest}&nummer={page}"

    for href, image in images.items():
        assert image.status == 200, href
        assert image.headers["Content-Type"] == "image/png"
        assert Image.open(io.BytesIO(image.body)).format == "PNG"

    # The hashes in the links are random hex, so the page number is looked
    # for outside them.
    outside_hashes = tree.text() + " ".join(
        value or ""
        for node in tree.iter()
        for name, value in node.attrs.items()
        if name != "href"
    )
    for absent in ("87.654", "87,654", "0.73114", "0.65227", "7.3114"):
        assert absent not in response.text, absent
    assert "4321" not in outside_hashes


def test_a_source_outside_the_map_has_no_file_name_and_no_links(three_documents):
    raw, overrides, _ = three_documents
    result = answer_result(
        sources=[source(1, file_sha256="e5" * 32, pages=[page_entry(4)])]
    )

    with running_server(FakeAnswerer(result), raw, type_overrides=overrides) as server:
        tree = parse_html(ask(server.port, "Fictieve vraag?").text)

    card = tree.find(cls="bron")
    assert card.find(cls="bron-type").text() == texts.NO_DOCUMENT_TYPE
    assert card.find_all(cls="bron-bestand") == []
    assert card.find(cls="herkomst").text() == texts.LABEL_TEXT_LAYER
    assert card.find_all("a") == []
    assert tree.find(cls="antwoordtekst").find_all("a") != []


def test_a_null_label_shows_no_hint_and_an_empty_pages_list_no_line(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    result = answer_result(sources=[source(1, clause_label=None, pages=[])])

    with running_server(FakeAnswerer(result), raw) as server:
        tree = parse_html(ask(server.port, "Fictieve vraag?").text)

    assert tree.find_all(cls="bron-hint") == []
    assert tree.find(cls="bron-paginas").find_all("li") == []
