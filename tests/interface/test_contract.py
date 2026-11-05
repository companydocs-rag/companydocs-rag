"""The contract with the answer layer (spec 009, D3, D9; A15).

Records from a real ``Answerer`` with spec 007's fake Retriever and a
MockTransport hold every field the interface reads, with the types it
expects; the fragments built from them pass A4-A7's checks. A change to
these fields in the answering package fails here.
"""

from __future__ import annotations

import re

import pytest

from companydocs_rag.answering.answerer import Answerer
from companydocs_rag.interface import texts
from companydocs_rag.interface.fragments import result_fragment
from tests.answering._helpers import (
    FakeRetriever,
    Generator,
    chunk,
    reply,
    search_result,
    span,
    write_manifest,
)
from tests.interface._helpers import parse_html

CZECH = chr(0x0159) + chr(0x010D) + chr(0x0161)
LABEL = "<img src=x onerror=alert(1)> Artikel 2 " + CZECH
SHA = "a1" * 32
CARD_ID = re.compile("bron-[0-9]+-[0-9]+")


@pytest.fixture
def records(tmp_path):
    """An answered and an insufficient_evidence record from a real
    ``Answerer``."""
    found = chunk(
        "c01",
        "Fictieve tekst.",
        file_sha256=SHA,
        file_path="raw/fictief-huur.pdf",
        page_start=1,
        page_end=3,
        page_spans=[
            span(1, origin="native", tier="Clean"),
            span(2, origin="ocr", tier=None, confidence=87.654),
            span(3, origin="native", tier="Garbage"),
        ],
        clause_label=LABEL,
    )
    other = chunk("c02", "Andere fictieve tekst.", file_sha256="b2" * 32)
    write_manifest(tmp_path / "index")
    generator = Generator(
        reply("<b>Fictief</b> & 'zo' [1], niet [2] of [9]."),
        reply("ONVOLDOENDE BEWIJS: niets fictiefs [1]."),
    )
    retriever = FakeRetriever(
        search_result(dense=[(found, 0.9), (other, 0.8)], sparse=[(found, 3.0)])
    )
    with Answerer(
        tmp_path / "index", retriever=retriever, transport=generator.transport
    ) as answerer:
        return answerer.answer("Vraag <i>een</i>?"), answerer.answer("Vraag twee?")


def _check_types(record):
    assert isinstance(record.status, str)
    assert record.abstention_source is None or isinstance(record.abstention_source, str)
    assert record.answer_text is None or isinstance(record.answer_text, str)
    assert all(type(number) is int for number in record.citations["valid"])
    assert record.error is None or isinstance(record.error["type"], str)
    for source in record.sources:
        assert type(source["prompt_number"]) is int
        assert isinstance(source["file_sha256"], str)
        assert source["clause_label"] is None or isinstance(source["clause_label"], str)
        assert isinstance(source["pages"], list)
        for page in source["pages"]:
            assert type(page["page"]) is int
            assert isinstance(page["text_origin"], str)
            assert page["tier"] is None or isinstance(page["tier"], str)


def test_the_records_hold_the_fields_the_interface_reads(records):
    answered, abstained = records

    for record in records:
        _check_types(record)
    assert (answered.status, answered.abstention_source, answered.error) == (
        "answered",
        None,
        None,
    )
    assert answered.citations["valid"] == [1, 2]
    assert [source["prompt_number"] for source in answered.sources] == [1, 2]
    assert answered.sources[0]["clause_label"] == LABEL
    assert [
        (page["page"], page["text_origin"], page["tier"])
        for page in answered.sources[0]["pages"]
    ] == [(1, "native", "Clean"), (2, "ocr", None), (3, "native", "Garbage")]
    assert (abstained.status, abstained.abstention_source) == (
        "insufficient_evidence",
        "prompt",
    )
    assert abstained.sources == []


def test_the_fragments_built_from_them_pass_the_checks(records, tmp_path):
    answered, abstained = records
    pdf = tmp_path / "fictief & 'huur'.pdf"
    files = {SHA: pdf}
    types = {SHA: "<script>x</script>"}

    tree = parse_html(result_fragment("Vraag <i>een</i>?", answered, 1, files, types))

    for node in tree.iter():
        assert node.tag not in ("script", "img", "b", "i"), node.tag
        assert not any(attribute.startswith("on") for attribute in node.attrs)
        if "id" in node.attrs:
            assert CARD_ID.fullmatch(node.attrs["id"])
    assert tree.find(cls="vraag").text() == "Vraag <i>een</i>?"
    assert tree.find(cls="antwoordtekst").text() == answered.answer_text
    links = [
        (a.text(), a.attrs["href"])
        for a in tree.find(cls="antwoordtekst").find_all("a")
    ]
    assert links == [("1", "#bron-1-1"), ("2", "#bron-1-2")]
    cards = tree.find_all(cls="bron")
    assert [card.attrs["id"] for card in cards] == ["bron-1-1", "bron-1-2"]
    first = cards[0]
    assert first.find(cls="bron-type").text() == "<script>x</script>"
    assert first.find(cls="bron-bestand").text() == pdf.name
    assert first.find(cls="bron-label").text() == LABEL
    assert [line.find(cls="herkomst").text() for line in first.find_all("li")] == [
        texts.LABEL_TEXT_LAYER,
        texts.label_scan(2),
        texts.label_low_quality(3),
    ]
    assert [a.attrs["href"] for a in first.find_all("a")] == [
        f"/pagina?sha256={SHA}&nummer={n}" for n in (1, 2, 3)
    ]
    second = cards[1]
    assert second.find(cls="bron-type").text() == texts.NO_DOCUMENT_TYPE
    assert second.find_all(cls="bron-bestand") == []
    assert second.find_all("a") == []
    assert "87.654" not in result_fragment("x", answered, 1, files, types)

    abstained_tree = parse_html(
        result_fragment("Vraag twee?", abstained, 2, files, types)
    )
    assert (
        abstained_tree.find(cls="banner").text() == texts.BANNER_INSUFFICIENT_EVIDENCE
    )
    assert abstained_tree.find(cls="antwoordtekst").text() == abstained.answer_text
    assert abstained_tree.find_all("a") == []
    assert abstained_tree.find_all(cls="bron") == []
