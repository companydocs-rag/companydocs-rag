"""The provenance of a cited source (spec 007, D9, R13, A9)."""

from __future__ import annotations

from companydocs_rag.answering import cli
from companydocs_rag.answering.answerer import Answerer
from tests.answering._helpers import (
    FakeRetriever,
    Generator,
    chunk,
    reply,
    search_result,
    settings_for,
    span,
    use_settings,
    write_manifest,
)

LABEL = "Artikel 9 Fictieve waarborg"
SOURCE_KEYS = [
    "prompt_number",
    "chunk_id",
    "file_sha256",
    "file_path",
    "page_start",
    "page_end",
    "clause_label",
    "pages",
]
PAGE_KEYS = ["page", "text_origin", "tier", "ocr_mean_confidence", "provenance"]


def _mixed(label=LABEL):
    """Pages 3 to 7: ocr, native Clean, native Garbage, native without a
    tier, and a page without text."""
    return chunk(
        "c01",
        "Fictieve tekst over een waarborgsom.",
        file_path="invoer/fictief-gemengd.pdf",
        page_start=3,
        page_end=7,
        page_spans=[
            span(3, origin="ocr", tier=None, confidence=87.25),
            span(4, origin="native", tier="Clean"),
            span(5, origin="native", tier="Garbage"),
            span(6, origin="native", tier=None),
            span(7, origin="none", tier=None, length=0),
        ],
        clause_label=label,
    )


def _result(item):
    return search_result(dense=[(item, 0.8)], sparse=[(item, 4.0)])


def test_a_source_carries_its_pages_and_their_provenance(tmp_path):
    write_manifest(tmp_path / "index")
    with Answerer(
        tmp_path / "index",
        settings=settings_for(tmp_path),
        retriever=FakeRetriever(_result(_mixed())),
        transport=Generator(reply("Fictief [1].")).transport,
    ) as answerer:
        record = answerer.answer("Fictieve vraag?")

    (source,) = record.sources
    assert list(source) == SOURCE_KEYS
    assert source["clause_label"] == LABEL
    assert (source["page_start"], source["page_end"]) == (3, 7)
    assert source["file_path"] == "invoer/fictief-gemengd.pdf"
    assert [list(page) for page in source["pages"]] == [PAGE_KEYS] * 4
    assert [
        (page["page"], page["text_origin"], page["tier"], page["provenance"])
        for page in source["pages"]
    ] == [
        (3, "ocr", None, "ocr"),
        (4, "native", "Clean", "native"),
        (5, "native", "Garbage", "native_garbage"),
        (6, "native", None, "native"),
    ]
    assert source["pages"][0]["ocr_mean_confidence"] == 87.25


def _command_line(tmp_path, monkeypatch, capsys, item):
    use_settings(tmp_path, monkeypatch, settings_for(tmp_path))
    write_manifest(tmp_path / "index")
    code = cli.main(
        ["Fictieve vraag?", "--index-dir", str(tmp_path / "index")],
        retriever=FakeRetriever(_result(item)),
        transport=Generator(reply("Fictief [1].")).transport,
    )
    assert code == 0
    return capsys.readouterr().out.splitlines()[-1]


def test_the_command_line_shows_the_distinct_provenance_and_the_label(
    tmp_path, monkeypatch, capsys
):
    line = _command_line(tmp_path, monkeypatch, capsys, _mixed())

    assert line == (
        "[1] invoer/fictief-gemengd.pdf pagina 3-7 (ocr, native, native_garbage)"
        " kop: Artikel 9 Fictieve waarborg"
    )


def test_without_a_label_the_line_ends_with_the_provenance(
    tmp_path, monkeypatch, capsys
):
    line = _command_line(tmp_path, monkeypatch, capsys, _mixed(label=None))

    assert line == (
        "[1] invoer/fictief-gemengd.pdf pagina 3-7 (ocr, native, native_garbage)"
    )


def test_a_single_page_is_written_without_a_range(tmp_path, monkeypatch, capsys):
    item = chunk("c01", "Fictief.", file_path="invoer/een.pdf", page_start=2)

    line = _command_line(tmp_path, monkeypatch, capsys, item)

    assert line == "[1] invoer/een.pdf pagina 2 (native)"
