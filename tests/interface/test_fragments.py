"""Answer fragments: escaping, citation links and statuses (spec 009, D7, D8,
R8, R9; Q1, Q5; A4, A5, A6)."""

from __future__ import annotations

import logging
import re
import time

import pytest

from companydocs_rag.ingestion.parse_pass import file_sha256
from companydocs_rag.interface import texts
from companydocs_rag.interface.fragments import answer_text_html, result_fragment
from companydocs_rag.interface.htmlpage import build_page
from companydocs_rag.interface.server import DemoApp
from companydocs_rag.interface.settings import interface_settings
from tests.interface._helpers import (
    FakeAnswerer,
    answer_result,
    ask,
    page_entry,
    parse_html,
    running_server,
    source,
    wait_until,
    write_overrides,
    write_pdf,
)

CZECH = chr(0x0159) + chr(0x010D) + chr(0x0161) + chr(0x011B)
POLISH = chr(0x0142) + chr(0x0105) + chr(0x017C) + chr(0x0107)
CARD_ID = re.compile("bron-[0-9]+-[0-9]+")
PAGE_URL = re.compile("/pagina[?]sha256=[0-9a-f]{64}&nummer=[1-9][0-9]*")
SIX_GENERATOR_ERRORS = [
    "http_error",
    "read_timeout",
    "transport_error",
    "malformed_response",
    "empty_answer",
    "prompt_truncation_suspected",
]


def _fragment(result, *, files=None, types=None, number=1, question="Vraag?"):
    return result_fragment(question, result, number, files or {}, types or {})


# --- A4: escaping ---------------------------------------------------------------------


@pytest.fixture
def hostile(tmp_path):
    """A4's fixture: a mapped PDF with ``&`` and ``'`` in its name, typed
    ``<script>x</script>`` by a fabricated override file."""
    name = "fictief & 'huur'.pdf"
    raw = tmp_path / "raw"
    pdf = write_pdf(raw / "map" / name, (200, 300))
    overrides = write_overrides(
        tmp_path / "handmatig" / "typen.csv", [(name, "<script>x</script>")]
    )
    return raw, overrides, name, file_sha256(pdf)


def test_every_inserted_text_is_escaped_and_comes_through_unchanged(hostile):
    raw, overrides, name, digest = hostile
    answer = (
        "<script>alert(1)</script> <b>vet</b> &amp; \"dubbel\" 'enkel'\n"
        f"Tweede regel {CZECH} {POLISH} [1]"
    )
    label = "<img src=x onerror=alert(1)> " + CZECH + POLISH
    question = "Wat staat er in <b>artikel</b> 3?"
    result = answer_result(
        answer_text=answer,
        sources=[source(1, file_sha256=digest, clause_label=label)],
    )
    answerer = FakeAnswerer(result)

    with running_server(answerer, raw, type_overrides=overrides) as server:
        response = ask(server.port, question)

    assert response.status == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
    tree = parse_html(response.text)
    for node in tree.iter():
        assert node.tag not in ("script", "img", "b", "i"), node.tag
        assert not any(attribute.startswith("on") for attribute in node.attrs)
    assert tree.find(cls="antwoordtekst").text() == answer
    assert "\n" in tree.find(cls="antwoordtekst").text()
    assert tree.find(cls="bron-label").text() == label
    assert tree.find(cls="bron-hint").text() == texts.clause_label(label)
    assert tree.find(cls="bron-type").text() == "<script>x</script>"
    assert tree.find(cls="bron-bestand").text() == name
    assert tree.find(cls="vraag").text() == question
    for node in tree.iter():
        if "id" in node.attrs:
            assert CARD_ID.fullmatch(node.attrs["id"]), node.attrs["id"]
        if "href" in node.attrs:
            href = node.attrs["href"]
            assert CARD_ID.fullmatch(href[1:]) or PAGE_URL.fullmatch(href), href
    assert answerer.questions == [question]


# --- A5: citation links (R9) ----------------------------------------------------------


def _links(fragment_html):
    return [
        (node.text(), node.attrs["href"])
        for node in parse_html(fragment_html).find_all("a")
    ]


def test_each_cited_number_with_a_card_links_to_it():
    text = "[1] [2, 3] [3][7] [01] [1-3] [a] []"
    html = answer_text_html(text, {1: "bron-4-1", 3: "bron-4-3"})

    assert _links(html) == [
        ("1", "#bron-4-1"),
        ("3", "#bron-4-3"),
        ("3", "#bron-4-3"),
        ("01", "#bron-4-1"),
    ]
    assert parse_html(html).text() == text


def test_a_citation_in_a_fragment_links_to_its_card():
    result = answer_result(
        answer_text="Zie [1] [2, 3] [3][7] [01] [1-3] [a] [].",
        sources=[source(3), source(1)],
    )

    tree = parse_html(_fragment(result, number=4))

    ids = [node.attrs["id"] for node in tree.iter() if "id" in node.attrs]
    assert ids == ["bron-4-1", "bron-4-3"]
    links = [
        (a.text(), a.attrs["href"])
        for a in tree.find(cls="antwoordtekst").find_all("a")
    ]
    assert links == [
        ("1", "#bron-4-1"),
        ("3", "#bron-4-3"),
        ("3", "#bron-4-3"),
        ("01", "#bron-4-1"),
    ]
    assert tree.find(cls="antwoordtekst").text() == (
        "Zie [1] [2, 3] [3][7] [01] [1-3] [a] []."
    )


def _now_ms():
    return time.time_ns() // 1_000_000


def _card_ids(fragment_html):
    tree = parse_html(fragment_html)
    return {node.attrs["id"] for node in tree.iter() if "id" in node.attrs}


def _answer_number(ids):
    (number,) = {int(card_id.split("-")[1]) for card_id in ids}
    return number


def test_two_answers_in_one_run_have_their_own_card_ids(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    first = answer_result(answer_text="Een [1] en [2].", sources=[source(1), source(2)])
    second = answer_result(answer_text="Twee [2].", sources=[source(2)])

    before = _now_ms()
    with running_server(FakeAnswerer(first, second), raw) as server:
        after = _now_ms()
        fragments = [ask(server.port, "Een?").text, ask(server.port, "Twee?").text]

    all_ids = []
    for fragment in fragments:
        tree = parse_html(fragment)
        ids = {node.attrs["id"] for node in tree.iter() if "id" in node.attrs}
        all_ids.append(ids)
        for link in tree.find(cls="antwoordtekst").find_all("a"):
            assert link.attrs["href"][1:] in ids
    # M1: the counter starts at the server's start time in whole milliseconds
    # since the Unix epoch and goes up by one per counted answer.
    start = _answer_number(all_ids[0])
    assert before <= start <= after
    assert all_ids == [
        {f"bron-{start}-1", f"bron-{start}-2"},
        {f"bron-{start + 1}-2"},
    ]


def test_two_server_runs_give_different_ids_for_their_first_answers(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    result = answer_result(answer_text="Een [1].", sources=[source(1)])

    with running_server(FakeAnswerer(result), raw) as server:
        first_run = _card_ids(ask(server.port, "Een?").text)
    first_start = _answer_number(first_run)
    # The system clock advances in steps (15.625 ms on the development
    # machine's Python), so the second run starts once it reads a later
    # millisecond (M1).
    wait_until(lambda: _now_ms() > first_start)
    with running_server(FakeAnswerer(result), raw) as server:
        second_run = _card_ids(ask(server.port, "Een?").text)
    second_start = _answer_number(second_run)

    assert first_run == {f"bron-{first_start}-1"}
    assert second_run == {f"bron-{second_start}-1"}
    assert second_start > first_start


# --- A6: statuses (D8, Q5) ------------------------------------------------------------


def _banner_classes(tree):
    return [node.classes for node in tree.iter() if "banner" in node.classes]


def test_an_answered_fragment_has_no_banner_its_text_and_its_cards():
    tree = parse_html(_fragment(answer_result(sources=[source(1)])))

    assert _banner_classes(tree) == []
    assert tree.find(cls="antwoordtekst").text() == "Fictief antwoord [1]."
    assert len(tree.find_all(cls="bron")) == 1
    assert tree.find(cls="bronnen-kop").text() == texts.SOURCES_HEADING


def test_an_abstention_by_the_model_shows_the_banner_then_the_text():
    text = "ONVOLDOENDE BEWIJS: de fictieve bronnen zeggen niets [1]."
    result = answer_result(
        "insufficient_evidence", abstention_source="prompt", answer_text=text
    )

    tree = parse_html(_fragment(result))

    answer = tree.find(cls="antwoord")
    banner = answer.children[0]
    assert "banner-onvoldoende" in banner.classes
    assert banner.text() == texts.BANNER_INSUFFICIENT_EVIDENCE
    assert tree.find(cls="antwoordtekst").text() == text
    assert tree.find_all("a") == []
    assert tree.find_all(cls="bron") == []


@pytest.mark.parametrize("abstention", ["score_gate", "no_candidates"])
def test_an_abstention_without_a_request_shows_the_banner_only(abstention):
    result = answer_result(
        "insufficient_evidence", abstention_source=abstention, answer_text=None
    )

    tree = parse_html(_fragment(result))

    assert tree.find(cls="banner").text() == texts.BANNER_INSUFFICIENT_EVIDENCE
    assert tree.find_all(cls="antwoordtekst") == []
    assert tree.find_all(cls="bron") == []


def test_an_uncited_answer_shows_its_banner_then_the_text_and_no_card():
    result = answer_result("uncited", answer_text="Fictief zonder bron.")

    tree = parse_html(_fragment(result))

    answer = tree.find(cls="antwoord")
    assert "banner-ongeciteerd" in answer.children[0].classes
    assert answer.children[0].text() == texts.BANNER_UNCITED
    assert tree.find(cls="antwoordtekst").text() == "Fictief zonder bron."
    assert tree.find_all(cls="bron") == []


def test_the_two_banners_differ_and_neither_occurs_in_an_answered_fragment():
    insufficient = parse_html(
        _fragment(
            answer_result(
                "insufficient_evidence", abstention_source="prompt", answer_text="x"
            )
        )
    )
    uncited = parse_html(_fragment(answer_result("uncited", answer_text="x")))
    answered = parse_html(_fragment(answer_result(sources=[source(1)])))

    (first,) = _banner_classes(insufficient)
    (second,) = _banner_classes(uncited)
    assert set(first) != set(second)
    used = {cls for node in answered.iter() for cls in node.classes}
    assert not (set(first) - set(second)) & used
    assert not (set(second) - set(first)) & used
    assert "banner" not in used


def _app(answerer, tmp_path):
    return DemoApp(answerer, {}, {}, interface_settings(), build_page())


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        ("context_budget_exceeded", texts.SERVER_ERROR),
        *[(error_type, texts.GENERATOR_ERROR) for error_type in SIX_GENERATOR_ERRORS],
    ],
)
def test_an_error_record_shows_its_d16_text_without_answer_or_card(
    tmp_path, caplog, capsys, error_type, message
):
    result = answer_result(
        "error",
        answer_text="Fictieve tekst die niet getoond wordt [1].",
        sources=[source(1)],
        error={"type": error_type, "message": "fictieve oorzaak"},
    )
    app = _app(FakeAnswerer(result), tmp_path)

    with caplog.at_level(logging.DEBUG):
        status, body = app.question("Fictieve vraag?")

    assert status == 200
    tree = parse_html(body.decode("utf-8"))
    assert tree.find(cls="melding").text() == message
    assert tree.find_all(cls="antwoordtekst") == []
    assert tree.find_all(cls="bron") == []
    assert tree.find_all("a") == []
    assert "fictieve tekst die niet getoond wordt" not in body.decode("utf-8").lower()
    assert [r for r in caplog.records if r.exc_info] == []
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("status", ["insufficient_evidence", "uncited"])
def test_other_statuses_show_no_card_and_their_numbers_stay_text(status):
    result = answer_result(
        status,
        abstention_source="prompt" if status == "insufficient_evidence" else None,
        answer_text="Fictief [1] en [2].",
        sources=[source(1), source(2)],
    )

    tree = parse_html(_fragment(result))

    assert tree.find_all(cls="bron") == []
    assert tree.find_all("a") == []
    assert tree.find(cls="antwoordtekst").text() == "Fictief [1] en [2]."


def test_a_page_entry_fills_its_label_with_its_page_number():
    result = answer_result(
        sources=[source(1, pages=[page_entry(7, origin="ocr", tier=None)])]
    )

    tree = parse_html(_fragment(result))

    assert tree.find(cls="herkomst").text() == texts.label_scan(7)
    assert tree.find(cls="paginanummer").text() == "7"
