"""Parsing the answer (spec 007, D7, R12, A5): pure functions, with N = 3."""

from __future__ import annotations

import pytest

from companydocs_rag.answering.parsing import (
    ParsedAnswer,
    cited_numbers,
    has_marker,
    is_empty,
    parse_answer,
    split_citations,
)


@pytest.mark.parametrize(
    ("content", "numbers"),
    [
        ("Zie [1].", [1]),
        ("Zie [1, 2].", [1, 2]),
        ("Zie [1,2].", [1, 2]),
        ("Zie [1][3].", [1, 3]),
        ("Zie [01].", [1]),
        ("Zie [1 ,  2].", [1, 2]),
    ],
)
def test_citations(content, numbers):
    assert cited_numbers(content) == numbers


@pytest.mark.parametrize("content", ["Zie [1-3].", "Zie [2.1].", "Zie [a].", "Zie []."])
def test_brackets_that_are_not_citations(content):
    assert cited_numbers(content) == []


def test_valid_and_invalid_are_listed_once_in_order_of_first_appearance():
    content = "Eerst [2], dan [4], weer [2], dan [0] en ten slotte [1]."

    assert cited_numbers(content) == [2, 4, 2, 0, 1]
    assert split_citations(cited_numbers(content), 3) == ([2, 1], [4, 0])
    parsed = parse_answer(content, 3)
    assert parsed == ParsedAnswer("answered", None, [2, 1], [4, 0])


@pytest.mark.parametrize(
    "content",
    [
        "ONVOLDOENDE BEWIJS, de bronnen noemen geen fictieve termijn.",
        "Onvoldoende bewijs. De bronnen zwijgen.",
        "**ONVOLDOENDE BEWIJS**",
        "  # ONVOLDOENDE BEWIJS: geen bron.",
        "\n_onvoldoende bewijs_",
        "   \n\t\n* _ # onvoldoende BEWIJS",
    ],
)
def test_the_marker_is_found(content):
    assert has_marker(content)
    assert parse_answer(content, 3).status == "insufficient_evidence"
    assert parse_answer(content, 3).abstention_source == "prompt"


@pytest.mark.parametrize(
    "content",
    [
        "De huur is fictief [1].\nONVOLDOENDE BEWIJS voor de rest.",
        "Helaas: ONVOLDOENDE BEWIJS.",
        "Er is ONVOLDOENDE BEWIJS in de bronnen.",
        "> ONVOLDOENDE BEWIJS",
        "ONVOLDOENDE  BEWIJS",
    ],
)
def test_the_marker_is_not_found(content):
    assert not has_marker(content)


def test_the_marker_with_a_citation_is_insufficient_evidence_without_sources():
    parsed = parse_answer("ONVOLDOENDE BEWIJS, alleen [2] raakt het onderwerp.", 3)

    assert parsed.status == "insufficient_evidence"
    assert parsed.abstention_source == "prompt"
    assert parsed.valid == [2]
    assert parsed.invalid == []


def test_no_valid_citation_is_uncited():
    assert parse_answer("Een fictief antwoord zonder bron.", 3) == ParsedAnswer(
        "uncited", None, [], []
    )
    assert parse_answer("Alleen [4] en [0].", 3) == ParsedAnswer(
        "uncited", None, [], [4, 0]
    )


@pytest.mark.parametrize("content", [None, "", "  \n\t "])
def test_empty_content_is_an_error(content):
    assert is_empty(content)
    assert parse_answer(content, 3) == ParsedAnswer(
        "error", None, [], [], "empty_answer"
    )
