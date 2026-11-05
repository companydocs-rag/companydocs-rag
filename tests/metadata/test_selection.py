"""Chunk selection (spec 008, D6, R8, R9, Patterns, A5)."""

from __future__ import annotations

import logging

import pytest

from companydocs_rag.metadata.selection import (
    chunk_score,
    has_valid_date,
    matched_stems,
    plan_batches,
    vocabulary,
)
from companydocs_rag.metadata.settings import KEYWORDS, MONTH_NAMES
from tests.metadata._helpers import (
    LEASE,
    LOGGER,
    SHA_A,
    Generator,
    chunk,
    chunk_record,
    extract,
    extraction_files,
    messages,
    use_settings,
    write_json_file,
    write_overrides,
)

NLD = vocabulary("nld", KEYWORDS, MONTH_NAMES)
ENG = vocabulary("eng", KEYWORDS, MONTH_NAMES)
UNION = vocabulary(None, KEYWORDS, MONTH_NAMES)


# --- Stems (R9) -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("words", "text", "stems"),
    [
        (NLD, "De huuropzegging volgt.", {"opzeg"}),
        (NLD, "Ondertekend te Fictiefstad.", {"ondertekend"}),
        (NLD, "De ondertekende partij en de ondergetekende.", set()),
        (NLD, "Notice of termination, terminated.", set()),
        (ENG, "Terminated.", {"term", "terminat"}),
        (ENG, "To determine the rent.", set()),
        (ENG, "De huuropzegging volgt.", set()),
        (UNION, "De huuropzegging, terminated.", {"opzeg", "term", "terminat"}),
        (NLD, "Einde, einde\nen nog eens einde.", {"einde"}),
    ],
    ids=[
        "infix",
        "whole-token",
        "longer-tokens",
        "english-in-nld",
        "two-prefixes",
        "no-prefix",
        "dutch-in-eng",
        "union",
        "counted-once",
    ],
)
def test_stems_match_by_their_mode(words, text, stems):
    assert matched_stems(text, words) == stems


def test_a_language_without_keywords_has_no_stems():
    words = vocabulary("deu", KEYWORDS, MONTH_NAMES)

    assert words.stems == ()
    u_diaeresis = chr(0x00FC)  # U+00FC LATIN SMALL LETTER U WITH DIAERESIS
    text = f"Die K{u_diaeresis}ndigung, opzegging, termination."

    assert chunk_score(text, words) == 0


def test_a_stem_listed_for_two_languages_counts_once_under_the_union():
    keywords = {"nld": {"infix": ["term"]}, "eng": {"prefix": ["term"]}}
    words = vocabulary(None, keywords, MONTH_NAMES)

    assert matched_stems("De termijn, the term.", words) == {"term"}


def test_nothing_matches_across_a_line_break():
    assert matched_stems("opz\neg", NLD) == set()


# --- Dates (Patterns) -----------------------------------------------------------------


def _with_lines(*lines):
    return "\n".join(["Fictief", *lines, "slot"])


@pytest.mark.parametrize(
    "date_text",
    [
        "1-4-2031",
        "01/04/31",
        "1.4.2031",
        "1 - 4 - 2031",
        "2031-04-01",
        "1 april 2031",
        "1 april",
        "3 mrt 2031",
        "3juni 2031",
    ],
)
def test_a_valid_nld_date_gives_the_bonus(date_text):
    text = _with_lines(date_text)

    assert matched_stems(text, NLD) == set()
    assert has_valid_date(text, NLD)
    assert chunk_score(text, NLD) == 1


@pytest.mark.parametrize(
    "lines",
    [
        ["1.4.31"],
        ["1-4/2031"],
        ["32-1-2031"],
        ["1-13-2031"],
        ["2031-13-01"],
        ["3 mrt"],
        ["2.2.10"],
        ["1-4-", "2031"],
    ],
    ids=lambda lines: "|".join(lines),
)
def test_an_invalid_nld_date_gives_no_bonus(lines):
    text = _with_lines(*lines)

    assert not has_valid_date(text, NLD)
    assert chunk_score(text, NLD) == 0


@pytest.mark.parametrize("date_text", ["1 march 2031", "march 1st, 2031", "march 2031"])
def test_a_valid_eng_date_gives_the_bonus(date_text):
    assert has_valid_date(_with_lines(date_text), ENG)


def test_may_without_a_year_gives_no_bonus():
    assert not has_valid_date(_with_lines("4 may"), ENG)


def test_month_names_follow_the_language():
    text = _with_lines("1 maart 2031")

    assert not has_valid_date(text, ENG)
    assert has_valid_date(text, UNION)
    assert has_valid_date(text, NLD)


def test_three_dates_still_add_one():
    text = _with_lines("1-4-2031", "2031-04-01", "1 april 2031")

    assert chunk_score(text, NLD) == 1
    assert chunk_score(text + "\nDe opzegging en de looptijd.", NLD) == 3


# --- Batches (D6) ---------------------------------------------------------------------


def _chunks(*sizes):
    return [chunk(SHA_A, index, "x" * size) for index, size in enumerate(sizes)]


def _sizes(batches):
    return [[len(item["text"]) for item in batch] for batch in batches]


def test_batches_fill_in_rank_order_and_never_reopen():
    chunks = _chunks(400, 400, 300, 100)

    batches = plan_batches(chunks, [4, 3, 2, 1], budget=1000, max_batches=3)

    assert _sizes(batches) == [[400, 400], [300, 100]]


def test_a_chunk_that_fills_the_budget_exactly_joins():
    batches = plan_batches(_chunks(600, 400), [2, 1], budget=1000, max_batches=3)

    assert _sizes(batches) == [[600, 400]]


def test_a_chunk_longer_than_the_budget_is_a_batch_of_its_own():
    batches = plan_batches(
        _chunks(300, 1200, 200), [3, 2, 1], budget=1000, max_batches=3
    )

    assert _sizes(batches) == [[300], [1200], [200]]


def test_at_most_max_batches_and_a_score_of_zero_is_never_selected():
    chunks = _chunks(600, 600, 600, 10)

    batches = plan_batches(chunks, [3, 2, 1, 0], budget=1000, max_batches=2)

    assert _sizes(batches) == [[600], [600]]
    assert plan_batches(_chunks(10, 10), [0, 0], budget=1000, max_batches=3) == []


def test_ties_go_by_chunk_index_and_a_batch_is_in_chunk_index_order():
    chunks = _chunks(100, 200, 300, 400)

    batches = plan_batches(chunks, [1, 2, 2, 1], budget=500, max_batches=3)

    assert [[item["chunk_index"] for item in batch] for batch in batches] == [
        [1, 2],
        [0, 3],
    ]


def test_the_command_sends_each_batch_in_chunk_index_order(tmp_path, monkeypatch):
    texts = [
        "Fictief zonder iets.",
        "Opzegging: de looptijd en de termijn, ingang 1 april 2030.",
        "Fictieve opzegging.",
        "De looptijd en de opzegging.",
    ]
    chunks = [
        chunk(SHA_A, index, text, page_start=index + 1)
        for index, text in enumerate(texts)
    ]
    chunks[1] = chunk(SHA_A, 1, texts[1], page_start=2, page_end=3)
    budget = len(texts[1]) + len(texts[3])
    use_settings(tmp_path, monkeypatch, selection_budget_chars=budget)
    write_json_file(
        tmp_path / "chunks",
        "huur.json",
        chunk_record(tmp_path / "los" / "fictief-huur.pdf", SHA_A, *chunks),
    )
    overrides = write_overrides(
        tmp_path / "handmatig" / "types.csv", [("fictief-huur.pdf", LEASE)]
    )
    generator = Generator()

    assert extract(tmp_path, overrides=overrides, generator=generator) == 0

    # Scores: chunk 1 has 4 stems and a date (5), chunk 3 two stems (2) and
    # chunk 2 one (1); chunk 0 scores 0. Batch 1 holds chunk 1 and chunk 3,
    # which fill the budget exactly; chunk 2 starts batch 2.
    assert generator.user_messages() == [
        f"source [1], pages 2-3\n{texts[1]}\n\nsource [2], page 4\n{texts[3]}",
        f"source [1], page 3\n{texts[2]}",
    ]
    record = extraction_files(tmp_path / "extracted")["huur.json"]
    assert [batch["chunk_ids"] for batch in record["batches"]] == [
        [chunks[1]["chunk_id"], chunks[3]["chunk_id"]],
        [chunks[2]["chunk_id"]],
    ]
    assert [batch["chars"] for batch in record["batches"]] == [
        len(texts[1]) + len(texts[3]),
        len(texts[2]),
    ]


def test_chunk_files_go_in_forward_slash_order_and_dot_names_are_skipped(
    tmp_path, caplog
):
    for relative in (
        "a/c.json",
        "a-b.json",
        ".verborgen.json",
        "a/.ook-verborgen.json",
    ):
        write_json_file(
            tmp_path / "chunks",
            relative,
            chunk_record(tmp_path / "los" / "fictief.pdf", SHA_A, "Fictief."),
        )

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert extract(tmp_path) == 0

    lines = [m for m in messages(caplog, logging.INFO) if m.startswith("[")]
    assert lines == [
        "[1/2] a-b.json: - (not_in_scope)",
        "[2/2] a/c.json: - (not_in_scope)",
    ]
    assert sorted(extraction_files(tmp_path / "extracted")) == ["a-b.json", "a/c.json"]
