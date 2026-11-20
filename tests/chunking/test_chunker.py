"""The chunker (spec 005) at function level: K2-K13 and the settings.

Inline fabricated normalized records only (``_helpers.py``); characters
outside ASCII are built with ``chr``. Every document a test chunks goes
through :func:`chunk`, which also checks A3's key order and A4's and A12's
invariants on every chunk. Tests pass a settings mapping: ``DEFAULTS`` gives
the Python defaults, ``settings_with`` overrides keys.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from companydocs_rag.chunking import chunker
from companydocs_rag.chunking.chunker import (
    ChunkingConfigError,
    NormalizedRecordError,
    chunk_document,
    chunking_settings,
    decompose_runs,
    validate_normalized_record,
)
from companydocs_rag.ingestion.settings import find_project_root
from tests.chunking._helpers import (
    CHUNK_KEYS,
    DOCUMENT_KEYS,
    PAGE_SPAN_KEYS,
    RUN_KEYS,
    block,
    body,
    check_chunks,
    filler,
    lines,
    normalized_page,
    normalized_record,
    stream_of,
)

DEFAULTS = {"chunking": {}}

# The K3 keywords, written with chr() for the characters outside ASCII.
KEYWORDS = [
    "Artikel",
    "Article",
    "Art.",
    "Artyku" + chr(0x0142),  # U+0142 LATIN SMALL LETTER L WITH STROKE
    # U+010C LATIN CAPITAL LETTER C WITH CARON, U+00E1 LATIN SMALL LETTER A
    # WITH ACUTE
    chr(0x010C) + "l" + chr(0x00E1) + "nek",
    chr(0x010C) + "l" + chr(0x00E1) + "nok",
    chr(0x010C) + "l.",
    "Section",
    "Clause",
    chr(0x00A7),  # U+00A7 SECTION SIGN
    "$",
]

# The integer settings, their defaults and minimums (External surface, R6).
INTEGER_SETTINGS = [
    ("heading_max_gap", 0, 0),
    ("min_heading_run_length", 3, 1),
    ("min_run_median_section_chars", 400, 0),
    ("max_label_inherit_chars", 6000, 1),
    ("chunk_target_chars", 1200, 1),
    ("chunk_max_chars", 2000, 1),
    ("chunk_overlap_chars", 150, 0),
    ("clause_label_max_chars", 80, 1),
]


def settings_with(**values):
    return {"chunking": values}


def chunk(record, settings=DEFAULTS, **kwargs):
    """Chunk ``record`` and check every chunk (A3, A4, A12)."""
    document = chunk_document(record, settings, **kwargs)
    check_chunks(
        record, document, max_chars=chunking_settings(settings).chunk_max_chars
    )
    return document


def three_articles(length=500):
    """Text of three keyword sections, each ``length`` long."""
    return (
        block("Artikel 1 Begrippen", length)
        + block("Artikel 2 Huurprijs", length)
        + block("Artikel 3 Duur", length)
    )


def spans_of(document):
    return [(chunk["char_start"], chunk["char_end"]) for chunk in document["chunks"]]


def methods_of(document):
    return [chunk["chunk_method"] for chunk in document["chunks"]]


def runs_of(document):
    return document["heading_evidence"]["runs"]


# --- The output (A3) ------------------------------------------------------------------


def test_the_document_carries_the_input_fields():
    record = normalized_record(
        block("Artikel 1 Begrippen", 500) + block("Artikel 2 Huurprijs", 500),
        normalized_page(2, block("Artikel 3 Duur", 500), text_origin="ocr"),
        path="map/fictief.pdf",
    )

    document = chunk(record)

    assert list(document) == DOCUMENT_KEYS
    assert (document["schema_version"], document["chunker_version"]) == (1, 1)
    assert document["file"] == record["file"]
    assert document["status"] == "processed"
    assert document["normalized_hash"] == record["normalized_sha256"]
    assert document["structure_hint"] == "unknown"
    assert (document["language"], document["language_source"]) == ("nld", "index")
    assert document["route"] == "structured"
    assert document["empty_pages"] == []
    assert document["chunk_count"] == 3
    assert [chunk["page_start"] for chunk in document["chunks"]] == [1, 1, 2]
    assert [chunk["text_origins"] for chunk in document["chunks"]] == [
        ["native"],
        ["native"],
        ["ocr"],
    ]
    (run,) = runs_of(document)
    assert list(run) == RUN_KEYS
    assert (run["start_page"], run["first_number"], run["length"]) == (1, 1, 3)


@pytest.mark.parametrize(
    ("second", "expected"),
    [
        (normalized_page(2, "Meer tekst."), ("nld", "index")),
        (normalized_page(2, "", language="nld"), ("nld", "index")),
        (normalized_page(2, "Meer tekst.", language="deu"), (None, "index")),
        (normalized_page(2, "", language="eng"), (None, "index")),
        (normalized_page(2, "Meer.", language_source="override"), ("nld", None)),
        (normalized_page(2, "Meer.", language=None), (None, "index")),
    ],
    ids=[
        "shared",
        "shared-with-an-empty-page",
        "one-page-differs",
        "an-empty-page-differs",
        "source-differs",
        "one-page-null",
    ],
)
def test_language_is_the_value_all_pages_share_and_null_when_one_differs(
    second, expected
):
    document = chunk(normalized_record("Fictieve tekst.", second))

    assert (document["language"], document["language_source"]) == expected


def test_language_is_null_without_pages():
    document = chunk(normalized_record())

    assert (document["language"], document["language_source"]) == (None, None)


@pytest.mark.parametrize("hint", ["structured", "Unknown", "", None])
def test_a_structure_hint_other_than_unknown_raises_value_error(hint):
    with pytest.raises(ValueError, match="structure_hint") as raised:
        chunk_document(normalized_record("Tekst."), DEFAULTS, structure_hint=hint)

    assert raised.type is ValueError


def test_the_structure_hint_unknown_is_accepted_and_recorded():
    document = chunk(normalized_record("Tekst."), structure_hint="unknown")

    assert document["structure_hint"] == "unknown"


def _keys(value, parent=None):
    """Every (parent key, key) pair in a JSON value."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield parent, key
            yield from _keys(item, key)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item, parent)


def test_no_key_holds_a_structure_score_the_detected_type_or_a_chunk_tier():
    record = normalized_record(
        three_articles(),
        normalized_page(2, filler(300), text_origin="ocr", ocr_mean_confidence=81.0),
    )

    pairs = set(_keys(chunk(record)))

    names = {key for _, key in pairs}
    assert not any("score" in name or "type" in name for name in names)
    assert names - {"structure_hint"} == {
        name for name in names if "structure" not in name
    }
    assert {
        parent for parent, key in pairs if key in ("tier", "ocr_mean_confidence")
    } == {"page_spans"}
    assert not any("confidence" in name for name in names - {"ocr_mean_confidence"})


# --- Text is never changed (A4) -------------------------------------------------------


def test_chunks_are_slices_of_the_stream_whatever_the_characters():
    """Tabs, a line feed at the end of a page, characters outside ASCII and a
    "$" heading stay as written; the stream joins the pages with one line
    feed."""
    e_acute = chr(0x00E9)  # U+00E9 LATIN SMALL LETTER E WITH ACUTE
    en_dash = chr(0x2013)  # U+2013 EN DASH
    first = block("$ 1 Begrippen", 450, inner=(f"Caf{e_acute}\tterras {en_dash} 3 m",))
    second = block("$ 2 Huurprijs", 450) + "\n"
    third = block("$ 3 Duur", 450)
    record = normalized_record(
        first,
        normalized_page(2, second, text_origin="ocr"),
        normalized_page(3, ""),
        normalized_page(4, third, text_origin="ocr"),
    )
    before = copy.deepcopy(record)

    document = chunk(record)

    assert record == before
    assert "".join(chunk["text"] for chunk in document["chunks"]) == stream_of(record)
    assert stream_of(record) == first + "\n" + second + "\n" + third
    assert document["empty_pages"] == [3]


def test_a_region_of_blank_lines_only_before_the_first_heading_gives_no_chunk():
    record = normalized_record("\n\n\n" + three_articles())

    document = chunk(record)

    assert document["route"] == "structured"
    assert spans_of(document) == [(3, 503), (503, 1003), (1003, 1503)]
    assert methods_of(document) == ["heading_section"] * 3


def test_a_document_without_text_has_no_chunks():
    record = normalized_record("", normalized_page(2, ""))

    document = chunk(record)

    assert (document["chunk_count"], document["chunks"]) == (0, [])
    assert document["route"] == "generic"
    assert document["empty_pages"] == [1, 2]
    assert document["heading_evidence"] == {
        "family": None,
        "candidate_counts": {"keyword": 0, "decimal": 0},
        "runs": [],
    }


def test_a_failed_document_without_pages_is_chunked_like_any_other():
    record = normalized_record(status="failed", sha256=None)

    document = chunk(record)

    assert list(document) == DOCUMENT_KEYS
    assert document["status"] == "failed"
    assert document["file"] == {"path": "invoer/fictief.pdf", "sha256": None}
    assert (document["route"], document["chunk_count"], document["chunks"]) == (
        "generic",
        0,
        [],
    )
    assert document["heading_evidence"]["runs"] == []


# --- Heading candidates (A5) ----------------------------------------------------------


def test_the_keyword_default_is_the_k3_list():
    assert list(chunker.HEADING_KEYWORDS) == KEYWORDS


@pytest.mark.parametrize("case", ["upper", "lower"])
@pytest.mark.parametrize("keyword", KEYWORDS)
def test_each_keyword_in_either_case_followed_by_a_number_is_a_keyword_candidate(
    keyword, case
):
    written = keyword.upper() if case == "upper" else keyword.lower()
    record = normalized_record(lines(f"{written} 7 Fictieve titel", body(40)))

    evidence = chunk(record)["heading_evidence"]

    assert evidence["candidate_counts"] == {"keyword": 1, "decimal": 0}
    assert evidence["family"] == "keyword"
    assert [(run["first_number"], run["length"]) for run in evidence["runs"]] == [
        (7, 1)
    ]


def test_a_decimal_heading_has_its_top_level_integer_as_chain_value():
    record = normalized_record(
        block("1.2 Begrippen", 500)
        + block("2.1 Huurprijs", 500)
        + block("3.4 Duur", 500)
    )

    document = chunk(record)

    evidence = document["heading_evidence"]
    assert evidence["candidate_counts"] == {"keyword": 0, "decimal": 3}
    assert evidence["family"] == "decimal"
    (run,) = evidence["runs"]
    assert (run["first_number"], run["length"], run["accepted"]) == (1, 3, True)
    assert [chunk["clause_number"] for chunk in document["chunks"]] == [
        "1.2",
        "2.1",
        "3.4",
    ]
    assert [chunk["heading_family"] for chunk in document["chunks"]] == ["decimal"] * 3


def test_a_dollar_heading_keeps_its_dollar_in_text_and_clause_label():
    record = normalized_record(
        block("$ 1 Begrippen", 500)
        + block("$ 2 Huurprijs", 500)
        + block("$ 3 Duur", 500)
    )

    document = chunk(record)

    third = document["chunks"][2]
    assert third["text"].startswith("$ 3 Duur\n")
    assert (third["clause_label"], third["clause_number"]) == ("$ 3 Duur", "3")
    assert document["heading_evidence"]["family"] == "keyword"


def test_references_standalone_numbers_and_the_other_family_start_no_section():
    first = block(
        "Artikel 1 Begrippen",
        600,
        inner=(
            "Zoals bedoeld in artikel 2 van deze fictieve tekst.",
            "Artikel 7:230 is hier niet van toepassing.",
            "7",
            "7.",
            "7)",
            "1. Eerste fictieve lid.",
            "2. Tweede fictieve lid.",
            "3. Derde fictieve lid.",
        ),
    )
    record = normalized_record(
        first + block("Artikel 2 Huurprijs", 500) + block("Artikel 3 Duur", 500)
    )

    document = chunk(record)

    evidence = document["heading_evidence"]
    assert evidence["candidate_counts"] == {"keyword": 4, "decimal": 3}
    assert evidence["family"] == "keyword"
    (run,) = evidence["runs"]
    assert (run["first_number"], run["length"], run["accepted"]) == (1, 3, True)
    assert spans_of(document) == [(0, 600), (600, 1100), (1100, 1600)]
    assert methods_of(document) == ["heading_section"] * 3
    assert [chunk["clause_number"] for chunk in document["chunks"]] == ["1", "2", "3"]


# --- Runs (A6) ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            [*range(1, 6), *range(1, 13)],
            [list(range(0, 5)), list(range(5, 17))],
        ),
        ([1, 2, 3, 4, 5, 1, 6, 7, 8, 9, 10], [[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]]),
        ([1, 2, 1, 2, 3], [[0, 1, 4]]),
        ([1, 2, 3, 1], [[0, 1, 2], [3]]),
        # Both [1, 2, 3] + [1] and [1, 2, 3, 4] cover four: fewer runs wins.
        ([1, 2, 3, 1, 4], [[0, 1, 2, 4]]),
        ([], []),
    ],
    ids=[
        "two-numberings",
        "a-lone-restart",
        "earliest-candidates",
        "short-last",
        "fewer-runs",
        "none",
    ],
)
def test_runs_cover_the_most_candidates(values, expected):
    assert decompose_runs(values) == expected


def test_heading_max_gap_lets_a_run_skip_numbers():
    assert decompose_runs([1, 2, 4, 5], max_gap=1) == [[0, 1, 2, 3]]
    assert decompose_runs([1, 2, 4, 5], max_gap=0) == [[0, 1]]


def test_on_a_tie_the_smaller_step_wins():
    # The 3 can follow the 2 (+1) or the 1 (+2), both for a run of two.
    assert decompose_runs([2, 1, 3], max_gap=1) == [[0, 2]]


def _articles(values, length):
    return "".join(block(f"Artikel {value} Titel", length) for value in values)


def test_two_numberings_are_two_runs_and_the_second_starts_at_1():
    record = normalized_record(_articles([*range(1, 6), *range(1, 13)], 60))

    runs = runs_of(chunk(record))

    assert [(run["run_index"], run["first_number"], run["length"]) for run in runs] == [
        (1, 1, 5),
        (2, 1, 12),
    ]


def test_start_page_is_the_page_of_the_runs_first_heading():
    """R3. Run 1 starts on the first line of page 3, after a page of body text
    and an empty page, so its heading starts exactly where page 3's stretch
    of the stream starts. Run 2 starts in the middle of page 4. Neither run
    starts on the first page with text."""
    record = normalized_record(
        filler(300),
        normalized_page(2, ""),
        three_articles(),
        filler(200) + _articles([1, 2, 3], 450),
    )

    document = chunk(record)

    assert document["empty_pages"] == [2]
    assert [
        (run["run_index"], run["first_number"], run["length"], run["start_page"])
        for run in runs_of(document)
    ] == [(1, 1, 3, 3), (2, 1, 3, 4)]


def test_a_lone_1_inside_a_run_is_body_text():
    values = [1, 2, 3, 4, 5, 1, 6, 7, 8, 9, 10]
    record = normalized_record(_articles(values, 450))

    document = chunk(record)

    (run,) = runs_of(document)
    assert (run["first_number"], run["length"], run["accepted"]) == (1, 10, True)
    # The fifth section runs across the lone 1 to the 6.
    starts = [chunk["char_start"] for chunk in document["chunks"]]
    assert starts == [450 * position for position in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10)]
    assert spans_of(document)[4] == (1800, 2700)


def test_a_run_takes_the_earliest_candidate_of_each_value():
    record = normalized_record(_articles([1, 2, 1, 2, 3], 450))

    document = chunk(record)

    (run,) = runs_of(document)
    assert (run["length"], run["accepted"]) == (3, True)
    assert spans_of(document) == [(0, 450), (450, 1800), (1800, 2250)]


def test_with_heading_max_gap_1_a_run_skips_a_number():
    record = normalized_record(_articles([1, 2, 4, 5], 450))

    with_gap = chunk(record, settings_with(heading_max_gap=1))
    without = chunk(record)

    (run,) = runs_of(with_gap)
    assert (run["length"], run["accepted"]) == (4, True)
    assert [chunk["clause_number"] for chunk in with_gap["chunks"]] == [
        "1",
        "2",
        "4",
        "5",
    ]
    (run,) = runs_of(without)
    assert (run["first_number"], run["length"]) == (1, 2)
    assert run["reject_reason"] == "run_too_short"


# --- The family (A7) ------------------------------------------------------------------


def _mixed(*decimal_lines):
    """Three keyword sections with ``decimal_lines`` spread over the first two."""
    return (
        block("Artikel 1 Begrippen", 500, inner=decimal_lines[:2])
        + block("Artikel 2 Huurprijs", 500, inner=decimal_lines[2:])
        + block("Artikel 3 Duur", 500)
    )


def test_a_longer_decimal_run_wins():
    record = normalized_record(
        _mixed("1. Een lid.", "2. Twee leden.", "3. Drie leden.", "4. Vier leden.")
    )

    evidence = chunk(record)["heading_evidence"]

    assert evidence["family"] == "decimal"
    assert evidence["candidate_counts"] == {"keyword": 3, "decimal": 4}
    assert [(run["first_number"], run["length"]) for run in evidence["runs"]] == [
        (1, 4)
    ]


def test_keyword_wins_a_tie():
    record = normalized_record(
        _mixed("1. Een lid.", "2. Twee leden.", "3. Drie leden.")
    )

    document = chunk(record)

    evidence = document["heading_evidence"]
    assert evidence["family"] == "keyword"
    assert evidence["candidate_counts"] == {"keyword": 3, "decimal": 3}
    assert document["route"] == "structured"
    assert [chunk["heading_family"] for chunk in document["chunks"]] == ["keyword"] * 3


def test_a_document_without_candidates_has_no_family_and_the_generic_route():
    record = normalized_record(lines(body(50), body(70)))

    document = chunk(record)

    assert document["heading_evidence"] == {
        "family": None,
        "candidate_counts": {"keyword": 0, "decimal": 0},
        "runs": [],
    }
    assert document["route"] == "generic"
    assert methods_of(document) == ["generic"]


# --- Sections and acceptance (A8) -----------------------------------------------------


def test_a_run_of_2_with_long_sections_is_too_short():
    record = normalized_record(
        block("Artikel 1 Begrippen", 600) + block("Artikel 2 Huurprijs", 600)
    )

    document = chunk(record)

    (run,) = runs_of(document)
    assert (run["length"], run["median_section_chars"]) == (2, 600)
    assert (run["accepted"], run["reject_reason"]) == (False, "run_too_short")
    assert document["route"] == "generic"
    assert methods_of(document) == ["generic"]


def test_a_short_run_is_too_short_whatever_its_median():
    document = chunk(normalized_record(lines("1 Begrippen", "2 Huurprijs")))

    (run,) = runs_of(document)
    assert (run["length"], run["median_section_chars"]) == (2, 11)
    assert run["reject_reason"] == "run_too_short"


def test_a_contents_list_run_has_sections_too_short():
    record = normalized_record(
        lines("1 Begrippen", "2 Huurprijs", "3 Duur", "4 Opzegging", "5 Slotbepalingen")
    )

    document = chunk(record)

    (run,) = runs_of(document)
    assert (run["length"], run["median_section_chars"]) == (5, 12)
    assert (run["accepted"], run["reject_reason"]) == (False, "sections_too_short")
    assert document["route"] == "generic"
    assert methods_of(document) == ["generic"]


def test_a_run_whose_median_section_reaches_the_minimum_is_accepted():
    document = chunk(normalized_record(three_articles(400)))

    (run,) = runs_of(document)
    assert (run["median_section_chars"], run["accepted"]) == (400, True)
    assert run["reject_reason"] is None
    assert document["route"] == "structured"


def _sections(lengths):
    return "".join(
        block(f"Artikel {number} Titel", length)
        for number, length in enumerate(lengths, start=1)
    )


@pytest.mark.parametrize(("last", "accepted"), [(390, False), (400, True)])
def test_the_median_of_four_sections_is_the_lower_middle_one_including_the_last(
    last, accepted
):
    """Sorted, the sections are 100, ``last``, 410 and 1000 long: the median is
    the second-smallest, the last section, which runs to the end of the
    stream. The mean of the two middle ones would be at least 400 both
    times."""
    record = normalized_record(_sections([100, 1000, 410, last]))

    document = chunk(record)

    (run,) = runs_of(document)
    assert run["median_section_chars"] == last
    assert run["accepted"] is accepted
    assert run["reject_reason"] == (None if accepted else "sections_too_short")


def test_the_last_section_of_a_run_ends_at_the_next_runs_first_heading():
    record = normalized_record(_sections([100, 900, 450]) + _sections([500, 500, 500]))

    document = chunk(record)

    first, second = runs_of(document)
    assert (first["median_section_chars"], first["accepted"]) == (450, True)
    assert (second["median_section_chars"], second["accepted"]) == (500, True)
    assert spans_of(document) == [
        (0, 100),
        (100, 1000),
        (1000, 1450),
        (1450, 1950),
        (1950, 2450),
        (2450, 2950),
    ]
    assert [chunk["numbering_run"] for chunk in document["chunks"]] == [
        1,
        1,
        1,
        2,
        2,
        2,
    ]


def test_a_contents_list_before_the_body_is_generic_and_the_body_is_structured():
    preamble = "Fictieve huurovereenkomst\nInhoud\n"
    contents = "1 Begrippen\n2 Huurprijs\n3 Duur\n4 Opzegging\n5 Slotbepalingen\n"
    titles = ["Begrippen", "Huurprijs", "Duur", "Opzegging", "Slotbepalingen"]
    text = (
        preamble
        + contents
        + "".join(
            block(f"{number} {title}", 500)
            for number, title in enumerate(titles, start=1)
        )
    )
    record = normalized_record(text)

    document = chunk(record)

    contents_run, body_run = runs_of(document)
    assert (contents_run["run_index"], contents_run["reject_reason"]) == (
        1,
        "sections_too_short",
    )
    assert (body_run["run_index"], body_run["accepted"]) == (2, True)
    assert document["route"] == "structured"
    body_start = len(preamble + contents)
    assert spans_of(document) == [(0, body_start)] + [
        (body_start + 500 * position, body_start + 500 * (position + 1))
        for position in range(5)
    ]
    assert methods_of(document) == ["generic"] + ["heading_section"] * 5
    assert [chunk["numbering_run"] for chunk in document["chunks"]] == [None] + [2] * 5
    assert [chunk["clause_number"] for chunk in document["chunks"][1:]] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]


def test_a_document_whose_runs_are_all_rejected_has_only_generic_chunks():
    record = normalized_record(
        lines("1 Begrippen", "2 Huurprijs", "3 Duur", body(60))
        + "\n"
        + block("Artikel 1 Begrippen", 700)
        + block("Artikel 2 Huurprijs", 700)
    )

    document = chunk(record)

    assert [run["accepted"] for run in runs_of(document)] == [False]
    assert document["route"] == "generic"
    assert set(methods_of(document)) == {"generic"}


# --- The generic splitter (A9) --------------------------------------------------------

# Lines of 20 characters, line feed included, make the sizes below visible.
SMALL = {
    "chunk_target_chars": 100,
    "chunk_max_chars": 160,
    "chunk_overlap_chars": 30,
    "min_run_median_section_chars": 50,
    "max_label_inherit_chars": 150,
    "clause_label_max_chars": 20,
}


def test_overlap_joins_two_pieces_of_a_region_but_not_the_region_and_a_section():
    region = filler(160, line_length=20)  # eight lines of 20
    sections = (
        block("Artikel 1 Begrippen", 60, line_length=20)
        + block("Artikel 2 Huurprijs", 60, line_length=20)
        + block("Artikel 3 Duur", 60, line_length=20)
    )
    record = normalized_record(region + sections)

    document = chunk(record, settings_with(**SMALL))

    # The second piece repeats the first piece's last line (20 <= 30); the
    # section after the region starts at its heading, without overlap.
    assert spans_of(document) == [
        (0, 100),
        (80, 160),
        (160, 220),
        (220, 280),
        (280, 340),
    ]
    assert methods_of(document) == ["generic"] * 2 + ["heading_section"] * 3


def test_overlap_lines_are_dropped_from_the_front_until_the_next_line_fits():
    # Five lines of 20, a line of 50, a line of 20. The full overlap (three
    # lines, 60) and the 50 exceed 100: two overlap lines are left.
    text = filler(100, line_length=20) + body(49) + "\n" + body(19) + "\n"
    record = normalized_record(text)

    document = chunk(
        record, settings_with(chunk_target_chars=100, chunk_overlap_chars=60)
    )

    assert spans_of(document) == [(0, 100), (60, 150), (100, 170)]
    assert methods_of(document) == ["generic"] * 3


def test_a_line_longer_than_the_target_is_a_piece_of_its_own_without_overlap():
    text = lines(body(19), body(19), body(149), body(19), body(19))
    record = normalized_record(text)

    document = chunk(
        record, settings_with(chunk_target_chars=100, chunk_overlap_chars=30)
    )

    assert spans_of(document) == [(0, 40), (40, 190), (190, 229)]


def test_every_piece_is_whole_lines_within_the_target_and_has_a_line_of_its_own():
    text = (
        filler(3000, line_length=70) + body(1500) + "\n" + filler(2500, line_length=45)
    )
    record = normalized_record(text)

    document = chunk(record)

    stream = stream_of(record)
    previous_end = 0
    for piece in document["chunks"]:
        size = piece["char_end"] - piece["char_start"]
        single_line = "\n" not in piece["text"].rstrip("\n")
        assert size <= 1200 or single_line
        assert piece["char_end"] > previous_end  # a line of its own
        if single_line and size > 1200:
            assert piece["char_start"] == previous_end  # no overlap
        previous_end = piece["char_end"]
    assert previous_end == len(stream)


# --- Sections into chunks and clause fields (A10) -------------------------------------


def test_a_long_section_gives_labelled_parts_then_overflow():
    """The first section, 300 long, is cut into pieces that start at 0, 80,
    160 and 240, overlap included. With max_label_inherit_chars 170 the third
    piece is a part: its start, 160, counts, not the 180 where its own lines
    begin (R9)."""
    sections = (
        block("Artikel 1 Begrippen", 300, line_length=20)
        + block("Artikel 2 Een bijzonder lange fictieve titel", 100, line_length=20)
        + block("  Artikel 3 Duur", 100, line_length=20)
    )
    record = normalized_record(sections)

    document = chunk(record, settings_with(**{**SMALL, "max_label_inherit_chars": 170}))

    chunks = document["chunks"]
    assert spans_of(document) == [
        (0, 100),
        (80, 180),
        (160, 260),
        (240, 300),
        (300, 400),
        (400, 500),
    ]
    assert methods_of(document) == [
        "heading_section_part",
        "heading_section_part",
        "heading_section_part",
        "heading_section_overflow",
        "heading_section",
        "heading_section",
    ]
    assert [(chunk["part"], chunk["part_count"]) for chunk in chunks] == [
        (1, 3),
        (2, 3),
        (3, 3),
        (None, None),
        (None, None),
        (None, None),
    ]
    clause = [
        (
            chunk["clause_label"],
            chunk["clause_number"],
            chunk["numbering_run"],
            chunk["heading_family"],
        )
        for chunk in chunks
    ]
    assert clause == [
        ("Artikel 1 Begrippen", "1", 1, "keyword"),
        ("Artikel 1 Begrippen", "1", 1, "keyword"),
        ("Artikel 1 Begrippen", "1", 1, "keyword"),
        (None, None, None, None),
        ("Artikel 2 Een bijzon", "2", 1, "keyword"),
        ("Artikel 3 Duur", "3", 1, "keyword"),
    ]


def test_a_section_up_to_the_maximum_is_one_chunk():
    sections = (
        block("Artikel 1 Begrippen", 2000)
        + block("Artikel 2 Huurprijs", 2001)
        + block("Artikel 3 Duur", 500)
    )
    record = normalized_record(sections)

    document = chunk(record)

    assert methods_of(document)[0] == "heading_section"
    assert spans_of(document)[0] == (0, 2000)
    assert set(methods_of(document)[1:-1]) == {"heading_section_part"}
    assert methods_of(document)[-1] == "heading_section"


def test_a_line_longer_than_the_maximum_is_a_chunk_of_its_own_marked_as_such():
    text = filler(600) + body(2500) + "\n" + filler(600)
    record = normalized_record(text)

    document = chunk(record)

    flagged = [chunk for chunk in document["chunks"] if chunk["exceeds_max_chars"]]
    (long_line,) = flagged
    assert (long_line["char_start"], long_line["char_end"]) == (600, 3101)
    assert long_line["chunk_method"] == "generic"


# --- Empty pages and provenance (A11) -------------------------------------------------


def test_a_section_across_an_empty_page_has_one_span_per_page_with_text():
    first = block("Artikel 1 Begrippen", 500) + "Artikel 2 Huurprijs\n" + body(200)
    third = body(250) + "\n" + block("Artikel 3 Duur", 500)
    record = normalized_record(
        first,
        normalized_page(2, "", text_origin="none"),
        normalized_page(3, third, text_origin="ocr", ocr_mean_confidence=87.5),
    )

    document = chunk(record)

    assert document["empty_pages"] == [2]
    assert document["route"] == "structured"
    assert spans_of(document) == [(0, 500), (500, 972), (972, 1472)]
    across = document["chunks"][1]
    assert across["chunk_method"] == "heading_section"
    assert across["page_spans"] == [
        {
            "page_number": 1,
            "char_start": 500,
            "char_end": 720,
            "text_origin": "native",
            "tier": "Clean",
            "ocr_mean_confidence": None,
        },
        {
            "page_number": 3,
            "char_start": 0,
            "char_end": 251,
            "text_origin": "ocr",
            "tier": None,
            "ocr_mean_confidence": 87.5,
        },
    ]
    assert list(across["page_spans"][0]) == PAGE_SPAN_KEYS
    assert (across["page_start"], across["page_end"]) == (1, 3)
    assert across["text_origins"] == ["native", "ocr"]


# --- Chunk ids and determinism (A12) --------------------------------------------------


def test_the_chunk_id_is_recomputable_from_file_hash_start_and_text_hash():
    file_sha256 = "0123abcd" * 8
    # The third section runs on over the second page, past chunk_max_chars.
    record = normalized_record(three_articles(), filler(2000), sha256=file_sha256)

    document = chunk(record)

    ids = []
    for piece in document["chunks"]:
        text_sha256 = hashlib.sha256(piece["text"].encode("utf-8")).hexdigest()
        payload = f"{file_sha256}:{piece['char_start']}:{text_sha256}"
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        assert (piece["text_sha256"], piece["chunk_id"]) == (text_sha256, expected)
        ids.append(piece["chunk_id"])
    assert len(ids) > 3
    assert len(set(ids)) == len(ids)


def test_chunking_the_same_record_twice_gives_the_same_document():
    record = normalized_record(three_articles(), normalized_page(2, filler(2500)))

    first = chunk(record)
    second = chunk(copy.deepcopy(record))

    assert json.dumps(first, ensure_ascii=False) == json.dumps(
        second, ensure_ascii=False
    )


# --- The normalized record (R7) -------------------------------------------------------

DELETE = object()


def _altered(path, value):
    record = normalized_record("Tekst.", normalized_page(2, "Meer.", text_origin="ocr"))
    target = record
    for key in path[:-1]:
        target = target[key]
    if value is DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return record


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("file",), DELETE, "the record lacks the key 'file'"),
        (("file",), "invoer/fictief.pdf", "'file' is not an object"),
        (("file", "path"), None, "'file.path' is not a string"),
        (("file", "sha256"), DELETE, "'file' lacks the key 'sha256'"),
        (("file", "sha256"), None, "'file.sha256' is not a string"),
        (("status",), "klaar", "'status' is \"klaar\", not processed or failed"),
        (
            ("normalized_sha256",),
            DELETE,
            "the record lacks the key 'normalized_sha256'",
        ),
        (("normalized_sha256",), None, "'normalized_sha256' is not a string"),
        (("pages",), DELETE, "the record lacks the key 'pages'"),
        (("pages",), {}, "'pages' is not a list"),
        (("pages", 0), "Tekst.", "pages[0] is not an object"),
        (("pages", 0, "page_number"), "1", "pages[0].page_number is not an integer"),
        (("pages", 0, "page_number"), True, "pages[0].page_number is not an integer"),
        (("pages", 0, "text"), None, "pages[0].text is not a string"),
        (("pages", 1, "text_origin"), "scan", 'pages[1].text_origin is "scan"'),
        (("pages", 0, "tier"), 1, "pages[0].tier is not a string or null"),
        (("pages", 0, "language"), DELETE, "pages[0] lacks the key 'language'"),
        (
            ("pages", 1, "language_source"),
            ["index"],
            "pages[1].language_source is not a string or null",
        ),
        (
            ("pages", 1, "ocr_mean_confidence"),
            "90",
            "pages[1].ocr_mean_confidence is not a number or null",
        ),
        (
            ("pages", 1, "ocr_mean_confidence"),
            True,
            "pages[1].ocr_mean_confidence is not a number or null",
        ),
    ],
)
def test_an_invalid_record_raises_normalized_record_error(path, value, message):
    record = _altered(path, value)

    with pytest.raises(NormalizedRecordError) as raised:
        chunk_document(record, DEFAULTS)

    assert str(raised.value).startswith(message)


def test_a_top_level_that_is_not_an_object_is_invalid():
    with pytest.raises(NormalizedRecordError, match="not a JSON object"):
        validate_normalized_record([1])


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("pages", 1, "ocr_mean_confidence"), 88),
        (("pages", 0, "language"), None),
        (("pages", 0, "tier"), "Decent"),
    ],
)
def test_these_values_are_valid(path, value):
    chunk(_altered(path, value))


def test_a_failed_document_may_have_no_file_hash():
    validate_normalized_record(normalized_record(status="failed", sha256=None))


# --- Settings (A13, R6) ---------------------------------------------------------------


def _settings_yaml():
    path = find_project_root() / "config" / "settings.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_settings_yaml_has_the_chunking_section_with_the_python_defaults():
    section = _settings_yaml()["chunking"]

    assert list(section) == ["strategy", "heading_keywords"] + [
        key for key, _, _ in INTEGER_SETTINGS
    ]
    assert section["strategy"] == "structure_aware" == chunker.STRATEGY
    assert section["heading_keywords"] == KEYWORDS
    assert list(chunker.HEADING_KEYWORDS) == section["heading_keywords"]
    for key, default, _ in INTEGER_SETTINGS:
        assert section[key] == default
        assert getattr(chunker, key.upper()) == section[key]


def test_heading_keywords_writes_the_characters_outside_ascii_with_chr():
    source = Path(chunker.__file__).read_text(encoding="utf-8")
    definition = source[
        source.index("HEADING_KEYWORDS: ") : source.index("HEADING_MAX_GAP = ")
    ]

    assert source.isascii()
    backslash = chr(0x5C)  # U+005C REVERSE SOLIDUS
    for escape in ("u", "U", "N{", "x"):
        assert backslash + escape not in source
    for code_point in ("0x0142", "0x010C", "0x00E1", "0x00A7"):
        assert f"chr({code_point})" in definition


def test_settings_are_read_at_call_time():
    generic = normalized_record(filler(3000))
    structured = normalized_record(three_articles(450))

    assert chunk(generic)["chunk_count"] == 3
    assert chunk(generic, settings_with(chunk_target_chars=600))["chunk_count"] == 6
    assert chunk(structured)["route"] == "structured"
    stricter = chunk(structured, settings_with(min_run_median_section_chars=451))
    assert stricter["route"] == "generic"
    assert runs_of(stricter)[0]["reject_reason"] == "sections_too_short"


def test_without_a_chunking_section_the_python_defaults_apply():
    config = chunking_settings({})

    assert config.strategy == "structure_aware"
    assert config.heading_keywords == chunker.HEADING_KEYWORDS
    for key, default, _ in INTEGER_SETTINGS:
        assert getattr(config, key) == default


@pytest.mark.parametrize(
    "keywords",
    ["Artikel", [], [""], ["Artikel", 3], None],
    ids=["a-string", "empty", "an-empty-keyword", "a-number", "null"],
)
def test_heading_keywords_must_be_a_non_empty_list_of_non_empty_strings(keywords):
    with pytest.raises(ChunkingConfigError, match="heading_keywords"):
        chunking_settings(settings_with(heading_keywords=keywords))


@pytest.mark.parametrize(("key", "default", "minimum"), INTEGER_SETTINGS)
def test_each_integer_setting_must_be_an_integer_at_its_minimum_or_above(
    key, default, minimum
):
    for value in (True, float(default), str(default), None, minimum - 1):
        with pytest.raises(ChunkingConfigError, match=key):
            chunking_settings(settings_with(**{key: value}))
    assert getattr(chunking_settings(settings_with(**{key: minimum})), key) == minimum


def test_a_configured_keyword_list_replaces_the_default():
    text = _articles([1, 2, 3], 450).replace("Artikel", "Paragraaf")
    record = normalized_record(text)

    assert chunk(record)["heading_evidence"]["family"] is None
    document = chunk(record, settings_with(heading_keywords=["Paragraaf"]))
    assert document["route"] == "structured"
    assert document["chunks"][0]["clause_label"] == "Paragraaf 1 Titel"


def test_chunk_keys_are_the_documented_ones():
    document = chunk(normalized_record(three_articles()))

    assert [list(piece) for piece in document["chunks"]] == [CHUNK_KEYS] * 3


# --- The strategy (spec 002) ----------------------------------------------------------

FIXED_LENGTH = settings_with(strategy="fixed_length")


def test_the_strategies_are_structure_aware_and_fixed_length():
    assert chunker.STRATEGIES == ("structure_aware", "fixed_length")
    for strategy in chunker.STRATEGIES:
        assert chunking_settings(settings_with(strategy=strategy)).strategy == strategy


@pytest.mark.parametrize(
    "value",
    ["fixed", "Fixed_Length", "", None, 1, True, ["fixed_length"]],
    ids=["unknown", "other-case", "empty", "null", "a-number", "a-bool", "a-list"],
)
def test_strategy_must_be_one_of_the_strategies(value):
    with pytest.raises(ChunkingConfigError, match="strategy"):
        chunking_settings(settings_with(strategy=value))


def test_fixed_length_cuts_a_structured_document_as_if_its_run_were_rejected():
    # Two pages, so the generic pieces also cross a page join.
    text = three_articles(900)
    record = normalized_record(text[:1300], text[1300:])
    rejected = settings_with(min_run_median_section_chars=10**6)

    structured = chunk(record)
    fixed = chunk(record, FIXED_LENGTH)

    assert structured["route"] == "structured"
    assert fixed["route"] == "generic"
    assert fixed["chunks"] == chunk(record, rejected)["chunks"]
    assert len(fixed["chunks"]) > 1
    assert {piece["chunk_method"] for piece in fixed["chunks"]} == {"generic"}
    assert {piece["clause_label"] for piece in fixed["chunks"]} == {None}
    assert fixed["chunks"][-1]["page_end"] == 2


def test_fixed_length_still_records_the_heading_evidence():
    record = normalized_record(three_articles())

    structured = chunk(record)
    fixed = chunk(record, FIXED_LENGTH)

    assert fixed["heading_evidence"] == structured["heading_evidence"]
    assert runs_of(fixed)[0]["accepted"] is True


@pytest.mark.parametrize(
    "text",
    [filler(3000), three_articles(300)],
    ids=["no-headings", "a-rejected-run"],
)
def test_fixed_length_leaves_a_document_without_an_accepted_run_unchanged(text):
    record = normalized_record(text)

    document = chunk(record)

    assert document["route"] == "generic"
    assert chunk(record, FIXED_LENGTH) == document


def test_fixed_length_takes_the_splitter_sizes_from_the_settings():
    record = normalized_record(three_articles(900))
    small = settings_with(strategy="fixed_length", chunk_target_chars=600)

    sizes = [len(piece["text"]) for piece in chunk(record, small)["chunks"]]

    assert len(sizes) > len(chunk(record, FIXED_LENGTH)["chunks"])
    assert max(sizes) <= 600
