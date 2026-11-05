"""Candidates and field status (spec 008, D8, R12, R13, A7, A8).

Pure functions over fabricated responses, and one run through the command
for the order across batches.
"""

from __future__ import annotations

import pytest

from companydocs_rag.metadata.checks import candidate_record, provision_record
from companydocs_rag.metadata.fields import field_status
from companydocs_rag.metadata.report import ExtractionFile, field_views
from tests.metadata._helpers import (
    LEASE,
    SHA_A,
    Generator,
    answer,
    candidate,
    chunk,
    chunk_record,
    counted,
    extract,
    extraction_files,
    extraction_record,
    months,
    provision,
    reply,
    use_settings,
    write_json_file,
    write_overrides,
    years,
)

EN_DASH = chr(0x2013)  # U+2013 EN DASH
LEFT_QUOTE = chr(0x201C)  # U+201C LEFT DOUBLE QUOTATION MARK
RIGHT_QUOTE = chr(0x201D)  # U+201D RIGHT DOUBLE QUOTATION MARK
FIRST = (
    f"De Huurder zegt op met een termijn van\ndrie maanden {EN_DASH} zie "
    f"artikel {LEFT_QUOTE}4{RIGHT_QUOTE}."
)
SECOND = "De fictieve huur gaat in op 1 april 2030."
SOURCES = [
    chunk(SHA_A, 0, FIRST, page_start=2),
    chunk(SHA_A, 1, SECOND, page_start=3, page_end=4),
]
VERIFIED_QUOTE = 'de huurder zegt op met een termijn van drie maanden - zie artikel "4"'


def _record(field, value, kind, quote=VERIFIED_QUOTE, source=1):
    return candidate_record(
        field, candidate(value, kind, quote, source=source), 1, SOURCES
    )


# --- value_valid and the stored value (A7) --------------------------------------------


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("2031-02-28", True),
        ("2031-02-29", False),
        ("2031-2-28", False),
        ("20310228", False),
        ("28-02-2031", False),
        (None, False),
    ],
)
def test_a_date_must_be_a_real_iso_date(value, valid):
    record = _record("signing_date", value, "date")

    assert record["value_valid"] is valid
    assert record["value"] == value


@pytest.mark.parametrize(
    ("field", "kind"),
    [
        ("initial_term", "duration"),
        ("notice_period", "duration"),
        ("renewal", "fixed_period"),
    ],
)
@pytest.mark.parametrize(
    ("value", "valid", "stored"),
    [
        (years(3), True, {"amount": 3, "unit": "years"}),
        (years(3.0), True, {"amount": 3, "unit": "years"}),
        (years(0), False, years(0)),
        (years(-1), False, years(-1)),
        (years(1.5), False, years(1.5)),
        (years(True), False, years(True)),
        (years("3"), False, years("3")),
        ({"amount": 3, "unit": "decennia"}, False, {"amount": 3, "unit": "decennia"}),
        (
            {"amount": 3, "unit": "years", "extra": 1},
            False,
            {"amount": 3, "unit": "years", "extra": 1},
        ),
        (None, False, None),
    ],
    ids=[
        "3",
        "3.0",
        "0",
        "-1",
        "1.5",
        "true",
        "string",
        "unit",
        "extra-key",
        "null",
    ],
)
def test_a_duration_needs_a_whole_amount_of_at_least_one_and_a_unit(
    field, kind, value, valid, stored
):
    record = _record(field, value, kind)

    assert record["value_valid"] is valid
    assert record["value"] == stored
    if valid:
        assert type(record["value"]["amount"]) is int


@pytest.mark.parametrize(
    ("field", "kind"),
    [
        ("signing_date", "conditional"),
        ("commencement_date", "blank"),
        ("initial_term_end", "illegible"),
        ("initial_term", "indefinite"),
        ("renewal", "none"),
        ("renewal", "indefinite"),
    ],
)
@pytest.mark.parametrize("value", ["2031-01-01", {"x": 1}, None, 5])
def test_the_null_kinds_are_valid_whatever_the_value_and_store_null(field, kind, value):
    record = _record(field, value, kind)

    assert record["value_valid"] is True
    assert record["value"] is None


@pytest.mark.parametrize(
    ("field", "kind", "value"),
    [
        ("signing_date", "duration", years(3)),
        ("notice_period", "date", "2031-01-01"),
        ("initial_term", "none", None),
        ("renewal", "duration", years(1)),
        ("signing_date", "datum", "2031-01-01"),
    ],
)
def test_a_kind_the_field_does_not_allow_is_invalid(field, kind, value):
    record = _record(field, value, kind)

    assert record["value_valid"] is False
    assert record["counted"] is False
    assert record["value"] == value


# --- quote_verified and the source (A7) -----------------------------------------------


def test_a_quote_that_differs_only_in_case_whitespace_quotes_and_dashes_verifies():
    record = _record("notice_period", months(3), "duration")

    assert record["quote_verified"] is True
    assert record["counted"] is True
    assert record["chunk_id"] == SOURCES[0]["chunk_id"]
    assert record["page_spans"] == SOURCES[0]["page_spans"]
    assert record["batch_index"] == 1


@pytest.mark.parametrize(
    ("quote", "source"),
    [
        ("gaat in op 1 april 2030", 1),
        ("", 1),
        (" \n\t ", 1),
    ],
    ids=["other-source", "empty", "whitespace"],
)
def test_a_quote_that_is_not_in_its_source_does_not_verify(quote, source):
    record = _record("notice_period", months(3), "duration", quote=quote, source=source)

    assert record["quote_verified"] is False
    assert record["counted"] is False
    assert record["chunk_id"] == SOURCES[0]["chunk_id"]


@pytest.mark.parametrize("source", [0, 3, "1", True, None, 1.0])
def test_a_source_outside_the_batch_verifies_nothing(source):
    record = _record("notice_period", months(3), "duration", source=source)

    assert record["quote_verified"] is False
    assert record["counted"] is False
    assert (record["chunk_id"], record["page_spans"]) == (None, [])


def test_the_second_source_is_the_second_chunk():
    record = _record(
        "commencement_date", "2030-04-01", "date", quote=SECOND.upper(), source=2
    )

    assert record["counted"] is True
    assert record["chunk_id"] == SOURCES[1]["chunk_id"]
    assert [span["page_number"] for span in record["page_spans"]] == [3, 4]


@pytest.mark.parametrize(
    ("value", "quote", "counted"),
    [
        (months(3), VERIFIED_QUOTE, True),
        (months(3), "elders", False),
        (months(0), VERIFIED_QUOTE, False),
        (months(0), "elders", False),
    ],
)
def test_counted_is_value_valid_and_quote_verified(value, quote, counted):
    record = _record("notice_period", value, "duration", quote=quote)

    assert record["counted"] is counted
    assert record["counted"] is (record["value_valid"] and record["quote_verified"])


def test_provision_items_are_checked_the_same_way():
    verified = provision_record(
        provision("tenant", VERIFIED_QUOTE, source=1), 2, SOURCES
    )
    elsewhere = provision_record(provision("both", "elders", source=2), 2, SOURCES)
    outside = provision_record(
        provision("landlord", VERIFIED_QUOTE, source=5), 2, SOURCES
    )

    assert list(verified) == [
        "party",
        "as_written",
        "quote",
        "quote_verified",
        "batch_index",
        "chunk_id",
        "page_spans",
    ]
    assert (verified["quote_verified"], verified["batch_index"]) == (True, 2)
    assert verified["chunk_id"] == SOURCES[0]["chunk_id"]
    assert verified["page_spans"] == SOURCES[0]["page_spans"]
    assert elsewhere["quote_verified"] is False
    assert elsewhere["chunk_id"] == SOURCES[1]["chunk_id"]
    assert (outside["quote_verified"], outside["chunk_id"], outside["page_spans"]) == (
        False,
        None,
        [],
    )


# --- Field status (A8) ----------------------------------------------------------------


def _counted(value, kind):
    return {"value": value, "value_kind": kind, "counted": True}


def _uncounted(value, kind):
    return {"value": value, "value_kind": kind, "counted": False}


@pytest.mark.parametrize(
    ("candidates", "status"),
    [
        ([], "not_found"),
        ([_uncounted("2031-01-01", "date"), _uncounted(None, "blank")], "not_found"),
        ([_counted("2031-01-01", "date"), _uncounted("2032-01-01", "date")], "found"),
        ([_counted(years(1), "duration"), _counted(months(12), "duration")], "found"),
        (
            [
                _counted({"amount": 2, "unit": "weeks"}, "duration"),
                _counted({"amount": 14, "unit": "days"}, "duration"),
            ],
            "found",
        ),
        ([_counted(None, "conditional"), _counted(None, "conditional")], "found"),
        (
            [_counted("2031-01-01", "date"), _counted(None, "conditional")],
            "conflicting",
        ),
        (
            [
                _counted(years(1), "duration"),
                _counted({"amount": 365, "unit": "days"}, "duration"),
            ],
            "conflicting",
        ),
        (
            [_counted("2031-01-01", "date"), _counted("2031-01-02", "date")],
            "conflicting",
        ),
    ],
    ids=[
        "none",
        "only-uncounted",
        "one-counted",
        "year-and-months",
        "weeks-and-days",
        "two-conditional",
        "date-and-conditional",
        "year-and-days",
        "two-dates",
    ],
)
def test_the_status_comes_from_the_counted_candidates(candidates, status):
    assert field_status(candidates) == status


def test_the_chosen_value_is_the_first_counted_candidates_stored_value():
    record = extraction_record(
        "bron/fictief.pdf",
        SHA_A,
        fields={
            "initial_term": {
                "status": "found",
                "candidates": [
                    {**counted(years(2), "duration"), "counted": False},
                    counted(years(1), "duration"),
                    counted(months(12), "duration"),
                ],
            }
        },
    )

    view = field_views(ExtractionFile("fictief.json", record).record)["initial_term"]

    assert (view.status, view.kind, view.value) == ("found", "duration", years(1))


def test_the_candidates_of_two_ok_batches_are_merged_in_batch_order(
    tmp_path, monkeypatch
):
    # Scores 3, 2 and 1, so the ranking follows chunk_index; the budget holds
    # one chunk per batch.
    texts = [
        "Opzegging, termijn, looptijd: drie maanden, fictief.",
        "De looptijd en de opzegging van de huur.",
        "Fictieve opzegging per maand.",
    ]
    chunks = [chunk(SHA_A, index, text) for index, text in enumerate(texts)]
    write_json_file(
        tmp_path / "chunks",
        "huur.json",
        chunk_record(tmp_path / "los" / "fictief-huur.pdf", SHA_A, *chunks),
    )
    overrides = write_overrides(
        tmp_path / "handmatig" / "types.csv", [("fictief-huur.pdf", LEASE)]
    )
    use_settings(tmp_path, monkeypatch, selection_budget_chars=len(texts[0]))
    generator = Generator(
        reply(
            answer(
                notice_period=[
                    candidate(months(3), "duration", "termijn, looptijd: drie maanden"),
                    candidate(months(1), "duration", "opzegging, termijn"),
                ]
            )
        ),
        reply("geen json"),
        reply(
            answer(
                notice_period=[
                    candidate(months(2), "duration", "fictieve opzegging per maand")
                ]
            )
        ),
    )

    assert extract(tmp_path, overrides=overrides, generator=generator) == 4

    record = extraction_files(tmp_path / "extracted")["huur.json"]
    assert record["extraction_status"] == "partial"
    assert [batch["chunk_ids"] for batch in record["batches"]] == [
        [item["chunk_id"]] for item in chunks
    ]
    assert [batch["status"] for batch in record["batches"]] == ["ok", "error", "ok"]
    found = record["fields"]["notice_period"]["candidates"]
    assert [(c["batch_index"], c["value"], c["counted"]) for c in found] == [
        (1, months(3), True),
        (1, months(1), True),
        (3, months(2), True),
    ]
    assert record["fields"]["notice_period"]["status"] == "conflicting"
