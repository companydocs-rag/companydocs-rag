"""Derived dates for a reference date (spec 008, D12, R20, R22; Q4, I1, A12).

Pure functions over fabricated field views; dates are fabricated. I1's cases
through ``report`` are in ``test_date_range.py``.
"""

from __future__ import annotations

from datetime import date

import pytest

from companydocs_rag.metadata.dates import (
    DateOutOfRange,
    FieldView,
    add_days,
    add_months,
    derive,
    effective_term_end,
    first_term_end,
    range_failures,
    shift,
)

MINUS = chr(0x2212)  # U+2212 MINUS SIGN
FIELDS = [
    "signing_date",
    "commencement_date",
    "initial_term",
    "initial_term_end",
    "renewal",
    "notice_period",
]
ENTRY_KEYS = [
    "status",
    "value",
    "reason",
    "missing_inputs",
    "inputs",
    "calculation",
    "for_term_end",
    "expired",
    "since",
]
DERIVED = ["expiry_date", "next_renewal_date", "latest_notice_date"]


def duration(amount, unit):
    return {"amount": amount, "unit": unit}


def views(**given):
    """Field views: every field ``not_found`` unless given. A given value is
    a status (``"conflicting"``), or (kind, value) or (kind, value, pages)
    for a found field."""
    result = {}
    for name in FIELDS:
        item = given.get(name, "not_found")
        if isinstance(item, str):
            pages = [[2, 2], [5, 6]] if item == "conflicting" else []
            result[name] = FieldView(name, item, None, None, pages)
        else:
            kind, value, *rest = item
            pages = [list(pair) for pair in (rest[0] if rest else [(2, 2)])]
            result[name] = FieldView(name, "found", kind, value, pages)
    return result


def run(reference, *, amendment=False, **given):
    found = views(**given)
    term_end = effective_term_end(found)
    return term_end, derive(found, term_end, reference, amendment=amendment)


D = date(2031, 9, 1)


# --- Arithmetic -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "period", "factor", "result"),
    [
        (date(2031, 1, 31), duration(1, "months"), 1, date(2031, 2, 28)),
        (date(2032, 1, 31), duration(1, "months"), 1, date(2032, 2, 29)),
        (date(2032, 2, 29), duration(1, "years"), 1, date(2033, 2, 28)),
        (date(2031, 5, 31), duration(3, "months"), -1, date(2031, 2, 28)),
        (date(2031, 1, 1), duration(2, "weeks"), 1, date(2031, 1, 15)),
        (date(2031, 1, 1), duration(14, "days"), 1, date(2031, 1, 15)),
        (date(2031, 8, 31), duration(6, "months"), 2, date(2032, 8, 31)),
    ],
)
def test_month_arithmetic_clamps_to_the_months_last_day(start, period, factor, result):
    assert shift(start, period, factor) == result


def test_e_k_is_computed_from_e0_never_step_by_step():
    e0 = date(2031, 8, 31)
    period = duration(6, "months")

    assert shift(e0, period, 1) == date(2032, 2, 29)
    assert shift(e0, period, 2) == date(2032, 8, 31)
    assert shift(shift(e0, period, 1), period, 1) == date(2032, 8, 29)
    assert add_months(date(2031, 12, 31), -6) == date(2031, 6, 30)


# --- E0 (D12, Q4) ---------------------------------------------------------------------


def test_a_stated_end_is_used():
    term_end, _ = run(D, initial_term_end=("date", "2034-03-14"))

    assert term_end.to_dict() == {
        "value": "2034-03-14",
        "source": "stated",
        "term_end_mismatch": False,
    }


def test_the_fallback_is_commencement_plus_term_minus_one_day():
    term_end, _ = run(
        D,
        commencement_date=("date", "2030-03-15"),
        initial_term=("duration", duration(4, "years")),
    )

    assert term_end.to_dict() == {
        "value": "2034-03-14",
        "source": "computed",
        "term_end_mismatch": False,
    }


@pytest.mark.parametrize(
    ("stated", "mismatch"), [("2034-03-14", False), ("2034-03-31", True)]
)
def test_a_stated_end_wins_and_a_difference_is_flagged(stated, mismatch):
    term_end, _ = run(
        D,
        commencement_date=("date", "2030-03-15"),
        initial_term=("duration", duration(4, "years")),
        initial_term_end=("date", stated),
    )

    assert term_end.to_dict() == {
        "value": stated,
        "source": "stated",
        "term_end_mismatch": mismatch,
    }
    assert term_end.computed == date(2034, 3, 14)


@pytest.mark.parametrize(
    "end",
    ["not_found", ("blank", None), ("illegible", None)],
    ids=["not_found", "blank", "illegible"],
)
def test_not_found_blank_and_illegible_fall_back(end):
    term_end, _ = run(
        D,
        commencement_date=("date", "2030-03-15"),
        initial_term=("duration", duration(4, "years")),
        initial_term_end=end,
    )

    assert (term_end.value, term_end.source) == (date(2034, 3, 14), "computed")


@pytest.mark.parametrize(
    ("end", "reason"),
    [("conflicting", "conflicting"), (("conditional", None), "conditional")],
    ids=["conflicting", "conditional"],
)
def test_conflicting_and_conditional_do_not_fall_back(end, reason):
    term_end, derived = run(
        D,
        commencement_date=("date", "2030-03-15"),
        initial_term=("duration", duration(4, "years")),
        initial_term_end=end,
        renewal=("fixed_period", duration(1, "years")),
        notice_period=("duration", duration(3, "months")),
    )

    assert term_end.to_dict() == {
        "value": None,
        "source": None,
        "term_end_mismatch": False,
    }
    for name in DERIVED:
        assert derived[name]["status"] == "not_computable"
        assert derived[name]["missing_inputs"] == [
            {"field": "initial_term_end", "reason": reason}
        ]


# --- renewal none ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "expired"), [(date(2034, 3, 14), False), (date(2034, 3, 15), True)]
)
def test_no_renewal_expires_at_e0(reference, expired):
    _, derived = run(
        reference,
        initial_term_end=("date", "2034-03-14", [(2, 2)]),
        renewal=("none", None, [(3, 3)]),
        notice_period=("duration", duration(3, "months")),
    )

    assert derived["expiry_date"] == {
        "status": "computed",
        "value": "2034-03-14",
        "reason": None,
        "missing_inputs": [],
        "inputs": [
            {"field": "initial_term_end", "value": "2034-03-14", "pages": [[2, 2]]},
            {"field": "renewal", "value": None, "pages": [[3, 3]]},
        ],
        "calculation": (
            "14-03-2034 (invoer: einde eerste termijn p. 2, verlenging p. 3)"
        ),
        "for_term_end": None,
        "expired": expired,
        "since": None,
    }
    for name in ("next_renewal_date", "latest_notice_date"):
        assert derived[name]["status"] == "not_applicable"
        assert derived[name]["reason"] == "no_renewal"


def test_no_renewal_needs_no_e0_for_the_other_two():
    _, derived = run(D, renewal=("none", None))

    assert derived["expiry_date"]["status"] == "not_computable"
    for name in ("next_renewal_date", "latest_notice_date"):
        assert (derived[name]["status"], derived[name]["reason"]) == (
            "not_applicable",
            "no_renewal",
        )


# --- renewal fixed_period -------------------------------------------------------------


def test_a_fixed_period_before_e0_is_k_0():
    _, derived = run(
        D,
        initial_term_end=("date", "2034-03-14", [(2, 2)]),
        renewal=("fixed_period", duration(1, "years"), [(2, 2)]),
    )

    assert derived["expiry_date"]["value"] == "2034-03-14"
    assert derived["expiry_date"]["calculation"] == (
        "14-03-2034 (invoer: einde eerste termijn p. 2, verlenging p. 2)"
    )
    assert derived["next_renewal_date"]["value"] == "2034-03-15"
    assert derived["next_renewal_date"]["calculation"] == (
        "14-03-2034 + 1 dag = 15-03-2034 (invoer: einde eerste termijn p. 2, "
        "verlenging p. 2)"
    )


@pytest.mark.parametrize(
    ("reference", "expiry", "step"),
    [
        (date(2031, 9, 1), "2032-02-29", "31-08-2031 + 6 maanden = 29-02-2032"),
        (date(2032, 3, 1), "2032-08-31", "31-08-2031 + 12 maanden = 31-08-2032"),
    ],
)
def test_a_fixed_period_after_e0_takes_the_smallest_e_k_from_e0(
    reference, expiry, step
):
    _, derived = run(
        reference,
        initial_term_end=("date", "2031-08-31", [(4, 4)]),
        renewal=("fixed_period", duration(6, "months"), [(5, 5)]),
    )

    entry = derived["expiry_date"]
    assert (entry["status"], entry["value"]) == ("computed", expiry)
    assert entry["calculation"] == (
        f"{step} (invoer: einde eerste termijn p. 4, verlenging p. 5)"
    )
    assert entry["expired"] is None
    following = derived["next_renewal_date"]
    assert (
        following["value"]
        == shift(date.fromisoformat(expiry), duration(1, "days")).isoformat()
    )


def test_the_latest_notice_date_looks_for_the_first_term_end_still_ahead():
    _, derived = run(
        D,
        initial_term_end=("date", "2031-12-31", [(2, 2)]),
        renewal=("fixed_period", duration(1, "years"), [(2, 2)]),
        notice_period=("duration", duration(6, "months"), [(4, 4)]),
    )

    assert derived["expiry_date"]["value"] == "2031-12-31"
    assert derived["latest_notice_date"] == {
        "status": "computed",
        "value": "2032-06-30",
        "reason": None,
        "missing_inputs": [],
        "inputs": [
            {"field": "initial_term_end", "value": "2031-12-31", "pages": [[2, 2]]},
            {
                "field": "renewal",
                "value": duration(1, "years"),
                "pages": [[2, 2]],
            },
            {
                "field": "notice_period",
                "value": duration(6, "months"),
                "pages": [[4, 4]],
            },
        ],
        "calculation": (
            f"31-12-2032 {MINUS} 6 maanden = 30-06-2032 (termijneinde 31-12-2032; "
            "invoer: einde eerste termijn p. 2, verlenging p. 2, opzegtermijn p. 4)"
        ),
        "for_term_end": "2032-12-31",
        "expired": None,
        "since": None,
    }


def test_d13s_example_form():
    _, derived = run(
        date(2033, 7, 31),
        initial_term_end=("date", "2033-10-31", [(2, 2)]),
        renewal=("fixed_period", duration(5, "years"), [(2, 2)]),
        notice_period=("duration", duration(3, "months"), [(4, 4)]),
    )

    assert derived["latest_notice_date"]["calculation"] == (
        f"31-10-2033 {MINUS} 3 maanden = 31-07-2033 (termijneinde 31-10-2033; "
        "invoer: einde eerste termijn p. 2, verlenging p. 2, opzegtermijn p. 4)"
    )


def test_a_computed_e0_precedes_or_replaces_the_step():
    _, derived = run(
        D,
        commencement_date=("date", "2030-03-15", [(1, 1)]),
        initial_term=("duration", duration(4, "years"), [(1, 1)]),
        renewal=("fixed_period", duration(1, "years"), [(2, 2)]),
        notice_period=("duration", duration(3, "months"), [(4, 4)]),
    )

    assert derived["expiry_date"]["calculation"] == (
        f"15-03-2030 + 4 jaar {MINUS} 1 dag = 14-03-2034 (invoer: ingangsdatum "
        "p. 1, looptijd p. 1, verlenging p. 2)"
    )
    assert derived["next_renewal_date"]["calculation"] == (
        f"15-03-2030 + 4 jaar {MINUS} 1 dag = 14-03-2034; 14-03-2034 + 1 dag = "
        "15-03-2034 (invoer: ingangsdatum p. 1, looptijd p. 1, verlenging p. 2)"
    )
    assert derived["latest_notice_date"]["calculation"] == (
        f"15-03-2030 + 4 jaar {MINUS} 1 dag = 14-03-2034; 14-03-2034 {MINUS} "
        "3 maanden = 14-12-2033 (termijneinde 14-03-2034; invoer: ingangsdatum "
        "p. 1, looptijd p. 1, verlenging p. 2, opzegtermijn p. 4)"
    )
    assert [item["field"] for item in derived["expiry_date"]["inputs"]] == [
        "commencement_date",
        "initial_term",
        "renewal",
    ]


# --- renewal indefinite ---------------------------------------------------------------


def test_indefinite_with_e0_ahead():
    _, derived = run(
        date(2031, 9, 1),
        initial_term_end=("date", "2031-12-31"),
        renewal=("indefinite", None),
        notice_period=("duration", duration(3, "months")),
    )

    assert derived["expiry_date"]["value"] == "2031-12-31"
    assert derived["next_renewal_date"]["value"] == "2032-01-01"
    latest = derived["latest_notice_date"]
    assert (latest["value"], latest["for_term_end"]) == ("2031-09-30", "2031-12-31")


def test_indefinite_with_the_notice_date_passed():
    _, derived = run(
        date(2031, 11, 1),
        initial_term_end=("date", "2031-12-31"),
        renewal=("indefinite", None),
        notice_period=("duration", duration(3, "months")),
    )

    assert derived["expiry_date"]["value"] == "2031-12-31"
    assert derived["latest_notice_date"] == {
        "status": "not_applicable",
        "value": None,
        "reason": "indefinite_after_initial_term",
        "missing_inputs": [],
        "inputs": [],
        "calculation": None,
        "for_term_end": None,
        "expired": None,
        "since": "2032-01-01",
    }


def test_indefinite_after_e0():
    _, derived = run(
        date(2032, 1, 1),
        initial_term_end=("date", "2031-12-31"),
        renewal=("indefinite", None),
        notice_period=("duration", duration(3, "months")),
    )

    for name in DERIVED:
        entry = derived[name]
        assert (entry["status"], entry["reason"], entry["since"]) == (
            "not_applicable",
            "indefinite_after_initial_term",
            "2032-01-01",
        )


# --- not_applicable before everything else --------------------------------------------


def test_an_amendment_is_not_applicable_before_every_other_rule():
    _, derived = run(
        D,
        amendment=True,
        initial_term=("indefinite", None),
        renewal=("none", None),
    )

    for name in DERIVED:
        assert (derived[name]["status"], derived[name]["reason"]) == (
            "not_applicable",
            "amendment",
        )


def test_an_indefinite_term_is_not_applicable():
    _, derived = run(D, initial_term=("indefinite", None), renewal=("none", None))

    for name in DERIVED:
        assert (derived[name]["status"], derived[name]["reason"]) == (
            "not_applicable",
            "indefinite_term",
        )


# --- not_computable -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("end", "reason", "with_fallback"),
    [
        ("not_found", "not_found", True),
        ("conflicting", "conflicting", False),
        (("conditional", None), "conditional", False),
        (("blank", None), "blank", True),
        (("illegible", None), "illegible", True),
    ],
    ids=["not_found", "conflicting", "conditional", "blank", "illegible"],
)
def test_each_reason_without_e0(end, reason, with_fallback):
    _, derived = run(
        D,
        commencement_date=("blank", None),
        initial_term_end=end,
        renewal="conflicting",
    )

    expected = [{"field": "initial_term_end", "reason": reason}]
    if with_fallback:
        expected = [
            {"field": "commencement_date", "reason": "blank"},
            {"field": "initial_term", "reason": "not_found"},
            *expected,
        ]
    expected.append({"field": "renewal", "reason": "conflicting"})
    for name in ("expiry_date", "next_renewal_date"):
        assert derived[name]["missing_inputs"] == expected
    assert derived["latest_notice_date"]["missing_inputs"] == [
        *expected,
        {"field": "notice_period", "reason": "not_found"},
    ]
    for name in DERIVED:
        entry = derived[name]
        assert list(entry) == ENTRY_KEYS
        assert (entry["status"], entry["value"], entry["reason"]) == (
            "not_computable",
            None,
            None,
        )
        assert (entry["inputs"], entry["calculation"]) == ([], None)


def test_a_missing_notice_period_affects_the_latest_notice_date_only():
    _, derived = run(
        D,
        initial_term_end=("date", "2034-03-14"),
        renewal=("fixed_period", duration(1, "years")),
    )

    assert derived["expiry_date"]["status"] == "computed"
    assert derived["next_renewal_date"]["status"] == "computed"
    assert derived["latest_notice_date"]["missing_inputs"] == [
        {"field": "notice_period", "reason": "not_found"}
    ]


def test_every_entry_has_the_nine_keys_in_every_branch():
    cases = [
        run(D, amendment=True)[1],
        run(D, initial_term=("indefinite", None))[1],
        run(D, renewal=("none", None), initial_term_end=("date", "2031-01-01"))[1],
        run(D)[1],
        run(
            D,
            initial_term_end=("date", "2031-01-01"),
            renewal=("indefinite", None),
            notice_period=("duration", duration(1, "months")),
        )[1],
    ]
    for derived in cases:
        assert list(derived) == DERIVED
        for entry in derived.values():
            assert list(entry) == ENTRY_KEYS


# --- Outside years 1-9999 (I1) --------------------------------------------------------


def oor(*fields):
    """``missing_inputs`` of an entry whose computation left the range."""
    return [{"field": name, "reason": "out_of_range"} for name in fields]


@pytest.mark.parametrize(
    ("compute", "result"),
    [
        (lambda: add_days(date(9999, 12, 30), 1), date(9999, 12, 31)),
        (lambda: add_days(date(1, 1, 2), -1), date(1, 1, 1)),
        (lambda: add_months(date(9999, 11, 30), 1), date(9999, 12, 30)),
        (lambda: add_months(date(1, 2, 28), -1), date(1, 1, 28)),
        (
            lambda: shift(date(2034, 12, 31), duration(7965, "years")),
            date(9999, 12, 31),
        ),
    ],
    ids=["last-day", "first-day", "last-month", "first-month", "last-year"],
)
def test_arithmetic_up_to_the_ends_of_the_range(compute, result):
    assert compute() == result


@pytest.mark.parametrize(
    "compute",
    [
        lambda: add_days(date(9999, 12, 31), 1),
        lambda: add_days(date(1, 1, 1), -1),
        lambda: add_months(date(9999, 12, 31), 1),
        lambda: add_months(date(1, 1, 31), -1),
        lambda: shift(date(2030, 1, 1), duration(10000, "years")),
        lambda: shift(date(2034, 12, 31), duration(30000, "months"), -1),
        lambda: shift(date(2031, 1, 1), duration(10**18, "days")),
        lambda: shift(date(2031, 1, 1), duration(10**18, "weeks"), -1),
    ],
    ids=[
        "past-the-last-day",
        "before-the-first-day",
        "past-the-last-month",
        "before-the-first-month",
        "10000-years",
        "minus-30000-months",
        "10-to-the-18-days",
        "minus-10-to-the-18-weeks",
    ],
)
def test_arithmetic_outside_years_1_to_9999_raises(compute):
    with pytest.raises(DateOutOfRange):
        compute()


def literal_search(e0, period, reference, notice=None):
    """I1 (c) read literally: k counts up from 0, and the first E_k or
    E_k - N outside the range ends the search (None)."""
    k = 0
    while True:
        try:
            end = shift(e0, period, k)
            target = end if notice is None else shift(end, notice, -1)
        except DateOutOfRange:
            return None
        if target >= reference:
            return k, end
        k += 1


def searched(e0, period, reference, notice=None):
    try:
        return first_term_end(e0, period, reference, notice)
    except DateOutOfRange:
        return None


@pytest.mark.parametrize(
    ("e0", "period", "reference", "notice", "expected"),
    [
        (date(2031, 12, 31), duration(1, "years"), D, None, (0, date(2031, 12, 31))),
        (
            date(2031, 8, 31),
            duration(6, "months"),
            date(2032, 3, 1),
            None,
            (2, date(2032, 8, 31)),
        ),
        (
            date(2031, 12, 31),
            duration(1, "years"),
            D,
            duration(6, "months"),
            (1, date(2032, 12, 31)),
        ),
        (
            date(2034, 12, 31),
            duration(1, "years"),
            date(5000, 1, 1),
            None,
            (2966, date(5000, 12, 31)),
        ),
        (
            date(2034, 12, 31),
            duration(1, "years"),
            date(9999, 12, 31),
            None,
            (7965, date(9999, 12, 31)),
        ),
        (date(2034, 12, 30), duration(1, "years"), date(9999, 12, 31), None, None),
        (
            date(2034, 12, 31),
            duration(1, "years"),
            date(9999, 12, 31),
            duration(3, "months"),
            None,
        ),
        (
            date(2034, 12, 31),
            duration(1, "years"),
            date(5000, 1, 1),
            duration(30000, "months"),
            None,
        ),
        (date(2031, 1, 1), duration(10**18, "days"), D, None, None),
        (
            date(2031, 1, 1),
            duration(10, "days"),
            date(2031, 3, 1),
            None,
            (6, date(2031, 3, 2)),
        ),
        (
            date(2031, 1, 1),
            duration(2, "weeks"),
            date(2031, 3, 1),
            duration(3, "days"),
            (5, date(2031, 3, 12)),
        ),
    ],
    ids=[
        "k-0",
        "skip-ahead",
        "notice-k-1",
        "reference-5000",
        "reference-9999-12-31",
        "past-the-range-before-the-reference",
        "notice-past-the-range",
        "notice-below-the-range-at-k-0",
        "first-step-past-the-range",
        "days",
        "weeks-with-notice",
    ],
)
def test_the_search_counts_k_up_from_0(e0, period, reference, notice, expected):
    assert searched(e0, period, reference, notice) == expected
    assert literal_search(e0, period, reference, notice) == expected


def test_e0_minus_n_below_the_range_at_k_0_ends_the_search_before_a_later_k():
    e0 = date(2034, 12, 31)
    period = duration(1, "years")
    notice = duration(30000, "months")
    reference = date(5000, 1, 1)

    # k = 5466 would give 31-12-5000, the first E_k - N on or after D (I1 (c)).
    assert shift(shift(e0, period, 5466), notice, -1) == date(5000, 12, 31)
    assert shift(shift(e0, period, 5465), notice, -1) < reference
    with pytest.raises(DateOutOfRange):
        shift(e0, notice, -1)
    with pytest.raises(DateOutOfRange):
        first_term_end(e0, period, reference, notice)


_GRID_E0 = [date(1, 1, 31), date(2031, 8, 31), date(2034, 12, 31), date(9999, 12, 31)]
_GRID_PERIODS = [
    duration(10, "days"),
    duration(2, "weeks"),
    duration(1, "months"),
    duration(6, "months"),
    duration(1, "years"),
    duration(5, "years"),
    duration(4000, "years"),
]
_GRID_NOTICES = [
    None,
    duration(1, "days"),
    duration(3, "months"),
    duration(2, "years"),
    duration(30000, "months"),
]
_GRID_REFERENCES = [date(1, 6, 1), D, date(5000, 1, 1), date(9999, 12, 31)]
_DAYS_PER_UNIT = {"days": 1, "weeks": 7, "months": 28, "years": 365}


def test_the_search_equals_the_literal_search_on_a_grid():
    compared = 0
    for e0 in _GRID_E0:
        for period in _GRID_PERIODS:
            step = period["amount"] * _DAYS_PER_UNIT[period["unit"]]
            for reference in _GRID_REFERENCES:
                if (reference - e0).days / step > 8000:
                    continue  # the literal search would take too long
                for notice in _GRID_NOTICES:
                    assert searched(e0, period, reference, notice) == literal_search(
                        e0, period, reference, notice
                    ), (e0, period, reference, notice)
                    compared += 1

    assert compared > 400


def test_a_fallback_that_leaves_the_range_gives_no_e0_and_no_missing_input():
    term_end, derived = run(
        D,
        commencement_date=("date", "2030-01-01"),
        initial_term=("duration", duration(10000, "years")),
        renewal=("fixed_period", duration(1, "years")),
        notice_period=("duration", duration(3, "months")),
    )

    assert term_end.to_dict() == {
        "value": None,
        "source": None,
        "term_end_mismatch": False,
    }
    assert (term_end.missing, term_end.computed) == ([], None)
    assert (term_end.fallback_out_of_range, term_end.mismatch_out_of_range) == (
        True,
        False,
    )
    assert derived["expiry_date"] == {
        "status": "not_computable",
        "value": None,
        "reason": None,
        "missing_inputs": oor("commencement_date", "initial_term", "renewal"),
        "inputs": [],
        "calculation": None,
        "for_term_end": None,
        "expired": None,
        "since": None,
    }
    assert derived["next_renewal_date"]["missing_inputs"] == oor(
        "commencement_date", "initial_term", "renewal"
    )
    assert derived["latest_notice_date"]["missing_inputs"] == oor(
        "commencement_date", "initial_term", "renewal", "notice_period"
    )
    assert range_failures(term_end, derived) == [
        "expiry_date",
        "next_renewal_date",
        "latest_notice_date",
    ]


def test_c_plus_t_counts_as_leaving_the_range_before_its_minus_one_day():
    term_end, _ = run(
        D,
        commencement_date=("date", "9999-01-01"),
        initial_term=("duration", duration(1, "years")),
    )

    assert term_end.fallback_out_of_range is True
    assert term_end.value is None


def test_a_missing_input_comes_before_a_fallback_that_leaves_the_range():
    term_end, derived = run(
        D,
        commencement_date=("date", "2030-01-01"),
        initial_term=("duration", duration(10000, "years")),
        renewal=("fixed_period", duration(1, "years")),
    )

    assert derived["expiry_date"]["missing_inputs"] == oor(
        "commencement_date", "initial_term", "renewal"
    )
    assert derived["latest_notice_date"]["missing_inputs"] == [
        {"field": "notice_period", "reason": "not_found"}
    ]
    assert range_failures(term_end, derived) == ["expiry_date", "next_renewal_date"]


@pytest.mark.parametrize(
    ("given", "amendment"),
    [({"renewal": "not_found"}, False), ({}, True)],
    ids=["renewal-missing", "amendment"],
)
def test_a_fallback_no_entry_shows_is_named_effective_initial_term_end(
    given, amendment
):
    fields = {
        "commencement_date": ("date", "2030-01-01"),
        "initial_term": ("duration", duration(10000, "years")),
        "renewal": ("fixed_period", duration(1, "years")),
        "notice_period": ("duration", duration(3, "months")),
        **given,
    }
    term_end, derived = run(D, amendment=amendment, **fields)

    assert range_failures(term_end, derived) == ["effective_initial_term_end"]


def test_no_renewal_with_a_fallback_that_leaves_the_range():
    term_end, derived = run(
        D,
        commencement_date=("date", "2030-01-01"),
        initial_term=("duration", duration(10000, "years")),
        renewal=("none", None),
    )

    assert derived["expiry_date"]["missing_inputs"] == oor(
        "commencement_date", "initial_term", "renewal"
    )
    for name in ("next_renewal_date", "latest_notice_date"):
        assert derived[name]["reason"] == "no_renewal"
    assert range_failures(term_end, derived) == ["expiry_date"]


def test_c_plus_t_beside_a_stated_end_that_leaves_the_range_is_no_mismatch():
    term_end, derived = run(
        D,
        commencement_date=("date", "2030-01-01"),
        initial_term=("duration", duration(10000, "years")),
        initial_term_end=("date", "2034-12-31"),
        renewal=("fixed_period", duration(1, "years")),
        notice_period=("duration", duration(3, "months")),
    )
    _, without = run(
        D,
        initial_term_end=("date", "2034-12-31"),
        renewal=("fixed_period", duration(1, "years")),
        notice_period=("duration", duration(3, "months")),
    )

    assert term_end.to_dict() == {
        "value": "2034-12-31",
        "source": "stated",
        "term_end_mismatch": False,
    }
    assert term_end.mismatch_out_of_range is True
    assert derived == without
    assert range_failures(term_end, derived) == ["term_end_mismatch"]


@pytest.mark.parametrize(
    "end",
    ["conflicting", ("conditional", None)],
    ids=["conflicting", "conditional"],
)
def test_c_plus_t_is_not_computed_where_it_is_not_used(end):
    term_end, derived = run(
        D,
        commencement_date=("date", "2030-01-01"),
        initial_term=("duration", duration(10000, "years")),
        initial_term_end=end,
        renewal=("fixed_period", duration(1, "years")),
    )

    assert (term_end.fallback_out_of_range, term_end.mismatch_out_of_range) == (
        False,
        False,
    )
    assert derived["expiry_date"]["missing_inputs"][0]["field"] == "initial_term_end"
    assert range_failures(term_end, derived) == []


def test_indefinite_with_e0_minus_n_below_the_range():
    term_end, derived = run(
        D,
        initial_term_end=("date", "2034-12-31"),
        renewal=("indefinite", None),
        notice_period=("duration", duration(30000, "months")),
    )

    assert derived["expiry_date"]["value"] == "2034-12-31"
    assert derived["next_renewal_date"]["value"] == "2035-01-01"
    assert derived["latest_notice_date"]["missing_inputs"] == oor(
        "initial_term_end", "renewal", "notice_period"
    )
    assert range_failures(term_end, derived) == ["latest_notice_date"]


def test_a_since_outside_the_range_is_not_computable():
    term_end, derived = run(
        D,
        initial_term_end=("date", "9999-12-31"),
        renewal=("indefinite", None),
        notice_period=("duration", duration(9000, "years")),
    )

    assert derived["expiry_date"]["value"] == "9999-12-31"
    assert derived["next_renewal_date"]["missing_inputs"] == oor(
        "initial_term_end", "renewal"
    )
    assert derived["latest_notice_date"]["missing_inputs"] == oor(
        "initial_term_end", "renewal", "notice_period"
    )
    assert range_failures(term_end, derived) == [
        "next_renewal_date",
        "latest_notice_date",
    ]


def test_every_out_of_range_entry_has_the_nine_keys():
    _, derived = run(
        date(9999, 12, 31),
        initial_term_end=("date", "2034-12-31"),
        renewal=("fixed_period", duration(1, "years")),
        notice_period=("duration", duration(3, "months")),
    )

    assert derived["expiry_date"]["value"] == "9999-12-31"
    for name in ("next_renewal_date", "latest_notice_date"):
        entry = derived[name]
        assert list(entry) == ENTRY_KEYS
        assert (entry["status"], entry["value"], entry["inputs"]) == (
            "not_computable",
            None,
            [],
        )
