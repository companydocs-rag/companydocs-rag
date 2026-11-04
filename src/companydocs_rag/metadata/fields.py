"""The six fields, their candidates and their status (spec 008, D7, D8, R12, R13, R21).

Both subcommands use this module: ``extract`` to judge the model's candidates
and decide each field's status, ``report`` to check what it reads.

A candidate's ``value_kind`` must be allowed for its field
(:data:`FIELD_KINDS`). Its value fits (R12) when it is, for ``date``, a
string ``YYYY-MM-DD`` that is a real calendar date; for ``duration`` and
``fixed_period``, an object with exactly ``amount`` and ``unit``, the amount a
JSON number that is a whole number of at least 1 (a bool is not one) and the
unit one of :data:`UNITS`; for the null kinds (:data:`NULL_KINDS`), any value.
The stored value is null for an allowed null kind, the string for a valid
date, ``{"amount": <integer>, "unit": ...}`` for a valid duration, and the
value as returned otherwise.

A field's status (D8, R13) comes from its counted candidates only: none gives
``not_found``, one distinct (``value_kind``, normalized value) ``found``, more
``conflicting``. Normalization turns years into months (times 12) and weeks
into days (times 7); the null kinds have the value null. A field's pages
(R21) are the distinct ``[page_start, page_end]`` pairs of its counted
candidates' chunks, in candidate order, the pair read from the chunk's
``page_spans`` (its first and last page, as the chunker sets them).
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SIGNING_DATE = "signing_date"
COMMENCEMENT_DATE = "commencement_date"
INITIAL_TERM = "initial_term"
INITIAL_TERM_END = "initial_term_end"
RENEWAL = "renewal"
NOTICE_PERIOD = "notice_period"
# D10 order, which is also the order of the report's columns (D13).
FIELDS = (
    SIGNING_DATE,
    COMMENCEMENT_DATE,
    INITIAL_TERM,
    INITIAL_TERM_END,
    RENEWAL,
    NOTICE_PERIOD,
)
PROVISIONS = "other_termination_provisions"

KIND_DATE = "date"
KIND_CONDITIONAL = "conditional"
KIND_BLANK = "blank"
KIND_ILLEGIBLE = "illegible"
KIND_DURATION = "duration"
KIND_INDEFINITE = "indefinite"
KIND_NONE = "none"
KIND_FIXED_PERIOD = "fixed_period"
DATE_KINDS = (KIND_DATE, KIND_CONDITIONAL, KIND_BLANK, KIND_ILLEGIBLE)
FIELD_KINDS: Dict[str, Tuple[str, ...]] = {
    SIGNING_DATE: DATE_KINDS,
    COMMENCEMENT_DATE: DATE_KINDS,
    INITIAL_TERM: (KIND_DURATION, KIND_INDEFINITE),
    INITIAL_TERM_END: DATE_KINDS,
    RENEWAL: (KIND_NONE, KIND_FIXED_PERIOD, KIND_INDEFINITE),
    NOTICE_PERIOD: (KIND_DURATION,),
}
NULL_KINDS = (KIND_CONDITIONAL, KIND_BLANK, KIND_ILLEGIBLE, KIND_INDEFINITE, KIND_NONE)
DURATION_KINDS = (KIND_DURATION, KIND_FIXED_PERIOD)

UNIT_DAYS = "days"
UNIT_WEEKS = "weeks"
UNIT_MONTHS = "months"
UNIT_YEARS = "years"
UNITS = (UNIT_DAYS, UNIT_WEEKS, UNIT_MONTHS, UNIT_YEARS)
DURATION_KEYS = ("amount", "unit")

PARTIES = ("tenant", "landlord", "both", "unspecified")

# D9: a document's extraction_status, and a batch's status.
EXTRACTION_EXTRACTED = "extracted"
EXTRACTION_PARTIAL = "partial"
EXTRACTION_ERROR = "error"
EXTRACTION_NOT_IN_SCOPE = "not_in_scope"
EXTRACTION_STATUSES = (
    EXTRACTION_EXTRACTED,
    EXTRACTION_PARTIAL,
    EXTRACTION_ERROR,
    EXTRACTION_NOT_IN_SCOPE,
)
BATCH_OK = "ok"
BATCH_ERROR = "error"
BATCH_STATUSES = (BATCH_OK, BATCH_ERROR)

STATUS_NOT_FOUND = "not_found"
STATUS_FOUND = "found"
STATUS_CONFLICTING = "conflicting"
FIELD_STATUSES = (STATUS_NOT_FOUND, STATUS_FOUND, STATUS_CONFLICTING)

_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def parse_iso_date(value: Any) -> Optional[date]:
    """The date a ``YYYY-MM-DD`` string names, or None when ``value`` is not
    one or names no calendar date (``date.fromisoformat`` alone accepts other
    forms too)."""
    if not isinstance(value, str) or _ISO_DATE.fullmatch(value) is None:
        return None
    try:
        return date(int(value[0:4]), int(value[5:7]), int(value[8:10]))
    except ValueError:
        return None


def _whole_amount(value: Any) -> Optional[int]:
    """A JSON number that is a whole number of at least 1, as an int."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    amount = int(value)
    return amount if amount >= 1 else None


def parse_duration(value: Any) -> Optional[Dict[str, Any]]:
    """``{"amount": <integer>, "unit": ...}`` when ``value`` is a valid
    duration (R12), else None."""
    if not isinstance(value, Mapping) or set(value) != set(DURATION_KEYS):
        return None
    amount = _whole_amount(value["amount"])
    unit = value["unit"]
    if amount is None or not isinstance(unit, str) or unit not in UNITS:
        return None
    return {"amount": amount, "unit": unit}


def value_fits(field: str, kind: Any, value: Any) -> bool:
    """``value_valid`` (D8, R12): the kind is allowed for ``field`` and the
    value fits it."""
    if not isinstance(kind, str) or kind not in FIELD_KINDS[field]:
        return False
    if kind == KIND_DATE:
        return parse_iso_date(value) is not None
    if kind in DURATION_KINDS:
        return parse_duration(value) is not None
    return True


def stored_value(field: str, kind: Any, value: Any) -> Any:
    """The value a candidate stores (R12): null for an allowed null kind, a
    valid date as its string, a valid duration normalized to an integer
    amount, an invalid value as returned."""
    if not value_fits(field, kind, value):
        return value
    if kind in NULL_KINDS:
        return None
    if kind in DURATION_KINDS:
        return parse_duration(value)
    return value


def normalized_value(kind: str, stored: Any) -> Any:
    """The value a counted candidate is compared by (D8): years as months,
    weeks as days, a duration as an (amount, unit) pair."""
    if kind not in DURATION_KINDS:
        return stored
    amount, unit = stored["amount"], stored["unit"]
    if unit == UNIT_YEARS:
        return (amount * 12, UNIT_MONTHS)
    if unit == UNIT_WEEKS:
        return (amount * 7, UNIT_DAYS)
    return (amount, unit)


def counted_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    return [candidate for candidate in candidates if candidate["counted"] is True]


def field_status(candidates: Sequence[Mapping[str, Any]]) -> str:
    """D8: the status from the counted candidates only."""
    distinct = {
        (
            candidate["value_kind"],
            normalized_value(candidate["value_kind"], candidate["value"]),
        )
        for candidate in counted_candidates(candidates)
    }
    if not distinct:
        return STATUS_NOT_FOUND
    if len(distinct) == 1:
        return STATUS_FOUND
    return STATUS_CONFLICTING


def page_pair(page_spans: Sequence[Mapping[str, Any]]) -> Optional[List[int]]:
    """``[page_start, page_end]`` of a chunk, from its ``page_spans``; None
    when it has none."""
    if not page_spans:
        return None
    return [page_spans[0]["page_number"], page_spans[-1]["page_number"]]


def distinct_pairs(items: Sequence[Mapping[str, Any]]) -> List[List[int]]:
    """The distinct page pairs of ``items`` (candidates or provision items),
    in their order (R21)."""
    pairs: List[List[int]] = []
    for item in items:
        pair = page_pair(item["page_spans"])
        if pair is not None and pair not in pairs:
            pairs.append(pair)
    return pairs
