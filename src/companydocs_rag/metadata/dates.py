"""Derived dates for a reference date (spec 008, D12, R20, R22; Q4, I1).

Month arithmetic keeps the day number, clamped to the month's last day;
years are 12 months, weeks 7 days. E_k = E0 + k R is always computed from E0
(:func:`shift`), never step by step, k counting up from 0.

Dates lie in years 1-9999 (I1). Arithmetic whose result lies outside that
range raises :class:`DateOutOfRange`, also for a step whose next step would
come back into it: C + T before its - 1 day, E_k before its - N or + 1 day.

E0, the effective end of the first term (:func:`effective_term_end`), is
``initial_term_end`` when it is found as a date (``stated``). When it is
``not_found``, ``blank`` or ``illegible``, it falls back to C + T - 1 day,
when ``commencement_date`` is a date and ``initial_term`` a duration
(``computed``). When it is ``conflicting`` or ``conditional`` there is no
fallback and no E0. When the stated and the computed end both exist and
differ, the stated one is used and ``term_end_mismatch`` is true. C + T - 1
day is computed only where it is used: as the fallback, and beside a stated
end for ``term_end_mismatch``, which is false when it leaves the range.

Each derived date (:func:`derive`) is decided in this order: a document of an
``amendment_types`` type (``not_applicable``, ``amendment``);
``initial_term`` found as ``indefinite`` (``indefinite_term``); for the next
renewal and the latest notice date, ``renewal`` found as ``none``
(``no_renewal``); missing inputs (``not_computable``); then the branch of
``renewal``. The expiry and the next renewal date need E0 and ``renewal``,
the latest notice date ``notice_period`` too. An entry whose inputs are all
usable and whose computation leaves the range (C + T - 1 day as its E0,
``since``, the E_k and E_k - N of its search) is ``not_computable``, with
every field its ``inputs`` would hold in ``missing_inputs``, reason
``out_of_range`` (I1). :func:`range_failures` names what the report logs for
each range failure.

The calculation text (R22) joins dates (``dd-mm-jjjj``) and durations with
" + ", " - " (U+2212 MINUS SIGN) and " = "; with E0 computed, the step is
preceded by its derivation, or replaced by it when the step is E0 alone; then
the inputs with their first page pair.

This module reads no clock: its only date is the reference date it is given.
"""

from __future__ import annotations

import calendar
import copy
from dataclasses import dataclass
from datetime import MAXYEAR, MINYEAR, date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .fields import (
    COMMENCEMENT_DATE,
    FIELDS,
    INITIAL_TERM,
    INITIAL_TERM_END,
    KIND_BLANK,
    KIND_CONDITIONAL,
    KIND_DATE,
    KIND_DURATION,
    KIND_FIXED_PERIOD,
    KIND_ILLEGIBLE,
    KIND_INDEFINITE,
    KIND_NONE,
    NOTICE_PERIOD,
    RENEWAL,
    STATUS_FOUND,
    STATUS_NOT_FOUND,
    UNIT_DAYS,
    UNIT_MONTHS,
    UNIT_WEEKS,
    UNIT_YEARS,
    parse_iso_date,
)
from .texts import (
    DERIVED,
    DERIVED_COMPUTED,
    DERIVED_NOT_APPLICABLE,
    DERIVED_NOT_COMPUTABLE,
    EXPIRY_DATE,
    LATEST_NOTICE_DATE,
    MINUS_SIGN,
    NEXT_RENEWAL_DATE,
    ONE_DAY,
    REASON_AMENDMENT,
    REASON_INDEFINITE_AFTER_INITIAL_TERM,
    REASON_INDEFINITE_TERM,
    REASON_NO_RENEWAL,
    REASON_OUT_OF_RANGE,
    date_text,
    duration_text,
    label_lower,
    pages_text,
)

SOURCE_STATED = "stated"
SOURCE_COMPUTED = "computed"

# What the report's WARNING names for a range failure that makes no derived
# entry not_computable (I1, R20).
TERM_END_MISMATCH = "term_end_mismatch"
EFFECTIVE_INITIAL_TERM_END = "effective_initial_term_end"

# The kinds of a found value that make it unusable as an input (R20).
_UNUSABLE_KINDS = (KIND_CONDITIONAL, KIND_BLANK, KIND_ILLEGIBLE)
_FALLBACK_REASONS = (STATUS_NOT_FOUND, KIND_BLANK, KIND_ILLEGIBLE)
_LAST_ORDINAL = date.max.toordinal()


# --- Date arithmetic ------------------------------------------------------------------


class DateOutOfRange(ArithmeticError):
    """Date arithmetic whose result lies outside years 1-9999 (I1)."""


def add_days(day: date, days: int) -> date:
    """``day`` plus ``days`` (negative to subtract)."""
    ordinal = day.toordinal() + days
    if not 1 <= ordinal <= _LAST_ORDINAL:
        raise DateOutOfRange(f"{day.isoformat()} plus {days} day(s)")
    return date.fromordinal(ordinal)


def add_months(day: date, months: int) -> date:
    """``day`` plus ``months`` (negative to subtract), the day number clamped
    to the month's last day."""
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    month += 1
    if not MINYEAR <= year <= MAXYEAR:
        raise DateOutOfRange(f"{day.isoformat()} plus {months} month(s)")
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def shift(day: date, duration: Mapping[str, Any], factor: int = 1) -> date:
    """``day`` plus ``factor`` times ``duration``; years are 12 months, weeks
    7 days."""
    amount = duration["amount"] * factor
    unit = duration["unit"]
    if unit == UNIT_YEARS:
        return add_months(day, amount * 12)
    if unit == UNIT_MONTHS:
        return add_months(day, amount)
    if unit == UNIT_WEEKS:
        return add_days(day, amount * 7)
    if unit == UNIT_DAYS:
        return add_days(day, amount)
    raise ValueError(f"unknown unit {unit!r}")


def _safe_start(e0: date, period: Mapping[str, Any], reference: date) -> int:
    """A k at which E_k < ``reference`` is certain, or 0: the search for the
    smallest qualifying k may start there."""
    amount, unit = period["amount"], period["unit"]
    if unit in (UNIT_YEARS, UNIT_MONTHS):
        step = amount * (12 if unit == UNIT_YEARS else 1)
        gap = (reference.year - e0.year) * 12 + reference.month - e0.month
    else:
        step = amount * (7 if unit == UNIT_WEEKS else 1)
        gap = (reference - e0).days
    return max(0, gap // step - 1)


def first_term_end(
    e0: date,
    period: Mapping[str, Any],
    reference: date,
    notice: Optional[Mapping[str, Any]] = None,
) -> Tuple[int, date]:
    """The smallest k >= 0 whose E_k = E0 + k ``period`` lies on or after
    ``reference``, or, with a ``notice`` period N, whose E_k - N does; and
    that E_k.

    The search counts k up from 0 (I1 (c)): when it meets an E_k or an
    E_k - N outside the range before it reaches that k, it raises
    :class:`DateOutOfRange`, even where a later k would have given a date.
    After k = 0 it goes on from :func:`_safe_start`'s k, with the outcome of
    trying every k: E_k grows with k and E_k - N never falls as k grows, so
    each k it skips has E0 <= E_k < ``reference`` and
    E_0 - N <= E_k - N <= E_k, all inside the range once k = 0 passed, and
    none accepted.
    """

    def accepted(k: int) -> Tuple[bool, date]:
        end = shift(e0, period, k)
        target = end if notice is None else shift(end, notice, -1)
        return target >= reference, end

    found, end = accepted(0)
    if found:
        return 0, end
    k = max(1, _safe_start(e0, period, reference))
    while True:
        found, end = accepted(k)
        if found:
            return k, end
        k += 1


# --- The fields as the report reads them ----------------------------------------------


@dataclass(frozen=True)
class FieldView:
    """One field of an extraction file, as the derivation reads it: its
    status, the chosen kind and value when found (R13), and its pages."""

    name: str
    status: str
    kind: Optional[str]
    value: Any
    pages: List[List[int]]

    @property
    def found(self) -> bool:
        return self.status == STATUS_FOUND

    def is_kind(self, kind: str) -> bool:
        return self.found and self.kind == kind

    def missing_reason(self) -> Optional[str]:
        """Why this field cannot be an input: its status, or its found kind
        (R20); None when it can."""
        if not self.found:
            return self.status
        if self.kind in _UNUSABLE_KINDS:
            return self.kind
        return None


def _missing(view: FieldView, reason: Optional[str] = None) -> Dict[str, Any]:
    return {"field": view.name, "reason": reason or view.missing_reason()}


def _in_column_order(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda item: FIELDS.index(item["field"]))


@dataclass(frozen=True)
class TermEnd:
    """E0 (R20). ``computed`` is C + T - 1 day where it was computed (as the
    fallback, or beside a stated end); ``missing`` the missing inputs when
    there is no E0.

    ``fallback_out_of_range``: the fallback was tried with C and T usable
    and left the range, so there is no E0 and no input is missing (I1).
    ``mismatch_out_of_range``: C + T - 1 day left the range beside a stated
    end, so ``mismatch`` is false (I1).
    """

    value: Optional[date]
    source: Optional[str]
    mismatch: bool
    computed: Optional[date]
    missing: List[Dict[str, Any]]
    fallback_out_of_range: bool = False
    mismatch_out_of_range: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """``effective_initial_term_end`` in ``followup.json``."""
        return {
            "value": None if self.value is None else self.value.isoformat(),
            "source": self.source,
            "term_end_mismatch": self.mismatch,
        }


def _start_plus_term(start: FieldView, term: FieldView) -> date:
    """C + T - 1 day; raises :class:`DateOutOfRange` when C + T leaves the
    range."""
    return add_days(shift(parse_iso_date(start.value), term.value), -1)


def effective_term_end(views: Mapping[str, FieldView]) -> TermEnd:
    """E0 per D12, Q4 and I1."""
    end = views[INITIAL_TERM_END]
    start = views[COMMENCEMENT_DATE]
    term = views[INITIAL_TERM]
    usable = start.is_kind(KIND_DATE) and term.is_kind(KIND_DURATION)
    if end.is_kind(KIND_DATE):
        stated = parse_iso_date(end.value)
        if not usable:
            return TermEnd(stated, SOURCE_STATED, False, None, [])
        try:
            computed = _start_plus_term(start, term)
        except DateOutOfRange:
            return TermEnd(
                stated, SOURCE_STATED, False, None, [], mismatch_out_of_range=True
            )
        return TermEnd(stated, SOURCE_STATED, computed != stated, computed, [])
    reason = end.missing_reason()
    if reason not in _FALLBACK_REASONS:
        return TermEnd(None, None, False, None, [_missing(end, reason)])
    if usable:
        try:
            computed = _start_plus_term(start, term)
        except DateOutOfRange:
            return TermEnd(None, None, False, None, [], fallback_out_of_range=True)
        return TermEnd(computed, SOURCE_COMPUTED, False, computed, [])
    missing = [_missing(end, reason)]
    if not start.is_kind(KIND_DATE):
        missing.append(_missing(start, start.missing_reason() or start.kind))
    if not term.is_kind(KIND_DURATION):
        missing.append(_missing(term, term.missing_reason() or term.kind))
    return TermEnd(None, None, False, None, _in_column_order(missing))


# --- Derived entries ------------------------------------------------------------------


def _entry(
    status: str,
    *,
    value: Optional[date] = None,
    reason: Optional[str] = None,
    missing_inputs: Sequence[Dict[str, Any]] = (),
    inputs: Sequence[Dict[str, Any]] = (),
    calculation: Optional[str] = None,
    for_term_end: Optional[date] = None,
    expired: Optional[bool] = None,
    since: Optional[date] = None,
) -> Dict[str, Any]:
    """A derived entry with its nine keys, in order (R20)."""
    return {
        "status": status,
        "value": None if value is None else value.isoformat(),
        "reason": reason,
        "missing_inputs": [dict(item) for item in missing_inputs],
        "inputs": [copy.deepcopy(item) for item in inputs],
        "calculation": calculation,
        "for_term_end": None if for_term_end is None else for_term_end.isoformat(),
        "expired": expired,
        "since": None if since is None else since.isoformat(),
    }


def _not_applicable(reason: str, *, since: Optional[date] = None) -> Dict[str, Any]:
    return _entry(DERIVED_NOT_APPLICABLE, reason=reason, since=since)


def _input_fields(name: str, term_end: TermEnd) -> List[str]:
    """The fields a derived entry is computed from, in column order (R20):
    ``initial_term_end``, or C and T when E0 is, or would be, C + T - 1 day;
    ``renewal``; ``notice_period`` for the latest notice date."""
    fields = (
        [INITIAL_TERM_END]
        if term_end.source == SOURCE_STATED
        else [COMMENCEMENT_DATE, INITIAL_TERM]
    )
    fields.append(RENEWAL)
    if name == LATEST_NOTICE_DATE:
        fields.append(NOTICE_PERIOD)
    return fields


def _out_of_range(name: str, term_end: TermEnd) -> Dict[str, Any]:
    """I1: every input of the entry, reason ``out_of_range``."""
    return _entry(
        DERIVED_NOT_COMPUTABLE,
        missing_inputs=[
            {"field": field, "reason": REASON_OUT_OF_RANGE}
            for field in _input_fields(name, term_end)
        ],
    )


def is_out_of_range(entry: Mapping[str, Any]) -> bool:
    """Whether a derived entry is ``not_computable`` because its computation
    left the range (I1)."""
    return any(
        item["reason"] == REASON_OUT_OF_RANGE for item in entry["missing_inputs"]
    )


def _input(view: FieldView) -> Dict[str, Any]:
    return {
        "field": view.name,
        "value": copy.deepcopy(view.value),
        "pages": [list(pair) for pair in view.pages],
    }


def _inputs_text(inputs: Sequence[Dict[str, Any]]) -> str:
    return ", ".join(
        label_lower(item["field"])
        + (f" p. {pages_text(item['pages'][0])}" if item["pages"] else "")
        for item in inputs
    )


def calculation_text(
    step: str,
    term_end: TermEnd,
    views: Mapping[str, FieldView],
    inputs: Sequence[Dict[str, Any]],
    *,
    step_is_e0: bool,
    shown_term_end: Optional[date] = None,
) -> str:
    """R22: the step, preceded or replaced by E0's derivation when E0 is
    computed, then the inputs (and the term end, for the notice date)."""
    text = step
    if term_end.source == SOURCE_COMPUTED:
        derivation = (
            f"{date_text(views[COMMENCEMENT_DATE].value)} + "
            f"{duration_text(views[INITIAL_TERM].value)} {MINUS_SIGN} {ONE_DAY} = "
            f"{date_text(term_end.value)}"
        )
        text = derivation if step_is_e0 else f"{derivation}; {step}"
    if shown_term_end is not None:
        return (
            f"{text} (termijneinde {date_text(shown_term_end)}; "
            f"invoer: {_inputs_text(inputs)})"
        )
    return f"{text} (invoer: {_inputs_text(inputs)})"


def derive(
    views: Mapping[str, FieldView],
    term_end: TermEnd,
    reference: date,
    *,
    amendment: bool,
) -> Dict[str, Dict[str, Any]]:
    """The three derived entries of one document (D12, R20, R22, I1)."""
    renewal = views[RENEWAL]
    notice = views[NOTICE_PERIOD]
    entries: Dict[str, Dict[str, Any]] = {}
    for name in DERIVED:
        if amendment:
            entries[name] = _not_applicable(REASON_AMENDMENT)
        elif views[INITIAL_TERM].is_kind(KIND_INDEFINITE):
            entries[name] = _not_applicable(REASON_INDEFINITE_TERM)
        elif name != EXPIRY_DATE and renewal.is_kind(KIND_NONE):
            entries[name] = _not_applicable(REASON_NO_RENEWAL)
        else:
            missing = list(term_end.missing) if term_end.value is None else []
            if renewal.missing_reason() is not None:
                missing.append(_missing(renewal))
            if name == LATEST_NOTICE_DATE and notice.missing_reason() is not None:
                missing.append(_missing(notice))
            if missing:
                entries[name] = _entry(
                    DERIVED_NOT_COMPUTABLE, missing_inputs=_in_column_order(missing)
                )
            elif term_end.fallback_out_of_range:
                entries[name] = _out_of_range(name, term_end)
            else:
                try:
                    entries[name] = _branch(name, views, term_end, reference)
                except DateOutOfRange:
                    entries[name] = _out_of_range(name, term_end)
    return entries


def _branch(
    name: str,
    views: Mapping[str, FieldView],
    term_end: TermEnd,
    reference: date,
) -> Dict[str, Any]:
    """D12's branch of ``renewal`` for one derived date, all inputs usable.
    Raises :class:`DateOutOfRange` when the computation leaves the range."""
    renewal = views[RENEWAL]
    notice = views[NOTICE_PERIOD]
    e0 = term_end.value
    inputs = [_input(views[field]) for field in _input_fields(name, term_end)]

    def computed(
        value: date,
        step: str,
        *,
        step_is_e0: bool = False,
        shown_term_end: Optional[date] = None,
        expired: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return _entry(
            DERIVED_COMPUTED,
            value=value,
            inputs=inputs,
            calculation=calculation_text(
                step,
                term_end,
                views,
                inputs,
                step_is_e0=step_is_e0,
                shown_term_end=shown_term_end,
            ),
            for_term_end=shown_term_end,
            expired=expired,
        )

    def after_initial_term() -> Dict[str, Any]:
        # ``since`` is computed only on an entry that shows it (I1).
        return _not_applicable(
            REASON_INDEFINITE_AFTER_INITIAL_TERM, since=add_days(e0, 1)
        )

    if name == EXPIRY_DATE:
        if renewal.is_kind(KIND_NONE):
            return computed(e0, date_text(e0), step_is_e0=True, expired=e0 < reference)
        if renewal.is_kind(KIND_FIXED_PERIOD):
            period = renewal.value
            k, end = first_term_end(e0, period, reference)
            if k == 0:
                return computed(end, date_text(e0), step_is_e0=True)
            step = f"{date_text(e0)} + {duration_text(period, k)} = {date_text(end)}"
            return computed(end, step)
        if e0 >= reference:
            return computed(e0, date_text(e0), step_is_e0=True)
        return after_initial_term()

    if name == NEXT_RENEWAL_DATE:
        if renewal.is_kind(KIND_FIXED_PERIOD):
            _, end = first_term_end(e0, renewal.value, reference)
        elif e0 >= reference:
            end = e0
        else:
            return after_initial_term()
        value = add_days(end, 1)
        step = f"{date_text(end)} + {ONE_DAY} = {date_text(value)}"
        return computed(value, step)

    if renewal.is_kind(KIND_FIXED_PERIOD):
        _, end = first_term_end(e0, renewal.value, reference, notice.value)
    elif shift(e0, notice.value, -1) >= reference:
        end = e0
    else:
        return after_initial_term()
    value = shift(end, notice.value, -1)
    step = (
        f"{date_text(end)} {MINUS_SIGN} {duration_text(notice.value)} = "
        f"{date_text(value)}"
    )
    return computed(value, step, shown_term_end=end)


def range_failures(
    term_end: TermEnd, derived: Mapping[str, Mapping[str, Any]]
) -> List[str]:
    """What the report's WARNING names for each range failure of one document
    (I1, R20), in the order of the computation: ``term_end_mismatch`` when
    C + T - 1 day left the range beside a stated end; each derived entry
    the range made ``not_computable``; ``effective_initial_term_end`` when
    the fallback left the range and no derived entry shows it."""
    names = [TERM_END_MISMATCH] if term_end.mismatch_out_of_range else []
    entries = [name for name in DERIVED if is_out_of_range(derived[name])]
    names += entries
    if term_end.fallback_out_of_range and not entries:
        names.append(EFFECTIVE_INITIAL_TERM_END)
    return names
