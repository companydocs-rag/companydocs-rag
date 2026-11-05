"""The report's Dutch texts (spec 008, D13, R21-R25).

Dates are written ``dd-mm-jjjj``; durations ``N jaar``, ``N maanden`` (``1
maand``), ``N weken`` (``1 week``) and ``N dagen`` (``1 dag``). A found value
is followed by `` (p. N)`` or `` (p. N-M)``: its first page pair; a
conflicting field shows every pair. A derived date shows its value (with ``
(verlopen)`` when it expired), why it cannot be computed, or why it does not
apply. The texts are fixed by D13; the two characters outside ASCII are
written as ``chr`` of their code point: U+00EB LATIN SMALL LETTER E WITH
DIAERESIS and U+2212 MINUS SIGN.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Mapping, Optional, Sequence

from .fields import (
    COMMENCEMENT_DATE,
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
    SIGNING_DATE,
    STATUS_CONFLICTING,
    STATUS_FOUND,
    STATUS_NOT_FOUND,
    UNIT_DAYS,
    UNIT_MONTHS,
    UNIT_WEEKS,
    UNIT_YEARS,
    parse_iso_date,
)

MINUS_SIGN = chr(0x2212)  # U+2212 MINUS SIGN
E_DIAERESIS = chr(0x00EB)  # U+00EB LATIN SMALL LETTER E WITH DIAERESIS

EXPIRY_DATE = "expiry_date"
NEXT_RENEWAL_DATE = "next_renewal_date"
LATEST_NOTICE_DATE = "latest_notice_date"
DERIVED = (EXPIRY_DATE, NEXT_RENEWAL_DATE, LATEST_NOTICE_DATE)

DERIVED_COMPUTED = "computed"
DERIVED_NOT_COMPUTABLE = "not_computable"
DERIVED_NOT_APPLICABLE = "not_applicable"
REASON_AMENDMENT = "amendment"
REASON_INDEFINITE_TERM = "indefinite_term"
REASON_NO_RENEWAL = "no_renewal"
REASON_INDEFINITE_AFTER_INITIAL_TERM = "indefinite_after_initial_term"
# A missing input's reason when the entry's date arithmetic left years
# 1-9999 (I1, R20).
REASON_OUT_OF_RANGE = "out_of_range"

# D13: the table's columns, in order.
COLUMN_DOCUMENT = "Document"
COLUMN_TYPE = "Type"
FIELD_LABELS: Dict[str, str] = {
    SIGNING_DATE: "Ondertekend",
    COMMENCEMENT_DATE: "Ingangsdatum",
    INITIAL_TERM: "Looptijd",
    INITIAL_TERM_END: "Einde eerste termijn",
    RENEWAL: "Verlenging",
    NOTICE_PERIOD: "Opzegtermijn",
}
DERIVED_LABELS: Dict[str, str] = {
    EXPIRY_DATE: "Vervaldatum",
    NEXT_RENEWAL_DATE: "Eerstvolgende verlenging",
    LATEST_NOTICE_DATE: "Uiterste opzegdatum",
}
COLUMNS = (
    COLUMN_DOCUMENT,
    COLUMN_TYPE,
    *FIELD_LABELS.values(),
    *DERIVED_LABELS.values(),
)

NOT_FOUND_TEXT = "niet gevonden"
CONFLICTING_TEXT = "tegenstrijdig"
INDEFINITE_TEXT = "onbepaalde tijd"
NO_RENEWAL_TEXT = "geen verlenging"
RENEWAL_PREFIX = "telkens "
KIND_TEXTS: Dict[str, str] = {
    KIND_CONDITIONAL: "voorwaardelijk",
    KIND_BLANK: "leeg veld in het document",
    KIND_ILLEGIBLE: "onleesbaar",
}
# D13: per missing input, its reason.
MISSING_TEXTS: Dict[str, str] = {
    STATUS_NOT_FOUND: "ontbreekt",
    STATUS_CONFLICTING: "tegenstrijdig",
    KIND_CONDITIONAL: "voorwaardelijk",
    KIND_BLANK: "leeg veld",
    KIND_ILLEGIBLE: "onleesbaar",
}
NOT_COMPUTABLE_PREFIX = "niet berekenbaar: "
OUT_OF_RANGE_TEXT = NOT_COMPUTABLE_PREFIX + "buiten datumbereik"
EXPIRED_SUFFIX = " (verlopen)"
NOT_APPLICABLE_TEXTS: Dict[str, str] = {
    REASON_AMENDMENT: "n.v.t.: wijzigingsdocument, zie relatie",
    REASON_INDEFINITE_TERM: "n.v.t.: onbepaalde tijd",
    REASON_NO_RENEWAL: "n.v.t.: geen verlenging",
}
PARTY_NAMES: Dict[str, str] = {
    "tenant": "huurder",
    "landlord": "verhuurder",
    "both": "beide",
    "unspecified": "niet gespecificeerd",
}
PROVISIONS_HEADING = (
    "Overige be" + E_DIAERESIS + "indigingsbepalingen (niet in de berekening)"
)
SCAN_NOTE = "Gelezen van een scan; controleer de waarden op de genoemde pagina's."
ONE_DAY = "1 dag"
_UNIT_WORDS = {
    UNIT_YEARS: ("jaar", "jaar"),
    UNIT_MONTHS: ("maand", "maanden"),
    UNIT_WEEKS: ("week", "weken"),
    UNIT_DAYS: ("dag", "dagen"),
}


def date_text(value: Any) -> str:
    """``dd-mm-jjjj`` for a date or a ``YYYY-MM-DD`` string."""
    day = value if isinstance(value, date) else parse_iso_date(value)
    return f"{day.day:02d}-{day.month:02d}-{day.year:04d}"


def duration_text(duration: Mapping[str, Any], factor: int = 1) -> str:
    """D13's form of ``duration`` times ``factor``: ``3 jaar``, ``1 maand``,
    ``6 maanden``, ``1 week``, ``2 weken``, ``1 dag``, ``10 dagen``."""
    amount = duration["amount"] * factor
    singular, plural = _UNIT_WORDS[duration["unit"]]
    return f"{amount} {singular if amount == 1 else plural}"


def pages_text(pair: Sequence[int]) -> str:
    """``N`` for one page, ``N-M`` for a range (R21)."""
    start, end = pair
    return f"{start}" if start == end else f"{start}-{end}"


def value_text(field: str, kind: str, value: Any) -> str:
    """A candidate's cell text without pages (D13, R23)."""
    if kind == KIND_DATE:
        return date_text(value)
    if kind in KIND_TEXTS:
        return KIND_TEXTS[kind]
    if kind == KIND_DURATION:
        return duration_text(value)
    if kind == KIND_INDEFINITE:
        return INDEFINITE_TEXT
    if kind == KIND_NONE:
        return NO_RENEWAL_TEXT
    if kind == KIND_FIXED_PERIOD:
        return RENEWAL_PREFIX + duration_text(value)
    raise ValueError(f"no text for {field!r} of kind {kind!r}")


def field_cell(
    field: str,
    status: str,
    kind: Optional[str],
    value: Any,
    pages: Sequence[Sequence[int]],
) -> str:
    """A field's cell (D13, R21)."""
    if status == STATUS_NOT_FOUND:
        return NOT_FOUND_TEXT
    if status == STATUS_CONFLICTING:
        if not pages:
            return CONFLICTING_TEXT
        return f"{CONFLICTING_TEXT} (p. {', '.join(pages_text(p) for p in pages)})"
    if status != STATUS_FOUND:
        raise ValueError(f"unknown field status {status!r}")
    text = value_text(field, kind, value)
    return text if not pages else f"{text} (p. {pages_text(pages[0])})"


def label_lower(field: str) -> str:
    """A field's column label in lower case."""
    return FIELD_LABELS[field].lower()


def derived_cell(entry: Mapping[str, Any]) -> str:
    """A derived date's cell (D13). An entry whose computation left the range
    reads "niet berekenbaar: buiten datumbereik" (I1)."""
    status = entry["status"]
    if status == DERIVED_COMPUTED:
        text = date_text(entry["value"])
        return text + EXPIRED_SUFFIX if entry["expired"] is True else text
    if status == DERIVED_NOT_COMPUTABLE:
        if any(
            item["reason"] == REASON_OUT_OF_RANGE for item in entry["missing_inputs"]
        ):
            return OUT_OF_RANGE_TEXT
        return NOT_COMPUTABLE_PREFIX + "; ".join(
            f"{label_lower(item['field'])} {MISSING_TEXTS[item['reason']]}"
            for item in entry["missing_inputs"]
        )
    reason = entry["reason"]
    if reason == REASON_INDEFINITE_AFTER_INITIAL_TERM:
        return f"n.v.t.: sinds {date_text(entry['since'])} voor onbepaalde tijd"
    return NOT_APPLICABLE_TEXTS[reason]


def mismatch_note(stated: Any, pages: Sequence[Sequence[int]], computed: Any) -> str:
    """Q7's note, when the stated end of the first term differs from
    commencement + term - 1 day."""
    shown = f" (p. {pages_text(pages[0])})" if pages else ""
    return (
        f"Let op: einde eerste termijn {date_text(stated)}{shown} wijkt af van "
        f"ingangsdatum + looptijd {MINUS_SIGN} 1 dag = {date_text(computed)}; "
        "de genoemde datum is gebruikt."
    )


def _with_detail(text: str, detail: str) -> str:
    return text if detail == "" else f"{text}: {detail}"


def amends_note(target_name: str, detail: str) -> str:
    """On the source of a resolved ``amends`` (R24)."""
    return (
        _with_detail(f"Wijzigt {target_name}", detail)
        + "; niet verwerkt in de berekening."
    )


def amended_by_note(source_name: str, detail: str) -> str:
    """On the target of a resolved ``amends`` (R24)."""
    return (
        _with_detail(f"Gewijzigd door {source_name}", detail)
        + "; niet verwerkt in de berekening."
    )


def duplicate_note(target_name: str) -> str:
    """On the source of a resolved ``duplicate_of`` (R24) [CUT-2]."""
    return f"Zelfde document als {target_name}."


def incomplete_note(failed: int, total: int) -> str:
    """For a ``partial`` or ``error`` document (D13)."""
    return f"Extractie onvolledig: {failed} van {total} batches mislukt."


def footer(reference: date) -> str:
    """The report's last line (D13)."""
    return (
        f"Referentiedatum: {date_text(reference)}. Rekenregel: kalenderrekenen, "
        "dag afgekapt op het maandeinde; geen juridische toets."
    )
