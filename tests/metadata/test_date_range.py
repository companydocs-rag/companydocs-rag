"""Date arithmetic outside years 1-9999, through ``report`` (spec 008, I1,
R20, D13; A12).

Each case runs ``report`` in a run of its own over a folder holding an
ordinary extraction file and the case's file, as in review F1's evidence,
and once over the ordinary file alone, with the case's reference date. The
ordinary document has no range failure at any of these dates. Dates and file
names are fabricated.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest

from tests.metadata._helpers import (
    AMENDMENT,
    LEASE,
    LOGGER,
    SHA_A,
    SHA_B,
    extraction_record,
    found,
    messages,
    months,
    not_found,
    report,
    write_json_file,
    years,
)

OUT_OF_RANGE = "niet berekenbaar: buiten datumbereik"
NO_RENEWAL = "n.v.t.: geen verlenging"
AMENDMENT_TEXT = "n.v.t.: wijzigingsdocument, zie relatie"
RENEWAL_MISSING = "niet berekenbaar: verlenging ontbreekt"
DERIVED_LABELS = {
    "expiry_date": "Vervaldatum",
    "next_renewal_date": "Eerstvolgende verlenging",
    "latest_notice_date": "Uiterste opzegdatum",
}
CASE_RELATIVE = "geval/b.json"
CASE_NAME = "fictief-geval.pdf"
ORDINARY_NAME = "fictief-gewoon.pdf"
ORDINARY = extraction_record(
    "bron/" + ORDINARY_NAME,
    SHA_A,
    fields={
        "commencement_date": found("2030-01-01", "date", pages=(1, 1)),
        "initial_term": found(years(5), "duration", pages=(1, 1)),
        "initial_term_end": found("2034-12-31", "date", pages=(2, 2)),
        "renewal": found(None, "indefinite", pages=(2, 2)),
        "notice_period": found(months(3), "duration", pages=(3, 3)),
    },
)


def oor(*fields):
    """``missing_inputs`` of an entry whose computation left the range."""
    return [{"field": name, "reason": "out_of_range"} for name in fields]


def effective(value=None, source=None):
    return {"value": value, "source": source, "term_end_mismatch": False}


@dataclass(frozen=True)
class Case:
    """One case of A12's I1 list: the case's fields, the reference date,
    per derived entry (status, value, missing_inputs or None), the three
    derived cells, ``effective_initial_term_end`` and the entries the
    WARNINGs name."""

    fields: Dict[str, Any]
    reference: str
    derived: Dict[str, Tuple[str, Optional[str], Optional[List[Dict[str, str]]]]]
    cells: List[str]
    effective: Dict[str, Any]
    warnings: List[str]
    document_type: str = LEASE


def _fallback_fields(**changes):
    """C 01-01-2030 and T 10000 jaar, ``initial_term_end`` not found, R 1
    jaar, N 3 maanden: C + T - 1 day leaves the range, every input usable."""
    fields = {
        "commencement_date": found("2030-01-01", "date", pages=(1, 1)),
        "initial_term": found(years(10000), "duration", pages=(1, 1)),
        "initial_term_end": not_found(),
        "renewal": found(years(1), "fixed_period", pages=(2, 2)),
        "notice_period": found(months(3), "duration", pages=(3, 3)),
    }
    fields.update(changes)
    return fields


def _notice_fields():
    """A stated end 31-12-2034, R 1 jaar and N 30000 maanden."""
    return {
        "initial_term_end": found("2034-12-31", "date"),
        "renewal": found(years(1), "fixed_period"),
        "notice_period": found(months(30000), "duration"),
    }


CASES = {
    "stated-end-9999-12-31-renewal-none": Case(
        fields={
            "initial_term_end": found("9999-12-31", "date"),
            "renewal": found(None, "none"),
        },
        reference="2031-09-01",
        derived={
            "expiry_date": ("computed", "9999-12-31", None),
            "next_renewal_date": ("not_applicable", None, None),
            "latest_notice_date": ("not_applicable", None, None),
        },
        cells=["31-12-9999", NO_RENEWAL, NO_RENEWAL],
        effective=effective("9999-12-31", "stated"),
        warnings=[],
    ),
    "stated-end-9999-12-31-renewal-indefinite": Case(
        fields={
            "initial_term_end": found("9999-12-31", "date"),
            "renewal": found(None, "indefinite"),
            "notice_period": found(months(3), "duration"),
        },
        reference="2031-09-01",
        derived={
            "expiry_date": ("computed", "9999-12-31", None),
            "next_renewal_date": (
                "not_computable",
                None,
                oor("initial_term_end", "renewal"),
            ),
            "latest_notice_date": ("computed", "9999-09-30", None),
        },
        cells=["31-12-9999", OUT_OF_RANGE, "30-09-9999"],
        effective=effective("9999-12-31", "stated"),
        warnings=["next_renewal_date"],
    ),
    "term-of-10000-years-beside-a-stated-end": Case(
        fields={
            "commencement_date": found("2030-01-01", "date", pages=(1, 1)),
            "initial_term": found(years(10000), "duration", pages=(1, 1)),
            "initial_term_end": found("2034-12-31", "date"),
            "renewal": found(years(1), "fixed_period"),
            "notice_period": found(months(3), "duration"),
        },
        reference="2031-09-01",
        derived={
            "expiry_date": ("computed", "2034-12-31", None),
            "next_renewal_date": ("computed", "2035-01-01", None),
            "latest_notice_date": ("computed", "2034-09-30", None),
        },
        cells=["31-12-2034", "01-01-2035", "30-09-2034"],
        effective=effective("2034-12-31", "stated"),
        warnings=["term_end_mismatch"],
    ),
    "fallback-out-of-range-every-input-usable": Case(
        fields=_fallback_fields(),
        reference="2031-09-01",
        derived={
            "expiry_date": (
                "not_computable",
                None,
                oor("commencement_date", "initial_term", "renewal"),
            ),
            "next_renewal_date": (
                "not_computable",
                None,
                oor("commencement_date", "initial_term", "renewal"),
            ),
            "latest_notice_date": (
                "not_computable",
                None,
                oor("commencement_date", "initial_term", "renewal", "notice_period"),
            ),
        },
        cells=[OUT_OF_RANGE, OUT_OF_RANGE, OUT_OF_RANGE],
        effective=effective(),
        warnings=["expiry_date", "next_renewal_date", "latest_notice_date"],
    ),
    "fallback-out-of-range-renewal-not-found": Case(
        fields=_fallback_fields(renewal=not_found()),
        reference="2031-09-01",
        derived={
            name: (
                "not_computable",
                None,
                [{"field": "renewal", "reason": "not_found"}],
            )
            for name in DERIVED_LABELS
        },
        cells=[RENEWAL_MISSING, RENEWAL_MISSING, RENEWAL_MISSING],
        effective=effective(),
        warnings=["effective_initial_term_end"],
    ),
    "fallback-out-of-range-amendment": Case(
        fields=_fallback_fields(),
        reference="2031-09-01",
        derived={name: ("not_applicable", None, None) for name in DERIVED_LABELS},
        cells=[AMENDMENT_TEXT, AMENDMENT_TEXT, AMENDMENT_TEXT],
        effective=effective(),
        warnings=["effective_initial_term_end"],
        document_type=AMENDMENT,
    ),
    "notice-of-30000-months": Case(
        fields=_notice_fields(),
        reference="2031-09-01",
        derived={
            "expiry_date": ("computed", "2034-12-31", None),
            "next_renewal_date": ("computed", "2035-01-01", None),
            "latest_notice_date": (
                "not_computable",
                None,
                oor("initial_term_end", "renewal", "notice_period"),
            ),
        },
        cells=["31-12-2034", "01-01-2035", OUT_OF_RANGE],
        effective=effective("2034-12-31", "stated"),
        warnings=["latest_notice_date"],
    ),
    "notice-of-30000-months-reference-date-01-01-5000": Case(
        fields=_notice_fields(),
        reference="5000-01-01",
        derived={
            "expiry_date": ("computed", "5000-12-31", None),
            "next_renewal_date": ("computed", "5001-01-01", None),
            "latest_notice_date": (
                "not_computable",
                None,
                oor("initial_term_end", "renewal", "notice_period"),
            ),
        },
        cells=["31-12-5000", "01-01-5001", OUT_OF_RANGE],
        effective=effective("2034-12-31", "stated"),
        warnings=["latest_notice_date"],
    ),
    "reference-date-31-12-9999": Case(
        fields={
            "initial_term_end": found("2034-12-31", "date"),
            "renewal": found(years(1), "fixed_period"),
            "notice_period": found(months(3), "duration"),
        },
        reference="9999-12-31",
        derived={
            "expiry_date": ("computed", "9999-12-31", None),
            "next_renewal_date": (
                "not_computable",
                None,
                oor("initial_term_end", "renewal"),
            ),
            "latest_notice_date": (
                "not_computable",
                None,
                oor("initial_term_end", "renewal", "notice_period"),
            ),
        },
        cells=["31-12-9999", OUT_OF_RANGE, OUT_OF_RANGE],
        effective=effective("2034-12-31", "stated"),
        warnings=["next_renewal_date", "latest_notice_date"],
    ),
    "control-stated-end-30-12-9999-renewal-none": Case(
        fields={
            "initial_term_end": found("9999-12-30", "date"),
            "renewal": found(None, "none"),
        },
        reference="2031-09-01",
        derived={
            "expiry_date": ("computed", "9999-12-30", None),
            "next_renewal_date": ("not_applicable", None, None),
            "latest_notice_date": ("not_applicable", None, None),
        },
        cells=["30-12-9999", NO_RENEWAL, NO_RENEWAL],
        effective=effective("9999-12-30", "stated"),
        warnings=[],
    ),
}


def _case_record(case, **changes):
    fields = {**case.fields, **changes}
    return extraction_record(
        "bron/" + CASE_NAME, SHA_B, document_type=case.document_type, fields=fields
    )


def _outputs(folder):
    """The CSV rows, the md sections and the JSON documents, by file name."""
    text = (folder / "followup.csv").read_bytes().decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=";"))
    markdown = (folder / "followup.md").read_bytes().decode("utf-8")
    body = markdown.rsplit("\n\nReferentiedatum: ", 1)[0].split("\n## ")[1:]
    sections = {}
    for part in body:
        lines = ("## " + part).split("\n")
        while lines[-1] == "":
            lines.pop()
        sections[part.split("\n", 1)[0]] = lines
    documents = json.loads((folder / "followup.json").read_text(encoding="utf-8"))
    return {
        "rows": {row[0]: row for row in rows[1:]},
        "sections": sections,
        "documents": {item["file_name"]: item for item in documents["documents"]},
    }


def _run(tmp_path, caplog, case):
    """``report`` over the ordinary file alone, then over it and the case's
    file. Returns both outputs, each run's WARNINGs and the second run's
    exit code."""
    alone = tmp_path / "alleen"
    write_json_file(alone, "a.json", ORDINARY)
    both = tmp_path / "beide"
    write_json_file(both, "a.json", ORDINARY)
    write_json_file(both, CASE_RELATIVE, _case_record(case))
    warnings = []
    for folder, report_dir in [(alone, "rapport-alleen"), (both, "rapport-beide")]:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=LOGGER):
            code = report(
                tmp_path,
                case.reference,
                extracted_dir=folder,
                report_dir=tmp_path / report_dir,
            )
        warnings.append(messages(caplog, logging.WARNING))
    return (
        code,
        _outputs(tmp_path / "rapport-alleen"),
        _outputs(tmp_path / "rapport-beide"),
        warnings,
    )


@pytest.mark.parametrize("name", list(CASES))
def test_date_arithmetic_outside_the_range_through_report(tmp_path, caplog, name):
    case = CASES[name]

    code, alone, both, (alone_warnings, warnings) = _run(tmp_path, caplog, case)

    assert code == 0
    assert sorted(path.name for path in (tmp_path / "rapport-beide").iterdir()) == [
        "followup.csv",
        "followup.json",
        "followup.md",
    ]
    assert alone_warnings == []
    for part in ("rows", "sections", "documents"):
        assert both[part][ORDINARY_NAME] == alone[part][ORDINARY_NAME]
    document = both["documents"][CASE_NAME]
    assert document["effective_initial_term_end"] == case.effective
    for entry_name, (status, value, missing) in case.derived.items():
        entry = document["derived"][entry_name]
        assert (entry["status"], entry["value"]) == (status, value)
        if missing is not None:
            assert entry["missing_inputs"] == missing
        if status != "computed":
            assert (entry["inputs"], entry["calculation"]) == ([], None)
    assert both["rows"][CASE_NAME][8:] == case.cells
    section = both["sections"][CASE_NAME]
    for (entry_name, label), cell in zip(DERIVED_LABELS.items(), case.cells):
        line = section.index(f"- {label}: {cell}")
        calculated = section[line + 1].startswith("  - ")
        assert calculated == (case.derived[entry_name][0] == "computed")
    assert warnings == [
        f"{CASE_RELATIVE}: {entry}: date out of range; not computed"
        for entry in case.warnings
    ]


@pytest.mark.parametrize(
    "name",
    [
        "stated-end-9999-12-31-renewal-none",
        "control-stated-end-30-12-9999-renewal-none",
    ],
)
def test_an_end_late_in_9999_without_renewal_has_not_expired(tmp_path, caplog, name):
    _, _, both, _ = _run(tmp_path, caplog, CASES[name])

    assert both["documents"][CASE_NAME]["derived"]["expiry_date"]["expired"] is False


def test_a_term_that_leaves_the_range_beside_a_stated_end_changes_no_derived_date(
    tmp_path, caplog
):
    case = CASES["term-of-10000-years-beside-a-stated-end"]
    _, _, both, _ = _run(tmp_path, caplog, case)
    plain = tmp_path / "zonder-c-en-t"
    write_json_file(
        plain,
        CASE_RELATIVE,
        _case_record(case, commencement_date=not_found(), initial_term=not_found()),
    )

    assert (
        report(
            tmp_path,
            case.reference,
            extracted_dir=plain,
            report_dir=tmp_path / "rapport-zonder",
        )
        == 0
    )

    without = _outputs(tmp_path / "rapport-zonder")["documents"][CASE_NAME]
    assert both["documents"][CASE_NAME]["derived"] == without["derived"]
    assert not any(line.startswith("Let op") for line in both["sections"][CASE_NAME])


def test_the_expiry_at_reference_date_31_12_9999_is_e0_plus_7965_years(
    tmp_path, caplog
):
    _, _, both, _ = _run(tmp_path, caplog, CASES["reference-date-31-12-9999"])

    expiry = both["documents"][CASE_NAME]["derived"]["expiry_date"]
    assert expiry["calculation"] == (
        "31-12-2034 + 7965 jaar = 31-12-9999 (invoer: einde eerste termijn p. 2, "
        "verlenging p. 2)"
    )
