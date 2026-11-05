"""The report over fabricated extraction files (spec 008, D12, D13, R19-R25;
Q7, Q8, I1, A13).

Ten in-scope documents and one out of scope, in ``tmp_path / "extracted"``,
written in an order other than their file paths'. The reference date is
01-09-2031. The cells, the CSV and the sections are written out by hand from
D13's texts.
"""

from __future__ import annotations

import json

import pytest

from companydocs_rag.metadata.texts import derived_cell
from tests.metadata._helpers import (
    AMENDMENT,
    LEASE,
    SHA_A,
    SHA_B,
    SHA_C,
    SHA_D,
    batch_entries,
    counted,
    extraction_record,
    field,
    found,
    months,
    not_found,
    provision_item,
    relation,
    report,
    write_json_file,
    years,
)

MINUS = chr(0x2212)  # U+2212 MINUS SIGN
E_DIAERESIS = chr(0x00EB)  # U+00EB LATIN SMALL LETTER E WITH DIAERESIS
SHA_E, SHA_F, SHA_G, SHA_H, SHA_I, SHA_J, SHA_K = (
    "e5" * 32,
    "f6" * 32,
    "a7" * 32,
    "b8" * 32,
    "c9" * 32,
    "d0" * 32,
    "e1" * 32,
)
REFERENCE = "2031-09-01"
DETAIL = "Fictieve wijziging van de huurprijs"


def duration(amount, unit):
    return {"amount": amount, "unit": unit}


def _documents():
    """Relative path -> extraction record."""
    return {
        "z/a.json": extraction_record(
            "bron/a-vast.pdf",
            SHA_A,
            fields={
                "signing_date": found(
                    "2030-03-01", "date", pages=(1, 1), quote="Getekend op 1 maart 2030"
                ),
                "commencement_date": found(
                    "2030-03-15", "date", pages=(1, 1), quote="Ingang 15 maart 2030"
                ),
                "initial_term": found(
                    years(4), "duration", pages=(1, 1), quote="Looptijd vier jaar"
                ),
                "initial_term_end": found(
                    "2034-03-14", "date", pages=(2, 2), quote="Tot en met 14 maart 2034"
                ),
                "renewal": found(
                    years(3),
                    "fixed_period",
                    pages=(2, 2),
                    quote="Telkens\nverlengd met drie jaar",
                ),
                "notice_period": found(
                    months(6),
                    "duration",
                    pages=(2, 3),
                    quote="Opzegtermijn zes maanden",
                ),
            },
            provisions=[
                provision_item(
                    "tenant", "De huurder mag tussentijds opzeggen", pages=(3, 3)
                ),
                provision_item(
                    "tenant", "DE HUURDER  mag tussentijds opzeggen", pages=(4, 4)
                ),
                provision_item("landlord", "De verhuurder mag ontbinden", pages=(5, 5)),
                provision_item("both", "Beide partijen | fictief", pages=(6, 6)),
                provision_item("unspecified", "Een partij mag opzeggen", pages=(7, 8)),
                provision_item("tenant", "Niet geverifieerd", verified=False),
            ],
        ),
        "b-geen.json": extraction_record(
            "bron/b-geen.pdf",
            SHA_B,
            fields={
                "commencement_date": found(
                    "2030-01-01", "date", pages=(1, 1), origin="native"
                ),
                "initial_term": found(years(1), "duration", pages=(1, 1)),
                "initial_term_end": found("2030-12-31", "date", pages=(1, 1)),
                "renewal": found(None, "none", pages=(2, 2)),
                "notice_period": found(months(1), "duration", pages=(2, 2)),
            },
        ),
        "c-onbepaald.json": extraction_record(
            "bron/c-onbepaald.pdf",
            SHA_C,
            fields={
                "commencement_date": found("2029-05-01", "date", pages=(1, 1)),
                "initial_term": found(None, "indefinite", pages=(1, 1)),
                "notice_period": found(duration(1, "weeks"), "duration", pages=(3, 3)),
            },
        ),
        "d-voortgezet.json": extraction_record(
            "bron/d-voortgezet.pdf",
            SHA_D,
            fields={
                "commencement_date": found("2030-06-21", "date", pages=(1, 1)),
                "initial_term": found(duration(10, "days"), "duration", pages=(1, 1)),
                "renewal": found(None, "indefinite", pages=(2, 2)),
                "notice_period": found(duration(2, "weeks"), "duration", pages=(2, 2)),
            },
        ),
        "e-allonge.json": extraction_record(
            "bron/e-allonge.pdf",
            SHA_E,
            document_type=AMENDMENT,
            fields={"signing_date": found("2031-02-01", "date", pages=(1, 1))},
            relations=[relation("amends", SHA_A, DETAIL)],
        ),
        "f-kopie.json": extraction_record(
            "bron/f-kopie.pdf",
            SHA_F,
            relations=[relation("duplicate_of", SHA_B)],
        ),
        "g-onvolledig.json": extraction_record(
            "bron/g-onvolledig.pdf",
            SHA_G,
            status="partial",
            batches=batch_entries("ok", "error", "ok"),
            fields={
                "signing_date": found(None, "conditional", pages=(1, 1)),
                "commencement_date": found(None, "blank", pages=(4, 4)),
                "initial_term_end": found(None, "illegible", pages=(7, 7)),
                "renewal": field(
                    "conflicting",
                    counted(years(1), "fixed_period", pages=(2, 2), quote="een jaar"),
                    counted(years(2), "fixed_period", pages=(5, 6), quote="twee jaar"),
                ),
                "notice_period": found(duration(1, "days"), "duration", pages=(3, 3)),
            },
        ),
        "h-afwijkend.json": extraction_record(
            "bron/h-afwijkend.pdf",
            SHA_H,
            document_type="Huurovereenkomst | fictief",
            fields={
                "commencement_date": found("2030-03-15", "date", pages=(1, 1)),
                "initial_term": found(years(3), "duration", pages=(1, 1)),
                "initial_term_end": found("2033-03-31", "date", pages=(2, 2)),
            },
        ),
        "i-statuten.json": extraction_record(
            "bron/i-statuten.pdf",
            SHA_I,
            document_type="Statuten",
            status="not_in_scope",
            batches=[],
            relations=[relation("amends", SHA_A)],
        ),
        "j-voorwaardelijk.json": extraction_record(
            "bron/j-voorwaardelijk.pdf",
            SHA_J,
            fields={
                "commencement_date": found(None, "conditional", pages=(1, 1)),
                "initial_term": found(years(5), "duration", pages=(1, 1)),
                "initial_term_end": not_found(),
                "renewal": field(
                    "conflicting",
                    counted(None, "none", pages=(2, 2), quote="geen verlenging"),
                    counted(years(1), "fixed_period", pages=(3, 3), quote="een jaar"),
                ),
                "notice_period": found(months(3), "duration", pages=(3, 3)),
            },
        ),
        "k-buiten.json": extraction_record(
            "bron/k-buiten.pdf",
            SHA_K,
            fields={
                "commencement_date": found("2030-01-01", "date", pages=(1, 1)),
                "initial_term": found(years(10000), "duration", pages=(1, 1)),
                "renewal": found(years(1), "fixed_period", pages=(2, 2)),
                "notice_period": found(months(3), "duration", pages=(3, 3)),
            },
        ),
    }


def _write(folder):
    for relative, record in _documents().items():
        write_json_file(folder, relative, record)


NOT_FOUND = "niet gevonden"
AMENDMENT_TEXT = "n.v.t.: wijzigingsdocument, zie relatie"
F_MISSING = (
    "niet berekenbaar: ingangsdatum ontbreekt; looptijd ontbreekt; einde eerste "
    "termijn ontbreekt; verlenging ontbreekt"
)
G_MISSING = (
    "niet berekenbaar: ingangsdatum leeg veld; looptijd ontbreekt; einde eerste "
    "termijn onleesbaar; verlenging tegenstrijdig"
)
J_MISSING = (
    "niet berekenbaar: ingangsdatum voorwaardelijk; einde eerste termijn "
    "ontbreekt; verlenging tegenstrijdig"
)
SINCE = "n.v.t.: sinds 01-07-2030 voor onbepaalde tijd"
OUT_OF_RANGE = "niet berekenbaar: buiten datumbereik"
CELLS = [
    [
        "a-vast.pdf",
        LEASE,
        "01-03-2030 (p. 1)",
        "15-03-2030 (p. 1)",
        "4 jaar (p. 1)",
        "14-03-2034 (p. 2)",
        "telkens 3 jaar (p. 2)",
        "6 maanden (p. 2-3)",
        "14-03-2034",
        "15-03-2034",
        "14-09-2033",
    ],
    [
        "b-geen.pdf",
        LEASE,
        NOT_FOUND,
        "01-01-2030 (p. 1)",
        "1 jaar (p. 1)",
        "31-12-2030 (p. 1)",
        "geen verlenging (p. 2)",
        "1 maand (p. 2)",
        "31-12-2030 (verlopen)",
        "n.v.t.: geen verlenging",
        "n.v.t.: geen verlenging",
    ],
    [
        "c-onbepaald.pdf",
        LEASE,
        NOT_FOUND,
        "01-05-2029 (p. 1)",
        "onbepaalde tijd (p. 1)",
        NOT_FOUND,
        NOT_FOUND,
        "1 week (p. 3)",
        "n.v.t.: onbepaalde tijd",
        "n.v.t.: onbepaalde tijd",
        "n.v.t.: onbepaalde tijd",
    ],
    [
        "d-voortgezet.pdf",
        LEASE,
        NOT_FOUND,
        "21-06-2030 (p. 1)",
        "10 dagen (p. 1)",
        NOT_FOUND,
        "onbepaalde tijd (p. 2)",
        "2 weken (p. 2)",
        SINCE,
        SINCE,
        SINCE,
    ],
    [
        "e-allonge.pdf",
        AMENDMENT,
        "01-02-2031 (p. 1)",
        NOT_FOUND,
        NOT_FOUND,
        NOT_FOUND,
        NOT_FOUND,
        NOT_FOUND,
        AMENDMENT_TEXT,
        AMENDMENT_TEXT,
        AMENDMENT_TEXT,
    ],
    [
        "f-kopie.pdf",
        LEASE,
        NOT_FOUND,
        NOT_FOUND,
        NOT_FOUND,
        NOT_FOUND,
        NOT_FOUND,
        NOT_FOUND,
        F_MISSING,
        F_MISSING,
        F_MISSING + "; opzegtermijn ontbreekt",
    ],
    [
        "g-onvolledig.pdf",
        LEASE,
        "voorwaardelijk (p. 1)",
        "leeg veld in het document (p. 4)",
        NOT_FOUND,
        "onleesbaar (p. 7)",
        "tegenstrijdig (p. 2, 5-6)",
        "1 dag (p. 3)",
        G_MISSING,
        G_MISSING,
        G_MISSING,
    ],
    [
        "h-afwijkend.pdf",
        "Huurovereenkomst | fictief",
        NOT_FOUND,
        "15-03-2030 (p. 1)",
        "3 jaar (p. 1)",
        "31-03-2033 (p. 2)",
        NOT_FOUND,
        NOT_FOUND,
        "niet berekenbaar: verlenging ontbreekt",
        "niet berekenbaar: verlenging ontbreekt",
        "niet berekenbaar: verlenging ontbreekt; opzegtermijn ontbreekt",
    ],
    [
        "j-voorwaardelijk.pdf",
        LEASE,
        NOT_FOUND,
        "voorwaardelijk (p. 1)",
        "5 jaar (p. 1)",
        NOT_FOUND,
        "tegenstrijdig (p. 2, 3)",
        "3 maanden (p. 3)",
        J_MISSING,
        J_MISSING,
        J_MISSING,
    ],
    [
        "k-buiten.pdf",
        LEASE,
        NOT_FOUND,
        "01-01-2030 (p. 1)",
        "10000 jaar (p. 1)",
        NOT_FOUND,
        "telkens 1 jaar (p. 2)",
        "3 maanden (p. 3)",
        OUT_OF_RANGE,
        OUT_OF_RANGE,
        OUT_OF_RANGE,
    ],
]
CSV = b"\xef\xbb\xbf" + (
    "Document;Type;Ondertekend;Ingangsdatum;Looptijd;Einde eerste termijn;"
    "Verlenging;Opzegtermijn;Vervaldatum;Eerstvolgende verlenging;"
    "Uiterste opzegdatum\r\n"
    "a-vast.pdf;Huurovereenkomst;01-03-2030 (p. 1);15-03-2030 (p. 1);"
    "4 jaar (p. 1);14-03-2034 (p. 2);telkens 3 jaar (p. 2);6 maanden (p. 2-3);"
    "14-03-2034;15-03-2034;14-09-2033\r\n"
    "b-geen.pdf;Huurovereenkomst;niet gevonden;01-01-2030 (p. 1);1 jaar (p. 1);"
    "31-12-2030 (p. 1);geen verlenging (p. 2);1 maand (p. 2);"
    "31-12-2030 (verlopen);n.v.t.: geen verlenging;n.v.t.: geen verlenging\r\n"
    "c-onbepaald.pdf;Huurovereenkomst;niet gevonden;01-05-2029 (p. 1);"
    "onbepaalde tijd (p. 1);niet gevonden;niet gevonden;1 week (p. 3);"
    "n.v.t.: onbepaalde tijd;n.v.t.: onbepaalde tijd;n.v.t.: onbepaalde tijd\r\n"
    "d-voortgezet.pdf;Huurovereenkomst;niet gevonden;21-06-2030 (p. 1);"
    "10 dagen (p. 1);niet gevonden;onbepaalde tijd (p. 2);2 weken (p. 2);"
    "n.v.t.: sinds 01-07-2030 voor onbepaalde tijd;"
    "n.v.t.: sinds 01-07-2030 voor onbepaalde tijd;"
    "n.v.t.: sinds 01-07-2030 voor onbepaalde tijd\r\n"
    "e-allonge.pdf;Allonge;01-02-2031 (p. 1);niet gevonden;niet gevonden;"
    "niet gevonden;niet gevonden;niet gevonden;"
    "n.v.t.: wijzigingsdocument, zie relatie;"
    "n.v.t.: wijzigingsdocument, zie relatie;"
    "n.v.t.: wijzigingsdocument, zie relatie\r\n"
    "f-kopie.pdf;Huurovereenkomst;niet gevonden;niet gevonden;niet gevonden;"
    "niet gevonden;niet gevonden;niet gevonden;"
    '"niet berekenbaar: ingangsdatum ontbreekt; looptijd ontbreekt; einde '
    'eerste termijn ontbreekt; verlenging ontbreekt";'
    '"niet berekenbaar: ingangsdatum ontbreekt; looptijd ontbreekt; einde '
    'eerste termijn ontbreekt; verlenging ontbreekt";'
    '"niet berekenbaar: ingangsdatum ontbreekt; looptijd ontbreekt; einde '
    'eerste termijn ontbreekt; verlenging ontbreekt; opzegtermijn ontbreekt"'
    "\r\n"
    "g-onvolledig.pdf;Huurovereenkomst;voorwaardelijk (p. 1);"
    "leeg veld in het document (p. 4);niet gevonden;onleesbaar (p. 7);"
    "tegenstrijdig (p. 2, 5-6);1 dag (p. 3);"
    '"niet berekenbaar: ingangsdatum leeg veld; looptijd ontbreekt; einde '
    'eerste termijn onleesbaar; verlenging tegenstrijdig";'
    '"niet berekenbaar: ingangsdatum leeg veld; looptijd ontbreekt; einde '
    'eerste termijn onleesbaar; verlenging tegenstrijdig";'
    '"niet berekenbaar: ingangsdatum leeg veld; looptijd ontbreekt; einde '
    'eerste termijn onleesbaar; verlenging tegenstrijdig"\r\n'
    "h-afwijkend.pdf;Huurovereenkomst | fictief;niet gevonden;"
    "15-03-2030 (p. 1);3 jaar (p. 1);31-03-2033 (p. 2);niet gevonden;"
    "niet gevonden;niet berekenbaar: verlenging ontbreekt;"
    "niet berekenbaar: verlenging ontbreekt;"
    '"niet berekenbaar: verlenging ontbreekt; opzegtermijn ontbreekt"\r\n'
    "j-voorwaardelijk.pdf;Huurovereenkomst;niet gevonden;voorwaardelijk (p. 1);"
    "5 jaar (p. 1);niet gevonden;tegenstrijdig (p. 2, 3);3 maanden (p. 3);"
    '"niet berekenbaar: ingangsdatum voorwaardelijk; einde eerste termijn '
    'ontbreekt; verlenging tegenstrijdig";'
    '"niet berekenbaar: ingangsdatum voorwaardelijk; einde eerste termijn '
    'ontbreekt; verlenging tegenstrijdig";'
    '"niet berekenbaar: ingangsdatum voorwaardelijk; einde eerste termijn '
    'ontbreekt; verlenging tegenstrijdig"\r\n'
    "k-buiten.pdf;Huurovereenkomst;niet gevonden;01-01-2030 (p. 1);"
    "10000 jaar (p. 1);niet gevonden;telkens 1 jaar (p. 2);3 maanden (p. 3);"
    "niet berekenbaar: buiten datumbereik;niet berekenbaar: buiten datumbereik;"
    "niet berekenbaar: buiten datumbereik\r\n"
).encode("utf-8")
SCAN_NOTE = "Gelezen van een scan; controleer de waarden op de genoemde pagina's."
SECTION_A = [
    "## a-vast.pdf",
    "",
    "- Ondertekend: 01-03-2030 (p. 1)",
    '  - 01-03-2030: "Getekend op 1 maart 2030" (p. 1)',
    "- Ingangsdatum: 15-03-2030 (p. 1)",
    '  - 15-03-2030: "Ingang 15 maart 2030" (p. 1)',
    "- Looptijd: 4 jaar (p. 1)",
    '  - 4 jaar: "Looptijd vier jaar" (p. 1)',
    "- Einde eerste termijn: 14-03-2034 (p. 2)",
    '  - 14-03-2034: "Tot en met 14 maart 2034" (p. 2)',
    "- Verlenging: telkens 3 jaar (p. 2)",
    '  - telkens 3 jaar: "Telkens verlengd met drie jaar" (p. 2)',
    "- Opzegtermijn: 6 maanden (p. 2-3)",
    '  - 6 maanden: "Opzegtermijn zes maanden" (p. 2-3)',
    "- Vervaldatum: 14-03-2034",
    "  - 14-03-2034 (invoer: einde eerste termijn p. 2, verlenging p. 2)",
    "- Eerstvolgende verlenging: 15-03-2034",
    "  - 14-03-2034 + 1 dag = 15-03-2034 (invoer: einde eerste termijn p. 2, "
    "verlenging p. 2)",
    "- Uiterste opzegdatum: 14-09-2033",
    f"  - 14-03-2034 {MINUS} 6 maanden = 14-09-2033 (termijneinde 14-03-2034; "
    "invoer: einde eerste termijn p. 2, verlenging p. 2, opzegtermijn p. 2-3)",
    "",
    f"### Overige be{E_DIAERESIS}indigingsbepalingen (niet in de berekening)",
    "",
    '- huurder: "De huurder mag tussentijds opzeggen" (p. 3, 4)',
    '- verhuurder: "De verhuurder mag ontbinden" (p. 5)',
    '- beide: "Beide partijen | fictief" (p. 6)',
    '- niet gespecificeerd: "Een partij mag opzeggen" (p. 7-8)',
    "",
    f"Gewijzigd door e-allonge.pdf: {DETAIL}; niet verwerkt in de berekening.",
    "",
    "Gewijzigd door i-statuten.pdf; niet verwerkt in de berekening.",
    "",
    SCAN_NOTE,
]


@pytest.fixture
def written(tmp_path):
    _write(tmp_path / "extracted")
    assert report(tmp_path, REFERENCE) == 0
    folder = tmp_path / "rapport"
    return {
        "md": (folder / "followup.md").read_bytes(),
        "json": (folder / "followup.json").read_bytes(),
        "csv": (folder / "followup.csv").read_bytes(),
    }


def _sections(markdown):
    """The lines of each section, by file name; the footer is no section's."""
    body = markdown.rsplit("\n\nReferentiedatum: ", 1)[0].split("\n## ")[1:]
    return {part.split("\n", 1)[0]: ("## " + part).split("\n") for part in body}


# --- followup.csv [CUT-1] -------------------------------------------------------------


def test_the_csv_is_byte_for_byte_the_table(written):
    assert written["csv"] == CSV


# --- followup.md ----------------------------------------------------------------------


def test_the_markdown_table_holds_the_same_cells_with_pipes_escaped(written):
    lines = written["md"].decode("utf-8").split("\n")

    assert lines[0] == (
        "| Document | Type | Ondertekend | Ingangsdatum | Looptijd | Einde eerste "
        "termijn | Verlenging | Opzegtermijn | Vervaldatum | Eerstvolgende "
        "verlenging | Uiterste opzegdatum |"
    )
    assert lines[1] == "|---|---|---|---|---|---|---|---|---|---|---|"
    assert lines[2] == (
        "| a-vast.pdf | Huurovereenkomst | 01-03-2030 (p. 1) | 15-03-2030 (p. 1) | "
        "4 jaar (p. 1) | 14-03-2034 (p. 2) | telkens 3 jaar (p. 2) | "
        "6 maanden (p. 2-3) | 14-03-2034 | 15-03-2034 | 14-09-2033 |"
    )
    for line, cells in zip(lines[2:12], CELLS, strict=True):
        assert (
            line == "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"
        )
    assert "| h-afwijkend.pdf | Huurovereenkomst \\| fictief |" in lines[9]
    assert lines[12] == ""


def test_the_markdown_is_utf_8_with_lf_and_one_final_lf(written):
    raw = written["md"]

    assert b"\r" not in raw
    assert raw.endswith(b".\n") and not raw.endswith(b"\n\n")
    text = raw.decode("utf-8")
    assert text.split("\n")[-2] == (
        "Referentiedatum: 01-09-2031. Rekenregel: kalenderrekenen, dag afgekapt op "
        "het maandeinde; geen juridische toets."
    )
    assert "i-statuten.pdf |" not in text
    assert list(_sections(text)) == [cells[0] for cells in CELLS]


def test_a_section_holds_fields_quotes_dates_provisions_and_notes(written):
    sections = _sections(written["md"].decode("utf-8"))

    assert sections["a-vast.pdf"][: len(SECTION_A)] == SECTION_A
    assert sections["a-vast.pdf"][len(SECTION_A) :] == [""]


def test_the_notes_of_each_document(written):
    sections = _sections(written["md"].decode("utf-8"))

    def notes(name):
        lines = sections[name]
        last_item = max(i for i, line in enumerate(lines) if line.startswith("- "))
        return [line for line in lines[last_item + 1 :] if line]

    assert notes("b-geen.pdf") == []
    assert notes("c-onbepaald.pdf") == [SCAN_NOTE]
    assert notes("e-allonge.pdf") == [
        f"Wijzigt a-vast.pdf: {DETAIL}; niet verwerkt in de berekening.",
        SCAN_NOTE,
    ]
    assert notes("f-kopie.pdf") == ["Zelfde document als b-geen.pdf."]
    assert notes("g-onvolledig.pdf") == [
        "Extractie onvolledig: 1 van 3 batches mislukt.",
        SCAN_NOTE,
    ]
    assert notes("h-afwijkend.pdf") == [
        "Let op: einde eerste termijn 31-03-2033 (p. 2) wijkt af van ingangsdatum + "
        f"looptijd {MINUS} 1 dag = 14-03-2033; de genoemde datum is gebruikt.",
        SCAN_NOTE,
    ]
    assert notes("k-buiten.pdf") == [SCAN_NOTE]


def test_an_out_of_range_date_shows_its_text_without_a_calculation(written):
    section = _sections(written["md"].decode("utf-8"))["k-buiten.pdf"]

    position = section.index("- Vervaldatum: niet berekenbaar: buiten datumbereik")
    assert section[position : position + 3] == [
        "- Vervaldatum: niet berekenbaar: buiten datumbereik",
        "- Eerstvolgende verlenging: niet berekenbaar: buiten datumbereik",
        "- Uiterste opzegdatum: niet berekenbaar: buiten datumbereik",
    ]


def test_the_mismatch_note_comes_before_the_relation_notes(tmp_path):
    documents = _documents()
    record = documents["h-afwijkend.json"]
    record["relations"] = [relation("amends", SHA_B, "Fictief")]
    write_json_file(tmp_path / "extracted", "h.json", record)
    write_json_file(tmp_path / "extracted", "b.json", documents["b-geen.json"])

    assert report(tmp_path, REFERENCE) == 0

    text = (tmp_path / "rapport" / "followup.md").read_text(encoding="utf-8")
    section = _sections(text)["h-afwijkend.pdf"]
    notes = [line for line in section if line.startswith(("Let op", "Wijzigt"))]
    assert [note.split(":")[0] for note in notes] == [
        "Let op",
        "Wijzigt b-geen.pdf",
    ]


def test_conflicting_quotes_and_line_feeds_in_quotes(written):
    section = _sections(written["md"].decode("utf-8"))["g-onvolledig.pdf"]

    assert section[2:4] == [
        "- Ondertekend: voorwaardelijk (p. 1)",
        '  - voorwaardelijk: "fictief citaat" (p. 1)',
    ]
    position = section.index("- Verlenging: tegenstrijdig (p. 2, 5-6)")
    assert section[position + 1 : position + 3] == [
        '  - telkens 1 jaar: "een jaar" (p. 2)',
        '  - telkens 2 jaar: "twee jaar" (p. 5-6)',
    ]


def test_the_scan_note_needs_every_evidence_page_to_be_ocr(tmp_path):
    documents = _documents()
    record = documents["b-geen.json"]
    write_json_file(tmp_path / "extracted", "native.json", record)
    record = json.loads(json.dumps(record))
    record["file"]["path"] = "bron/b-ocr.pdf"
    record["fields"]["commencement_date"] = found("2030-01-01", "date", pages=(1, 1))
    write_json_file(tmp_path / "extracted", "ocr.json", record)

    assert report(tmp_path, REFERENCE) == 0

    sections = _sections(
        (tmp_path / "rapport" / "followup.md").read_text(encoding="utf-8")
    )
    assert SCAN_NOTE not in sections["b-geen.pdf"]
    assert SCAN_NOTE in sections["b-ocr.pdf"]


def test_the_niet_berekenbaar_text_of_a13():
    entry = {
        "status": "not_computable",
        "value": None,
        "reason": None,
        "missing_inputs": [
            {"field": "commencement_date", "reason": "conditional"},
            {"field": "initial_term_end", "reason": "not_found"},
            {"field": "renewal", "reason": "conflicting"},
        ],
        "inputs": [],
        "calculation": None,
        "for_term_end": None,
        "expired": None,
        "since": None,
    }

    assert derived_cell(entry) == (
        "niet berekenbaar: ingangsdatum voorwaardelijk; einde eerste termijn "
        "ontbreekt; verlenging tegenstrijdig"
    )


def test_the_out_of_range_text_of_i1():
    entry = {
        "status": "not_computable",
        "value": None,
        "reason": None,
        "missing_inputs": [
            {"field": "initial_term_end", "reason": "out_of_range"},
            {"field": "renewal", "reason": "out_of_range"},
        ],
        "inputs": [],
        "calculation": None,
        "for_term_end": None,
        "expired": None,
        "since": None,
    }

    assert derived_cell(entry) == "niet berekenbaar: buiten datumbereik"


# --- followup.json --------------------------------------------------------------------

DOCUMENT_KEYS = [
    "file",
    "file_name",
    "document_type",
    "language",
    "extraction_status",
    "relations",
    "fields",
    "effective_initial_term_end",
    "derived",
    "other_termination_provisions",
]


def test_the_json_holds_d12s_keys_in_order(written):
    text = written["json"].decode("utf-8")
    content = json.loads(text)

    assert text.replace("\r\n", "\n") == json.dumps(
        content, indent=2, ensure_ascii=False
    )
    assert list(content) == ["schema_version", "reference_date", "documents"]
    assert (content["schema_version"], content["reference_date"]) == (1, REFERENCE)
    documents = content["documents"]
    assert [document["file_name"] for document in documents] == [
        cells[0] for cells in CELLS
    ]
    for document in documents:
        assert list(document) == DOCUMENT_KEYS
        for entry in document["fields"].values():
            assert list(entry) == ["status", "value", "value_kind", "pages"]
        assert list(document["derived"]) == [
            "expiry_date",
            "next_renewal_date",
            "latest_notice_date",
        ]
    first = documents[0]
    assert first["file"] == {"path": "bron/a-vast.pdf", "sha256": SHA_A}
    assert first["fields"]["notice_period"] == {
        "status": "found",
        "value": months(6),
        "value_kind": "duration",
        "pages": [[2, 3]],
    }
    assert first["fields"]["signing_date"]["value"] == "2030-03-01"
    assert first["effective_initial_term_end"] == {
        "value": "2034-03-14",
        "source": "stated",
        "term_end_mismatch": False,
    }
    assert first["derived"]["latest_notice_date"]["for_term_end"] == "2034-03-14"
    assert first["other_termination_provisions"][0] == {
        "party": "tenant",
        "as_written": "fictief",
        "quote": "De huurder mag tussentijds opzeggen",
        "pages": [[3, 3], [4, 4]],
    }
    assert len(first["other_termination_provisions"]) == 4
    conflicting = documents[6]["fields"]["renewal"]
    assert conflicting == {
        "status": "conflicting",
        "value": None,
        "value_kind": None,
        "pages": [[2, 2], [5, 6]],
    }
    mismatch = documents[7]["effective_initial_term_end"]
    assert mismatch == {
        "value": "2033-03-31",
        "source": "stated",
        "term_end_mismatch": True,
    }
    conditional = documents[8]
    assert conditional["effective_initial_term_end"] == {
        "value": None,
        "source": None,
        "term_end_mismatch": False,
    }
    for entry in conditional["derived"].values():
        assert (entry["status"], entry["missing_inputs"]) == (
            "not_computable",
            [
                {"field": "commencement_date", "reason": "conditional"},
                {"field": "initial_term_end", "reason": "not_found"},
                {"field": "renewal", "reason": "conflicting"},
            ],
        )
    out_of_range = documents[9]
    assert out_of_range["effective_initial_term_end"] == {
        "value": None,
        "source": None,
        "term_end_mismatch": False,
    }
    inputs = ["commencement_date", "initial_term", "renewal"]
    for name, fields in [
        ("expiry_date", inputs),
        ("next_renewal_date", inputs),
        ("latest_notice_date", [*inputs, "notice_period"]),
    ]:
        entry = out_of_range["derived"][name]
        assert (entry["status"], entry["missing_inputs"]) == (
            "not_computable",
            [{"field": field, "reason": "out_of_range"} for field in fields],
        )


def test_two_runs_write_byte_identical_files(tmp_path, written):
    assert report(tmp_path, REFERENCE, report_dir=tmp_path / "tweede") == 0

    for name, suffix in [("md", "md"), ("json", "json"), ("csv", "csv")]:
        assert (tmp_path / "tweede" / f"followup.{suffix}").read_bytes() == written[
            name
        ]
