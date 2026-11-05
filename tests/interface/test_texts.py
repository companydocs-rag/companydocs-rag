"""The interface's Dutch texts (spec 009, D10, D16; A14).

Each text is written out here as D10 and D16 give it; each is one line with
single spaces, all ASCII.
"""

from __future__ import annotations

from companydocs_rag.interface import texts

D16 = {
    "PAGE_TITLE": "Bedrijfsdocumenten: vragen (demo)",
    "INPUT_PLACEHOLDER": "Stel een vraag over de documenten",
    "BUTTON": "Vraag stellen",
    "UNDER_INPUT": (
        "Elke vraag wordt los beantwoord. Eerdere vragen en antwoorden op deze "
        "pagina worden niet meegenomen."
    ),
    "BUSY": "Bezig met zoeken en formuleren. Dit kan een minuut duren.",
    "BANNER_INSUFFICIENT_EVIDENCE": (
        "Geen antwoord: de gevonden passages bevatten onvoldoende bewijs."
    ),
    "BANNER_UNCITED": (
        "Let op: dit antwoord verwijst naar geen enkele bron en is niet te controleren."
    ),
    "GENERATOR_UNAVAILABLE": (
        "Het taalmodel is niet bereikbaar. Controleer of Ollama draait en stel de "
        "vraag opnieuw."
    ),
    "GENERATOR_ERROR": "Het taalmodel gaf een fout. De vraag is niet beantwoord.",
    "SERVER_ERROR": "Er ging iets mis in de server. De vraag is niet beantwoord.",
    "EMPTY_QUESTION": "Typ eerst een vraag.",
    "QUESTION_TOO_LONG": "De vraag is te lang (maximaal {N} tekens).",
    "SOURCES_HEADING": "Bronnen",
    "CLAUSE_LABEL": "kop (hint): {label}",
    "NO_DOCUMENT_TYPE": "type onbekend",
    "PAGE_LINK": "pagina {N} bekijken",
    "PAGE_NOT_FOUND": "Pagina niet gevonden.",
    "MALFORMED_REQUEST": "Ongeldig verzoek.",
}
D10 = {
    "LABEL_TEXT_LAYER": "uit de tekstlaag van het PDF",
    "LABEL_LOW_QUALITY": "tekstlaag van slechte kwaliteit, controleer op pagina {N}",
    "LABEL_SCAN": "gelezen van een scan, controleer cijfers op pagina {N}",
    "LABEL_NO_TEXT": "geen tekst op deze pagina",
    "LABEL_UNKNOWN": "herkomst onbekend, controleer op pagina {N}",
}


def test_the_module_holds_d16s_eighteen_texts():
    assert len(D16) == 18
    for name, text in D16.items():
        assert getattr(texts, name) == text, name


def test_the_module_holds_d10s_five_labels():
    assert len(D10) == 5
    for name, text in D10.items():
        assert getattr(texts, name) == text, name


def test_every_text_is_one_ascii_line_with_single_spaces():
    for text in [*D16.values(), *D10.values()]:
        assert text.isascii(), text
        assert "\n" not in text and "  " not in text, text
        assert text == text.strip(), text


def test_the_placeholders_are_filled():
    assert (
        texts.question_too_long(1000) == "De vraag is te lang (maximaal 1000 tekens)."
    )
    assert texts.clause_label("Artikel 3 {x}") == "kop (hint): Artikel 3 {x}"
    assert texts.page_link(7) == "pagina 7 bekijken"
    assert texts.label_low_quality(7) == (
        "tekstlaag van slechte kwaliteit, controleer op pagina 7"
    )
    assert texts.label_scan(7) == "gelezen van een scan, controleer cijfers op pagina 7"
    assert texts.label_unknown(7) == "herkomst onbekend, controleer op pagina 7"
