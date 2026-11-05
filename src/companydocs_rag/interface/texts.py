"""The interface's Dutch texts (spec 009, D10, D16), verbatim, all ASCII.

D16's eighteen texts and D10's five provenance labels. A template holds
``{N}`` (a number) or ``{label}`` (a clause label); the functions below fill
them. Nothing else in the package writes a Dutch text of its own.
"""

from __future__ import annotations

from typing import Any

# --- D16: the page ------------------------------------------------------------------
PAGE_TITLE = "Bedrijfsdocumenten: vragen (demo)"
INPUT_PLACEHOLDER = "Stel een vraag over de documenten"
BUTTON = "Vraag stellen"
UNDER_INPUT = (
    "Elke vraag wordt los beantwoord. Eerdere vragen en antwoorden op deze "
    "pagina worden niet meegenomen."
)
BUSY = "Bezig met zoeken en formuleren. Dit kan een minuut duren."

# --- D16: banners and messages ------------------------------------------------------
BANNER_INSUFFICIENT_EVIDENCE = (
    "Geen antwoord: de gevonden passages bevatten onvoldoende bewijs."
)
BANNER_UNCITED = (
    "Let op: dit antwoord verwijst naar geen enkele bron en is niet te controleren."
)
GENERATOR_UNAVAILABLE = (
    "Het taalmodel is niet bereikbaar. Controleer of Ollama draait en stel de "
    "vraag opnieuw."
)
GENERATOR_ERROR = "Het taalmodel gaf een fout. De vraag is niet beantwoord."
SERVER_ERROR = "Er ging iets mis in de server. De vraag is niet beantwoord."
EMPTY_QUESTION = "Typ eerst een vraag."
QUESTION_TOO_LONG = "De vraag is te lang (maximaal {N} tekens)."

# --- D16: source cards and the page view ---------------------------------------------
SOURCES_HEADING = "Bronnen"
CLAUSE_LABEL = "kop (hint): {label}"
NO_DOCUMENT_TYPE = "type onbekend"
PAGE_LINK = "pagina {N} bekijken"
PAGE_NOT_FOUND = "Pagina niet gevonden."
MALFORMED_REQUEST = "Ongeldig verzoek."

# --- D10: provenance labels, per page span -------------------------------------------
LABEL_TEXT_LAYER = "uit de tekstlaag van het PDF"
LABEL_LOW_QUALITY = "tekstlaag van slechte kwaliteit, controleer op pagina {N}"
LABEL_SCAN = "gelezen van een scan, controleer cijfers op pagina {N}"
LABEL_NO_TEXT = "geen tekst op deze pagina"
LABEL_UNKNOWN = "herkomst onbekend, controleer op pagina {N}"

# The placeholder ``{label}`` in CLAUSE_LABEL, for callers that put the label
# in an element of its own.
LABEL_PLACEHOLDER = "{label}"


def question_too_long(maximum: int) -> str:
    """D16's too-long text with ``{N}`` the maximum number of characters."""
    return QUESTION_TOO_LONG.format(N=maximum)


def clause_label(label: str) -> str:
    """D16's clause label text with ``{label}`` filled."""
    return CLAUSE_LABEL.format(label=label)


def page_link(page: Any) -> str:
    """D16's page link text with ``{N}`` the page number."""
    return PAGE_LINK.format(N=page)


def label_low_quality(page: Any) -> str:
    return LABEL_LOW_QUALITY.format(N=page)


def label_scan(page: Any) -> str:
    return LABEL_SCAN.format(N=page)


def label_unknown(page: Any) -> str:
    return LABEL_UNKNOWN.format(N=page)
