"""The interface's Dutch texts (spec 009, D10, D16; spec 011), verbatim, all
ASCII.

D16's eighteen texts, D10's five provenance labels and spec 011's texts of
the workspace: the side bar with the chat history and projects, the
documents view and the page viewer. A template holds ``{N}`` (a number) or
``{label}`` (a clause label); the functions below fill them. Nothing else in
the package writes a Dutch text of its own.
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

# --- Spec 011: the side bar ---------------------------------------------------------
BRAND = "Bedrijfsdocumenten RAG"
BRAND_SUBTITLE = "proof of concept"
NAV_DOCUMENTS = "Documenten"
STATUS_HEADING = "Systeem"
STATUS_LOCAL = "Lokaal model"
STATUS_API = "Extern model via API"
DOCUMENT_COUNT = "{N} documenten"
ACCOUNT_NAME = "Lokale gebruiker"
ACCOUNT_SUBTITLE = "Alleen op deze laptop, geen login"
ACCOUNT_INITIALS = "LG"

# --- Spec 011: the chat view --------------------------------------------------------
NEW_CHAT = "Nieuwe chat"
CHAT_EMPTY_TITLE = "Waar ben je naar op zoek?"
CHAT_EMPTY_TEXT = (
    "Elk antwoord verwijst naar zijn bronnen. Klik op een bron om de pagina "
    "rechts te openen. Noem in je vraag bij voorkeur een naam, plaats of nummer."
)

# --- Spec 011: the documents view ---------------------------------------------------
DOCUMENTS_SEARCH = "Zoek op naam of type"
DOCUMENTS_NONE = "Geen documenten gevonden."
UPLOAD_BUTTON = "Uploaden"
UPLOAD_TOOLTIP = (
    "Dummyknop: uploaden kan nog niet. Een nieuw document moet eerst door de "
    "hele pipeline (OCR, normaliseren, chunken), en daarna moet de index "
    "opnieuw worden gebouwd, wat ongeveer 20 minuten duurt. Het team beslist "
    "ook nog waar geuploade bestanden komen: data/raw is de levering van de "
    "klant."
)
PAGE_COUNT_ONE = "1 pagina"
PAGE_COUNT = "{N} pagina's"

# --- Spec 011: the chat history and projects ----------------------------------------
PROJECTS_HEADING = "Projecten"
CHATS_HEADING = "Chats"
NEW_PROJECT = "Nieuw project"
PROJECT_NAME = "Naam van het project"
NO_CHATS = "Nog geen chats. Stel een vraag om te beginnen."
PROJECT_EMPTY = "Nog geen chats in dit project"
HISTORY_NOTE = (
    "De chatgeschiedenis staat alleen in deze browser op deze laptop. Het model "
    "krijgt eerdere vragen niet mee."
)
MENU_OPTIONS = "Opties"
MENU_RENAME = "Naam wijzigen"
MENU_MOVE = "Verplaatsen naar"
MENU_NO_PROJECT = "Geen project"
MENU_DELETE = "Verwijderen"
MENU_NEW_CHAT_IN_PROJECT = "Nieuwe chat in dit project"
CONFIRM_DELETE_CHAT = "Deze chat verwijderen?"
CONFIRM_DELETE_PROJECT = (
    "Dit project verwijderen? De chats erin blijven bewaard, zonder project."
)

# --- Spec 011: the page viewer ------------------------------------------------------
VIEWER_EMPTY_TITLE = "Geen document geopend"
VIEWER_EMPTY_TEXT = (
    "Klik op een bron bij een antwoord, of op een document in Documenten, om "
    "het hier te bekijken."
)
VIEWER_LOADING = "Pagina laden..."
VIEWER_CITED = "Geciteerde pagina"
VIEWER_HIGHLIGHT = "Geel: de passage die het model las"
VIEWER_PREVIOUS = "Vorige pagina"
VIEWER_NEXT = "Volgende pagina"
VIEWER_PAGE = "Paginanummer"
VIEWER_ZOOM_OUT = "Uitzoomen"
VIEWER_ZOOM_IN = "Inzoomen"
VIEWER_FIT = "Passend"
VIEWER_OPEN_TAB = "Nieuw tabblad"
VIEWER_CLOSE = "Sluiten"
VIEWER_RESIZE = "Sleep om de breedte van de panelen te wijzigen"

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


def document_count(count: int) -> str:
    """Spec 011's document count with ``{N}`` the number of documents."""
    return DOCUMENT_COUNT.format(N=count)


def page_count(count: int) -> str:
    """Spec 011's page count of one document."""
    return PAGE_COUNT_ONE if count == 1 else PAGE_COUNT.format(N=count)
