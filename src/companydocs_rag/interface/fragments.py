"""The HTML fragment of one question and its answer (spec 009, D7-D10,
R7-R11; Q1, Q5, Q9).

A fragment is one ``section`` with the question, then the answer or one D16
message. Every inserted text (the question, the answer text, the document
type, the file name, the clause label) goes through
``html.escape(..., quote=True)``, also inside attribute values. Element ids
and link URLs are built only from integers and a hash that matches R12's
pattern. The answer text keeps its line breaks through the page's CSS
(``white-space: pre-wrap``); it is never rendered as Markdown.

- ``answered``: the answer text with its citations linked, then one card per
  entry of ``sources``, in order of n, under "Bronnen".
- ``insufficient_evidence`` and ``uncited``: a banner of its own class, then
  the answer text when it is not null; no card, and every ``[n]`` stays text.
- ``error``: D16's server text for ``context_budget_exceeded``, else D16's
  generator-error text; no answer text and no card.

A citation (R9) is spec 007's form: ``[``, one or more ASCII-digit numbers
separated by commas with optional spaces, ``]``. Each number that has a card
on this answer links to that card; everything else stays text.

Card ids are ``bron-<a>-<n>``, with ``<a>`` the server's answer counter and
``<n>`` the source's number.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from companydocs_rag.answering.answerer import (
    ERROR_CONTEXT_BUDGET_EXCEEDED,
    AnswerResult,
)
from companydocs_rag.answering.parsing import (
    STATUS_ANSWERED,
    STATUS_ERROR,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_UNCITED,
)
from companydocs_rag.ingestion.quality import (
    TEXT_ORIGIN_NATIVE,
    TEXT_ORIGIN_NONE,
    TEXT_ORIGIN_OCR,
    TIER_CLEAN,
    TIER_DECENT,
    TIER_GARBAGE,
)

from . import texts
from .pageview import SHA256_PATTERN, is_page_number, page_url

CLASS_EXCHANGE = "uitwisseling"
CLASS_QUESTION = "vraag"
CLASS_ANSWER = "antwoord"
CLASS_ANSWER_TEXT = "antwoordtekst"
CLASS_MESSAGE = "melding"
CLASS_BANNER_INSUFFICIENT_EVIDENCE = "banner-onvoldoende"
CLASS_BANNER_UNCITED = "banner-ongeciteerd"
CLASS_CITATION = "citaat"
CLASS_SOURCES = "bronnen"
CLASS_SOURCES_HEADING = "bronnen-kop"
CLASS_CARD = "bron"
CLASS_CARD_HEAD = "bron-kop"
CLASS_CARD_NUMBER = "bron-nummer"
CLASS_CARD_TYPE = "bron-type"
CLASS_CARD_FILE = "bron-bestand"
CLASS_CARD_HINT = "bron-hint"
CLASS_CARD_LABEL = "bron-label"
CLASS_CARD_PAGES = "bron-paginas"
CLASS_PAGE_NUMBER = "paginanummer"
CLASS_PROVENANCE = "herkomst"
CLASS_PAGE_LINK = "paginalink"

_CITATION = re.compile(r"\[([0-9]+(?: *, *[0-9]+)*)\]")
_NUMBER = re.compile("[0-9]+")
_BANNERS = {
    STATUS_INSUFFICIENT_EVIDENCE: (
        CLASS_BANNER_INSUFFICIENT_EVIDENCE,
        texts.BANNER_INSUFFICIENT_EVIDENCE,
    ),
    STATUS_UNCITED: (CLASS_BANNER_UNCITED, texts.BANNER_UNCITED),
}


def escape(text: str) -> str:
    """R8's escaping: ``html.escape`` with quotes, for text and attributes."""
    return html.escape(text, quote=True)


# --- D10: provenance labels (R11) -----------------------------------------------------


def provenance_label(text_origin: Any, tier: Any, page: Any) -> str:
    """D10's label of one page span, ``{N}`` filled with ``page``: a pure
    function of ``text_origin`` and ``tier``, compared as exact,
    case-sensitive strings (R11). A value that is not a string, or a key the
    caller did not find (None), is "anything else"."""
    origin = text_origin if isinstance(text_origin, str) else None
    level = tier if isinstance(tier, str) else None
    if origin == TEXT_ORIGIN_NATIVE:
        if level in (TIER_CLEAN, TIER_DECENT):
            return texts.LABEL_TEXT_LAYER
        if level == TIER_GARBAGE:
            return texts.label_low_quality(page)
        return texts.label_unknown(page)
    if origin == TEXT_ORIGIN_OCR:
        return texts.label_scan(page)
    if origin == TEXT_ORIGIN_NONE:
        return texts.LABEL_NO_TEXT
    return texts.label_unknown(page)


def span_label(entry: Mapping[str, Any]) -> str:
    """D10's label of one entry of a source's ``pages``; a missing key reads
    as None."""
    return provenance_label(
        entry.get("text_origin"), entry.get("tier"), entry.get("page")
    )


# --- Citations (D7, R9) ---------------------------------------------------------------


def card_id(answer_number: int, prompt_number: int) -> str:
    """R10's card id ``bron-<a>-<n>``, from two integers only."""
    if type(answer_number) is not int or type(prompt_number) is not int:
        raise ValueError(
            f"card ids are built from integers, not {answer_number!r}, "
            f"{prompt_number!r}"
        )
    return f"bron-{answer_number}-{prompt_number}"


def _citation_html(citation: str, card_ids: Mapping[int, str]) -> str:
    parts: List[str] = []
    position = 0
    for number in _NUMBER.finditer(citation):
        parts.append(escape(citation[position : number.start()]))
        digits = number.group(0)
        target = card_ids.get(int(digits))
        if target is None:
            parts.append(escape(digits))
        else:
            parts.append(
                f'<a class="{CLASS_CITATION}" href="#{target}">{escape(digits)}</a>'
            )
        position = number.end()
    parts.append(escape(citation[position:]))
    return "".join(parts)


def answer_text_html(text: str, card_ids: Mapping[int, str]) -> str:
    """The answer text, escaped, with each cited number that has a card
    linked to it (R9). The text's characters are unchanged."""
    parts: List[str] = []
    position = 0
    for match in _CITATION.finditer(text):
        parts.append(escape(text[position : match.start()]))
        parts.append(_citation_html(match.group(0), card_ids))
        position = match.end()
    parts.append(escape(text[position:]))
    return "".join(parts)


# --- Cards (D9, R10) ------------------------------------------------------------------


def _page_line(entry: Mapping[str, Any], mapped_sha256: Optional[str]) -> str:
    page = entry.get("page")
    line = (
        f'<li><span class="{CLASS_PAGE_NUMBER}">{escape(str(page))}</span>: '
        f'<span class="{CLASS_PROVENANCE}">{escape(span_label(entry))}</span>'
    )
    if mapped_sha256 is not None and is_page_number(page):
        line += (
            f' <a class="{CLASS_PAGE_LINK}" '
            f'href="{escape(page_url(mapped_sha256, page))}" '
            'target="_blank" rel="noopener noreferrer">'
            f"{escape(texts.page_link(page))}</a>"
        )
    return line + "</li>"


def _hint_html(label: str) -> str:
    before, _, after = texts.CLAUSE_LABEL.partition(texts.LABEL_PLACEHOLDER)
    return (
        f'<div class="{CLASS_CARD_HINT}">{escape(before)}'
        f'<span class="{CLASS_CARD_LABEL}">{escape(label)}</span>{escape(after)}</div>'
    )


def card_html(
    source: Mapping[str, Any],
    answer_number: int,
    files: Mapping[str, Path],
    types: Mapping[str, Optional[str]],
) -> str:
    """One source's card (D9, R10; Q9): ``[n]``, the document type or "type
    onbekend", the mapped file's name, the clause label as a hint, and one
    line per page with its D10 label and, when the file is mapped, its
    link."""
    number = source["prompt_number"]
    file_sha256 = source.get("file_sha256")
    mapped = (
        file_sha256
        if isinstance(file_sha256, str)
        and SHA256_PATTERN.fullmatch(file_sha256) is not None
        and file_sha256 in files
        else None
    )
    document_type = types.get(mapped) if mapped is not None else None
    parts = [
        f'<div class="{CLASS_CARD}" id="{card_id(answer_number, number)}">',
        f'<div class="{CLASS_CARD_HEAD}">',
        f'<span class="{CLASS_CARD_NUMBER}">[{escape(str(number))}]</span> ',
        f'<span class="{CLASS_CARD_TYPE}">',
        escape(texts.NO_DOCUMENT_TYPE if document_type is None else document_type),
        "</span></div>",
    ]
    if mapped is not None:
        parts.append(
            f'<div class="{CLASS_CARD_FILE}">{escape(files[mapped].name)}</div>'
        )
    label = source.get("clause_label")
    if label is not None:
        parts.append(_hint_html(str(label)))
    parts.append(f'<ul class="{CLASS_CARD_PAGES}">')
    parts.extend(_page_line(entry, mapped) for entry in source.get("pages") or [])
    parts.append("</ul></div>")
    return "".join(parts)


def _sources_html(
    sources: Sequence[Mapping[str, Any]],
    answer_number: int,
    files: Mapping[str, Path],
    types: Mapping[str, Optional[str]],
) -> str:
    cards = "".join(card_html(s, answer_number, files, types) for s in sources)
    return (
        f'<div class="{CLASS_SOURCES}">'
        f'<h3 class="{CLASS_SOURCES_HEADING}">{escape(texts.SOURCES_HEADING)}</h3>'
        f"{cards}</div>"
    )


# --- Fragments (R7) -------------------------------------------------------------------


def _exchange(question: str, answer: str) -> str:
    return (
        f'<section class="{CLASS_EXCHANGE}">'
        f'<div class="{CLASS_QUESTION}">{escape(question)}</div>'
        f'<div class="{CLASS_ANSWER}">{answer}</div>'
        "</section>"
    )


def message_fragment(question: str, message: str) -> str:
    """The question, then one D16 message."""
    return _exchange(question, f'<p class="{CLASS_MESSAGE}">{escape(message)}</p>')


def result_fragment(
    question: str,
    result: AnswerResult,
    answer_number: int,
    files: Mapping[str, Path],
    types: Mapping[str, Optional[str]],
) -> str:
    """The question, then the answer of ``result`` (D7-D9, Q5). Raises
    ``ValueError`` for a status the interface does not know."""
    status = result.status
    if status == STATUS_ERROR:
        error = result.error if isinstance(result.error, Mapping) else {}
        message = (
            texts.SERVER_ERROR
            if error.get("type") == ERROR_CONTEXT_BUDGET_EXCEEDED
            else texts.GENERATOR_ERROR
        )
        return message_fragment(question, message)
    if status != STATUS_ANSWERED and status not in _BANNERS:
        raise ValueError(f"unknown status {status!r}")

    sources = (
        sorted(result.sources, key=lambda source: source["prompt_number"])
        if status == STATUS_ANSWERED
        else []
    )
    card_ids: Dict[int, str] = {
        source["prompt_number"]: card_id(answer_number, source["prompt_number"])
        for source in sources
    }
    parts: List[str] = []
    if status in _BANNERS:
        banner_class, banner_text = _BANNERS[status]
        parts.append(f'<div class="banner {banner_class}">{escape(banner_text)}</div>')
    if result.answer_text is not None:
        parts.append(
            f'<div class="{CLASS_ANSWER_TEXT}">'
            f"{answer_text_html(result.answer_text, card_ids)}</div>"
        )
    if sources:
        parts.append(_sources_html(sources, answer_number, files, types))
    return _exchange(question, "".join(parts))
