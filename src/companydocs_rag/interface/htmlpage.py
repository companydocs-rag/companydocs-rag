"""The page (spec 009, D1, D4, D5, D16, R1, R17, R18; spec 011).

``page.html`` in this package is the template: one HTML document with one
inline ``<style>`` and one inline ``<script>``, ASCII only, without external
resources. :func:`build_page` fills its placeholders (``{{name}}``) with
D16's and spec 011's texts, escaped, and computes the
Content-Security-Policy, whose ``script-src`` names the sha256 of the inline
script. The script never contains an inserted text: it reads the busy text
and the server text from the form's ``data-`` attributes, the loading and
not-found texts from the body's, and every other text it shows comes from
the JSON views (:mod:`.library`), set with ``textContent``.
"""

from __future__ import annotations

import base64
import hashlib
import html
import re
from dataclasses import dataclass
from importlib import resources

from . import texts

TEMPLATE_NAME = "page.html"
PLACEHOLDERS = {
    "{{titel}}": texts.PAGE_TITLE,
    "{{invoertekst}}": texts.INPUT_PLACEHOLDER,
    "{{knop}}": texts.BUTTON,
    "{{toelichting}}": texts.UNDER_INPUT,
    "{{bezig}}": texts.BUSY,
    "{{serverfout}}": texts.SERVER_ERROR,
    # Spec 011: the side bar.
    "{{merk}}": texts.BRAND,
    "{{merk_sub}}": texts.BRAND_SUBTITLE,
    "{{nav_documenten}}": texts.NAV_DOCUMENTS,
    "{{status_kop}}": texts.STATUS_HEADING,
    "{{account_naam}}": texts.ACCOUNT_NAME,
    "{{account_sub}}": texts.ACCOUNT_SUBTITLE,
    "{{account_initialen}}": texts.ACCOUNT_INITIALS,
    # Spec 011: the chat view.
    "{{nieuwe_chat}}": texts.NEW_CHAT,
    "{{leeg_titel}}": texts.CHAT_EMPTY_TITLE,
    "{{leeg_tekst}}": texts.CHAT_EMPTY_TEXT,
    # Spec 011: the documents view.
    "{{zoek}}": texts.DOCUMENTS_SEARCH,
    "{{geen_documenten}}": texts.DOCUMENTS_NONE,
    "{{upload}}": texts.UPLOAD_BUTTON,
    "{{upload_uitleg}}": texts.UPLOAD_TOOLTIP,
    # Spec 011: the chat history and projects.
    "{{projecten}}": texts.PROJECTS_HEADING,
    "{{chats}}": texts.CHATS_HEADING,
    "{{nieuw_project}}": texts.NEW_PROJECT,
    "{{project_naam}}": texts.PROJECT_NAME,
    "{{geen_chats}}": texts.NO_CHATS,
    "{{project_leeg}}": texts.PROJECT_EMPTY,
    "{{geschiedenis_noot}}": texts.HISTORY_NOTE,
    "{{opties}}": texts.MENU_OPTIONS,
    "{{naam_wijzigen}}": texts.MENU_RENAME,
    "{{verplaatsen}}": texts.MENU_MOVE,
    "{{geen_project}}": texts.MENU_NO_PROJECT,
    "{{verwijderen}}": texts.MENU_DELETE,
    "{{nieuwe_chat_project}}": texts.MENU_NEW_CHAT_IN_PROJECT,
    "{{bevestig_chat}}": texts.CONFIRM_DELETE_CHAT,
    "{{bevestig_project}}": texts.CONFIRM_DELETE_PROJECT,
    # Spec 011: the page viewer.
    "{{viewer_leeg_titel}}": texts.VIEWER_EMPTY_TITLE,
    "{{viewer_leeg_tekst}}": texts.VIEWER_EMPTY_TEXT,
    "{{laden}}": texts.VIEWER_LOADING,
    "{{niet_gevonden}}": texts.PAGE_NOT_FOUND,
    "{{geciteerd}}": texts.VIEWER_CITED,
    "{{markering}}": texts.VIEWER_HIGHLIGHT,
    "{{vorige}}": texts.VIEWER_PREVIOUS,
    "{{volgende}}": texts.VIEWER_NEXT,
    "{{paginanummer}}": texts.VIEWER_PAGE,
    "{{uitzoomen}}": texts.VIEWER_ZOOM_OUT,
    "{{inzoomen}}": texts.VIEWER_ZOOM_IN,
    "{{passend}}": texts.VIEWER_FIT,
    "{{nieuw_tabblad}}": texts.VIEWER_OPEN_TAB,
    "{{sluiten}}": texts.VIEWER_CLOSE,
    "{{sleep}}": texts.VIEWER_RESIZE,
}
CSP_TEMPLATE = (
    "default-src 'none'; script-src 'sha256-{script_hash}'; "
    "style-src 'unsafe-inline'; img-src 'self'; connect-src 'self'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)

_SCRIPT = re.compile("<script>(.*?)</script>", re.DOTALL)
_LEFT_OVER = "{{"


@dataclass(frozen=True)
class Page:
    """The filled page as UTF-8 bytes and its Content-Security-Policy."""

    html: bytes
    content_security_policy: str


def script_hash(script: str) -> str:
    """The CSP hash source value of an inline script: base64 of its sha256."""
    digest = hashlib.sha256(script.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def build_page() -> Page:
    """Read the template, fill it and compute its CSP (R17)."""
    template = (
        resources.files(__package__).joinpath(TEMPLATE_NAME).read_text(encoding="utf-8")
    )
    filled = template
    for placeholder, text in PLACEHOLDERS.items():
        filled = filled.replace(placeholder, html.escape(text, quote=True))
    if _LEFT_OVER in filled:
        raise ValueError(f"{TEMPLATE_NAME} holds a placeholder without a text")
    scripts = _SCRIPT.findall(filled)
    if len(scripts) != 1:
        raise ValueError(f"{TEMPLATE_NAME} holds {len(scripts)} scripts, not 1")
    return Page(
        html=filled.encode("utf-8"),
        content_security_policy=CSP_TEMPLATE.format(
            script_hash=script_hash(scripts[0])
        ),
    )
