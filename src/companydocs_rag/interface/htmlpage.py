"""The page (spec 009, D1, D4, D5, D16, R1, R17, R18).

``page.html`` in this package is the template: one HTML document with one
inline ``<style>`` and one inline ``<script>``, ASCII only, without external
resources. :func:`build_page` fills its placeholders (``{{name}}``) with
D16's texts, escaped, and computes the Content-Security-Policy, whose
``script-src`` names the sha256 of the inline script. The script never
contains an inserted text: it reads the busy text and the server text from
the form's ``data-`` attributes.
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
