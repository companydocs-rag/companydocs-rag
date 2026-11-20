"""The page and the headers of every response (spec 009, D1, D16, R17, R18;
A4's script check, A13)."""

from __future__ import annotations

import base64
import hashlib
import re

import pytest

from companydocs_rag.ingestion.parse_pass import file_sha256
from companydocs_rag.ingestion.settings import find_project_root
from companydocs_rag.interface import texts
from companydocs_rag.interface.htmlpage import build_page
from tests.interface._helpers import (
    FakeAnswerer,
    ask,
    parse_html,
    request,
    running_server,
    write_pdf,
)

PACKAGE = find_project_root() / "src" / "companydocs_rag" / "interface"
CSP = (
    "default-src 'none'; script-src 'sha256-{hash}'; style-src 'unsafe-inline'; "
    "img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)
GENERIC_OR_SYSTEM_FAMILIES = {
    "system-ui",
    "sans-serif",
    "serif",
    "monospace",
    "ui-sans-serif",
    "ui-serif",
    "ui-monospace",
    "-apple-system",
    "blinkmacsystemfont",
    '"segoe ui"',
}
SCHEME = re.compile(r"^\s*([a-zA-Z][a-zA-Z0-9+.-]*:|//)")


@pytest.fixture
def raw(tmp_path):
    folder = tmp_path / "raw"
    write_pdf(folder / "fictief-huur.pdf", (200, 300))
    return folder


@pytest.fixture
def page(raw):
    with running_server(FakeAnswerer(), raw) as server:
        response = request(server.port, "GET", "/")
    assert response.status == 200
    return response


def _script(text):
    (script,) = re.findall("<script>(.*?)</script>", text, re.DOTALL)
    return script


def _style(text):
    (style,) = re.findall("<style>(.*?)</style>", text, re.DOTALL)
    return style


def test_the_page_is_utf8_html_in_dutch_with_d16s_texts(page):
    assert page.headers["Content-Type"] == "text/html; charset=utf-8"
    tree = parse_html(page.text)
    (html,) = tree.find_all("html")
    assert html.attrs["lang"] == "nl"
    assert tree.find("title").text() == texts.PAGE_TITLE
    assert [m.attrs.get("charset") for m in tree.find_all("meta")].count("utf-8") == 1
    (textarea,) = tree.find_all("textarea")
    assert textarea.attrs["placeholder"] == texts.INPUT_PLACEHOLDER
    form = tree.find("form")
    assert form.find("button").text() == texts.BUTTON
    assert tree.find("p", "toelichting").text() == texts.UNDER_INPUT
    assert form.attrs["data-bezig"] == texts.BUSY
    assert form.attrs["data-serverfout"] == texts.SERVER_ERROR
    assert "action" not in form.attrs


def test_the_page_has_one_inline_style_and_one_inline_script(page):
    tree = parse_html(page.text)

    assert len(tree.find_all("style")) == 1
    (script,) = tree.find_all("script")
    assert "src" not in script.attrs
    assert page.text.count("<script") == 1
    assert page.text.count("<style") == 1


def test_the_page_names_no_external_resource(page):
    tree = parse_html(page.text)
    style = _style(page.text)

    for node in tree.iter():
        for name in ("src", "href", "action"):
            value = node.attrs.get(name)
            assert value is None or SCHEME.match(value) is None, (name, value)
    for value in re.findall(r"url\(([^)]*)\)", page.text):
        assert SCHEME.match(value.strip("'\"")) is None, value
    assert "@import" not in style
    assert "@font-face" not in style
    families = re.findall(r"font-family\s*:\s*([^;}]+)", style)
    assert families
    for declaration in families:
        for family in declaration.split(","):
            assert family.strip().lower() in GENERIC_OR_SYSTEM_FAMILIES, family


def test_the_answer_text_and_the_question_keep_their_line_breaks(page):
    style = _style(page.text)

    for selector in (".antwoordtekst", ".vraag"):
        rule = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", style)
        assert rule is not None, selector
        assert re.search(r"white-space\s*:\s*pre-wrap", rule.group(1)), selector


def test_the_script_assigns_html_once_and_only_the_fragment(page):
    script = _script(page.text)

    assert script.count("innerHTML") == 1
    assert re.search(r"\.innerHTML\s*=\s*fragment\s*;", script)
    for forbidden in (
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    ):
        assert forbidden not in script, forbidden
    # The pending question is set as text.
    assert "asked.textContent = question" in script


def test_the_page_carries_the_csp_with_its_script_hash(page):
    script = _script(page.text)
    digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest())

    assert page.headers["Content-Security-Policy"] == CSP.format(
        hash=digest.decode("ascii")
    )
    assert build_page().content_security_policy == CSP.format(
        hash=digest.decode("ascii")
    )


def test_every_response_carries_r17s_headers(raw):
    digest = file_sha256(raw / "fictief-huur.pdf")
    with running_server(FakeAnswerer(), raw) as server:
        responses = {
            "page": request(server.port, "GET", "/"),
            "question": ask(server.port, "Fictieve vraag?"),
            "empty-question": ask(server.port, ""),
            "bad-question": ask(server.port, "x", content_type="text/plain"),
            "page-view": request(
                server.port, "GET", f"/pagina?sha256={digest}&nummer=1"
            ),
            "page-view-400": request(server.port, "GET", "/pagina?sha256=x&nummer=1"),
            "page-view-404": request(
                server.port, "GET", f"/pagina?sha256={'0' * 64}&nummer=1"
            ),
            "unknown-path": request(server.port, "GET", "/onbekend"),
            "post-elsewhere": request(server.port, "POST", "/", body=b""),
            "other-method": request(server.port, "PUT", "/vraag", body=b""),
        }

    statuses = {name: response.status for name, response in responses.items()}
    assert statuses == {
        "page": 200,
        "question": 200,
        "empty-question": 200,
        "bad-question": 400,
        "page-view": 200,
        "page-view-400": 400,
        "page-view-404": 404,
        "unknown-path": 404,
        "post-elsewhere": 400,
        "other-method": 501,
    }
    for name, response in responses.items():
        headers = response.headers
        assert headers["Cache-Control"] == "no-store", name
        assert headers["X-Content-Type-Options"] == "nosniff", name
        assert headers["Connection"] == "close", name
        assert int(headers["Content-Length"]) == len(response.body), name
    assert "Content-Security-Policy" in responses["page"].headers


def test_no_cloud_api_host_occurs_in_any_file_of_the_package():
    files = [path for path in PACKAGE.rglob("*") if path.is_file()]

    assert any(path.name == "page.html" for path in files)
    for path in files:
        content = path.read_bytes().decode("latin-1").lower()
        for word in ("api.openai.com", "googleapis.com", "openai.azure.com"):
            assert word not in content, (path.name, word)
