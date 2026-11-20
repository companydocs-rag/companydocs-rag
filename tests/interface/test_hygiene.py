"""Imports, the guard, characters and packaging (spec 009, R19; A16).

The import checks run in a fresh interpreter, as
``tests/embeddings/_helpers.run_import_check`` does for spec 006. The git
diffs of A16 and ``uv lock --locked`` are checked outside the suite, as
specs 007 and 008 checked theirs.
"""

from __future__ import annotations

import ast
import re
import socket
import sys
import tomllib

import httpx
import pytest

from companydocs_rag.ingestion.settings import find_project_root
from tests.embeddings._helpers import run_import_check
from tests.interface._helpers import FakeAnswerer, ask, running_server

ROOT = find_project_root()
PACKAGE = ROOT / "src" / "companydocs_rag" / "interface"
MODULES = sorted(
    "companydocs_rag.interface" + ("" if path.stem == "__init__" else f".{path.stem}")
    for path in PACKAGE.glob("*.py")
)
NEW_PYTHON = sorted(
    [*PACKAGE.glob("*.py"), *(ROOT / "tests" / "interface").glob("*.py")]
)
# The distribution of each third-party module the new code imports.
DISTRIBUTIONS = {
    "httpx": "httpx",
    "pytest": "pytest",
    "PIL": "pillow",
    "pdfplumber": "pdfplumber",
    "qdrant_client": "qdrant-client",
    "yaml": "pyyaml",
}


# --- Imports --------------------------------------------------------------------------


def test_the_package_has_the_modules_r1_names():
    names = {path.name for path in PACKAGE.iterdir() if path.is_file()}

    assert {"__init__.py", "cli.py", "settings.py", "texts.py", "page.html"} <= names


@pytest.mark.parametrize("module", MODULES)
def test_importing_imports_no_model_library_preocr_or_magic(module):
    run_import_check(f"import {module}")


def test_running_main_up_to_the_url_and_the_stop_imports_no_model_library(tmp_path):
    run_import_check(
        "import io\n"
        "import json\n"
        "import os\n"
        "import socket\n"
        "import threading\n"
        "from pathlib import Path\n"
        f"base = Path({str(tmp_path)!r})\n"
        "settings = base / 'instellingen.yaml'\n"
        "settings.write_text(json.dumps({'answering': {'warmup': True,\n"
        "    'log_dir': str(base / 'logs')}}), encoding='utf-8')\n"
        "os.environ['COMPANYDOCS_RAG_SETTINGS'] = str(settings)\n"
        "from tests.answering._helpers import FakeRetriever, Generator, "
        "simple_result, write_manifest\n"
        "from tests.interface._helpers import write_pdf\n"
        "from companydocs_rag.interface import cli\n"
        "write_manifest(base / 'index')\n"
        "write_pdf(base / 'raw' / 'fictief.pdf')\n"
        "probe = socket.socket()\n"
        "probe.bind(('127.0.0.1', 0))\n"
        "port = probe.getsockname()[1]\n"
        "probe.close()\n"
        "stop = threading.Event()\n"
        "stop.set()\n"
        "stdout = io.StringIO()\n"
        "real_stdout, sys.stdout = sys.stdout, stdout\n"
        "generator = Generator()\n"
        "try:\n"
        "    code = cli.main(\n"
        "        ['--index-dir', str(base / 'index'), '--port', str(port)],\n"
        "        raw_dir=base / 'raw', type_overrides=base / 'geen.csv',\n"
        "        chunks_dir=base / 'geen-chunks', ocr_cache_dir=base / 'geen-ocr',\n"
        "        retriever=FakeRetriever(simple_result()),\n"
        "        transport=generator.transport, stop=stop)\n"
        "finally:\n"
        "    sys.stdout = real_stdout\n"
        "assert code == 0, code\n"
        "url = f'http://127.0.0.1:{port}/\\n'\n"
        "assert stdout.getvalue() == url, stdout.getvalue()\n"
        "assert len(generator.requests) == 1\n"
    )


# --- No generator (the guard in conftest.py) ------------------------------------------


def test_a_request_through_a_real_httpx_transport_fails_its_test():
    with pytest.raises(pytest.fail.Exception, match="past the injected"):
        httpx.Client().post("http://127.0.0.1:11434/v1/chat/completions", json={})
    with pytest.raises(pytest.fail.Exception, match="past the injected"):
        socket.create_connection(("127.0.0.1", 11434), timeout=1)


def test_the_server_under_test_is_reachable_on_loopback(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()

    with running_server(FakeAnswerer(), raw) as server:
        assert ask(server.port, "Fictieve vraag?").status == 200


# --- Characters (docs/DEVELOPMENT.md, the chr() rule) ---------------------------------


@pytest.mark.parametrize(
    "path", NEW_PYTHON, ids=lambda path: path.relative_to(ROOT).as_posix()
)
def test_new_python_files_are_ascii_without_escapes_in_strings(path):
    source = path.read_text(encoding="utf-8")
    backslash = chr(0x5C)  # U+005C REVERSE SOLIDUS

    assert source.isascii()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            segment = ast.get_source_segment(source, node) or ""
            for escape in ("u", "U", "N{", "x"):
                assert backslash + escape not in segment, segment


def test_the_page_template_is_ascii():
    assert (PACKAGE / "page.html").read_bytes().isascii()


# --- Packaging ------------------------------------------------------------------------


def _pyproject():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _requirement_names(requirements):
    return {re.split(r"[=<>!~ \[;]", requirement)[0] for requirement in requirements}


def test_every_package_the_new_code_imports_is_declared():
    project = _pyproject()
    declared = _requirement_names(project["project"]["dependencies"])
    declared |= _requirement_names(project["dependency-groups"]["dev"])
    imported = set()
    for path in NEW_PYTHON:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add(node.module.split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - {"companydocs_rag", "tests"}

    assert third_party == set(DISTRIBUTIONS)
    assert {DISTRIBUTIONS[name] for name in third_party} <= declared


def test_pyproject_registers_the_command_and_adds_no_dependency():
    project = _pyproject()["project"]

    assert (
        project["scripts"]["companydocs-demo"] == "companydocs_rag.interface.cli:main"
    )
    assert len(project["dependencies"]) == 10


def test_the_package_init_holds_a_docstring_only():
    tree = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))

    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.Expr)
    assert isinstance(tree.body[0].value, ast.Constant)
