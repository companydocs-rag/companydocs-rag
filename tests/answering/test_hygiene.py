"""Imports, the guard, characters and packaging (spec 007, Constraints, A15).

The import checks run in a fresh interpreter, as
``tests/embeddings/_helpers.run_import_check`` does for spec 006.
"""

from __future__ import annotations

import ast
import re
import socket
import sys
import tomllib

import httpx
import pytest

from companydocs_rag.answering.answerer import Answerer
from companydocs_rag.ingestion.settings import find_project_root
from tests.answering._helpers import (
    FakeRetriever,
    settings_for,
    simple_result,
    write_manifest,
)
from tests.embeddings._helpers import run_import_check

ROOT = find_project_root()
MODULES = [
    "companydocs_rag.answering",
    "companydocs_rag.answering.answerer",
    "companydocs_rag.answering.settings",
    "companydocs_rag.answering.cli",
    "companydocs_rag.answering.prompt",
    "companydocs_rag.answering.generator",
    "companydocs_rag.answering.parsing",
    "companydocs_rag.answering.provenance",
    "companydocs_rag.answering.log",
    "companydocs_rag.answering.questions",
]
NEW_PYTHON = sorted(
    [
        *(ROOT / "src" / "companydocs_rag" / "answering").glob("*.py"),
        *(ROOT / "tests" / "answering").glob("*.py"),
        ROOT / "tests" / "retrieval" / "test_path_hits.py",
    ]
)
# The distribution of each third-party module the new code imports.
DISTRIBUTIONS = {"httpx": "httpx", "pytest": "pytest", "yaml": "pyyaml"}


# --- Imports --------------------------------------------------------------------------


@pytest.mark.parametrize("module", MODULES)
def test_importing_imports_no_model_library_preocr_or_magic(module):
    run_import_check(f"import {module}")


def test_running_the_command_with_the_fakes_imports_no_model_library(tmp_path):
    run_import_check(
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        f"base = Path({str(tmp_path)!r})\n"
        "settings = base / 'instellingen.yaml'\n"
        "settings.write_text(json.dumps({'answering': {'warmup': True,\n"
        "    'log_dir': str(base / 'logs')}}), encoding='utf-8')\n"
        "os.environ['COMPANYDOCS_RAG_SETTINGS'] = str(settings)\n"
        "from tests.answering._helpers import FakeRetriever, Generator, "
        "simple_result, write_manifest\n"
        "from companydocs_rag.answering import cli\n"
        "write_manifest(base / 'index')\n"
        "generator = Generator()\n"
        "code = cli.main(['Fictieve vraag?', '--index-dir', str(base / 'index')],\n"
        "    retriever=FakeRetriever(simple_result()), transport=generator.transport)\n"
        "assert code == 0, code\n"
        "assert len(generator.requests) == 2\n"
        "assert len(list((base / 'logs').iterdir())) == 1\n"
    )


# --- No server (the guard in conftest.py) ---------------------------------------------


def test_a_request_past_the_mock_transport_fails_its_test():
    with pytest.raises(pytest.fail.Exception, match="past the injected"):
        httpx.Client().post("http://127.0.0.1:11434/v1/chat/completions", json={})
    with pytest.raises(pytest.fail.Exception, match="past the injected"):
        socket.create_connection(("127.0.0.1", 11434), timeout=1)


def test_an_answerer_without_a_transport_cannot_reach_a_server(tmp_path):
    write_manifest(tmp_path / "index")
    retriever = FakeRetriever(simple_result())

    with pytest.raises(pytest.fail.Exception, match="past the injected"):
        Answerer(
            tmp_path / "index",
            settings=settings_for(tmp_path, warmup=True),
            retriever=retriever,
        )

    assert retriever.closed == 1


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


# --- Packaging ------------------------------------------------------------------------


def _requirement_names(requirements):
    return {re.split(r"[=<>!~ \[;]", requirement)[0] for requirement in requirements}


def test_every_package_the_new_code_imports_is_declared():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
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
