"""Imports, the client contract, the guard, characters and packaging
(spec 008, D1, D3, R1, R26; Q1, A15, A16, A17).

The import checks run in a fresh interpreter, as
``tests/embeddings/_helpers.run_import_check`` does for spec 006.
"""

from __future__ import annotations

import ast
import json
import re
import socket
import subprocess
import sys
import tomllib
from dataclasses import fields

import httpx
import pytest

from companydocs_rag.answering.generator import GeneratorClient
from companydocs_rag.ingestion.settings import find_project_root
from companydocs_rag.metadata.request import GeneratorConfig
from tests.embeddings._helpers import run_import_check
from tests.metadata._helpers import Generator, reply
from tests.metadata.test_report_cli import REPORT_MODULES

ROOT = find_project_root()
PACKAGE = ROOT / "src" / "companydocs_rag" / "metadata"
MODULES = sorted(
    "companydocs_rag.metadata" + ("" if path.stem == "__init__" else f".{path.stem}")
    for path in PACKAGE.glob("*.py")
)
REPORT_IMPORTS = sorted(
    "companydocs_rag.metadata" + ("" if name == "__init__.py" else f".{name[:-3]}")
    for name in REPORT_MODULES
)
NEW_PYTHON = sorted(
    [*PACKAGE.glob("*.py"), *(ROOT / "tests" / "metadata").glob("*.py")]
)
# The distribution of each third-party module the new code imports.
DISTRIBUTIONS = {"httpx": "httpx", "pytest": "pytest", "yaml": "pyyaml"}
D3_ATTRIBUTES = [
    "base_url",
    "connect_timeout_s",
    "read_timeout_s",
    "model",
    "temperature",
    "seed",
    "reasoning_effort",
    "extra_body",
]
NO_QDRANT = "assert 'qdrant_client' not in sys.modules, 'qdrant_client'\n"


def _settings_code(base, **metadata):
    values = {"warmup": True, "log_dir": str(base / "logs"), **metadata}
    return (
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        f"base = Path({str(base)!r})\n"
        "settings = base / 'instellingen.yaml'\n"
        f"content = json.dumps({{'metadata': {values!r}}})\n"
        "settings.write_text(content, encoding='utf-8')\n"
        "os.environ['COMPANYDOCS_RAG_SETTINGS'] = str(settings)\n"
    )


# --- Imports (A15) --------------------------------------------------------------------


@pytest.mark.parametrize("module", MODULES)
def test_importing_imports_no_model_library_preocr_or_magic(module):
    run_import_check(f"import {module}")


@pytest.mark.parametrize("module", REPORT_IMPORTS)
def test_the_modules_report_uses_leave_qdrant_out(module):
    run_import_check(f"import {module}\n{NO_QDRANT}")


def test_running_extract_imports_no_model_library(tmp_path):
    run_import_check(
        _settings_code(tmp_path)
        + "from tests.metadata._helpers import Generator, chunk_record, "
        "lease_text, write_json_file, write_overrides\n"
        "from companydocs_rag.metadata import cli\n"
        "write_json_file(base / 'chunks', 'huur.json', chunk_record(base / 'los' / "
        "'fictief-huur.pdf', 'a1' * 32, lease_text()))\n"
        "overrides = write_overrides(base / 'handmatig' / 'types.csv', "
        "[('fictief-huur.pdf', 'Huurovereenkomst')])\n"
        "generator = Generator()\n"
        "code = cli.main(['extract', '--chunks-dir', str(base / 'chunks'), "
        "'--extracted-dir', str(base / 'extracted'), '--type-overrides', "
        "str(overrides), '--relations', str(base / 'geen.csv')], "
        "transport=generator.transport)\n"
        "assert code == 0, code\n"
        "assert len(generator.requests) == 2, generator.requests\n"
    )


def test_running_report_imports_neither_a_model_library_nor_qdrant(tmp_path):
    code = (
        _settings_code(tmp_path)
        + "from tests.metadata._helpers import extraction_record, write_json_file\n"
        "from companydocs_rag.metadata import cli\n"
        "write_json_file(base / 'extracted', 'a.json', "
        "extraction_record('bron/fictief-a.pdf', 'a1' * 32))\n"
        "code = cli.main(['report', '--reference-date', '2031-09-01', "
        "'--extracted-dir', str(base / 'extracted'), '--report-dir', "
        "str(base / 'rapport')])\n"
        "assert code == 0, code\n" + NO_QDRANT
    )
    run_import_check(code)


def test_running_report_loads_only_the_modules_report_uses(tmp_path):
    code = (
        "import sys\n"
        + _settings_code(tmp_path)
        + "from tests.metadata._helpers import extraction_record, write_json_file\n"
        "from companydocs_rag.metadata import cli\n"
        "write_json_file(base / 'extracted', 'a.json', "
        "extraction_record('bron/fictief-a.pdf', 'a1' * 32))\n"
        "assert cli.main(['report', '--reference-date', '2031-09-01', "
        "'--extracted-dir', str(base / 'extracted'), '--report-dir', "
        "str(base / 'rapport')]) == 0\n"
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.startswith('companydocs_rag.metadata'))))\n"
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.startswith('companydocs_rag.answering'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stderr
    metadata_modules, answering_modules = (
        json.loads(line) for line in result.stdout.strip().splitlines()[-2:]
    )
    assert metadata_modules == REPORT_IMPORTS
    assert answering_modules == []


# --- The client contract (A16) --------------------------------------------------------


class Recording:
    """A configuration object that records every attribute read from it."""

    def __init__(self, values):
        self.__dict__["_values"] = values
        self.__dict__["read"] = []

    def __getattr__(self, name):
        self.__dict__["read"].append(name)
        return self.__dict__["_values"][name]


def test_generator_client_reads_exactly_d3s_attributes():
    recording = Recording(
        {
            "base_url": "http://127.0.0.1:11434/v1",
            "connect_timeout_s": 5.0,
            "read_timeout_s": 600.0,
            "model": "fictief-model",
            "temperature": 0.0,
            "seed": 0,
            "reasoning_effort": "high",
            "extra_body": {"keep_alive": "60m"},
        }
    )
    generator = Generator(reply("fictief"))
    client = GeneratorClient(recording, api_key=None, transport=generator.transport)
    try:
        client.complete([{"role": "user", "content": "ok"}], max_tokens=1)
    finally:
        client.close()

    assert len(generator.requests) == 1
    assert set(recording.read) == set(D3_ATTRIBUTES)


def test_the_metadata_dataclass_has_exactly_d3s_attributes_in_order():
    assert [item.name for item in fields(GeneratorConfig)] == D3_ATTRIBUTES


# --- No server (the guard in conftest.py, A17) ----------------------------------------


def test_a_request_past_the_mock_transport_fails_its_test():
    with pytest.raises(pytest.fail.Exception, match="past the injected"):
        httpx.Client().post("http://127.0.0.1:11434/v1/chat/completions", json={})
    with pytest.raises(pytest.fail.Exception, match="past the injected"):
        socket.create_connection(("127.0.0.1", 11434), timeout=1)


def test_extract_without_a_transport_cannot_reach_a_server(tmp_path):
    from companydocs_rag.metadata import cli
    from tests.metadata._helpers import (
        LEASE,
        chunk_record,
        lease_text,
        write_json_file,
        write_overrides,
    )

    write_json_file(
        tmp_path / "chunks",
        "huur.json",
        chunk_record(tmp_path / "los" / "fictief-huur.pdf", "a1" * 32, lease_text()),
    )
    overrides = write_overrides(
        tmp_path / "handmatig" / "types.csv", [("fictief-huur.pdf", LEASE)]
    )

    with pytest.raises(pytest.fail.Exception, match="past the injected"):
        cli.main(
            [
                "extract",
                "--chunks-dir",
                str(tmp_path / "chunks"),
                "--extracted-dir",
                str(tmp_path / "extracted"),
                "--type-overrides",
                str(overrides),
                "--relations",
                str(tmp_path / "geen.csv"),
            ]
        )


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


# --- Packaging (A17) ------------------------------------------------------------------


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

    assert project["scripts"]["companydocs-metadata"] == (
        "companydocs_rag.metadata.cli:main"
    )
    assert len(project["dependencies"]) == 10


def test_the_defaults_are_the_external_surfaces(monkeypatch, tmp_path):
    from companydocs_rag.metadata import cli
    from companydocs_rag.metadata import settings as metadata_settings_module

    monkeypatch.setattr(metadata_settings_module, "project_base", lambda: tmp_path)
    parser = cli.build_parser()

    extract = parser.parse_args(["extract"])
    assert extract.chunks_dir == tmp_path / "data" / "interim" / "chunks"
    assert extract.extracted_dir == (
        tmp_path / "data" / "processed" / "metadata" / "extracted"
    )
    assert extract.type_overrides == (
        tmp_path / "data" / "manual" / "document_type_overrides.csv"
    )
    assert extract.relations == tmp_path / "data" / "manual" / "relations.csv"
    report = parser.parse_args(["report", "--reference-date", "2031-09-01"])
    assert report.extracted_dir == extract.extracted_dir
    assert report.report_dir == tmp_path / "data" / "processed" / "metadata"
    assert str(report.reference_date) == "2031-09-01"


def test_the_package_init_holds_a_docstring_only():
    tree = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))

    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.Expr)
    assert isinstance(tree.body[0].value, ast.Constant)
