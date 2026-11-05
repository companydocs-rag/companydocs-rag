"""Packaging and source hygiene for spec 006 (A16).

``pyproject.toml`` pins the four runtime dependencies exactly, lists every
other package the new code imports directly, registers both commands and adds
no package index. The new Python files are ASCII only, and no str in them
holds a backslash escape of the u, U, N or x kind (docs/DEVELOPMENT.md, the chr() rule).
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from importlib import metadata

import pytest

from companydocs_rag.ingestion.settings import find_project_root

ROOT = find_project_root()
PINNED = ("qdrant-client", "sentence-transformers", "transformers", "torch")
NEW_SOURCE = [
    ROOT / "src" / "companydocs_rag" / package / name
    for package, names in {
        "embeddings": ["__init__.py", "dense.py", "sparse.py"],
        "vectorstore": ["__init__.py", "store.py", "manifest.py", "cli.py"],
        "retrieval": ["__init__.py", "fusion.py", "search.py", "cli.py"],
    }.items()
    for name in names
]
NEW_TESTS = sorted(
    path
    for folder in ("embeddings", "vectorstore", "retrieval")
    for path in (ROOT / "tests" / folder).glob("*.py")
)
# The distribution of each third-party module the new code imports.
DISTRIBUTIONS = {
    "qdrant_client": "qdrant-client",
    "sentence_transformers": "sentence-transformers",
    "yaml": "pyyaml",
    "pytest": "pytest",
}


def _pyproject():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _requirement_names(requirements):
    return {re.split(r"[=<>!~ \[;]", requirement)[0] for requirement in requirements}


def test_the_four_runtime_dependencies_are_pinned_to_the_installed_releases():
    requirements = _pyproject()["project"]["dependencies"]

    for name in PINNED:
        (pin,) = [item for item in requirements if item.startswith(name + "==")]
        assert pin == f"{name}=={metadata.version(name)}"


def test_every_package_the_new_code_imports_is_declared():
    project = _pyproject()
    declared = _requirement_names(project["project"]["dependencies"])
    declared |= _requirement_names(project["dependency-groups"]["dev"])
    imported = set()
    for path in NEW_SOURCE + NEW_TESTS:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add(node.module.split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - {"companydocs_rag", "tests"}

    assert third_party == set(DISTRIBUTIONS)
    assert {DISTRIBUTIONS[name] for name in third_party} <= declared


def test_both_commands_are_registered_and_no_package_index_is_added():
    project = _pyproject()
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert project["project"]["scripts"]["companydocs-index"] == (
        "companydocs_rag.vectorstore.cli:main"
    )
    assert project["project"]["scripts"]["companydocs-search"] == (
        "companydocs_rag.retrieval.cli:main"
    )
    assert "index" not in project.get("tool", {}).get("uv", {})
    assert "index-url" not in text


@pytest.mark.parametrize(
    "path", NEW_SOURCE + NEW_TESTS, ids=lambda path: path.relative_to(ROOT).as_posix()
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
