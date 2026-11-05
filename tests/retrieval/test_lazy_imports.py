"""Importing the retrieval modules, and indexing and searching with the stub,
import no model library (D26, R15, A16).

Each check runs in a fresh interpreter with a timeout, as in
``tests/chunking/test_lazy_imports.py``.
"""

from __future__ import annotations

import pytest

from tests.embeddings._helpers import run_import_check


@pytest.mark.parametrize(
    "module",
    [
        "companydocs_rag.retrieval",
        "companydocs_rag.retrieval.fusion",
        "companydocs_rag.retrieval.search",
        "companydocs_rag.retrieval.cli",
    ],
)
def test_importing_imports_no_model_library_preocr_or_magic(module):
    run_import_check(f"import {module}")


def test_indexing_and_searching_with_the_stub_import_no_model_library(tmp_path):
    run_import_check(
        "from pathlib import Path\n"
        "from tests.embeddings._helpers import StubEmbedder\n"
        "from tests.vectorstore._helpers import run_index, three_documents\n"
        "from companydocs_rag.retrieval import cli\n"
        "from companydocs_rag.retrieval.search import Retriever\n"
        f"base = Path({str(tmp_path)!r})\n"
        "three_documents(base / 'chunks')\n"
        "assert run_index(base / 'chunks', base / 'index', '-q') == 0\n"
        "with Retriever(base / 'index', embedder=StubEmbedder()) as retriever:\n"
        "    assert retriever.search('artikel 2 huurprijs').candidates\n"
        "code = cli.main(['huurprijs', '--index-dir', str(base / 'index'), '--json'],"
        " embedder=StubEmbedder())\n"
        "assert code == 0\n"
    )
