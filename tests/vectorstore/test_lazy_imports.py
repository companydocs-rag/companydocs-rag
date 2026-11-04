"""Importing the vectorstore modules imports no model library (D26, R15, A16).

Each import runs in a fresh interpreter with a timeout, as in
``tests/chunking/test_lazy_imports.py``.
"""

from __future__ import annotations

import pytest

from tests.embeddings._helpers import run_import_check


@pytest.mark.parametrize(
    "module",
    [
        "companydocs_rag.vectorstore",
        "companydocs_rag.vectorstore.store",
        "companydocs_rag.vectorstore.manifest",
        "companydocs_rag.vectorstore.cli",
    ],
)
def test_importing_imports_no_model_library_preocr_or_magic(module):
    run_import_check(f"import {module}")
