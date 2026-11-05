"""Importing the package and the embeddings modules imports no model library
(D26, R15, A16).

torch, transformers and sentence-transformers are imported only when
``SentenceTransformerEmbedder`` is constructed; preocr and python-magic never
(the known issue in docs/DEVELOPMENT.md: ``sparse.py`` loads the ingestion package).
Each import runs in a fresh interpreter with a timeout, as in
``tests/chunking/test_lazy_imports.py``.
"""

from __future__ import annotations

import pytest

from tests.embeddings._helpers import run_import_check


@pytest.mark.parametrize(
    "module",
    [
        "companydocs_rag",
        "companydocs_rag.embeddings",
        "companydocs_rag.embeddings.dense",
        "companydocs_rag.embeddings.sparse",
    ],
)
def test_importing_imports_no_model_library_preocr_or_magic(module):
    run_import_check(f"import {module}")
