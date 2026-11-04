"""Importing the chunking package or its modules must not import preocr (A13).

preocr pulls in python-magic, whose Windows loader can hang when Git's
``usr/bin`` is on PATH (the known issue in docs/DEVELOPMENT.md). The chunker imports
the ingestion package for its settings, so this guards that path too. Each
import runs in a subprocess with a timeout, as in
``tests/ingestion/test_lazy_imports.py``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_TIMEOUT_SECONDS = 30

_CHECK = (
    "import sys\n"
    "{imported}\n"
    "assert 'preocr' not in sys.modules, sorted(sys.modules)\n"
    "assert 'magic' not in sys.modules, sorted(sys.modules)\n"
)


@pytest.mark.parametrize(
    "module",
    [
        "companydocs_rag.chunking",
        "companydocs_rag.chunking.chunker",
        "companydocs_rag.chunking.cli",
    ],
)
def test_importing_does_not_import_preocr_or_magic(module):
    snippet = _CHECK.format(imported=f"import {module}")
    try:
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"'import {module}' did not return within {_TIMEOUT_SECONDS}s -- "
            "it likely imported preocr (and therefore python-magic) eagerly"
        )
    assert result.returncode == 0, result.stderr
