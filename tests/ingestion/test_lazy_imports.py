"""Fix: importing the ingestion package (or its submodules) must not import preocr.

preocr pulls in python-magic, whose Windows loader can hang indefinitely when
a directory containing a conflicting ``msys-magic-1.dll`` (Git for Windows'
``usr/bin``) is on PATH -- see the module docstring of
:mod:`companydocs_rag.ingestion.quality`. Every import checked here is
therefore run in a subprocess: a regression would hang that subprocess rather
than corrupting ``sys.modules`` for the rest of the test session, and a bounded
timeout turns "hangs forever" into a normal, visible test failure instead of
wedging the whole suite.
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


def _assert_import_does_not_pull_in_preocr(import_statement: str) -> None:
    snippet = _CHECK.format(imported=import_statement)
    try:
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"'{import_statement}' did not return within {_TIMEOUT_SECONDS}s -- "
            "it likely imported preocr (and therefore python-magic) eagerly"
        )
    assert result.returncode == 0, result.stderr


def test_importing_the_ingestion_package_does_not_import_preocr():
    _assert_import_does_not_pull_in_preocr("import companydocs_rag.ingestion")


def test_importing_structure_does_not_import_preocr():
    _assert_import_does_not_pull_in_preocr("import companydocs_rag.ingestion.structure")


def test_importing_type_detection_does_not_import_preocr():
    _assert_import_does_not_pull_in_preocr(
        "import companydocs_rag.ingestion.type_detection"
    )


def test_importing_quality_itself_does_not_import_preocr():
    """The quality module's own top-level import list must stay preocr-free.

    ``import companydocs_rag.ingestion`` does load ``quality`` (``.parse_pass``
    imports it at module level), so this is what actually keeps the package
    import preocr-free: ``quality.py`` only imports ``preocr`` lazily, inside
    the functions that call it. Checked directly against the submodule too, so
    a regression here is not masked by anything the package does.
    """
    _assert_import_does_not_pull_in_preocr("import companydocs_rag.ingestion.quality")


def test_importing_ocr_does_not_import_preocr():
    _assert_import_does_not_pull_in_preocr("import companydocs_rag.ingestion.ocr")


def test_importing_language_does_not_import_preocr():
    _assert_import_does_not_pull_in_preocr("import companydocs_rag.ingestion.language")


def test_importing_normalization_does_not_import_preocr():
    _assert_import_does_not_pull_in_preocr(
        "import companydocs_rag.ingestion.normalization"
    )


def test_importing_normalize_cli_does_not_import_preocr():
    _assert_import_does_not_pull_in_preocr(
        "import companydocs_rag.ingestion.normalize_cli"
    )


def test_quality_exports_are_still_reachable_through_the_package():
    """The quality names re-exported from the package must keep working."""
    _assert_import_does_not_pull_in_preocr(
        "import companydocs_rag.ingestion as ingestion\n"
        "assert ingestion.TIER_CLEAN == 'Clean'\n"
        "assert callable(ingestion.assess_quality)\n"
        "assert callable(ingestion.run_preocr)\n"
    )


def test_unknown_attribute_still_raises_attribute_error():
    import companydocs_rag.ingestion as ingestion

    with pytest.raises(AttributeError):
        ingestion.this_name_does_not_exist
