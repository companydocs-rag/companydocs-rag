"""The test script ``scripts/run-tests.ps1`` (spec 004, D1): its failure paths.

Each case starts ``powershell.exe`` by absolute path, from a temporary working
directory, with a PATH built here. None of them runs a test (the recursion
guard of spec 004, section 1): the script stops before pytest, or pytest gets a
nonexistent test path or a ``-k`` expression that selects nothing. The cases
that stop before pytest also pass that nonexistent path, so a script that
wrongly went on to pytest would exit 4 there, not run the suite.

Skipped when not on Windows. A case that needs ``uv`` (or, to get past the
script's checks, ``tesseract``) on this test's own PATH is skipped without it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="run-tests.ps1 is a Windows PowerShell script"
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run-tests.ps1"
NO_SUCH_TEST_PATH = "tests/no_such_folder_for_run_tests/test_nothing.py"
REMOVED_LINE_PREFIX = "run-tests.ps1: removed "
TIMEOUT_SECONDS = 600


def _powershell() -> Path:
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    return (
        Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )


def _run(path_value: str, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = path_value
    return subprocess.run(
        [
            str(_powershell()),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            *args,
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT_SECONDS,
    )


def _script_lines(stream: str) -> List[str]:
    return [line for line in stream.splitlines() if line.startswith("run-tests.ps1: ")]


def _require_on_own_path(*commands: str) -> None:
    for command in commands:
        if shutil.which(command) is None:
            pytest.skip(f"{command} is not on this test's PATH")


def test_only_git_usr_bin_entries_on_path_exits_90_before_pytest(tmp_path):
    path_value = ";".join(
        [
            r"C:\Fictief\Git\usr\bin",
            r"D:\fictief\GIT\USR\BIN\core_perl",
            r"c:\fictief\git\Usr\Bin\vendor_perl",
        ]
    )

    result = _run(path_value, NO_SUCH_TEST_PATH, cwd=tmp_path)

    assert result.returncode == 90
    assert result.stderr.splitlines() == [
        r"run-tests.ps1: PATH is empty after removing Git\usr\bin entries"
    ]
    assert REMOVED_LINE_PREFIX not in result.stdout


def test_without_uv_on_path_exits_91_before_pytest(tmp_path):
    empty = tmp_path / "leeg"
    empty.mkdir()
    path_value = ";".join([r"C:\Fictief\Git\usr\bin", str(empty)])

    result = _run(path_value, NO_SUCH_TEST_PATH, cwd=tmp_path)

    assert result.returncode == 91
    assert result.stderr.splitlines() == ["run-tests.ps1: uv not found on PATH"]
    assert REMOVED_LINE_PREFIX not in result.stdout


def test_without_tesseract_on_path_exits_92_before_pytest(tmp_path):
    _require_on_own_path("uv")
    uv_folder = Path(shutil.which("uv")).parent
    if shutil.which("tesseract", path=str(uv_folder)) is not None:
        pytest.skip("tesseract sits in the same folder as uv")
    path_value = ";".join([r"C:\Fictief\Git\usr\bin", str(uv_folder)])

    result = _run(path_value, NO_SUCH_TEST_PATH, cwd=tmp_path)

    assert result.returncode == 92
    assert result.stderr.splitlines() == ["run-tests.ps1: tesseract not found on PATH"]
    assert REMOVED_LINE_PREFIX not in result.stdout


def test_a_k_expression_that_selects_nothing_exits_5_with_the_message(tmp_path):
    """Exit 5 also shows the spaced expression arrived as one argument: split
    into three, pytest would get two nonexistent paths and exit 4."""
    _require_on_own_path("uv", "tesseract")

    result = _run(
        os.environ["PATH"], "-k", "no_such_test_a and no_such_test_b", cwd=tmp_path
    )

    assert result.returncode == 5
    assert _script_lines(result.stderr) == ["run-tests.ps1: pytest collected no tests"]
    assert result.stderr.splitlines()[-1] == "run-tests.ps1: pytest collected no tests"
    (removed_line,) = _script_lines(result.stdout)
    assert removed_line.startswith(REMOVED_LINE_PREFIX)
    assert removed_line.endswith(r" PATH entr(ies) containing Git\usr\bin")
    assert "deselected" in result.stdout


def test_a_nonexistent_test_path_passes_pytests_exit_code_4_on(tmp_path):
    _require_on_own_path("uv", "tesseract")

    result = _run(os.environ["PATH"], NO_SUCH_TEST_PATH, cwd=tmp_path)

    assert result.returncode == 4
    assert _script_lines(result.stderr) == []
    assert f"file or directory not found: {NO_SUCH_TEST_PATH}" in result.stderr
