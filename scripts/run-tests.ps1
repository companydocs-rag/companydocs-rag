# run-tests.ps1 - run the test suite without Git\usr\bin on PATH (spec 004, D1).
#
# Usage, from the repository root:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-tests.ps1 [pytest args]
#
# Within this process only, it removes every PATH entry that contains
# Git\usr\bin: a literal, case-insensitive substring comparison, never a
# regular expression. With that folder on PATH, python-magic (imported by
# preocr) loads an incompatible libmagic and hangs or crashes; see the
# Known-issue bullet in docs/DEVELOPMENT.md. It then checks that PATH still holds an
# entry and that uv and tesseract are on it, runs `uv run pytest` in the
# repository root with the given arguments, and exits with pytest's exit code.
#
# Exit codes and messages (one line on stderr each):
#   90  PATH is empty after removing Git\usr\bin entries
#   91  uv not found on PATH
#   92  tesseract not found on PATH
#    5  pytest collected no tests (pytest's own code 5)
# Every other pytest exit code is passed on unchanged.
#
# The script changes nothing outside its own process: no execution policy, no
# profile, no files. Keep this file ASCII only: PowerShell 5.1 reads a file
# without a byte order mark in the ANSI code page.

$prefix = 'run-tests.ps1: '
$gitUsrBin = 'Git\usr\bin'

function Stop-WithMessage([int] $code, [string] $message) {
    [Console]::Error.WriteLine($prefix + $message)
    exit $code
}

# The repository root is the parent of this script's folder.
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot) -ErrorAction Stop

$entries = ([string] $env:Path).Split(';')
$kept = @(
    $entries | Where-Object {
        $_.IndexOf($gitUsrBin, [StringComparison]::OrdinalIgnoreCase) -lt 0
    }
)
$removed = $entries.Count - $kept.Count
$env:Path = $kept -join ';'

if (-not ($kept | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
    Stop-WithMessage 90 'PATH is empty after removing Git\usr\bin entries'
}
if (-not (Get-Command uv -CommandType Application -ErrorAction SilentlyContinue)) {
    Stop-WithMessage 91 'uv not found on PATH'
}
if (-not (Get-Command tesseract -CommandType Application -ErrorAction SilentlyContinue)) {
    Stop-WithMessage 92 'tesseract not found on PATH'
}

[Console]::Out.WriteLine($prefix + "removed $removed PATH entr(ies) containing $gitUsrBin")

& uv run pytest @args
$code = $LASTEXITCODE
if ($code -eq 5) {
    Stop-WithMessage 5 'pytest collected no tests'
}
exit $code
