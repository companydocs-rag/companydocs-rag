"""Tests for the CLI wrapper: folder walking and CSV report writing.

``classify_and_assess`` is stubbed, so the placeholder ``.pdf`` files created
here are never opened -- they exist only to exercise folder discovery.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from companydocs_rag.ingestion import cli
from companydocs_rag.ingestion.settings import find_project_root


def _fake_result(path: Path, **kwargs) -> dict:
    return {
        "file": {
            "path": str(path),
            "filename": path.name,
            "page_count": 3,
            "extraction_error": None,
            "failed_pages": [],
        },
        "status": "processed",
        "status_reason": None,
        "detected_type": "lease_agreement",
        "type_label": "Lease/rental/hire agreement",
        "type_confidence": 0.87,
        "type_evidence": [
            {"term": "verhuurder", "language": "nl", "page": 1, "occurrences": 2}
        ],
        "type_scores": {"lease_agreement": 0.87},
        "type_runner_up": None,
        "type_fallback_reason": None,
        "quality_label": "Clean",
        "awaiting_ocr": None,
        "quality_signals": {"structure": {"matched_pattern": "numbered_clauses"}},
        "explanation": "Clean: strong extraction signals. Type lease_agreement.",
        "page_text": {
            "schema_version": 1,
            "file": {"path": str(path), "filename": path.name, "sha256": None},
            "status": "processed",
            "page_count": 3,
            "language": "nld",
            "language_source": "default",
            "ocr_engine": None,
            "pages": [],
        },
    }


def _make_placeholder_pdfs(folder: Path, *names: str) -> None:
    for name in names:
        (folder / name).write_bytes(b"")


def test_find_pdfs_is_sorted_and_non_recursive_by_default(tmp_path):
    _make_placeholder_pdfs(tmp_path, "b.pdf", "a.pdf")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    _make_placeholder_pdfs(nested, "c.pdf")

    assert [path.name for path in cli.find_pdfs(tmp_path)] == ["a.pdf", "b.pdf"]
    assert [path.name for path in cli.find_pdfs(tmp_path, recursive=True)] == [
        "a.pdf",
        "b.pdf",
        "c.pdf",
    ]


def test_report_has_the_required_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "classify_and_assess", _fake_result)
    _make_placeholder_pdfs(tmp_path, "lease.pdf")
    output = tmp_path / "report.csv"

    cli.process_folder(tmp_path, output)

    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == [
        "filename",
        "status",
        "detected_type",
        "type_confidence",
        "quality_label",
        "awaiting_ocr",
        "explanation",
        "language",
        "language_source",
        "ocr_pages",
        "no_text_pages",
        "ocr_mean_confidence",
        "no_text_reasons",
    ]
    assert rows[0]["filename"] == "lease.pdf"
    assert rows[0]["status"] == "processed"
    assert rows[0]["detected_type"] == "lease_agreement"
    assert rows[0]["type_confidence"] == "0.87"
    assert rows[0]["quality_label"] == "Clean"
    assert rows[0]["awaiting_ocr"] == ""
    assert rows[0]["explanation"].startswith("Clean:")


def test_empty_folder_still_writes_a_header_only_report(tmp_path):
    output = tmp_path / "report.csv"
    cli.process_folder(tmp_path, output)
    assert output.read_text(encoding="utf-8-sig").strip() == (
        "filename,status,detected_type,type_confidence,quality_label,"
        "awaiting_ocr,explanation,language,language_source,ocr_pages,"
        "no_text_pages,ocr_mean_confidence,no_text_reasons"
    )


def test_a_failing_document_does_not_stop_the_batch(tmp_path, monkeypatch):
    """A bug in our own code (classify_and_assess raising) must not be
    reported as a property of the document (Axis 1: status=failed, no
    quality tier -- never Garbage), and must still reach the CSV, the
    returned results list, and the signals JSON.
    """

    def explode_on_one(path: Path, **kwargs):
        if path.name == "broken.pdf":
            raise RuntimeError("boom")
        return _fake_result(path)

    monkeypatch.setattr(cli, "classify_and_assess", explode_on_one)
    _make_placeholder_pdfs(tmp_path, "broken.pdf", "fine.pdf")
    output = tmp_path / "report.csv"
    signals_dir = tmp_path / "signals"

    results = cli.process_folder(tmp_path, output, signals_dir=signals_dir)

    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["filename"]: row for row in csv.DictReader(handle)}

    # Present in the CSV, blank quality label rather than mislabelled.
    assert rows["broken.pdf"]["quality_label"] == ""
    assert "RuntimeError" in rows["broken.pdf"]["explanation"]
    assert "boom" in rows["broken.pdf"]["explanation"]
    assert rows["fine.pdf"]["quality_label"] == "Clean"

    # Present in the returned results list (previously silently dropped).
    by_filename = {result["file"]["filename"]: result for result in results}
    assert by_filename["broken.pdf"]["status"] == "failed"
    assert by_filename["broken.pdf"]["quality_label"] is None

    # Present in the signals JSON too.
    assert (signals_dir / "broken.json").is_file()
    stored = json.loads((signals_dir / "broken.json").read_text(encoding="utf-8"))
    assert stored["status"] == "failed"


def test_a_file_that_cannot_be_opened_is_status_failed_in_csv_results_and_signals(
    tmp_path,
):
    """Runs for real, no stubs: a genuinely corrupt .pdf through the real
    classify_and_assess -> build_failed_result path (Axis 1), not the CLI's
    own outer exception guard.
    """
    (tmp_path / "corrupt.pdf").write_bytes(b"not a real pdf")
    output = tmp_path / "report.csv"
    signals_dir = tmp_path / "signals"

    results = cli.process_folder(tmp_path, output, signals_dir=signals_dir)

    assert results[0]["status"] == "failed"
    assert results[0]["quality_label"] is None
    assert results[0]["status_reason"]

    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["filename"] == "corrupt.pdf"
    assert rows[0]["status"] == "failed"
    assert rows[0]["quality_label"] == ""

    stored = json.loads((signals_dir / "corrupt.json").read_text(encoding="utf-8"))
    assert stored["status"] == "failed"


def test_signals_dir_writes_the_full_result_per_document(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "classify_and_assess", _fake_result)
    _make_placeholder_pdfs(tmp_path, "lease.pdf")
    signals_dir = tmp_path / "signals"

    cli.process_folder(tmp_path, tmp_path / "report.csv", signals_dir=signals_dir)

    stored = json.loads((signals_dir / "lease.json").read_text(encoding="utf-8"))
    assert (
        stored["quality_signals"]["structure"]["matched_pattern"] == "numbered_clauses"
    )
    assert stored["type_evidence"][0]["page"] == 1


def test_recursive_run_gives_same_named_pdfs_in_different_subfolders_distinct_signals(
    tmp_path, monkeypatch
):
    """Two documents named the same but in different subfolders must not
    collide on one signals JSON -- the last one processed silently winning
    would drop a document's evidence from the run without any error.
    """
    monkeypatch.setattr(cli, "classify_and_assess", _fake_result)
    sub_a = tmp_path / "a"
    sub_b = tmp_path / "b"
    sub_a.mkdir()
    sub_b.mkdir()
    _make_placeholder_pdfs(sub_a, "huur.pdf")
    _make_placeholder_pdfs(sub_b, "huur.pdf")
    output = tmp_path / "report.csv"
    signals_dir = tmp_path / "signals"

    cli.process_folder(tmp_path, output, recursive=True, signals_dir=signals_dir)

    signals_files = sorted(signals_dir.glob("**/*.json"))
    assert len(signals_files) == 2

    stored_paths = {
        json.loads(path.read_text(encoding="utf-8"))["file"]["path"]
        for path in signals_files
    }
    assert stored_paths == {str(sub_a / "huur.pdf"), str(sub_b / "huur.pdf")}


def test_main_reports_a_missing_folder_without_raising(tmp_path):
    assert cli.main([str(tmp_path / "nope"), "-o", str(tmp_path / "r.csv"), "-q"]) == 2


def test_main_runs_end_to_end_on_an_empty_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "default_page_text_dir", lambda: tmp_path / "page-text")
    monkeypatch.setattr(cli, "default_ocr_cache_dir", lambda: tmp_path / "ocr-cache")
    monkeypatch.setattr(
        cli, "default_language_overrides_path", lambda: tmp_path / "overrides.csv"
    )
    output = tmp_path / "report.csv"
    signals_dir = tmp_path / "signals"
    # --signals-dir is passed explicitly here so this test never touches the
    # real, git-ignored data/interim/ the CLI default resolves to.
    assert (
        cli.main(
            [str(tmp_path), "-o", str(output), "--signals-dir", str(signals_dir), "-q"]
        )
        == 0
    )
    assert output.is_file()


# --- Signals JSON as a default output (spec 001) -----------------------------


def test_a_default_run_writes_one_signals_json_per_document_without_the_flag(
    tmp_path, monkeypatch
):
    """The signals JSON is a default output now: omitting --signals-dir must
    still write it, not silently skip it.

    The real default resolves under the repository's git-ignored
    data/interim/, which tests must never touch -- so the default itself is
    monkeypatched to a tmp_path here, and cli.main() is exercised exactly as a
    user would run it, with no --signals-dir flag at all.
    """
    monkeypatch.setattr(cli, "classify_and_assess", _fake_result)
    default_signals = tmp_path / "default-signals"
    monkeypatch.setattr(cli, "default_signals_dir", lambda: default_signals)
    monkeypatch.setattr(cli, "default_page_text_dir", lambda: tmp_path / "page-text")
    monkeypatch.setattr(cli, "default_ocr_cache_dir", lambda: tmp_path / "ocr-cache")
    monkeypatch.setattr(
        cli, "default_language_overrides_path", lambda: tmp_path / "overrides.csv"
    )

    _make_placeholder_pdfs(tmp_path, "lease.pdf")
    output = tmp_path / "report.csv"

    assert cli.main([str(tmp_path), "-o", str(output), "-q"]) == 0

    assert (default_signals / "lease.json").is_file()


# --- Default report location (fix 4: must not default into the repo root) ---


def test_default_report_path_is_under_the_gitignored_data_interim_directory():
    path = cli.default_report_path()

    assert path.name == "parse_pass_report.csv"
    assert path.parent.name == "interim"
    assert path.parent.parent.name == "data"


def test_default_report_path_is_resolved_against_the_project_root_not_the_cwd(
    monkeypatch, tmp_path
):
    root = find_project_root()
    assert root is not None

    monkeypatch.chdir(tmp_path)
    path = cli.default_report_path()

    assert path == root / "data" / "interim" / "parse_pass_report.csv"


def test_parser_default_output_matches_default_report_path():
    parser = cli.build_parser()
    args = parser.parse_args(["some-folder"])
    assert args.output == cli.default_report_path()


# --- Default signals-JSON location (spec 001) --------------------------------
# These only resolve a Path -- they never write anything, so they are safe to
# run without a tmp_path override.


def test_default_signals_dir_is_under_the_gitignored_data_interim_directory():
    path = cli.default_signals_dir()

    assert path.name == "parse_pass_signals"
    assert path.parent.name == "interim"
    assert path.parent.parent.name == "data"


def test_default_signals_dir_is_resolved_against_the_project_root_not_the_cwd(
    monkeypatch, tmp_path
):
    root = find_project_root()
    assert root is not None

    monkeypatch.chdir(tmp_path)
    path = cli.default_signals_dir()

    assert path == root / "data" / "interim" / "parse_pass_signals"


def test_parser_default_signals_dir_matches_default_signals_dir():
    parser = cli.build_parser()
    args = parser.parse_args(["some-folder"])
    assert args.signals_dir == cli.default_signals_dir()
