"""CLI runs through the real classify_and_assess (spec 003): the page text store,
the signals JSON, the CSV columns and the end-of-run summary.

The extraction and preocr are stubbed per file name, and the OCR seams return
fabricated text, except in the one test that runs over a PDF generated here
from a Pillow image (A24): there extraction and preocr run for real, and only
the OCR seams are stubbed. The placeholder files are read only to hash them.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw

from companydocs_rag.ingestion import cli
from companydocs_rag.ingestion import parse_pass as parse_pass_module
from companydocs_rag.ingestion.extraction import DocumentText
from tests.ingestion._helpers import (
    make_document,
    make_page,
    stub_ocr,
    tesseract_output,
    write_client_csv,
)
from tests.ingestion.test_quality import CLEAN_PAGE, DIGITAL_PREOCR

OCR_TEXT = "Fictieve OCR-tekst over de huurprijs en de verhuurder van dit pand."
STORE_KEYS = {
    "schema_version",
    "file",
    "status",
    "page_count",
    "language",
    "language_source",
    "ocr_engine",
    "pages",
}
STORE_FILE_KEYS = {"path", "filename", "sha256"}
STORE_PAGE_KEYS = {
    "page_number",
    "text",
    "source",
    "tier",
    "text_origin",
    "language",
    "language_source",
    "ocr_mean_confidence",
    "no_text_reason",
}
OCR_ENGINE_KEYS = {"name", "version", "dpi", "grayscale"}
NEW_COLUMNS = [
    "language",
    "language_source",
    "ocr_pages",
    "no_text_pages",
    "ocr_mean_confidence",
    "no_text_reasons",
]


def _placeholder(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _stub_documents(monkeypatch, documents):
    """Extraction by file name; a missing name is a document that failed to open."""

    def extract(path):
        path = Path(path)
        if path.name in documents:
            return documents[path.name]
        return DocumentText(path=str(path), extraction_error="fabricated: unreadable")

    monkeypatch.setattr(parse_pass_module, "extract_pages", extract)
    monkeypatch.setattr(parse_pass_module, "run_preocr", lambda path: DIGITAL_PREOCR)


def _record_results(monkeypatch):
    """Keep every result the CLI gets from the real classify_and_assess."""
    results = []
    real = cli.classify_and_assess

    def recording(path, **kwargs):
        result = real(path, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(cli, "classify_and_assess", recording)
    return results


def _cli_args(folder: Path, out: Path, *extra: str):
    return [
        str(folder),
        "-o",
        str(out / "report.csv"),
        "--signals-dir",
        str(out / "signals"),
        "--page-text-dir",
        str(out / "page_text"),
        "--ocr-cache-dir",
        str(out / "ocr_cache"),
        "--language-overrides",
        str(out / "geen_overrides.csv"),
        "-q",
        *extra,
    ]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# --- The page text store (A21) ----------------------------------------------------


def test_a_cli_run_writes_one_store_file_per_pdf_mirroring_the_input(
    tmp_path, monkeypatch
):
    folder = tmp_path / "invoer"
    _placeholder(folder / "a" / "fictief.pdf", b"fabricated bytes a")
    _placeholder(folder / "b" / "fictief.pdf", b"fabricated bytes b")
    _placeholder(folder / "kapot.pdf", b"fabricated bytes, unreadable")
    _stub_documents(monkeypatch, {"fictief.pdf": make_document(CLEAN_PAGE, "", "")})
    stub_ocr(monkeypatch, {2: OCR_TEXT})
    results = _record_results(monkeypatch)
    out = tmp_path / "uit"

    assert cli.main(_cli_args(folder, out, "--recursive")) == 0

    store_dir = out / "page_text"
    stored = sorted(p.relative_to(store_dir).as_posix() for p in store_dir.rglob("*"))
    assert stored == ["a", "a/fictief.json", "b", "b/fictief.json", "kapot.json"]
    by_path = {result["file"]["path"]: result for result in results}

    for relative in ("a/fictief", "b/fictief"):
        store = _read_json(store_dir / f"{relative}.json")
        pdf = folder / f"{relative}.pdf"
        assert store == by_path[str(pdf)]["page_text"]
        assert set(store) == STORE_KEYS
        assert set(store["file"]) == STORE_FILE_KEYS
        assert set(store["ocr_engine"]) == OCR_ENGINE_KEYS
        assert store["schema_version"] == 2
        assert store["status"] == "processed"
        assert store["page_count"] == 3
        assert store["file"]["path"] == str(pdf)
        assert store["file"]["filename"] == "fictief.pdf"
        assert store["ocr_engine"] == {
            "name": "tesseract",
            "version": "5.5.3",
            "dpi": 300,
            "grayscale": True,
        }
        native, ocr_page, none_page = store["pages"]
        assert all(set(page) == STORE_PAGE_KEYS for page in store["pages"])
        assert native == {
            "page_number": 1,
            "text": CLEAN_PAGE,
            "source": "native",
            "tier": "Clean",
            "text_origin": "native",
            "language": "nld",
            "language_source": "default",
            "ocr_mean_confidence": None,
            "no_text_reason": None,
        }
        assert ocr_page == {
            "page_number": 2,
            "text": OCR_TEXT,
            "source": "no_text_layer",
            "tier": None,
            "text_origin": "ocr",
            "language": "nld",
            "language_source": "default",
            "ocr_mean_confidence": 90.0,
            "no_text_reason": None,
        }
        assert none_page == {
            "page_number": 3,
            "text": "",
            "source": "no_text_layer",
            "tier": None,
            "text_origin": "none",
            "language": "nld",
            "language_source": "default",
            "ocr_mean_confidence": None,
            "no_text_reason": "ocr_empty",
        }

    failed = _read_json(store_dir / "kapot.json")
    assert set(failed) == STORE_KEYS
    assert failed["status"] == "failed"
    assert failed["pages"] == []
    assert failed["ocr_engine"] is None
    assert failed["language"] is None
    assert failed == by_path[str(folder / "kapot.pdf")]["page_text"]


def test_the_signals_json_contains_no_page_text(tmp_path, monkeypatch):
    folder = tmp_path / "invoer"
    _placeholder(folder / "fictief.pdf", b"fabricated bytes")
    _stub_documents(monkeypatch, {"fictief.pdf": make_document(CLEAN_PAGE, "")})
    stub_ocr(monkeypatch, {2: OCR_TEXT})
    out = tmp_path / "uit"

    assert cli.main(_cli_args(folder, out)) == 0

    signals_text = (out / "signals" / "fictief.json").read_text(encoding="utf-8")
    signals = json.loads(signals_text)
    assert "page_text" not in signals
    assert signals["quality_signals"]["pages"][1]["text_origin"] == "ocr"
    assert "Fictieve OCR-tekst" not in signals_text
    assert "gemeenschappelijke ruimten" not in signals_text
    store_text = (out / "page_text" / "fictief.json").read_text(encoding="utf-8")
    assert "Fictieve OCR-tekst" in store_text


def test_default_page_text_and_ocr_cache_locations_are_under_data_interim():
    for path, name in (
        (cli.default_page_text_dir(), "page_text"),
        (cli.default_ocr_cache_dir(), "ocr_cache"),
    ):
        assert path.name == name
        assert path.parent.name == "interim"
        assert path.parent.parent.name == "data"


def test_parser_defaults_match_the_default_functions():
    args = cli.build_parser().parse_args(["some-folder"])
    assert args.page_text_dir == cli.default_page_text_dir()
    assert args.ocr_cache_dir == cli.default_ocr_cache_dir()


# --- CSV columns (A25) and the run summary (A23) ------------------------------------


def _mixed_run(tmp_path, monkeypatch):
    """Three documents: one from the index with pages [native, OCR at 80, OCR
    at 90, none (ocr_empty), none (extraction_failed)], one with an override,
    and one that fails to open."""
    folder = tmp_path / "invoer"
    _placeholder(folder / "fictief_a.pdf", b"fabricated bytes a")
    _placeholder(folder / "fictief_b.pdf", b"fabricated bytes b")
    _placeholder(folder / "kapot.pdf", b"fabricated bytes, unreadable")
    write_client_csv(
        folder / "00_INDEX.csv",
        ("Map", "Bestandsnaam", "Taal"),
        [("", "fictief_a.pdf", "NL")],
    )
    overrides = write_client_csv(
        tmp_path / "overrides.csv", ("Bestandsnaam", "Taal"), [("fictief_b.pdf", "DE")]
    )
    failed_page = make_page(
        5, "", extraction_failed=True, extraction_error="ValueError: fabricated"
    )
    _stub_documents(
        monkeypatch,
        {
            "fictief_a.pdf": make_document(CLEAN_PAGE, "", "", "", failed_page),
            "fictief_b.pdf": make_document(CLEAN_PAGE),
        },
    )
    stub_ocr(
        monkeypatch,
        {
            2: tesseract_output(OCR_TEXT, 80),
            3: tesseract_output(OCR_TEXT, 90),
            4: "",
        },
    )
    output = tmp_path / "report.csv"
    results = cli.process_folder(folder, output, language_overrides=overrides)
    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["filename"]: row for row in csv.DictReader(handle)}
    return results, rows


def test_the_csv_carries_the_new_columns_after_explanation(tmp_path, monkeypatch):
    _, rows = _mixed_run(tmp_path, monkeypatch)

    assert list(rows["fictief_a.pdf"])[-7:] == ["explanation", *NEW_COLUMNS]
    row = rows["fictief_a.pdf"]
    assert [row[column] for column in NEW_COLUMNS] == [
        "nld",
        "index",
        "2",
        "2",
        "85.00",
        "extraction_failed:1;ocr_empty:1",
    ]
    native_only = rows["fictief_b.pdf"]
    assert [native_only[column] for column in NEW_COLUMNS] == [
        "deu",
        "override",
        "0",
        "0",
        "",
        "",
    ]
    failed = rows["kapot.pdf"]
    assert failed["status"] == "failed"
    assert [failed[column] for column in NEW_COLUMNS] == [""] * 6


def test_the_end_of_run_summary(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="companydocs_rag.ingestion.cli"):
        _mixed_run(tmp_path, monkeypatch)

    summaries = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and r.getMessage().startswith("Run summary")
    ]
    assert summaries == [
        "Run summary: 2 document(s) processed, 1 failed; "
        "pages by text_origin: native=2, ocr=2, none=2; "
        "pages by no_text_reason: extraction_failed=1, ocr_skipped=0, "
        "ocr_error=0, ocr_empty=1; "
        "pages by language_source: override=1, index=5, default=0"
    ]


# --- The input folder is read-only (A24) --------------------------------------------


def test_a_cli_run_leaves_the_input_folder_unchanged(tmp_path, monkeypatch):
    folder = tmp_path / "invoer"
    folder.mkdir()
    image = Image.new("RGB", (300, 400), "white")
    ImageDraw.Draw(image).rectangle((40, 40, 260, 120), fill="black")
    image.save(folder / "gegenereerd.pdf", "PDF", resolution=72.0)
    write_client_csv(
        folder / "00_INDEX.csv",
        ("Map", "Bestandsnaam", "Taal"),
        [("", "gegenereerd.pdf", "NL")],
    )
    stub = stub_ocr(monkeypatch, default=OCR_TEXT)
    before = {
        path.relative_to(folder): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
    }
    out = tmp_path / "uit"

    assert cli.main(_cli_args(folder, out)) == 0

    after = {
        path.relative_to(folder): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert stub.pages == [1]
    store = _read_json(out / "page_text" / "gegenereerd.json")
    assert store["pages"][0]["text_origin"] == "ocr"
    assert store["language_source"] == "index"


# --- The page tier in the store (spec 004, D2; A4) -----------------------------------

# Native and tiered Garbage: 150 times U+FFFD REPLACEMENT CHARACTER first.
GARBAGE_TIER_PAGE = chr(0xFFFD) * 150 + CLEAN_PAGE
GARBAGE_FLAG = {"ingestion": {"parse_pass": {"ocr": {"ocr_garbage_tier_pages": True}}}}
OCR_TEXT_PAGE_TWO = "Fictieve heldere OCR-tekst voor de tweede pagina van dit document."


def _tier_run(tmp_path, monkeypatch, settings=None):
    """One process_folder run over a document with pages [native Clean, native
    Garbage, no_text_layer with OCR text, no_text_layer with ocr_empty]; returns
    its store file and signals JSON, both at the relative path sub/fictief."""
    folder = tmp_path / "invoer"
    _placeholder(folder / "sub" / "fictief.pdf", b"fabricated bytes")
    _stub_documents(
        monkeypatch,
        {"fictief.pdf": make_document(CLEAN_PAGE, GARBAGE_TIER_PAGE, "", "")},
    )
    stub_ocr(monkeypatch, {2: OCR_TEXT_PAGE_TWO, 3: OCR_TEXT, 4: ""})
    if settings is not None:
        real = cli.classify_and_assess

        def with_settings(path, **kwargs):
            return real(path, settings=settings, **kwargs)

        monkeypatch.setattr(cli, "classify_and_assess", with_settings)
    out = tmp_path / "uit"
    cli.process_folder(
        folder,
        out / "report.csv",
        recursive=True,
        signals_dir=out / "signals",
        page_text_dir=out / "page_text",
    )
    store = _read_json(out / "page_text" / "sub" / "fictief.json")
    signals = _read_json(out / "signals" / "sub" / "fictief.json")
    return store, signals["quality_signals"]["pages"]


def test_the_store_carries_the_tier_of_native_pages_and_null_elsewhere(
    tmp_path, monkeypatch
):
    store, signal_pages = _tier_run(tmp_path, monkeypatch)

    assert store["schema_version"] == 2
    pages = store["pages"]
    assert [page["text_origin"] for page in pages] == [
        "native",
        "native",
        "ocr",
        "none",
    ]
    assert pages[3]["no_text_reason"] == "ocr_empty"
    assert [page["tier"] for page in pages] == ["Clean", "Garbage", None, None]
    for page, evidence in zip(pages, signal_pages):
        if page["text_origin"] == "native":
            assert page["tier"] == evidence["tier"]
        else:
            assert page["tier"] is None
    assert [list(page)[:5] for page in pages] == [
        ["page_number", "text", "source", "tier", "text_origin"]
    ] * 4


def test_a_garbage_page_that_got_ocr_text_has_no_store_tier_but_keeps_its_evidence(
    tmp_path, monkeypatch
):
    store, signal_pages = _tier_run(tmp_path, monkeypatch, settings=GARBAGE_FLAG)

    page, evidence = store["pages"][1], signal_pages[1]
    assert (page["source"], page["text_origin"]) == ("native", "ocr")
    assert page["text"] == OCR_TEXT_PAGE_TWO
    assert page["tier"] is None
    assert evidence["tier"] == "Garbage"
    assert [page["tier"] for page in store["pages"]] == ["Clean", None, None, None]
