"""Language resolution (spec 003): override file, client index, default.

Index and override files are fabricated in ``tmp_path``, with fabricated
folder and file names. The PDFs they name are empty placeholder files: the
resolver only looks at their paths, and the pipeline tests below stub the
extraction, preocr and the OCR seams, so no PDF is ever opened.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from companydocs_rag.ingestion import cli
from companydocs_rag.ingestion import language as language_module
from companydocs_rag.ingestion import parse_pass as parse_pass_module
from companydocs_rag.ingestion.language import (
    LANGUAGE_SOURCE_DEFAULT,
    LANGUAGE_SOURCE_INDEX,
    LANGUAGE_SOURCE_OVERRIDE,
    LanguageResolver,
)
from companydocs_rag.ingestion.parse_pass import classify_and_assess
from tests.ingestion._helpers import make_document, stub_ocr, write_client_csv
from tests.ingestion.test_quality import DIGITAL_PREOCR

_LOGGER = "companydocs_rag.ingestion.language"

# The index's own columns, among others (A18).
INDEX_HEADER = ("Nr", "Map", "Bestandsnaam", "Documenttype", "Taal", "Opmerking")
OVERRIDE_HEADER = ("Bestandsnaam", "Taal")


def _index_row(folder: str, filename: str, code: str, number: int = 1):
    return (str(number), folder, filename, "Fictief type", code, "")


def _placeholder(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]


# --- Client index ---------------------------------------------------------------


def test_the_index_resolves_a_pdf_found_by_ancestor_lookup(tmp_path):
    delivery = tmp_path / "levering_fictief"
    index = write_client_csv(
        delivery / "00_INDEX.csv",
        INDEX_HEADER,
        [_index_row("Deel A\\Submap", "fictief_contract_01.pdf", "SK")],
    )
    pdf = _placeholder(delivery / "Deel A" / "Submap" / "fictief_contract_01.pdf")

    # The fabricated index follows the client's conventions (A18).
    raw = index.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in raw
    assert b";" in raw

    assert LanguageResolver().resolve(pdf) == ("slk", LANGUAGE_SOURCE_INDEX)


def test_a_forward_slash_map_resolves_too(tmp_path):
    delivery = tmp_path / "levering_fictief"
    write_client_csv(
        delivery / "00_INDEX.csv",
        INDEX_HEADER,
        [_index_row("Deel A/Submap", "fictief_contract_01.pdf", "PL")],
    )
    pdf = _placeholder(delivery / "Deel A" / "Submap" / "fictief_contract_01.pdf")

    assert LanguageResolver().resolve(pdf) == ("pol", LANGUAGE_SOURCE_INDEX)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("NL", "nld"),
        ("EN", "eng"),
        ("DE", "deu"),
        ("FR", "fra"),
        ("PL", "pol"),
        ("SK", "slk"),
        ("CS", "ces"),
        (" sk ", "slk"),  # stripped and uppercased
    ],
)
def test_the_seven_index_codes_map_to_tesseract_codes(tmp_path, code, expected):
    write_client_csv(
        tmp_path / "00_INDEX.csv",
        INDEX_HEADER,
        [_index_row("Map", "fictief.pdf", code)],
    )
    pdf = _placeholder(tmp_path / "Map" / "fictief.pdf")

    assert LanguageResolver().resolve(pdf) == (expected, LANGUAGE_SOURCE_INDEX)


@pytest.mark.parametrize("code", ["", "XX"])
def test_an_empty_or_unmapped_code_falls_through_with_a_warning(tmp_path, caplog, code):
    write_client_csv(
        tmp_path / "00_INDEX.csv",
        INDEX_HEADER,
        [_index_row("Map", "fictief.pdf", code)],
    )
    pdf = _placeholder(tmp_path / "Map" / "fictief.pdf")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = LanguageResolver().resolve(pdf)

    assert result == ("nld", LANGUAGE_SOURCE_DEFAULT)
    assert any(repr(code) in message for message in _warnings(caplog))


def test_a_pdf_outside_any_indexed_tree_gets_the_default_with_a_warning(
    tmp_path, caplog
):
    write_client_csv(
        tmp_path / "geindexeerd" / "00_INDEX.csv",
        INDEX_HEADER,
        [_index_row("", "fictief.pdf", "SK")],
    )
    pdf = _placeholder(tmp_path / "los" / "fictief.pdf")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = LanguageResolver().resolve(pdf)

    assert result == ("nld", LANGUAGE_SOURCE_DEFAULT)
    assert any(str(pdf) in message for message in _warnings(caplog))


def test_the_nearest_index_is_the_one_used(tmp_path):
    write_client_csv(
        tmp_path / "00_INDEX.csv",
        INDEX_HEADER,
        [_index_row("binnen", "fictief.pdf", "EN")],
    )
    write_client_csv(
        tmp_path / "binnen" / "00_INDEX.csv",
        INDEX_HEADER,
        [_index_row("", "fictief.pdf", "DE")],
    )
    pdf = _placeholder(tmp_path / "binnen" / "fictief.pdf")

    assert LanguageResolver().resolve(pdf) == ("deu", LANGUAGE_SOURCE_INDEX)


def test_path_components_are_compared_with_exact_case(tmp_path):
    write_client_csv(
        tmp_path / "00_INDEX.csv",
        INDEX_HEADER,
        [_index_row("Map", "Fictief.pdf", "SK")],
    )
    pdf = _placeholder(tmp_path / "Map" / "fictief.pdf")

    assert LanguageResolver().resolve(pdf) == ("nld", LANGUAGE_SOURCE_DEFAULT)


def test_each_index_is_read_once_per_run(tmp_path, monkeypatch):
    write_client_csv(
        tmp_path / "00_INDEX.csv",
        INDEX_HEADER,
        [
            _index_row("", "fictief_a.pdf", "NL", 1),
            _index_row("", "fictief_b.pdf", "FR", 2),
        ],
    )
    reads = []
    real_read_rows = language_module._read_rows

    def counting_read_rows(path, columns):
        reads.append(path)
        return real_read_rows(path, columns)

    monkeypatch.setattr(language_module, "_read_rows", counting_read_rows)
    resolver = LanguageResolver()

    assert resolver.resolve(_placeholder(tmp_path / "fictief_a.pdf"))[0] == "nld"
    assert resolver.resolve(_placeholder(tmp_path / "fictief_b.pdf"))[0] == "fra"
    assert len(reads) == 1


def test_an_index_missing_a_column_logs_an_error_and_is_treated_as_absent(
    tmp_path, caplog
):
    write_client_csv(
        tmp_path / "00_INDEX.csv",
        ("Map", "Bestandsnaam"),
        [("", "fictief.pdf")],
    )
    pdf = _placeholder(tmp_path / "fictief.pdf")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = LanguageResolver().resolve(pdf)

    assert result == ("nld", LANGUAGE_SOURCE_DEFAULT)
    assert len(_errors(caplog)) == 1
    assert "00_INDEX.csv" in _errors(caplog)[0]


def test_an_unreadable_index_logs_an_error_and_is_treated_as_absent(tmp_path, caplog):
    (tmp_path / "00_INDEX.csv").write_bytes(b"Map;Bestandsnaam;Taal\r\n\xff\xfe;x;\r\n")
    pdf = _placeholder(tmp_path / "fictief.pdf")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = LanguageResolver().resolve(pdf)

    assert result == ("nld", LANGUAGE_SOURCE_DEFAULT)
    assert len(_errors(caplog)) == 1


def test_language_settings_are_read_at_call_time(tmp_path):
    write_client_csv(
        tmp_path / "geindexeerd" / "00_INDEX.csv",
        INDEX_HEADER,
        [_index_row("", "fictief.pdf", "NL")],
    )
    indexed = _placeholder(tmp_path / "geindexeerd" / "fictief.pdf")
    loose = _placeholder(tmp_path / "los" / "fictief.pdf")
    settings = {
        "ingestion": {
            "parse_pass": {
                "language": {"default_language": "eng", "language_map": {"NL": "deu"}}
            }
        }
    }
    resolver = LanguageResolver()

    assert resolver.resolve(indexed, settings=settings) == (
        "deu",
        LANGUAGE_SOURCE_INDEX,
    )
    assert resolver.resolve(loose, settings=settings) == (
        "eng",
        LANGUAGE_SOURCE_DEFAULT,
    )


# --- Override file (Q1) -----------------------------------------------------------


def test_the_override_file_follows_q1(tmp_path):
    overrides = write_client_csv(
        tmp_path / "manual" / "language_overrides.csv",
        OVERRIDE_HEADER,
        [("fictief_scan.pdf", "SK")],
    )
    pdf = _placeholder(tmp_path / "invoer" / "fictief_scan.pdf")

    assert overrides.read_bytes().startswith(b"\xef\xbb\xbfBestandsnaam;Taal\r\n")
    assert LanguageResolver(overrides).resolve(pdf) == (
        "slk",
        LANGUAGE_SOURCE_OVERRIDE,
    )


def test_an_override_entry_beats_the_index(tmp_path):
    write_client_csv(
        tmp_path / "invoer" / "00_INDEX.csv",
        INDEX_HEADER,
        [_index_row("", "fictief.pdf", "NL")],
    )
    overrides = write_client_csv(
        tmp_path / "overrides.csv", OVERRIDE_HEADER, [("fictief.pdf", "CS")]
    )
    pdf = _placeholder(tmp_path / "invoer" / "fictief.pdf")

    assert LanguageResolver(overrides).resolve(pdf) == (
        "ces",
        LANGUAGE_SOURCE_OVERRIDE,
    )
    # And without an override entry, the index applies.
    assert LanguageResolver().resolve(pdf) == ("nld", LANGUAGE_SOURCE_INDEX)


def test_an_unmapped_override_code_falls_through_to_the_index(tmp_path, caplog):
    write_client_csv(
        tmp_path / "invoer" / "00_INDEX.csv",
        INDEX_HEADER,
        [_index_row("", "fictief.pdf", "FR")],
    )
    overrides = write_client_csv(
        tmp_path / "overrides.csv", OVERRIDE_HEADER, [("fictief.pdf", "QQ")]
    )
    pdf = _placeholder(tmp_path / "invoer" / "fictief.pdf")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = LanguageResolver(overrides).resolve(pdf)

    assert result == ("fra", LANGUAGE_SOURCE_INDEX)
    assert any("'QQ'" in message for message in _warnings(caplog))


def test_a_shared_file_name_gets_the_override_everywhere_and_a_warning(
    tmp_path, caplog
):
    overrides = write_client_csv(
        tmp_path / "overrides.csv", OVERRIDE_HEADER, [("fictief_scan.pdf", "SK")]
    )
    first = _placeholder(tmp_path / "invoer" / "a" / "fictief_scan.pdf")
    second = _placeholder(tmp_path / "invoer" / "b" / "fictief_scan.pdf")
    resolver = LanguageResolver(overrides)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert resolver.resolve(first) == ("slk", LANGUAGE_SOURCE_OVERRIDE)
        assert resolver.resolve(second) == ("slk", LANGUAGE_SOURCE_OVERRIDE)
        caplog.clear()
        resolver.report_end_of_run()

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "fictief_scan.pdf" in warnings[0]
    assert str(first) in warnings[0]
    assert str(second) in warnings[0]


def test_a_file_name_listed_twice_uses_the_first_row_with_a_warning(tmp_path, caplog):
    overrides = write_client_csv(
        tmp_path / "overrides.csv",
        OVERRIDE_HEADER,
        [("fictief_scan.pdf", "SK"), ("fictief_scan.pdf", "PL")],
    )
    pdf = _placeholder(tmp_path / "invoer" / "fictief_scan.pdf")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        resolver = LanguageResolver(overrides)

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "fictief_scan.pdf" in warnings[0]
    assert "more than once" in warnings[0]
    assert resolver.resolve(pdf) == ("slk", LANGUAGE_SOURCE_OVERRIDE)


def test_a_missing_override_file_is_not_an_error(tmp_path, caplog):
    pdf = _placeholder(tmp_path / "invoer" / "fictief.pdf")

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        resolver = LanguageResolver(tmp_path / "bestaat_niet.csv")
        resolver.resolve(pdf)
        resolver.report_end_of_run()

    assert _errors(caplog) == []
    assert any(
        r.levelno == logging.INFO and "bestaat_niet.csv" in r.getMessage()
        for r in caplog.records
    )


def test_an_override_file_missing_a_column_logs_an_error_and_counts_as_empty(
    tmp_path, caplog
):
    overrides = write_client_csv(
        tmp_path / "overrides.csv", ("Bestandsnaam",), [("fictief.pdf",)]
    )
    pdf = _placeholder(tmp_path / "invoer" / "fictief.pdf")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = LanguageResolver(overrides).resolve(pdf)

    assert result == ("nld", LANGUAGE_SOURCE_DEFAULT)
    assert len(_errors(caplog)) == 1
    assert "overrides.csv" in _errors(caplog)[0]


def test_unmatched_override_entries_are_listed_at_the_end_of_the_run(tmp_path, caplog):
    overrides = write_client_csv(
        tmp_path / "overrides.csv",
        OVERRIDE_HEADER,
        [("fictief_a.pdf", "SK"), ("fictief_b.pdf", "PL"), ("fictief_c.pdf", "CS")],
    )
    resolver = LanguageResolver(overrides)
    resolver.resolve(_placeholder(tmp_path / "invoer" / "fictief_a.pdf"))

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        resolver.report_end_of_run()

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "fictief_b.pdf" in warnings[0]
    assert "fictief_c.pdf" in warnings[0]
    assert "fictief_a.pdf" not in warnings[0]


def test_the_default_override_path_is_under_data_manual():
    path = cli.default_language_overrides_path()

    assert path.name == "language_overrides.csv"
    assert path.parent.name == "manual"
    assert path.parent.parent.name == "data"


def test_parser_default_language_overrides_matches_the_default():
    args = cli.build_parser().parse_args(["some-folder"])
    assert args.language_overrides == cli.default_language_overrides_path()


# --- Through the parse pass (A17, A18) -------------------------------------------


def _stub_pipeline(monkeypatch, document):
    monkeypatch.setattr(parse_pass_module, "extract_pages", lambda path: document)
    monkeypatch.setattr(parse_pass_module, "run_preocr", lambda path: DIGITAL_PREOCR)


def test_the_ocr_seam_receives_the_resolved_code(tmp_path, monkeypatch):
    write_client_csv(
        tmp_path / "00_INDEX.csv",
        INDEX_HEADER,
        [_index_row("Map", "fictief_scan.pdf", "SK")],
    )
    pdf = _placeholder(tmp_path / "Map" / "fictief_scan.pdf")
    _stub_pipeline(monkeypatch, make_document("", ""))
    stub = stub_ocr(monkeypatch, {1: "Fictieve tekst", 2: "Meer fictieve tekst"})

    result = classify_and_assess(pdf)

    assert [call["language"] for call in stub.page_calls] == ["slk", "slk"]
    assert result["language"] == "slk"
    assert result["language_source"] == LANGUAGE_SOURCE_INDEX


def test_language_appears_at_top_level_on_every_page_and_every_store_page(
    tmp_path, monkeypatch
):
    overrides = write_client_csv(
        tmp_path / "overrides.csv", OVERRIDE_HEADER, [("fictief.pdf", "DE")]
    )
    pdf = _placeholder(tmp_path / "invoer" / "fictief.pdf")
    _stub_pipeline(monkeypatch, make_document("x" * 300, ""))
    stub_ocr(monkeypatch, {2: "Fiktiver Text"})

    result = classify_and_assess(pdf, language_resolver=LanguageResolver(overrides))

    assert (result["language"], result["language_source"]) == ("deu", "override")
    for page in result["quality_signals"]["pages"]:
        assert (page["language"], page["language_source"]) == ("deu", "override")
    store = result["page_text"]
    assert (store["language"], store["language_source"]) == ("deu", "override")
    for page in store["pages"]:
        assert (page["language"], page["language_source"]) == ("deu", "override")


def test_without_override_or_index_a_run_uses_nld_with_a_warning_and_continues(
    tmp_path, monkeypatch, caplog
):
    folder = tmp_path / "invoer"
    _placeholder(folder / "fictief_a.pdf")
    _placeholder(folder / "fictief_b.pdf")
    _stub_pipeline(monkeypatch, make_document("x" * 300, ""))
    stub_ocr(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        results = cli.process_folder(folder, tmp_path / "report.csv")

    assert [r["status"] for r in results] == ["processed", "processed"]
    assert [(r["language"], r["language_source"]) for r in results] == [
        ("nld", "default"),
        ("nld", "default"),
    ]
    default_warnings = [m for m in _warnings(caplog) if "default" in m]
    assert len(default_warnings) == 2
    assert any("fictief_a.pdf" in m for m in default_warnings)


def test_a_failed_document_records_no_language():
    result = classify_and_assess("dit-bestand-bestaat-niet.pdf")

    assert result["status"] == "failed"
    assert result["language"] is None
    assert result["language_source"] is None
    assert result["page_text"]["language"] is None
    assert result["page_text"]["language_source"] is None
