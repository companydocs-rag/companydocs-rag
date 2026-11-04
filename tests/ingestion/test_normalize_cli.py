"""The companydocs-normalize command (spec 004, D4 and D5).

Store files are fabricated in ``tmp_path`` (``_helpers.store_record``), or
written there by a parse-pass run whose extraction, preocr and OCR seams are
stubbed. Every run passes both folders, so nothing under ``data/`` is read or
written. Characters outside ASCII are built with ``unicodedata.lookup``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import tomllib
import unicodedata
from pathlib import Path

import pytest
import yaml

from companydocs_rag.ingestion import cli, normalization, normalize_cli
from companydocs_rag.ingestion import parse_pass as parse_pass_module
from companydocs_rag.ingestion.settings import find_project_root, load_settings
from tests.ingestion._helpers import make_document, store_page, store_record, stub_ocr
from tests.ingestion.test_quality import CLEAN_PAGE, DIGITAL_PREOCR

LOGGER = "companydocs_rag.ingestion.normalize_cli"

DOCUMENT_KEYS = [
    "schema_version",
    "normalization_version",
    "unicode_version",
    "file",
    "status",
    "page_count",
    "page_labels",
    "header_footer",
    "normalized_sha256",
    "pages",
]
PAGE_KEYS = [
    "page_number",
    "source",
    "text_origin",
    "no_text_reason",
    "language",
    "language_source",
    "tier",
    "ocr_mean_confidence",
    "input_sha256",
    "text",
    "normalized_sha256",
    "removed_lines",
    "joins",
    "counts",
]
CARRIED_KEYS = PAGE_KEYS[:8]
COUNT_KEYS = [
    "nfc_changed",
    "zero_width_removed",
    "soft_hyphens_removed",
    "nbsp_replaced",
    "space_runs_collapsed",
    "trailing_whitespace_stripped",
    "page_label_lines_removed",
    "page_number_run_lines_removed",
    "dehyphenation",
    "filler_runs_collapsed",
    "filler_chars_removed",
]

SOFT_HYPHEN = unicodedata.lookup("SOFT HYPHEN")
NO_BREAK_SPACE = unicodedata.lookup("NO-BREAK SPACE")
ZERO_WIDTH_SPACE = unicodedata.lookup("ZERO WIDTH SPACE")
COMBINING_ACUTE = unicodedata.lookup("COMBINING ACUTE ACCENT")


def _write_store(folder: Path, relative: str, record) -> Path:
    """Write one store file; a string is written as it is."""
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        record
        if isinstance(record, str)
        else json.dumps(record, indent=2, ensure_ascii=False)
    )
    path.write_text(text, encoding="utf-8")
    return path


def _run(store: Path, out: Path) -> int:
    return normalize_cli.main(
        ["--page-text-dir", str(store), "--normalized-dir", str(out)]
    )


def _messages(caplog, level):
    return [r.getMessage() for r in caplog.records if r.levelno == level]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _tree(folder: Path):
    """Every file and folder under ``folder``, with the bytes of each file."""
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        if path.is_file()
        else None
        for path in sorted(folder.rglob("*"))
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _written(out: Path):
    return sorted(path.relative_to(out).as_posix() for path in out.rglob("*.json"))


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """Point the settings at a YAML file written here, for one test."""

    def use(settings):
        path = tmp_path / "instellingen.yaml"
        path.write_text(
            yaml.safe_dump(settings, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        monkeypatch.setenv("COMPANYDOCS_RAG_SETTINGS", str(path))
        load_settings.cache_clear()

    yield use
    load_settings.cache_clear()


# --- The command and its options (A7) ---------------------------------------------


def test_pyproject_registers_the_command():
    with (find_project_root() / "pyproject.toml").open("rb") as handle:
        scripts = tomllib.load(handle)["project"]["scripts"]

    assert scripts["companydocs-normalize"] == (
        "companydocs_rag.ingestion.normalize_cli:main"
    )


def test_the_parser_has_the_two_folders_quiet_and_help_and_nothing_else():
    parser = normalize_cli.build_parser()

    options = sorted(
        option for action in parser._actions for option in action.option_strings
    )
    assert options == [
        "--help",
        "--normalized-dir",
        "--page-text-dir",
        "--quiet",
        "-h",
        "-q",
    ]
    assert [action for action in parser._actions if not action.option_strings] == []


def test_the_default_folders_are_under_data_interim_of_the_project_root(
    monkeypatch, tmp_path
):
    root = find_project_root()
    monkeypatch.chdir(tmp_path)

    args = normalize_cli.build_parser().parse_args([])

    assert args.page_text_dir == root / "data" / "interim" / "page_text"
    assert args.normalized_dir == root / "data" / "interim" / "normalized"
    assert args.page_text_dir == cli.default_page_text_dir()


# --- Store files in, normalized files out (A8) ----------------------------------------


def test_valid_files_mirror_the_store_and_the_others_are_refused_or_fail(
    tmp_path, caplog
):
    store, out = tmp_path / "page_text", tmp_path / "normalized"
    _write_store(store, "a/fictief-1.json", store_record("Tekst een."))
    _write_store(store, "a/b/fictief-2.json", store_record("Tekst twee."))
    _write_store(store, "fictief-3.json", store_record(status="failed", page_count=0))
    old = store_record("Oud.")
    old["schema_version"] = 1
    _write_store(store, "oud/v1.json", old)
    unversioned = store_record("Oud.")
    del unversioned["schema_version"]
    _write_store(store, "oud/zonder.json", unversioned)
    _write_store(store, "kapot.json", "{geen geldige json")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = _run(store, out)

    assert code == 1
    assert _written(out) == ["a/b/fictief-2.json", "a/fictief-1.json", "fictief-3.json"]
    errors = _messages(caplog, logging.ERROR)
    assert (
        f"Refused {Path('oud', 'v1.json')}: schema_version is 1, expected 2" in errors
    )
    assert (
        f"Refused {Path('oud', 'zonder.json')}: schema_version is missing, expected 2"
        in errors
    )
    (failed,) = [message for message in errors if message.startswith("Could not")]
    assert failed.startswith("Could not normalize kapot.json: JSONDecodeError: ")
    assert errors[-2:] == [
        "Refused 2 page text file(s) whose schema_version is not 2; "
        "rerun companydocs-parse-pass",
        "1 page text file(s) could not be normalized",
    ]
    assert len(errors) == 5


@pytest.mark.parametrize(
    ("value", "shown"),
    [(1, "1"), (2.0, "2.0"), ("2", '"2"'), (True, "true"), (None, "null")],
    ids=["one", "float", "string", "true", "null"],
)
def test_only_the_integer_2_is_accepted(tmp_path, caplog, value, shown):
    store, out = tmp_path / "page_text", tmp_path / "normalized"
    record = store_record("Tekst.")
    record["schema_version"] = value
    _write_store(store, "fictief.json", record)

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        code = _run(store, out)

    assert code == 1
    assert not out.exists()
    assert _messages(caplog, logging.ERROR)[0] == (
        f"Refused fictief.json: schema_version is {shown}, expected 2"
    )


def _without_pages():
    record = store_record("Tekst.")
    del record["pages"]
    return json.dumps(record)


def _with_text_origin(origin):
    record = store_record("Tekst.")
    record["pages"][0]["text_origin"] = origin
    return json.dumps(record)


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("[2]", "StoreRecordError: the top level is not a JSON object"),
        (_without_pages(), "StoreRecordError: the record lacks the key 'pages'"),
        (
            _with_text_origin("scan"),
            'StoreRecordError: pages[0].text_origin is "scan", not native, ocr or none',
        ),
    ],
    ids=["not-an-object", "no-pages", "unknown-text-origin"],
)
def test_valid_json_that_is_not_a_store_record_fails(tmp_path, caplog, content, error):
    store, out = tmp_path / "page_text", tmp_path / "normalized"
    _write_store(store, "fictief.json", content)
    _write_store(store, "goed.json", store_record("Tekst."))

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        code = _run(store, out)

    assert code == 1
    assert _written(out) == ["goed.json"]
    assert _messages(caplog, logging.ERROR) == [
        f"Could not normalize fictief.json: {error}",
        "1 page text file(s) could not be normalized",
    ]


def test_a_missing_page_text_folder_exits_2_before_anything_is_written(
    tmp_path, caplog
):
    not_a_folder = tmp_path / "bestand.json"
    not_a_folder.write_text("{}", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert _run(tmp_path / "bestaat-niet", tmp_path / "normalized") == 2
        assert _run(not_a_folder, tmp_path / "normalized") == 2

    assert len(_messages(caplog, logging.ERROR)) == 2
    assert not (tmp_path / "normalized").exists()


def test_an_empty_store_is_a_warning_and_exits_0(tmp_path, caplog):
    store = tmp_path / "page_text"
    store.mkdir()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert _run(store, tmp_path / "normalized") == 0

    assert _messages(caplog, logging.WARNING) == [
        f"No page text files found in {store}"
    ]
    assert "Summary: documents written=0, refused=0, failed=0" in _messages(
        caplog, logging.INFO
    )


def test_file_names_starting_with_a_dot_are_skipped_but_dot_folders_are_read(
    tmp_path,
):
    store, out = tmp_path / "page_text", tmp_path / "normalized"
    _write_store(store, "fictief.json", store_record("Tekst."))
    _write_store(store, ".verborgen.json", store_record("Tekst."))
    _write_store(store, ".map/fictief.json", store_record("Tekst."))

    assert _run(store, out) == 0
    assert _written(out) == [".map/fictief.json", "fictief.json"]


# --- The output (A9) ------------------------------------------------------------------


def test_every_output_has_the_documented_keys_and_carries_the_store_fields(tmp_path):
    store, out = tmp_path / "page_text", tmp_path / "normalized"
    record = store_record(
        "Tekst van de eerste pagina.",
        store_page(2, "Fictieve OCR-tekst.", text_origin="ocr"),
        store_page(3, "", text_origin="none"),
    )
    _write_store(store, "fictief.json", record)
    _write_store(store, "mislukt.json", store_record(status="failed", page_count=2))

    assert _run(store, out) == 0

    document = _read(out / "fictief.json")
    assert list(document) == DOCUMENT_KEYS
    assert document["file"] == {
        "path": record["file"]["path"],
        "sha256": record["file"]["sha256"],
    }
    assert (document["status"], document["page_count"]) == ("processed", 3)
    for page, stored in zip(document["pages"], record["pages"], strict=True):
        assert list(page) == PAGE_KEYS
        assert {key: page[key] for key in CARRIED_KEYS} == {
            key: stored[key] for key in CARRIED_KEYS
        }
        assert list(page["counts"]) == COUNT_KEYS
        assert page["input_sha256"] == _sha256(stored["text"])
        assert page["normalized_sha256"] == _sha256(page["text"])
    assert [page["tier"] for page in document["pages"]] == ["Clean", None, None]
    assert document["pages"][2]["text"] == ""
    pairs = [
        [page["page_number"], page["normalized_sha256"]] for page in document["pages"]
    ]
    assert document["normalized_sha256"] == _sha256(
        json.dumps(pairs, separators=(",", ":"), ensure_ascii=True)
    )
    assert list(document["page_labels"]) == [
        "status",
        "pages_considered",
        "constants",
        "runs",
    ]
    assert document["page_labels"]["status"] == "applied"
    assert list(document["page_labels"]["constants"]) == [
        "edge_lines",
        "min_run_pages",
        "bare_number_max_digits",
        "families",
    ]
    assert document["header_footer"] == {
        "status": "not_built",
        "pages_considered": 2,
        "constants": {},
    }

    failed = _read(out / "mislukt.json")
    assert list(failed) == DOCUMENT_KEYS
    assert (failed["status"], failed["page_count"], failed["pages"]) == (
        "failed",
        2,
        [],
    )
    assert failed["page_labels"]["status"] == "not_applicable"
    assert failed["header_footer"] == {
        "status": "not_applicable",
        "pages_considered": 0,
        "constants": {},
    }
    assert failed["normalized_sha256"] == _sha256("[]")


def test_the_output_is_written_as_the_store_is(tmp_path):
    store, out = tmp_path / "page_text", tmp_path / "normalized"
    text = f"Caf{unicodedata.lookup('LATIN SMALL LETTER E WITH ACUTE')} en huur."
    _write_store(store, "fictief.json", store_record(text))

    assert _run(store, out) == 0

    written = (out / "fictief.json").read_text(encoding="utf-8")
    document = json.loads(written)
    assert written == json.dumps(document, indent=2, ensure_ascii=False)
    assert text in written


def test_the_page_text_store_the_parse_pass_writes_is_normalized_with_its_tiers(
    tmp_path, monkeypatch
):
    """A parse-pass run (extraction, preocr and OCR stubbed) writes the store;
    the command reads it: native, OCR and none pages carry the tier of
    section 2 (A9)."""
    folder = tmp_path / "invoer"
    folder.mkdir()
    (folder / "fictief.pdf").write_bytes(b"fabricated bytes")
    document = make_document(CLEAN_PAGE, "", "")
    monkeypatch.setattr(parse_pass_module, "extract_pages", lambda path: document)
    monkeypatch.setattr(parse_pass_module, "run_preocr", lambda path: DIGITAL_PREOCR)
    stub_ocr(monkeypatch, {2: "Fictieve OCR-tekst over de huur-\nprijs van het pand."})
    store, out = tmp_path / "page_text", tmp_path / "normalized"
    cli.process_folder(folder, tmp_path / "rapport.csv", page_text_dir=store)

    assert _run(store, out) == 0

    stored = _read(store / "fictief.json")
    normalized = _read(out / "fictief.json")
    assert stored["schema_version"] == 2
    assert [page["text_origin"] for page in normalized["pages"]] == [
        "native",
        "ocr",
        "none",
    ]
    assert [page["tier"] for page in normalized["pages"]] == ["Clean", None, None]
    for page, stored_page in zip(normalized["pages"], stored["pages"], strict=True):
        assert {key: page[key] for key in CARRIED_KEYS} == {
            key: stored_page[key] for key in CARRIED_KEYS
        }
    assert normalized["pages"][1]["text"] == (
        "Fictieve OCR-tekst over de huurprijs van het pand."
    )


# --- Reruns, the input and hard normalization (A10, A11, A12) -------------------------


def _mixed_store(store: Path) -> None:
    _write_store(
        store,
        "a/fictief.json",
        store_record(
            "Pagina 1 van 2\nDe huur-\novereenkomst loopt.\nNaam: ........",
            store_page(2, "Fictieve OCR-tekst.\n- 2 -", text_origin="ocr"),
        ),
    )
    _write_store(store, "b/mislukt.json", store_record(status="failed", page_count=0))


def test_a_second_run_writes_byte_identical_files(tmp_path):
    store, out = tmp_path / "page_text", tmp_path / "normalized"
    _mixed_store(store)

    assert _run(store, out) == 0
    first = _tree(out)
    assert _run(store, out) == 0

    assert _tree(out) == first
    assert sorted(name for name, data in first.items() if data is not None) == [
        "a/fictief.json",
        "b/mislukt.json",
    ]


def test_the_run_writes_nothing_outside_the_normalized_folder(tmp_path):
    store = tmp_path / "page_text"
    out = tmp_path / "uit" / "normalized"
    _mixed_store(store)
    before = _tree(tmp_path)

    assert _run(store, out) == 0

    after = _tree(tmp_path)
    outside = {
        name: data
        for name, data in after.items()
        if name not in ("uit", "uit/normalized")
        and not name.startswith("uit/normalized/")
    }
    assert outside == before


def test_hard_normalization_is_not_on_the_output_path(tmp_path, monkeypatch):
    def must_not_run(text):
        raise AssertionError("hard_normalize was called")

    monkeypatch.setattr(normalization, "hard_normalize", must_not_run)
    store, out = tmp_path / "page_text", tmp_path / "normalized"
    _mixed_store(store)

    assert _run(store, out) == 0
    assert _written(out) == ["a/fictief.json", "b/mislukt.json"]


# --- Settings (A19) -------------------------------------------------------------------


@pytest.mark.parametrize("breakage", ["families", "conjunctions", "regex"])
def test_broken_settings_stop_the_command_before_it_writes_anything(
    tmp_path, caplog, settings_file, breakage
):
    settings = copy.deepcopy(load_settings())
    section = settings["ingestion"]["normalization"]
    if breakage == "families":
        del section["page_labels"]["families"]
    elif breakage == "conjunctions":
        del section["dehyphenation"]["conjunctions"]
    else:
        section["page_labels"]["families"]["pagina_van"] = "pagina (?P<page>[0-9"
    settings_file(settings)
    store, out = tmp_path / "page_text", tmp_path / "normalized"
    _mixed_store(store)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = _run(store, out)

    assert code == 2
    assert not out.exists()
    (error,) = _messages(caplog, logging.ERROR)
    assert error.startswith("Invalid settings: ")
    assert _messages(caplog, logging.INFO) == []


def test_the_page_text_folder_is_checked_before_the_settings(
    tmp_path, caplog, settings_file
):
    settings = copy.deepcopy(load_settings())
    del settings["ingestion"]["normalization"]["page_labels"]["families"]
    settings_file(settings)
    missing = tmp_path / "bestaat-niet"

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert _run(missing, tmp_path / "normalized") == 2

    assert _messages(caplog, logging.ERROR) == [f"Not a folder: {missing}"]


# --- The end of the run ---------------------------------------------------------------


def test_the_run_ends_with_the_five_summary_lines(tmp_path, caplog):
    store, out = tmp_path / "page_text", tmp_path / "normalized"
    first = "\n".join(
        [
            "Pagina 1 van 3",
            "De huur-",
            "overeenkomst  loopt. ",
            "Naam: ......",
            "1",
        ]
    )
    second = (
        f"Tek{SOFT_HYPHEN}st{NO_BREAK_SPACE}twee{ZERO_WIDTH_SPACE} "
        f"cafe{COMBINING_ACUTE}\n2"
    )
    third = "OCR-tekst met KvK-\nnummer.\n3"
    _write_store(
        store,
        "fictief.json",
        store_record(
            first,
            second,
            store_page(3, third, text_origin="ocr"),
            store_page(4, "", text_origin="none"),
        ),
    )
    _write_store(store, "mislukt.json", store_record(status="failed", page_count=0))
    old = store_record("Oud.")
    old["schema_version"] = 1
    _write_store(store, "oud.json", old)
    _write_store(store, "kapot.json", "niet eens json")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert _run(store, out) == 1

    summary = [
        message
        for message in _messages(caplog, logging.INFO)
        if message.startswith("Summary: ")
    ]
    assert summary == [
        "Summary: documents written=2, refused=1, failed=1",
        "Summary: pages total=4, native=2, ocr=1, none=1",
        "Summary: removed lines page_label=1, page_number_run=3; by family "
        "dash_number=0, pagina_van=1, slash_total=0, seite_von=0, strona_z=0, "
        "strana_z=0, strana=0, page_of=0",
        "Summary: dehyphenation a_not_lowercase=0, b_conjunction=0, "
        "c_inner_uppercase=1, d_hyphen_kept=0, e_hyphen_removed=1, "
        "e_joined_form_in_document=0, no_next_line=0",
        "Summary: operations nfc_changed=1, zero_width_removed=1, "
        "soft_hyphens_removed=1, nbsp_replaced=1, space_runs_collapsed=1, "
        "trailing_whitespace_stripped=1, filler_runs_collapsed=1, "
        "filler_chars_removed=3",
    ]


def test_a_page_language_without_conjunctions_warns_once_per_run(tmp_path, caplog):
    store, out = tmp_path / "page_text", tmp_path / "normalized"
    for name in ("een.json", "twee.json"):
        _write_store(
            store,
            name,
            store_record(store_page(1, "Het kantoor-\nen pand.", language="ita")),
        )

    with caplog.at_level(logging.WARNING):
        assert _run(store, out) == 0

    warnings = _messages(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "'ita'" in warnings[0]
