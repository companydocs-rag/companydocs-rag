"""The document type: override file, client index, none (spec 008, D4, R6, A2).

The tree under ``tmp_path / "bron"`` holds a client index two folders above
some documents and a nearer index above others. The documents' types are
not in the lease family unless a test says so, so most runs send no request.
"""

from __future__ import annotations

import logging

import pytest

from companydocs_rag.ingestion.settings import find_project_root
from companydocs_rag.metadata import settings as metadata_settings_module
from tests.metadata._helpers import (
    LOGGER,
    UTF8_BOM,
    Generator,
    chunk_record,
    extract,
    extraction_files,
    lease_text,
    messages,
    pdf,
    use_settings,
    write_csv,
    write_index,
    write_json_file,
    write_overrides,
)

OTHER = "Statuten"


def _document(tmp_path, relative, path, sha="a1" * 32, text="Fictieve tekst."):
    write_json_file(tmp_path / "chunks", relative, chunk_record(str(path), sha, text))


def _types(tmp_path):
    return {
        name: (
            record["index_type"],
            record["document_type"],
            record["document_type_source"],
        )
        for name, record in extraction_files(tmp_path / "extracted").items()
    }


@pytest.fixture
def tree(tmp_path):
    """A client index in ``bron`` and a nearer one in ``bron/x``."""
    root = tmp_path / "bron"
    write_index(
        root,
        [
            ("a/b", "fictief-diep.pdf", "Vergunning"),
            (".\\a/./b", "fictief-scheef.pdf", "Statuten"),
            ("x", "fictief-nabij.pdf", "Statuten"),
            ("a", "fictief-overschreven.pdf", "Statuten"),
            ("a", " fictief-spatie.pdf ", "  Vergunning  "),
            ("a", "fictief-leeg.pdf", ""),
            ("a/b", "fictief-diep.pdf", "Tweede rij"),
            ("kapot", "fictief-kapot.pdf", "Statuten"),
            ("zonder", "fictief-zonder.pdf", "Statuten"),
        ],
    )
    write_index(root / "x", [(".", "fictief-nabij.pdf", "Vergunning")])
    return root


def test_override_index_and_none(tmp_path, tree, caplog):
    documents = {
        "diep.json": pdf(tmp_path, "a", "b", "fictief-diep.pdf"),
        "scheef.json": pdf(tmp_path, "a", "b", "fictief-scheef.pdf"),
        "nabij.json": pdf(tmp_path, "x", "fictief-nabij.pdf"),
        "overschreven.json": pdf(tmp_path, "a", "fictief-overschreven.pdf"),
        "spatie.json": pdf(tmp_path, "a", "fictief-spatie.pdf"),
        "leeg.json": pdf(tmp_path, "a", "fictief-leeg.pdf"),
        "onbekend.json": pdf(tmp_path, "a", "fictief-onbekend.pdf"),
        "los.json": tmp_path / "los" / "fictief-los.pdf",
    }
    for relative, path in documents.items():
        _document(tmp_path, relative, path)
    overrides = write_overrides(
        tmp_path / "handmatig" / "types.csv",
        [("fictief-overschreven.pdf", " Vergunning "), ("fictief-los.pdf", "")],
    )

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(tmp_path, overrides=overrides)

    assert code == 0
    assert _types(tmp_path) == {
        "diep.json": ("Vergunning", "Vergunning", "index"),
        "scheef.json": ("Statuten", "Statuten", "index"),
        "nabij.json": ("Vergunning", "Vergunning", "index"),
        "overschreven.json": ("Statuten", "Vergunning", "override"),
        "spatie.json": ("Vergunning", "Vergunning", "index"),
        "leeg.json": (None, None, "none"),
        "onbekend.json": (None, None, "none"),
        "los.json": (None, None, "none"),
    }
    warnings = messages(caplog, logging.WARNING)
    index = tree / "00_INDEX.csv"
    assert f"Empty Documenttype for fictief-leeg.pdf in {index}" in warnings
    assert f"Empty Documenttype for fictief-los.pdf in {overrides}" in warnings
    assert f"No client index above {documents['los.json']}" in warnings
    for relative in ("leeg.json", "onbekend.json", "los.json"):
        assert f"No document type for {documents[relative]}" in warnings
    assert sum(w.startswith("No document type for") for w in warnings) == 3


def test_a_nearest_index_that_cannot_be_used_stops_the_walk(tmp_path, tree, caplog):
    (tree / "kapot").mkdir()
    (tree / "kapot" / "00_INDEX.csv").write_bytes(
        b"Map;Bestandsnaam;Documenttype\n.;fictief-kapot.pdf;Statuten\xff\n"
    )
    write_csv(
        tree / "zonder" / "00_INDEX.csv",
        [("Map", "Bestandsnaam"), (".", "fictief-zonder.pdf")],
    )
    for relative, folder, name in [
        ("kapot.json", "kapot", "fictief-kapot.pdf"),
        ("kapot-2.json", "kapot", "fictief-ander.pdf"),
        ("zonder.json", "zonder", "fictief-zonder.pdf"),
    ]:
        _document(tmp_path, relative, pdf(tmp_path, folder, name))

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(tmp_path)

    assert code == 0
    assert set(_types(tmp_path).values()) == {(None, None, "none")}
    warnings = messages(caplog, logging.WARNING)
    for folder in ("kapot", "zonder"):
        index = tree / folder / "00_INDEX.csv"
        unreadable = [w for w in warnings if w.startswith(f"Client index {index} ")]
        assert len(unreadable) == 1, folder
        assert "is unreadable: " in unreadable[0]


@pytest.mark.parametrize("bom", [True, False], ids=["bom", "no-bom"])
def test_the_override_file_with_and_without_a_byte_order_mark(tmp_path, bom):
    _document(tmp_path, "huur.json", tmp_path / "los" / "fictief-huur.pdf")
    overrides = write_overrides(
        tmp_path / "handmatig" / "types.csv", [("fictief-huur.pdf", OTHER)], bom=bom
    )

    assert extract(tmp_path, overrides=overrides) == 0
    assert _types(tmp_path)["huur.json"] == (None, OTHER, "override")


def test_a_repeated_name_warns_and_the_first_row_is_used(tmp_path, caplog):
    _document(tmp_path, "huur.json", tmp_path / "los" / "fictief-huur.pdf")
    overrides = write_overrides(
        tmp_path / "handmatig" / "types.csv",
        [
            ("fictief-huur.pdf", "Statuten"),
            ("", "Vergunning"),
            ("fictief-huur.pdf", "Vergunning"),
            ("fictief-weg.pdf", "Vergunning"),
        ],
    )

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert extract(tmp_path, overrides=overrides) == 0

    assert _types(tmp_path)["huur.json"] == (None, "Statuten", "override")
    warnings = messages(caplog, logging.WARNING)
    assert (
        f"{overrides} lists fictief-huur.pdf more than once; the first row is used"
        in warnings
    )
    assert f"{overrides}: line 3: empty Bestandsnaam; the row is ignored" in warnings
    assert f"{overrides}: no chunk file is named fictief-weg.pdf" in warnings


def test_a_name_that_matches_several_chunk_files_applies_to_each(tmp_path):
    _document(tmp_path, "a.json", tmp_path / "los" / "a" / "fictief-huur.pdf")
    _document(tmp_path, "b.json", tmp_path / "los" / "b" / "fictief-huur.pdf")
    overrides = write_overrides(
        tmp_path / "handmatig" / "types.csv", [("fictief-huur.pdf", "Statuten")]
    )

    assert extract(tmp_path, overrides=overrides) == 0
    assert set(_types(tmp_path).values()) == {(None, "Statuten", "override")}


def test_a_missing_override_file_is_info_and_the_run_goes_on(tmp_path, caplog):
    _document(tmp_path, "huur.json", tmp_path / "los" / "fictief-huur.pdf")
    missing = tmp_path / "handmatig" / "geen.csv"

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert extract(tmp_path, overrides=missing) == 0

    assert f"No type overrides file at {missing}" in messages(caplog, logging.INFO)
    assert list(_types(tmp_path)) == ["huur.json"]


def _broken_overrides(tmp_path, kind):
    path = tmp_path / "handmatig" / "types.csv"
    if kind == "missing-column":
        write_csv(path, [("Bestandsnaam", "Type"), ("fictief-huur.pdf", "Statuten")])
    elif kind == "not-utf-8":
        path.parent.mkdir(parents=True)
        path.write_bytes(UTF8_BOM + b"Bestandsnaam;Documenttype\nfictief\xff.pdf;X\n")
    else:
        path.mkdir(parents=True)
    return path


@pytest.mark.parametrize("kind", ["missing-column", "not-utf-8", "folder"])
def test_an_unusable_override_file_cannot_start(tmp_path, caplog, kind):
    _document(
        tmp_path, "huur.json", tmp_path / "los" / "fictief-huur.pdf", text=lease_text()
    )
    overrides = _broken_overrides(tmp_path, kind)
    generator = Generator()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(tmp_path, overrides=overrides, generator=generator)

    assert code == 2
    (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert error.startswith(f"Invalid type overrides file {overrides}: ")
    assert generator.requests == []
    assert not (tmp_path / "extracted").exists()
    assert not (tmp_path / "logs").exists()


def test_a_relative_file_path_resolves_against_the_project_root(tmp_path, monkeypatch):
    project = tmp_path / "project"
    write_index(
        project / "bron",
        [
            ("deel", "fictief-voorwaarts.pdf", "Statuten"),
            ("deel", "fictief-terug.pdf", "Vergunning"),
        ],
    )
    monkeypatch.setattr(metadata_settings_module, "project_base", lambda: project)
    prompt = find_project_root() / "config" / "prompts" / "metadata_extract_system.txt"
    use_settings(tmp_path, monkeypatch, prompt_file=str(prompt))
    _document(tmp_path, "voorwaarts.json", "bron/deel/fictief-voorwaarts.pdf")
    _document(tmp_path, "terug.json", "bron\\deel\\fictief-terug.pdf")

    assert extract(tmp_path) == 0
    assert _types(tmp_path) == {
        "terug.json": ("Vergunning", "Vergunning", "index"),
        "voorwaarts.json": ("Statuten", "Statuten", "index"),
    }
