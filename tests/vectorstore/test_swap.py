"""The staging folder and the swap (spec 006, D14, R8, Q1, A9).

Every run builds in ``<index-dir>.building``, writes ``manifest.json`` last,
then renames the previous index to ``<index-dir>.previous``, the staging
folder to ``<index-dir>``, and removes the previous index. Failures are forced
by replacing the command's own helpers for one test.
"""

from __future__ import annotations

import logging
import sys

import pytest
from qdrant_client import QdrantClient

from companydocs_rag.retrieval import cli as search_cli
from companydocs_rag.vectorstore import cli
from tests.embeddings._helpers import StubEmbedder
from tests.vectorstore._helpers import (
    LAST_ERROR,
    LOGGER,
    chunk_record,
    messages,
    run_index,
    three_documents,
    tree,
    write_chunk_file,
)

QDRANT_FILES = {".lock", "meta.json", "collection"}


@pytest.fixture
def folders(tmp_path):
    """The chunk files of a new index, and a previous index built from other
    chunk files. Returns (chunks, index, staging, previous, the previous
    index's tree)."""
    old = tmp_path / "eerder"
    write_chunk_file(
        old,
        "eerder.json",
        chunk_record("Een eerdere fictieve tekst.", sha256="c3" * 32),
    )
    index = tmp_path / "index"
    assert run_index(old, index) == 0
    chunks = tmp_path / "chunks"
    three_documents(chunks)
    return (
        chunks,
        index,
        tmp_path / "index.building",
        tmp_path / "index.previous",
        tree(index),
    )


def _names(folder):
    return {path.name for path in folder.iterdir()}


def test_a_leftover_staging_folder_is_removed(tmp_path):
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    three_documents(chunks)
    write_chunk_file(tmp_path / "index.building", "los/bestand.txt", "Iets anders.")

    assert run_index(chunks, index) == 0

    assert not (tmp_path / "index.building").exists()
    assert _names(index) == QDRANT_FILES | {"manifest.json"}


def test_a_run_over_a_previous_index_leaves_only_the_new_index(folders):
    chunks, index, staging, previous, old = folders

    assert run_index(chunks, index) == 0

    assert not staging.exists()
    assert not previous.exists()
    assert _names(index) == QDRANT_FILES | {"manifest.json"}
    assert tree(index) != old
    manifest = (index / "manifest.json").read_text(encoding="utf-8")
    assert '"chunk_file": "a/fictief-huur.json"' in manifest


def _assert_failed_build(caplog, code, expected, index, old, staging):
    assert code == 1
    assert messages(caplog, logging.ERROR) == [expected, LAST_ERROR]
    assert tree(index) == old
    assert _names(staging) == QDRANT_FILES
    assert messages(caplog, logging.INFO) == []


def test_a_point_count_mismatch_leaves_the_previous_index(folders, caplog, monkeypatch):
    chunks, index, staging, previous, old = folders
    upload = cli.upload_points

    def upload_all_but_the_last(client, points, batch_size):
        upload(client, list(points)[:-1], batch_size)

    monkeypatch.setattr(cli, "upload_points", upload_all_but_the_last)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, index)

    _assert_failed_build(
        caplog, code, "Index has 5 points, expected 6", index, old, staging
    )
    assert not previous.exists()


def test_an_error_while_the_manifest_is_written_leaves_the_previous_index(
    folders, caplog, monkeypatch
):
    chunks, index, staging, previous, old = folders

    def fail(folder, manifest):
        raise OSError("schijf vol")

    monkeypatch.setattr(cli, "write_manifest", fail)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, index)

    _assert_failed_build(
        caplog,
        code,
        "Could not build the index: OSError: schijf vol",
        index,
        old,
        staging,
    )
    assert not previous.exists()


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="only Windows refuses to rename a folder with an open file in it",
)
def test_an_index_held_open_is_not_replaced(folders, caplog):
    chunks, index, staging, previous, old = folders
    holder = QdrantClient(path=str(index))
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            code = run_index(chunks, index)
    finally:
        holder.close()

    errors = messages(caplog, logging.ERROR)
    assert code == 1
    assert errors[0].startswith("Could not build the index: PermissionError: ")
    assert errors[1:] == [LAST_ERROR]
    assert tree(index) == old
    assert not previous.exists()
    assert (staging / "manifest.json").is_file()  # the failure came in the swap


def test_a_failed_second_rename_puts_the_previous_index_back(
    folders, caplog, monkeypatch
):
    chunks, index, staging, previous, old = folders
    rename = cli._rename
    calls = []

    def fail_the_second(source, target):
        calls.append((source.name, target.name))
        if len(calls) == 2:
            raise OSError("hernoemen mislukt")
        rename(source, target)

    monkeypatch.setattr(cli, "_rename", fail_the_second)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, index)

    assert calls == [
        ("index", "index.previous"),
        ("index.building", "index"),
        ("index.previous", "index"),
    ]
    assert code == 1
    assert messages(caplog, logging.ERROR) == [
        "Could not build the index: OSError: hernoemen mislukt",
        LAST_ERROR,
    ]
    assert tree(index) == old
    assert not previous.exists()


def test_a_failed_final_removal_keeps_the_new_index_and_warns(
    folders, caplog, monkeypatch
):
    chunks, index, staging, previous, old = folders
    remove = cli._remove

    def fail_on_the_previous_index(path):
        if path == previous and path.exists():
            raise PermissionError("in gebruik")
        remove(path)

    monkeypatch.setattr(cli, "_remove", fail_on_the_previous_index)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, index)

    assert code == 0
    assert messages(caplog, logging.WARNING) == [
        f"Could not remove the previous index {previous}: PermissionError: in gebruik"
    ]
    assert messages(caplog, logging.ERROR) == []
    assert len(messages(caplog, logging.INFO)) == 2
    assert tree(previous) == old
    assert '"chunk_file": "a/fictief-huur.json"' in (index / "manifest.json").read_text(
        encoding="utf-8"
    )
    new = tree(index)

    monkeypatch.setattr(cli, "_remove", remove)
    assert run_index(chunks, index) == 0

    assert not previous.exists()
    assert not staging.exists()
    assert (index / "manifest.json").read_bytes() == new["manifest.json"]


def test_the_search_finds_no_index_in_a_folder_without_a_manifest(
    folders, caplog, capsys
):
    _, index, _, _, _ = folders
    (index / "manifest.json").unlink()

    with caplog.at_level(logging.ERROR):
        code = search_cli.main(
            ["huur", "--index-dir", str(index)], embedder=StubEmbedder()
        )

    assert code == 1
    assert [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR
    ] == [f"No index at {index}: manifest.json missing"]
    assert capsys.readouterr().out == ""
