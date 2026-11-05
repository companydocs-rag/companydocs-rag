"""Output files are replaced whole (spec 008 review F2; R14, R16, R19).

Every extraction file and follow-up file is written to a temporary file in
its folder first, which ``os.replace`` then moves over the old one. Texts
and file names are fabricated.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from companydocs_rag.metadata import files as files_module
from companydocs_rag.metadata.files import find_json_files, replace_file
from tests.metadata._helpers import (
    LEASE,
    LOGGER,
    SHA_A,
    Generator,
    answer,
    candidate,
    chunk_record,
    extract,
    extraction_record,
    found,
    lease_text,
    messages,
    months,
    reply,
    report,
    write_json_file,
    write_overrides,
)

LONE_SURROGATE = chr(0xD800)  # U+D800, a high surrogate without its pair
E_DIAERESIS = chr(0x00EB)  # U+00EB LATIN SMALL LETTER E WITH DIAERESIS
QUOTE = "termijn van drie maanden"
GOOD = answer(notice_period=[candidate(months(3), "duration", QUOTE)])
# The response's JSON escapes the lone surrogate, so it arrives intact and
# only the extraction file cannot hold it in UTF-8 (review F2's evidence).
UNWRITABLE = answer(
    notice_period=[
        candidate(months(3), "duration", QUOTE, as_written="fictief " + LONE_SURROGATE)
    ]
)


def _lease(tmp_path):
    write_json_file(
        tmp_path / "chunks",
        "huur.json",
        chunk_record(tmp_path / "los" / "fictief-huur.pdf", SHA_A, lease_text()),
    )
    return write_overrides(
        tmp_path / "handmatig" / "types.csv", [("fictief-huur.pdf", LEASE)]
    )


def _names(folder):
    return sorted(path.name for path in folder.iterdir())


# --- replace_file ---------------------------------------------------------------------


def test_a_str_is_written_as_write_text_writes_it_and_bytes_as_they_are(tmp_path):
    folder = tmp_path / "uit"
    text = f"fictieve be{E_DIAERESIS}indiging\nen een tweede regel\n"
    (tmp_path / "vergelijk.json").write_text(text, encoding="utf-8")

    replace_file(folder / "a" / "tekst.json", text)
    replace_file(folder / "bytes.md", b"regel\nzonder vertaling\n")

    written = (folder / "a" / "tekst.json").read_bytes()
    assert written == (tmp_path / "vergelijk.json").read_bytes()
    assert (folder / "bytes.md").read_bytes() == b"regel\nzonder vertaling\n"
    assert _names(folder) == ["a", "bytes.md"]
    assert _names(folder / "a") == ["tekst.json"]


def test_an_existing_file_is_replaced(tmp_path):
    target = tmp_path / "uit" / "huur.json"
    replace_file(target, "oud")

    replace_file(target, "nieuw")

    assert target.read_text(encoding="utf-8") == "nieuw"
    assert _names(target.parent) == ["huur.json"]


def test_a_failed_write_keeps_the_old_content_and_no_temporary_file(tmp_path):
    target = tmp_path / "uit" / "huur.json"
    replace_file(target, "oud")
    before = target.read_bytes()

    with pytest.raises(UnicodeEncodeError):
        replace_file(target, "nieuw " + LONE_SURROGATE)

    assert target.read_bytes() == before
    assert _names(target.parent) == ["huur.json"]


def test_a_failed_move_keeps_the_old_content_and_no_temporary_file(
    tmp_path, monkeypatch
):
    target = tmp_path / "uit" / "huur.json"
    replace_file(target, "oud")

    def refuse(source, destination):
        raise OSError("fictieve weigering")

    monkeypatch.setattr(files_module.os, "replace", refuse)

    with pytest.raises(OSError, match="fictieve weigering"):
        replace_file(target, "nieuw")

    assert target.read_text(encoding="utf-8") == "oud"
    assert _names(target.parent) == ["huur.json"]


# --- The extraction file (R14, R16) ---------------------------------------------------


def test_a_failed_write_leaves_the_older_extraction_file_whole(tmp_path, caplog):
    overrides = _lease(tmp_path)
    assert extract(tmp_path, overrides=overrides, generator=Generator(reply(GOOD))) == 0
    path = tmp_path / "extracted" / "huur.json"
    before = path.read_bytes()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(
            tmp_path, overrides=overrides, generator=Generator(reply(UNWRITABLE))
        )

    assert code == 4
    (error,) = messages(caplog, logging.ERROR)
    assert error.startswith("Could not extract huur.json: UnicodeEncodeError: ")
    assert path.read_bytes() == before
    assert _names(tmp_path / "extracted") == ["huur.json"]
    assert report(tmp_path) == 0


def test_a_temporary_file_left_behind_is_never_read_as_an_extraction_file(
    tmp_path, monkeypatch, caplog
):
    overrides = _lease(tmp_path)
    assert extract(tmp_path, overrides=overrides, generator=Generator(reply(GOOD))) == 0
    folder = tmp_path / "extracted"

    def interrupted(source, destination):
        raise OSError("fictieve onderbreking")

    def kept(path):
        raise OSError("fictief vergrendeld")

    # An interrupted process: the temporary file is written in full, then
    # neither moved over the extraction file nor removed.
    with monkeypatch.context() as patched:
        patched.setattr(files_module.os, "replace", interrupted)
        patched.setattr(files_module.os, "unlink", kept)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            code = extract(
                tmp_path, overrides=overrides, generator=Generator(reply(GOOD))
            )

    assert code == 4
    (error,) = messages(caplog, logging.ERROR)
    assert error == "Could not extract huur.json: OSError: fictieve onderbreking"
    (left,) = [path for path in folder.iterdir() if path.name != "huur.json"]
    assert left.name.startswith(".") and left.name.endswith(".tmp")
    held = json.loads(left.read_text(encoding="utf-8"))
    assert held["file"]["path"].endswith("fictief-huur.pdf")
    assert find_json_files(folder) == [folder / "huur.json"]
    assert report(tmp_path) == 0
    summary = json.loads((tmp_path / "rapport" / "followup.json").read_text("utf-8"))
    assert [item["file_name"] for item in summary["documents"]] == ["fictief-huur.pdf"]


# --- The follow-up files (R19) --------------------------------------------------------


def test_report_replaces_each_follow_up_file_through_a_temporary_file(
    tmp_path, monkeypatch
):
    write_json_file(
        tmp_path / "extracted",
        "a.json",
        extraction_record(
            "bron/fictief-a.pdf",
            SHA_A,
            fields={"signing_date": found("2031-01-01", "date")},
        ),
    )
    folder = tmp_path / "rapport"
    names = ["followup.md", "followup.json", "followup.csv"]
    for name in names:
        replace_file(folder / name, b"oud")
    moves = []
    move = os.replace

    def recording(source, destination):
        moves.append((Path(source), Path(destination)))
        move(source, destination)

    monkeypatch.setattr(files_module.os, "replace", recording)

    assert report(tmp_path) == 0

    assert [destination for _, destination in moves] == [
        folder / name for name in names
    ]
    for source, _ in moves:
        assert source.parent == folder
        assert source.name.startswith(".") and source.name.endswith(".tmp")
    assert _names(folder) == sorted(names)
    assert all((folder / name).read_bytes() != b"oud" for name in names)
