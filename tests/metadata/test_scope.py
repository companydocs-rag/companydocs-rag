"""Scope: only the lease family is extracted (spec 008, D5, A4)."""

from __future__ import annotations

from tests.metadata._helpers import (
    LEASE,
    OTHER_TYPE,
    SHA_A,
    SHA_B,
    SHA_C,
    Generator,
    chunk_record,
    extract,
    extraction_files,
    pdf,
    use_settings,
    write_index,
    write_json_file,
    write_overrides,
)

FIELDS = [
    "signing_date",
    "commencement_date",
    "initial_term",
    "initial_term_end",
    "renewal",
    "notice_period",
]
IN_SCOPE_TEXT = "De fictieve huur gaat in op 1 april 2030; opzegging per kwartaal."
OUT_OF_SCOPE_TEXT = (
    "Fictieve statuten: de looptijd van het bestuur eindigt op 1 mei 2031."
)


def _tree(tmp_path):
    write_index(
        tmp_path / "bron",
        [(".", "fictief-huur.pdf", LEASE), (".", "fictief-statuten.pdf", OTHER_TYPE)],
    )
    write_json_file(
        tmp_path / "chunks",
        "huur.json",
        chunk_record(pdf(tmp_path, "fictief-huur.pdf"), SHA_A, IN_SCOPE_TEXT),
    )
    write_json_file(
        tmp_path / "chunks",
        "statuten.json",
        chunk_record(pdf(tmp_path, "fictief-statuten.pdf"), SHA_B, OUT_OF_SCOPE_TEXT),
    )


def test_out_of_scope_documents_get_a_record_and_no_request(tmp_path):
    _tree(tmp_path)
    generator = Generator()

    assert extract(tmp_path, generator=generator) == 0

    files = extraction_files(tmp_path / "extracted")
    assert sorted(files) == ["huur.json", "statuten.json"]
    assert files["huur.json"]["extraction_status"] == "extracted"
    out = files["statuten.json"]
    assert out["extraction_status"] == "not_in_scope"
    assert out["document_type"] == OTHER_TYPE
    assert out["batches"] == []
    assert out["fields"] == {
        name: {"status": "not_found", "candidates": []} for name in FIELDS
    }
    assert out["other_termination_provisions"] == []
    sent = [request.content.decode("utf-8") for request in generator.requests]
    assert len(sent) == 1
    assert "fictieve huur" in sent[0]
    assert not any("Fictieve statuten" in body for body in sent)


def test_a_run_with_nothing_in_scope_sends_nothing_and_writes_no_log(
    tmp_path, monkeypatch
):
    use_settings(tmp_path, monkeypatch, warmup=True)
    write_index(tmp_path / "bron", [(".", "fictief-statuten.pdf", OTHER_TYPE)])
    write_json_file(
        tmp_path / "chunks",
        "statuten.json",
        chunk_record(pdf(tmp_path, "fictief-statuten.pdf"), SHA_B, OUT_OF_SCOPE_TEXT),
    )
    generator = Generator()

    assert extract(tmp_path, generator=generator) == 0

    assert generator.requests == []
    assert not (tmp_path / "logs").exists()
    record = extraction_files(tmp_path / "extracted")["statuten.json"]
    assert record["extraction_status"] == "not_in_scope"


def test_a_type_from_the_override_file_alone_puts_a_document_in_scope(tmp_path):
    write_json_file(
        tmp_path / "chunks",
        "los.json",
        chunk_record(tmp_path / "los" / "fictief-los.pdf", SHA_C, IN_SCOPE_TEXT),
    )
    overrides = write_overrides(
        tmp_path / "handmatig" / "types.csv", [("fictief-los.pdf", LEASE)]
    )
    generator = Generator()

    assert extract(tmp_path, overrides=overrides, generator=generator) == 0

    record = extraction_files(tmp_path / "extracted")["los.json"]
    assert (record["document_type"], record["document_type_source"]) == (
        LEASE,
        "override",
    )
    assert record["index_type"] is None
    assert record["extraction_status"] == "extracted"
    assert len(generator.requests) == 1
