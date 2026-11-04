"""Relations between documents (spec 008, D5, R7, A3)."""

from __future__ import annotations

import logging

import pytest

from tests.metadata._helpers import (
    AMENDMENT,
    LOGGER,
    SHA_A,
    SHA_B,
    SHA_C,
    SHA_D,
    Generator,
    chunk_record,
    extract,
    extraction_files,
    lease_text,
    messages,
    write_csv,
    write_json_file,
    write_overrides,
    write_relations,
)

DETAIL = "Wijzigt de fictieve huurprijs per 1 januari 2032"


def _document(tmp_path, relative, name, sha, **kwargs):
    path = tmp_path / "los" / relative.replace(".json", "") / name
    write_json_file(
        tmp_path / "chunks",
        relative,
        chunk_record(str(path), sha, lease_text(), **kwargs),
    )


def _documents(tmp_path):
    _document(tmp_path, "allonge.json", "fictief-allonge.pdf", SHA_A)
    _document(tmp_path, "huur.json", "fictief-huur.pdf", SHA_B)
    _document(tmp_path, "kopie.json", "fictief-kopie.pdf", SHA_C)
    _document(tmp_path, "geen-doel.json", "fictief-geen-doel.pdf", SHA_D)
    _document(tmp_path, "twee.json", "fictief-twee.pdf", "e5" * 32)
    _document(tmp_path, "dubbel-1.json", "fictief-dubbel.pdf", "f6" * 32)
    _document(tmp_path, "dubbel-2.json", "fictief-dubbel.pdf", "a7" * 32)
    _document(tmp_path, "leeg-doel.json", "fictief-leeg-doel.pdf", "b8" * 32)
    _document(tmp_path, "zoekt.json", "fictief-zoekt.pdf", "c9" * 32)
    path = tmp_path / "los" / "mislukt" / "fictief-mislukt.pdf"
    write_json_file(
        tmp_path / "chunks",
        "mislukt.json",
        chunk_record(str(path), None, status="failed"),
    )
    _document(tmp_path, "zonder-rij.json", "fictief-zonder-rij.pdf", "d0" * 32)
    return write_overrides(
        tmp_path / "handmatig" / "types.csv", [("fictief-allonge.pdf", AMENDMENT)]
    )


def test_relations_resolve_or_are_recorded_as_not_found(tmp_path, caplog):
    overrides = _documents(tmp_path)
    relations = write_relations(
        tmp_path / "handmatig" / "relaties.csv",
        [
            ("fictief-allonge.pdf", "amends", "fictief-huur.pdf", f" {DETAIL} "),
            ("fictief-kopie.pdf", "duplicate_of", "fictief-huur.pdf", ""),
            ("fictief-geen-doel.pdf", "amends", "fictief-bestaat-niet.pdf", "x"),
            ("fictief-twee.pdf", "amends", "fictief-dubbel.pdf", "y"),
            ("fictief-leeg-doel.pdf", "amends", "", "z"),
            ("fictief-zoekt.pdf", "amends", "fictief-mislukt.pdf", ""),
            ("", "amends", "fictief-huur.pdf", ""),
            ("fictief-allonge.pdf", "duplicate_of", "fictief-twee.pdf", ""),
            ("fictief-weg.pdf", "amends", "fictief-huur.pdf", ""),
        ],
    )
    generator = Generator()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(
            tmp_path, overrides=overrides, relations=relations, generator=generator
        )

    assert code == 0
    found = {
        name: record["relations"]
        for name, record in extraction_files(tmp_path / "extracted").items()
    }
    assert found["allonge.json"] == [
        {
            "relation": "amends",
            "target_file_sha256": SHA_B,
            "detail": DETAIL,
            "status": "resolved",
        }
    ]
    assert list(found["allonge.json"][0]) == [
        "relation",
        "target_file_sha256",
        "detail",
        "status",
    ]
    assert found["kopie.json"] == [
        {
            "relation": "duplicate_of",
            "target_file_sha256": SHA_B,
            "detail": "",
            "status": "resolved",
        }
    ]
    for name, relation, detail in [
        ("geen-doel.json", "amends", "x"),
        ("twee.json", "amends", "y"),
        ("leeg-doel.json", "amends", "z"),
        ("zoekt.json", "amends", ""),
    ]:
        assert found[name] == [
            {
                "relation": relation,
                "target_file_sha256": None,
                "detail": detail,
                "status": "target_not_found",
            }
        ], name
    for name in ("huur.json", "zonder-rij.json", "mislukt.json", "dubbel-1.json"):
        assert found[name] == [], name

    warnings = messages(caplog, logging.WARNING)
    for expected in (
        "Relation of fictief-geen-doel.pdf: target fictief-bestaat-niet.pdf not found",
        "Relation of fictief-twee.pdf: target fictief-dubbel.pdf matches 2 chunk files",
        "Relation of fictief-leeg-doel.pdf: target  not found",
        "Relation of fictief-zoekt.pdf: target fictief-mislukt.pdf has no file sha256",
        f"{relations}: line 8: empty Bestandsnaam; the row is ignored",
        f"{relations} lists fictief-allonge.pdf more than once; the first row is used",
        f"{relations}: no chunk file is named fictief-weg.pdf",
    ):
        assert expected in warnings, expected
    assert sum(w.startswith("Relation of") for w in warnings) == 4
    # The amendment is in scope, its target is not: only one document is sent.
    assert len(generator.user_messages()) == 1


def test_a_missing_relations_file_is_info(tmp_path, caplog):
    _document(tmp_path, "huur.json", "fictief-huur.pdf", SHA_B)
    missing = tmp_path / "handmatig" / "geen.csv"

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert extract(tmp_path, relations=missing) == 0

    assert f"No relations file at {missing}" in messages(caplog, logging.INFO)
    assert extraction_files(tmp_path / "extracted")["huur.json"]["relations"] == []


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [
                ("Bestandsnaam", "Relatie", "Doelbestand"),
                ("fictief-allonge.pdf", "amends", "fictief-huur.pdf"),
            ],
            ": missing column(s) Detail",
        ),
        (
            [
                ("Bestandsnaam", "Relatie", "Doelbestand", "Detail"),
                ("fictief-allonge.pdf", "amends", "fictief-huur.pdf", ""),
                ("fictief-kopie.pdf", "vervangt", "fictief-huur.pdf", ""),
            ],
            ": line 3: Relatie is 'vervangt', not amends or duplicate_of",
        ),
    ],
    ids=["missing-column", "unknown-relation"],
)
def test_a_malformed_relations_file_cannot_start(tmp_path, caplog, rows, expected):
    overrides = _documents(tmp_path)
    relations = write_csv(tmp_path / "handmatig" / "relaties.csv", rows)
    generator = Generator()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(
            tmp_path, overrides=overrides, relations=relations, generator=generator
        )

    assert code == 2
    (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert error == f"Invalid relations file {relations}{expected}"
    assert generator.requests == []
    assert not (tmp_path / "extracted").exists()
    assert not (tmp_path / "logs").exists()
