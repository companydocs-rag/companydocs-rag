"""The prompt (spec 007, D4, R8, R9, A3).

The user message is compared, character for character, with a string written
out here. The prompt file's expected text writes its two non-ASCII letters
with ``chr()``: U+00EB LATIN SMALL LETTER E WITH DIAERESIS and U+00E9 LATIN
SMALL LETTER E WITH ACUTE.
"""

from __future__ import annotations

import hashlib
import logging

import pytest

from companydocs_rag.answering import cli
from companydocs_rag.answering.answerer import Answerer
from companydocs_rag.embeddings.dense import SettingsError
from companydocs_rag.ingestion.settings import find_project_root
from tests.answering._helpers import (
    SHA_A,
    SHA_B,
    FakeRetriever,
    Generator,
    chunk,
    search_result,
    settings_for,
    span,
    use_settings,
    write_manifest,
)

E_DIAERESIS = chr(0x00EB)
E_ACUTE = chr(0x00E9)
PROMPT_TEXT = (
    "Je beantwoordt vragen over juridische en financi"
    + E_DIAERESIS
    + "le documenten uitsluitend op basis van de bronnen hieronder. Elke bron "
    "heeft een nummer tussen vierkante haken.\n"
    "Regels:\n"
    "1. Antwoord in het Nederlands, ook als de bronnen in een andere taal zijn.\n"
    "2. Gebruik alleen informatie uit de bronnen, geen eigen kennis.\n"
    "3. Zet achter elke bewering het nummer van de bron, bijvoorbeeld [2]. "
    "Meerdere bronnen: [2][3].\n"
    "4. Neem bedragen, datums, termijnen en namen over zoals ze in de bron "
    "staan. Reken niet en corrigeer niets; de tekst kan leesfouten van een scan "
    "bevatten.\n"
    "5. Spreken bronnen elkaar tegen, noem dan beide met hun nummer.\n"
    "6. Staat het antwoord niet in de bronnen, antwoord dan alleen met: "
    "ONVOLDOENDE BEWIJS, gevolgd door "
    + E_ACUTE
    + E_ACUTE
    + "n zin over wat ontbreekt.\n"
)
PROMPT_FILE = find_project_root() / "config" / "prompts" / "answer_system_nl.txt"
QUESTION = "  Wat is de fictieve opzegtermijn?  "

TEXT_1 = "Artikel 4\nDe fictieve huurder betaalt elke maand vooruit."
TEXT_2 = "Der fiktive Mieter zahlt monatlich im Voraus."
TEXT_3 = "De fictieve opzegtermijn is drie maanden."

# Values that occur in no text, and that the prompt may not show.
HIDDEN = {
    "file_path": [
        "verborgen-map/geheim-bestand-een.pdf",
        "verborgen-map/geheim-twee.pdf",
    ],
    "clause_label": ["Kopregel Zeta Een", "Kopregel Zeta Twee"],
    "chunk_id": [
        "qqqq1111qqqq1111qqqq",
        "wwww2222wwww2222wwww",
        "xxxx3333xxxx3333xxxx",
    ],
    "text_origin": ["native", "ocr"],
}


def _three_chunks():
    first = chunk(
        HIDDEN["chunk_id"][0],
        TEXT_1,
        file_sha256=SHA_A,
        file_path=HIDDEN["file_path"][0],
        language="nld",
        page_start=4,
        clause_label=HIDDEN["clause_label"][0],
    )
    second = chunk(
        HIDDEN["chunk_id"][1],
        TEXT_2,
        file_sha256=SHA_B,
        file_path=HIDDEN["file_path"][1],
        language="deu",
        page_start=2,
        page_end=3,
        page_spans=[span(2, origin="ocr", tier=None, confidence=88.5), span(3)],
        clause_label=HIDDEN["clause_label"][1],
    )
    third = chunk(
        HIDDEN["chunk_id"][2],
        TEXT_3,
        file_sha256=SHA_A,
        file_path=HIDDEN["file_path"][0],
        language="nld",
        page_start=7,
    )
    return [first, second, third]


def _answer(tmp_path, chunks, **answering):
    index = tmp_path / "index"
    write_manifest(index)
    generator = Generator()
    result = search_result(
        dense=[(item, 0.9 - n / 10) for n, item in enumerate(chunks)],
        sparse=[(item, 9.0 - n) for n, item in enumerate(chunks)],
        query=QUESTION,
    )
    with Answerer(
        index,
        settings=settings_for(tmp_path, **answering),
        retriever=FakeRetriever(result),
        transport=generator.transport,
    ) as answerer:
        record = answerer.answer(QUESTION)
    return record, generator


def test_the_user_message_character_for_character(tmp_path):
    record, generator = _answer(tmp_path, _three_chunks())

    expected = (
        "[1] (document D1, taal nld, pagina 4)\n"
        "Artikel 4\n"
        "De fictieve huurder betaalt elke maand vooruit.\n"
        "\n"
        "[2] (document D2, taal deu, pagina 2-3)\n"
        "Der fiktive Mieter zahlt monatlich im Voraus.\n"
        "\n"
        "[3] (document D1, taal nld, pagina 7)\n"
        "De fictieve opzegtermijn is drie maanden.\n"
        "\n"
        "Vraag:   Wat is de fictieve opzegtermijn?  "
    )
    (body,) = generator.bodies
    assert body["messages"][1] == {"role": "user", "content": expected}
    assert record.prompt["user_message"] == expected
    assert [entry["document_label"] for entry in record.context] == ["D1", "D2", "D1"]


def test_the_prompt_shows_no_file_name_label_origin_or_chunk_id(tmp_path):
    chunks = _three_chunks()
    texts = [item["text"] for item in chunks]
    for values in HIDDEN.values():
        for value in values:
            assert all(value not in text for text in texts), value

    record, generator = _answer(tmp_path, chunks)

    (body,) = generator.bodies
    system, user = (message["content"] for message in body["messages"])
    for values in HIDDEN.values():
        for value in values:
            assert value not in system, value
            assert value not in user, value


def test_the_system_message_is_the_prompt_file(tmp_path):
    record, generator = _answer(tmp_path, _three_chunks())

    raw = PROMPT_FILE.read_bytes()
    (body,) = generator.bodies
    assert body["messages"][0] == {"role": "system", "content": raw.decode("utf-8")}
    assert record.prompt["system_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record.prompt["system_file"] == "config/prompts/answer_system_nl.txt"


def test_an_absolute_prompt_file_is_used_as_is(tmp_path, monkeypatch):
    other = tmp_path / "prompt" / "systeem.txt"
    other.parent.mkdir()
    raw = ("Fictieve systeemprompt met " + E_ACUTE + "\n").encode("utf-8")
    other.write_bytes(raw)
    monkeypatch.chdir(tmp_path)

    record, generator = _answer(tmp_path, _three_chunks(), prompt_file=str(other))

    assert generator.bodies[0]["messages"][0]["content"] == raw.decode("utf-8")
    assert record.prompt["system_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record.prompt["system_file"] == str(other)


def test_a_relative_prompt_file_resolves_against_the_project_root(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    record, generator = _answer(tmp_path, _three_chunks())

    assert generator.bodies[0]["messages"][0]["content"] == PROMPT_TEXT


def test_the_committed_prompt_file_holds_d4s_text():
    raw = PROMPT_FILE.read_bytes()

    assert raw == PROMPT_TEXT.encode("utf-8")
    assert b"\r" not in raw
    assert raw.endswith(b".\n") and not raw.endswith(b"\n\n")
    assert len(PROMPT_TEXT) == 751


def _bad_prompt_files(tmp_path):
    not_utf8 = tmp_path / "niet-utf8.txt"
    not_utf8.write_bytes(b"Fictieve prompt met een \xe9 in latin-1.\n")
    return {"missing": tmp_path / "bestaat-niet.txt", "not-utf8": not_utf8}


@pytest.mark.parametrize("case", ["missing", "not-utf8"])
def test_a_bad_prompt_file_is_a_settings_error(tmp_path, case):
    path = _bad_prompt_files(tmp_path)[case]
    write_manifest(tmp_path / "index")
    retriever = FakeRetriever(search_result())

    with pytest.raises(SettingsError) as raised:
        Answerer(
            tmp_path / "index",
            settings=settings_for(tmp_path, prompt_file=str(path)),
            retriever=retriever,
            transport=Generator().transport,
        )

    assert (raised.value.section, raised.value.key) == ("answering", "prompt_file")
    assert retriever.calls == []


@pytest.mark.parametrize("case", ["missing", "not-utf8"])
def test_a_bad_prompt_file_makes_the_command_exit_2(
    tmp_path, monkeypatch, caplog, capsys, case
):
    path = _bad_prompt_files(tmp_path)[case]
    use_settings(tmp_path, monkeypatch, settings_for(tmp_path, prompt_file=str(path)))
    write_manifest(tmp_path / "index")
    retriever, generator = FakeRetriever(search_result()), Generator()

    with caplog.at_level(logging.WARNING, logger="companydocs_rag.answering.cli"):
        code = cli.main(
            ["Fictieve vraag?", "--index-dir", str(tmp_path / "index")],
            retriever=retriever,
            transport=generator.transport,
        )

    assert code == 2
    (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert error.startswith("Invalid settings: answering.prompt_file: ")
    assert capsys.readouterr().out == ""
    assert (retriever.calls, retriever.closed, generator.requests) == ([], 0, [])


def test_a_null_language_is_written_onbekend(tmp_path):
    unknown = chunk("c01", "Fictieve tekst zonder taal.", language=None, page_start=2)

    record, generator = _answer(tmp_path, [unknown])

    user = generator.bodies[0]["messages"][1]["content"]
    assert user.startswith("[1] (document D1, taal onbekend, pagina 2)\n")
