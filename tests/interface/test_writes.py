"""Writes and logs of a session (spec 009, D9, D12, R16; Q3; A12).

One run of ``main`` through the real ``Answerer`` (spec 007's fake
Retriever and a MockTransport): three questions, the page view of every
cited page, a 400 and a 404. The raw folder, the override file, the index
and the log are under ``tmp_path``. The texts are distinctive fabricated
strings, so their absence from stderr can be checked.
"""

from __future__ import annotations

import logging
import sys

from companydocs_rag.ingestion.parse_pass import file_sha256
from tests.answering._helpers import (
    FakeRetriever,
    Generator,
    chunk,
    reply,
    search_result,
    span,
    write_manifest,
)
from tests.interface._helpers import (
    MainRun,
    Recorder,
    ask,
    free_port,
    parse_html,
    request,
    snapshot,
    write_overrides,
    write_pdf,
)

QUESTIONS = [
    "Uniekevraagtekst-een over de huur?",
    "Uniekevraagtekst-twee over de duur?",
    "Uniekevraagtekst-drie over de statuten?",
]
ANSWERS = [
    "Uniekantwoordtekst-een [1] en [2].",
    "Uniekantwoordtekst-twee [2].",
    "Uniekantwoordtekst-drie [1].",
]
LABELS = ["Artikel 9 Uniekelabeltekst-een", "Uniekelabeltekst-twee"]
FILE_NAMES = ["fictief-uniekbestand-huur.pdf", "fictief-uniekbestand-statuten.pdf"]


def test_a_session_writes_only_the_answer_log_and_logs_no_question(
    tmp_path, monkeypatch, caplog
):
    raw = tmp_path / "raw"
    huur = write_pdf(raw / "fictiefmap" / FILE_NAMES[0], (200, 300), (300, 200))
    statuten = write_pdf(raw / FILE_NAMES[1], (250, 250))
    overrides = write_overrides(
        tmp_path / "handmatig" / "typen.csv", [(FILE_NAMES[0], "Huurovereenkomst")]
    )
    write_manifest(tmp_path / "index")
    chunks = [
        chunk(
            "c01",
            "Fictieve huurtekst.",
            file_sha256=file_sha256(huur),
            file_path=f"raw/fictiefmap/{FILE_NAMES[0]}",
            page_start=1,
            page_end=2,
            page_spans=[span(1), span(2, origin="ocr", tier=None, confidence=80.0)],
            clause_label=LABELS[0],
        ),
        chunk(
            "c02",
            "Fictieve statutentekst.",
            file_sha256=file_sha256(statuten),
            file_path=f"raw/{FILE_NAMES[1]}",
            clause_label=LABELS[1],
        ),
    ]
    retriever = FakeRetriever(
        search_result(
            dense=[(chunks[0], 0.9), (chunks[1], 0.8)],
            sparse=[(chunks[0], 3.0), (chunks[1], 2.0)],
        )
    )
    generator = Generator(*(reply(answer) for answer in ANSWERS))
    before = snapshot(tmp_path)
    port = free_port()
    stdout, stderr = Recorder(), Recorder()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    with caplog.at_level(logging.DEBUG):
        run = MainRun(
            ["--index-dir", str(tmp_path / "index"), "--port", str(port)],
            raw_dir=raw,
            type_overrides=overrides,
            retriever=retriever,
            transport=generator.transport,
        ).start()
        try:
            run.wait_for_url(stdout)
            startup_stderr = stderr.getvalue()
            startup_records = len(caplog.records)
            fragments = [ask(port, question) for question in QUESTIONS]
            links = [
                link.attrs["href"]
                for fragment in fragments
                for link in parse_html(fragment.text).find_all("a")
                if link.attrs["href"].startswith("/pagina")
            ]
            pages = [request(port, "GET", link) for link in links]
            malformed = request(port, "GET", "/pagina?sha256=fictief&nummer=1")
            unknown = request(port, "GET", "/onbekend")
        finally:
            code = run.finish()
    serving_stderr = stderr.getvalue()[len(startup_stderr) :]
    serving_records = [r.getMessage() for r in caplog.records[startup_records:]]

    assert code == 0
    assert [response.status for response in fragments] == [200, 200, 200]
    assert len(links) == 6
    assert [response.status for response in pages] == [200] * 6
    assert (malformed.status, unknown.status) == (400, 404)

    # Writes: one answer log, nothing else; the inputs are unchanged.
    after = snapshot(tmp_path)
    new = sorted(set(after) - set(before))
    assert len(new) == 1
    assert new[0].startswith("logs/answers-") and new[0].endswith(".jsonl")
    assert {name: after[name] for name in before} == before

    # stderr while serving: one line per request, no question or answer.
    lines = serving_stderr.strip().splitlines()
    assert len(lines) == len(QUESTIONS) + len(links) + 2
    assert [line.split()[:3] for line in lines[:3]] == [["POST", "/vraag", "200"]] * 3
    for line, link in zip(lines[3:-2], links, strict=True):
        assert line.startswith(f"GET {link} 200 ")
        assert line.endswith(" ms")
    assert lines[-2].startswith("GET /pagina 400 ")
    assert lines[-1].startswith("GET /onbekend 404 ")
    for text in [*QUESTIONS, *ANSWERS, *LABELS, *FILE_NAMES, "Uniek", str(raw)]:
        assert text not in serving_stderr, text
        assert not any(text in message for message in serving_records), text
