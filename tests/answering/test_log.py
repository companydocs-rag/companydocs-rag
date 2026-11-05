"""The answer log (spec 007, D10, R14, R15, Q5, Q6, A10).

The clock is fixed by replacing ``answerer._utc_now``. Records are compared
with ``to_dict()`` of the answers that wrote them.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

import pytest

from companydocs_rag.answering import answerer as answerer_module
from companydocs_rag.answering import cli
from companydocs_rag.answering.answerer import Answerer
from tests.answering._helpers import (
    FakeRetriever,
    Generator,
    chunk,
    reply,
    search_result,
    settings_for,
    simple_result,
    use_settings,
    write_manifest,
)

FIXED = datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=timezone.utc)

TOP_KEYS = [
    "schema_version",
    "timestamp_utc",
    "question_id",
    "question",
    "status",
    "abstention_source",
    "answer_text",
    "sources",
    "citations",
    "context",
    "dropped_for_budget",
    "candidates",
    "path_hits",
    "prompt",
    "generation",
    "retrieval_ms",
    "settings",
    "index",
    "error",
]
CONTEXT_KEYS = [
    "prompt_number",
    "document_label",
    "chunk_id",
    "file_sha256",
    "page_start",
    "page_end",
    "rank",
    "dense_rank",
    "sparse_rank",
    "selected_by",
]
CANDIDATE_KEYS = [
    "rank",
    "chunk_id",
    "rrf_score",
    "dense_rank",
    "dense_score",
    "sparse_rank",
    "sparse_score",
    "file_sha256",
    "page_start",
    "page_end",
]
GENERATION_KEYS = [
    "model",
    "base_url",
    "finish_reason",
    "truncated",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "context_nearly_full",
    "reasoning",
    "generation_ms",
]
ANSWERING_KEYS = [
    "base_url",
    "model",
    "api_key_env",
    "context_window",
    "max_tokens",
    "temperature",
    "seed",
    "reasoning_effort",
    "extra_body",
    "connect_timeout_s",
    "read_timeout_s",
    "warmup",
    "prompt_file",
    "selection_mode",
    "fused_k",
    "per_path_k",
    "max_context_chars",
    "max_chars_per_prompt_token",
    "min_dense_score",
    "log_dir",
]
INDEX_KEYS = [
    "model_name",
    "model_revision",
    "embedding_text_version",
    "tokenizer_version",
    "manifest_sha256",
]
NAME = re.compile(r"answers-\d{8}T\d{6}Z\.jsonl")
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")


@pytest.fixture
def fixed_clock(monkeypatch):
    monkeypatch.setattr(answerer_module, "_utc_now", lambda: FIXED)


def _open(tmp_path, log_dir=None, result=None, generator=None, **answering):
    write_manifest(tmp_path / "index")
    if log_dir is not None:
        answering["log_dir"] = str(log_dir)
    return Answerer(
        tmp_path / "index",
        settings=settings_for(tmp_path, **answering),
        retriever=FakeRetriever(result if result is not None else simple_result()),
        transport=(generator if generator is not None else Generator()).transport,
    )


def _lines(path):
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    return raw.decode("utf-8").split("\n")[:-1]


def test_a_run_writes_one_file_in_a_new_folder(tmp_path):
    folder = tmp_path / "nieuw" / "logs"
    with _open(tmp_path, folder) as answerer:
        answerer.answer("Fictieve vraag?")

    (path,) = folder.iterdir()
    assert NAME.fullmatch(path.name)


def test_with_the_clock_fixed_later_runs_add_a_suffix(
    tmp_path, monkeypatch, capsys, fixed_clock
):
    use_settings(tmp_path, monkeypatch, settings_for(tmp_path))
    write_manifest(tmp_path / "index")
    for _ in range(3):
        code = cli.main(
            ["Fictieve vraag?", "--index-dir", str(tmp_path / "index")],
            retriever=FakeRetriever(simple_result()),
            transport=Generator().transport,
        )
        assert code == 0

    assert sorted(path.name for path in (tmp_path / "logs").iterdir()) == [
        "answers-20260102T030405Z-2.jsonl",
        "answers-20260102T030405Z-3.jsonl",
        "answers-20260102T030405Z.jsonl",
    ]


def test_a_run_without_records_leaves_no_file(tmp_path):
    with _open(tmp_path):
        pass

    assert not (tmp_path / "logs").exists()


def test_each_line_is_the_compact_record_of_its_answer_in_order(tmp_path, fixed_clock):
    euro = chr(0x20AC)  # U+20AC EURO SIGN
    generator = Generator(
        reply("De fictieve huur is " + euro + " 100 [1]."),
        reply("ONVOLDOENDE BEWIJS, geen bron."),
        reply("Zonder bron."),
    )
    with _open(tmp_path, generator=generator) as answerer:
        answers = [
            answerer.answer(question, question_id=f"v{n}")
            for n, question in enumerate(["Een?", "Twee?", "Drie?"], start=1)
        ]
        path = answerer.log_path

    assert path.name == "answers-20260102T030405Z.jsonl"
    lines = _lines(path)
    assert len(lines) == 3
    for line, answer in zip(lines, answers):
        record = json.loads(line)
        assert record == answer.to_dict()
        assert line == json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    assert euro in lines[0]
    assert [json.loads(line)["question_id"] for line in lines] == ["v1", "v2", "v3"]
    assert answers[0].timestamp_utc == "2026-01-02T03:04:05.678Z"


def test_the_keys_of_a_record(tmp_path):
    result = simple_result()
    manifest = write_manifest(tmp_path / "index")
    with _open(tmp_path, result=result) as answerer:
        record = answerer.answer("Fictieve vraag?", question_id="v1").to_dict()

    assert list(record) == TOP_KEYS
    assert record["schema_version"] == 1
    assert TIMESTAMP.fullmatch(record["timestamp_utc"])
    assert isinstance(record["retrieval_ms"], float)
    assert isinstance(record["generation"]["generation_ms"], float)
    assert list(record["citations"]) == ["valid", "invalid"]
    assert [list(entry) for entry in record["context"]] == [CONTEXT_KEYS] * 3
    assert len(record["candidates"]) == len(result.candidates)
    for entry, candidate in zip(record["candidates"], result.candidates):
        assert list(entry) == CANDIDATE_KEYS
        assert entry == {key: candidate[key] for key in CANDIDATE_KEYS}
    assert record["path_hits"] is None
    assert list(record["prompt"]) == ["system_file", "system_sha256", "user_message"]
    assert list(record["generation"]) == GENERATION_KEYS
    assert list(record["settings"]) == ["answering", "retrieval"]
    assert list(record["settings"]["answering"]) == ANSWERING_KEYS
    assert record["settings"]["retrieval"] == result.settings
    assert list(record["index"]) == INDEX_KEYS
    assert record["index"]["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert {key: record["index"][key] for key in INDEX_KEYS[:4]} == result.index
    assert record["error"] is None


def test_path_hits_are_the_full_lists_in_per_path_mode(tmp_path):
    chunks = [chunk(f"c{n}", f"Fictief {n}.") for n in range(1, 5)]
    result = search_result(
        dense=[(item, 0.9 - n / 10) for n, item in enumerate(chunks)],
        sparse=[(chunks[2], 5.0), (chunks[0], 4.0)],
    )
    with _open(tmp_path, result=result, per_path_k=1) as answerer:
        record = answerer.answer("Fictieve vraag?", selection_mode="per_path")

    assert record.path_hits == {
        "dense": [
            {"rank": n + 1, "chunk_id": f"c{n + 1}", "score": 0.9 - n / 10}
            for n in range(4)
        ],
        "sparse": [
            {"rank": 1, "chunk_id": "c3", "score": 5.0},
            {"rank": 2, "chunk_id": "c1", "score": 4.0},
        ],
    }


def test_the_snapshot_holds_the_mode_the_answer_used(tmp_path, monkeypatch, capsys):
    with _open(tmp_path, selection_mode="fused") as answerer:
        fused = answerer.answer("Fictieve vraag?")
        per_path = answerer.answer("Fictieve vraag?", selection_mode="per_path")

    assert fused.settings["answering"]["selection_mode"] == "fused"
    assert per_path.settings["answering"]["selection_mode"] == "per_path"
    assert per_path.settings["answering"]["log_dir"] == str(tmp_path / "logs")

    use_settings(tmp_path, monkeypatch, settings_for(tmp_path, selection_mode="fused"))
    code = cli.main(
        ["Fictieve vraag?", "--index-dir", str(tmp_path / "index"), "--json"]
        + ["--selection-mode", "per_path"],
        retriever=FakeRetriever(simple_result()),
        transport=Generator().transport,
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["settings"]["answering"]["selection_mode"] == "per_path"


def test_an_answerer_appends_to_its_own_file(tmp_path):
    with _open(tmp_path) as first:
        first.answer("Een?")
        first.answer("Twee?")
        first_path = first.log_path
    with _open(tmp_path) as second:
        second.answer("Drie?")
        second_path = second.log_path

    assert first_path != second_path
    assert first_path.parent == second_path.parent == tmp_path / "logs"
    assert [json.loads(line)["question"] for line in _lines(first_path)] == [
        "Een?",
        "Twee?",
    ]
    assert [json.loads(line)["question"] for line in _lines(second_path)] == ["Drie?"]


def test_a_closed_answerer_refuses_to_answer(tmp_path):
    answerer = _open(tmp_path)
    answerer.close()
    answerer.close()

    with pytest.raises(ValueError, match="The answerer is closed"):
        answerer.answer("Fictieve vraag?")
    with pytest.raises(ValueError, match="Empty question"):
        answerer.answer("  ")
