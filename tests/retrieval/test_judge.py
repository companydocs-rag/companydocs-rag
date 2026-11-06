"""The judge route (retrieval/judge.py) with a fake judge: no model is loaded."""

from __future__ import annotations

import pytest

from companydocs_rag.embeddings.dense import SettingsError
from companydocs_rag.retrieval import search
from companydocs_rag.retrieval.judge import (
    _budget_batches,
    document_title,
    judge_rank,
    passage_inputs,
    windows,
)


def test_a_short_text_is_one_window():
    assert windows("abc", 10) == ["abc"]


def test_windows_overlap_by_half_and_end_at_the_text_end():
    text = "".join(str(i % 10) for i in range(25))
    parts = windows(text, 10)

    assert parts[0] == text[:10]
    assert parts[1] == text[5:15]
    assert parts[-1] == text[-10:]
    assert all(len(part) <= 10 for part in parts)


def test_the_document_title_comes_from_the_file_name():
    payload = {"file_path": "data/raw/uitzend-cao/cao-abu-2026.pdf"}

    assert document_title(payload) == "cao abu 2026"


def test_every_judged_input_starts_with_the_title_and_fits_the_window():
    inputs = passage_inputs("cao abu 2026", "x" * 5000, 1600)

    assert len(inputs) > 1
    assert all(item.startswith("Document: cao abu 2026\n\n") for item in inputs)
    assert all(len(item) <= 1600 for item in inputs)


def test_budget_batches_cover_every_index_once():
    lengths = [5, 300, 40, 300, 12]
    batches = _budget_batches(lengths, budget=600)

    assert sorted(i for batch in batches for i in batch) == list(range(5))
    assert all(lengths[b[0]] * len(b) <= 600 or len(b) == 1 for b in batches)


def _candidate(chunk_id, file_sha256, text, rank=None):
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "file_sha256": file_sha256,
        "file_path": f"data/raw/x/{file_sha256}.pdf",
        "text": text,
    }


def _fake_judge(items):
    """P(yes) = share of the query's words in the text (document or passage)."""
    out = []
    for query, _kind, text in items:
        words = set(query.lower().split())
        out.append(len(words & set(text.lower().split())) / max(1, len(words)))
    return out


def test_the_judge_route_ranks_the_pool_by_the_judge_and_renumbers():
    documents = {
        "aaa": {"title": "proeftijd regels", "opening": "proeftijd ontslag"},
        "bbb": {"title": "vakantie", "opening": "vakantiedagen"},
    }
    hybrid = [_candidate("c1", "bbb", "vakantiedagen per jaar", rank=1)]
    inside = {
        "aaa": [_candidate("c2", "aaa", "de proeftijd duurt twee maanden")],
        "bbb": [],
    }

    ranked = judge_rank(
        "hoe lang duurt de proeftijd",
        documents,
        hybrid,
        lambda key: inside[key],
        _fake_judge,
        top_documents=1,
        window_chars=1600,
        result_count=20,
    )

    assert [c["chunk_id"] for c in ranked] == ["c2", "c1"]
    assert [c["rank"] for c in ranked] == [1, 2]
    assert ranked[0]["judge_score"] >= ranked[1]["judge_score"]
    assert ranked[0]["judge_document_score"] is not None


def test_only_the_best_documents_are_searched_inside():
    documents = {
        "aaa": {"title": "a", "opening": "proeftijd"},
        "bbb": {"title": "b", "opening": "niets"},
    }
    asked = []

    def doc_chunks(key):
        asked.append(key)
        return []

    judge_rank(
        "proeftijd",
        documents,
        [],
        doc_chunks,
        _fake_judge,
        top_documents=1,
        window_chars=1600,
        result_count=5,
    )

    assert asked == ["aaa"]


def test_a_long_passage_gets_its_best_window():
    long_text = "niets " * 600 + "de proeftijd duurt twee maanden"
    ranked = judge_rank(
        "proeftijd duurt",
        {},
        [_candidate("c1", "aaa", long_text, rank=1)],
        lambda key: [],
        _fake_judge,
        top_documents=1,
        window_chars=400,
        result_count=5,
    )

    assert ranked[0]["judge_score"] == 1.0


def test_the_result_holds_at_most_result_count():
    hybrid = [_candidate(f"c{i}", "aaa", f"tekst {i}", rank=i) for i in range(1, 30)]
    ranked = judge_rank(
        "tekst",
        {},
        hybrid,
        lambda key: [],
        _fake_judge,
        top_documents=1,
        window_chars=1600,
        result_count=20,
    )

    assert len(ranked) == 20


def test_the_judge_route_is_off_by_default():
    config = search.retrieval_settings({"retrieval": {}})

    assert config.reranker == "none"


def test_an_unknown_reranker_is_refused():
    with pytest.raises(SettingsError):
        search.retrieval_settings({"retrieval": {"reranker": "magic"}})


def test_the_judge_route_needs_a_model_folder():
    with pytest.raises(SettingsError):
        search.retrieval_settings({"retrieval": {"reranker": "judge"}})


def test_a_too_small_window_is_refused():
    with pytest.raises(SettingsError):
        search.retrieval_settings({"retrieval": {"judge_window_chars": 50}})
