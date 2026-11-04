"""``Retriever.search(query, include_path_hits=...)`` (spec 007, D2, R5, A1).

Indexes are built in ``tmp_path`` with spec 006's stub embedder and searched
with the stub injected. To show the lists are ranked as fusion ranks a leg
(spec 006 D20), several chunks share one dense vector and the same BM25
statistics, so they tie on both legs, and both legs are wrapped to hand
their hits over in reverse ``chunk_id`` order.
"""

from __future__ import annotations

import json

import pytest

from companydocs_rag.retrieval import cli, search
from companydocs_rag.retrieval.search import Retriever
from companydocs_rag.vectorstore.store import chunk_id_for
from tests.embeddings._helpers import StubEmbedder
from tests.vectorstore._helpers import (
    PAYLOAD_KEYS,
    hand_record,
    run_index,
    stored_points,
    write_chunk_file,
)

D22_KEYS = [
    "query",
    "query_terms",
    "query_instruction",
    "index",
    "settings",
    "leg_counts",
    "candidates",
]
ENTRY_KEYS = ["rank", "chunk_id", "score", *PAYLOAD_KEYS[1:]]

# Four chunks that tie: one dense vector, and for the query "huur" the same
# BM25 statistics (three terms, "huur" once).
TIED = [f"Fictieve huur {letter}." for letter in "abcd"]
OTHERS = [
    "Een bepaling zonder dat woord over het pand.",
    "Nog een fictieve bepaling, over de huur van een pand met een tuin.",
    "Iets heel anders.",
]
SHARED_VECTOR = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _stub():
    vectors = {text: SHARED_VECTOR for text in TIED}
    vectors["huur"] = [1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return StubEmbedder(vectors=vectors)


@pytest.fixture
def index(tmp_path):
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    write_chunk_file(
        chunks, "fictief.json", hand_record(*TIED, *OTHERS, sha256="77" * 32)
    )
    assert run_index(chunks, index, embedder=_stub()) == 0
    return index


@pytest.fixture
def reversed_legs(monkeypatch):
    """Both legs hand their hits over in reverse chunk_id order."""
    real_dense, real_sparse = search.query_dense, search.query_sparse

    def reverse(hits):
        return sorted(hits, key=lambda hit: hit.chunk_id, reverse=True)

    monkeypatch.setattr(search, "query_dense", lambda *args: reverse(real_dense(*args)))
    monkeypatch.setattr(
        search, "query_sparse", lambda *args: reverse(real_sparse(*args))
    )


def _search(index, query, *, retrieval=None, **kwargs):
    with Retriever(
        index, embedder=_stub(), settings={"retrieval": retrieval or {}}
    ) as retriever:
        return retriever.search(query, **kwargs)


def test_without_path_hits_the_result_is_spec_006s(index):
    for result in (
        _search(index, "huur"),
        _search(index, "huur", include_path_hits=False),
    ):
        assert result.path_hits is None
        assert list(result.to_dict()) == D22_KEYS


def test_the_command_output_keeps_its_seven_keys(index, capsys):
    assert (
        cli.main(["huur", "--index-dir", str(index), "--json"], embedder=_stub()) == 0
    )

    assert list(json.loads(capsys.readouterr().out)) == D22_KEYS


def test_path_hits_are_each_legs_ranked_list(index, reversed_legs):
    plain = _search(index, "huur")
    result = _search(index, "huur", include_path_hits=True)

    assert result.to_dict()["candidates"] == plain.to_dict()["candidates"]
    assert list(result.path_hits) == ["dense", "sparse"]
    payloads = {
        chunk_id_for(point_id): point.payload
        for point_id, point in stored_points(index).items()
    }
    for path in ("dense", "sparse"):
        entries = result.path_hits[path]
        assert len(entries) == result.leg_counts[path]
        assert [entry["rank"] for entry in entries] == list(range(1, len(entries) + 1))
        assert [(entry["score"], entry["chunk_id"]) for entry in entries] == sorted(
            ((entry["score"], entry["chunk_id"]) for entry in entries),
            key=lambda pair: (-pair[0], pair[1]),
        )
        for entry in entries:
            assert list(entry) == ENTRY_KEYS
            stored = payloads[entry["chunk_id"]]
            assert {key: entry[key] for key in PAYLOAD_KEYS[1:]} == {
                key: stored[key] for key in PAYLOAD_KEYS[1:]
            }

    # The fixture ties: four chunks share a dense score and a sparse score,
    # and arrive in reverse chunk_id order; they are ranked by chunk_id.
    tied_ids = sorted(
        entry["chunk_id"]
        for entry in result.path_hits["dense"]
        if entry["text"] in TIED
    )
    for path in ("dense", "sparse"):
        tied = [e for e in result.path_hits[path] if e["text"] in TIED]
        assert len({entry["score"] for entry in tied}) == 1
        assert [entry["chunk_id"] for entry in tied] == tied_ids
        ranks = [entry["rank"] for entry in tied]
        assert ranks == list(range(ranks[0], ranks[0] + len(TIED)))


def test_each_candidates_leg_rank_and_score_are_its_path_hit(index, reversed_legs):
    result = _search(index, "huur", include_path_hits=True)

    by_path = {
        path: {entry["chunk_id"]: entry for entry in result.path_hits[path]}
        for path in ("dense", "sparse")
    }
    assert result.candidates
    assert any(candidate["sparse_rank"] is None for candidate in result.candidates)
    for candidate in result.candidates:
        for path in ("dense", "sparse"):
            entry = by_path[path].get(candidate["chunk_id"])
            if candidate[f"{path}_rank"] is None:
                assert entry is None
            else:
                assert (entry["rank"], entry["score"]) == (
                    candidate[f"{path}_rank"],
                    candidate[f"{path}_score"],
                )


def test_the_limits_bound_the_lists_and_a_query_without_terms_has_no_sparse_list(
    index,
):
    limited = _search(
        index,
        "huur",
        retrieval={"dense_limit": 2, "sparse_limit": 1, "candidate_count": 3},
        include_path_hits=True,
    )
    no_terms = _search(index, "?! ... --", include_path_hits=True)

    assert [len(limited.path_hits[path]) for path in ("dense", "sparse")] == [2, 1]
    assert no_terms.path_hits["sparse"] == []
    assert no_terms.leg_counts["sparse"] == 0
    assert len(no_terms.path_hits["dense"]) == no_terms.leg_counts["dense"] == 7


def test_to_dict_puts_path_hits_eighth(index):
    result = _search(index, "huur", include_path_hits=True)

    as_dict = result.to_dict()

    assert list(as_dict) == [*D22_KEYS, "path_hits"]
    assert as_dict["path_hits"] == result.path_hits
    as_dict["path_hits"]["dense"].clear()
    assert result.path_hits["dense"]  # to_dict hands out a copy
