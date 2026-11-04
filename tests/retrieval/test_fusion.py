"""Reciprocal rank fusion as a pure function (spec 006, D20, A12).

Leg results are fabricated (chunk_id, score) pairs. Each tie-break of the
fused order has a fixture in which it decides alone.
"""

from __future__ import annotations

import pytest

from companydocs_rag.retrieval.fusion import FusedCandidate, fuse, rank_leg


def test_a_leg_is_sorted_by_score_descending_then_chunk_id():
    """Equal scores arrive in reverse chunk_id order."""
    ranked = rank_leg([("c", 0.5), ("b", 0.5), ("a", 0.5), ("d", 0.9)])

    assert ranked == {"d": (1, 0.9), "a": (2, 0.5), "b": (3, 0.5), "c": (4, 0.5)}


def test_ranks_are_1_based_and_the_rrf_score_sums_over_the_legs():
    fused = fuse(
        [("a", 0.9), ("b", 0.8)],
        [("c", 1.0), ("b", 3.0)],
        rrf_k=60,
        candidate_count=10,
    )

    assert fused == [
        FusedCandidate(
            rank=1,
            chunk_id="b",
            rrf_score=1 / (60 + 2) + 1 / (60 + 1),
            dense_rank=2,
            dense_score=0.8,
            sparse_rank=1,
            sparse_score=3.0,
        ),
        FusedCandidate(
            rank=2,
            chunk_id="a",
            rrf_score=1 / (60 + 1),
            dense_rank=1,
            dense_score=0.9,
            sparse_rank=None,
            sparse_score=None,
        ),
        FusedCandidate(
            rank=3,
            chunk_id="c",
            rrf_score=1 / (60 + 2),
            dense_rank=None,
            dense_score=None,
            sparse_rank=2,
            sparse_score=1.0,
        ),
    ]


def test_the_rrf_score_decides_first():
    """ "b" is in both legs at rank 2, "a" only in the dense leg at rank 1:
    "a" has the better rank and the smaller id, "b" the larger rrf_score."""
    fused = fuse(
        [("a", 0.9), ("b", 0.8)],
        [("x", 2.0), ("b", 1.0)],
        rrf_k=60,
        candidate_count=10,
    )

    order = [candidate.chunk_id for candidate in fused]
    assert order.index("b") < order.index("a")
    assert fused[order.index("b")].rrf_score == pytest.approx(2 / 62)


def test_on_equal_rrf_scores_the_best_single_leg_rank_decides():
    """With rrf_k 1, "z" at dense rank 1 alone scores 1/2, and "a" at rank 3
    in both legs 1/4 + 1/4 = 1/2: equal, also in floating point. "z" has the
    better best rank and the larger id."""
    fused = fuse(
        [("z", 0.9), ("x", 0.8), ("a", 0.7)],
        [("y", 3.0), ("w", 2.0), ("a", 1.0)],
        rrf_k=1,
        candidate_count=10,
    )

    by_id = {candidate.chunk_id: candidate for candidate in fused}
    assert by_id["z"].rrf_score == by_id["a"].rrf_score == 0.5
    order = [candidate.chunk_id for candidate in fused]
    assert order.index("z") < order.index("a")


def test_on_equal_scores_and_best_ranks_the_chunk_id_decides():
    """ "q" at dense rank 1 and "p" at sparse rank 1: the same rrf_score and
    the same best rank."""
    fused = fuse([("q", 0.9)], [("p", 5.0)], rrf_k=60, candidate_count=10)

    assert [candidate.chunk_id for candidate in fused] == ["p", "q"]
    assert fused[0].rrf_score == fused[1].rrf_score


def test_the_list_is_cut_to_candidate_count():
    dense = [(f"d{position}", 1.0 - position / 10) for position in range(5)]
    sparse = [(f"s{position}", 5.0 - position) for position in range(5)]

    fused = fuse(dense, sparse, rrf_k=60, candidate_count=3)

    assert [candidate.rank for candidate in fused] == [1, 2, 3]
    assert [candidate.chunk_id for candidate in fused] == ["d0", "s0", "d1"]


def test_empty_legs_give_no_candidates():
    assert fuse([], [], rrf_k=60, candidate_count=20) == []


def test_a_chunk_twice_in_one_leg_is_refused():
    with pytest.raises(ValueError):
        rank_leg([("a", 1.0), ("a", 0.5)])
