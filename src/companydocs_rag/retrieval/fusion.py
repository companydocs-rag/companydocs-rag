"""Reciprocal rank fusion of the two legs (spec 006, D20): a pure function.

Each leg is sorted by score descending, then ``chunk_id``, and ranked from 1.
A chunk's ``rrf_score`` is the sum, over the legs that returned it, of
``1/(rrf_k + rank)``. The fused order is ``rrf_score`` descending, then the
best single-leg rank ascending, then ``chunk_id``, cut to
``candidate_count``. A leg that did not return a chunk leaves its rank and
score None. Qdrant's built-in fusion is not used: it returns fused scores
only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# A leg's result: (chunk_id, score) per returned point, in any order.
LegHits = Iterable[Tuple[str, float]]


@dataclass(frozen=True)
class FusedCandidate:
    """One chunk in fused order, with both legs' ranks and scores."""

    rank: int
    chunk_id: str
    rrf_score: float
    dense_rank: Optional[int]
    dense_score: Optional[float]
    sparse_rank: Optional[int]
    sparse_score: Optional[float]


def rank_leg(hits: LegHits) -> Dict[str, Tuple[int, float]]:
    """``chunk_id`` -> (rank, score): sorted by score descending, then
    ``chunk_id``, ranks from 1. A chunk may occur once per leg."""
    ranked: Dict[str, Tuple[int, float]] = {}
    ordered = sorted(hits, key=lambda hit: (-hit[1], hit[0]))
    for rank, (chunk_id, score) in enumerate(ordered, start=1):
        if chunk_id in ranked:
            raise ValueError(f"chunk {chunk_id} occurs twice in one leg")
        ranked[chunk_id] = (rank, score)
    return ranked


def fuse(
    dense: LegHits, sparse: LegHits, *, rrf_k: int, candidate_count: int
) -> List[FusedCandidate]:
    """The fused candidates of the two legs (D20), at most
    ``candidate_count``."""
    dense_ranked = rank_leg(dense)
    sparse_ranked = rank_leg(sparse)
    entries = []
    for chunk_id in {*dense_ranked, *sparse_ranked}:
        legs = (dense_ranked.get(chunk_id), sparse_ranked.get(chunk_id))
        found = [leg for leg in legs if leg is not None]
        rrf_score = sum(1.0 / (rrf_k + rank) for rank, _ in found)
        best_rank = min(rank for rank, _ in found)
        entries.append((rrf_score, best_rank, chunk_id, legs))
    entries.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))

    fused: List[FusedCandidate] = []
    for rank, (rrf_score, _, chunk_id, (dense_leg, sparse_leg)) in enumerate(
        entries[:candidate_count], start=1
    ):
        fused.append(
            FusedCandidate(
                rank=rank,
                chunk_id=chunk_id,
                rrf_score=rrf_score,
                dense_rank=None if dense_leg is None else dense_leg[0],
                dense_score=None if dense_leg is None else dense_leg[1],
                sparse_rank=None if sparse_leg is None else sparse_leg[0],
                sparse_score=None if sparse_leg is None else sparse_leg[1],
            )
        )
    return fused
