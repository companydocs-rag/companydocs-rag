"""Selection and the budget (spec 007, D3, R6, R7, A2).

Fabricated search results go through the answerer with a fake Retriever and
a recording MockTransport; the record's ``context`` and
``dropped_for_budget`` show what was selected. The expected block lengths are
computed from header strings written out here.
"""

from __future__ import annotations

import pytest

from companydocs_rag.answering.answerer import Answerer
from companydocs_rag.answering.prompt import select_per_path
from companydocs_rag.answering.settings import answering_settings
from companydocs_rag.embeddings.dense import SettingsError
from tests.answering._helpers import (
    FakeRetriever,
    Generator,
    chunk,
    numbered_chunks,
    search_result,
    settings_for,
    write_manifest,
)


def _answer(tmp_path, result, *, selection_mode=None, **answering):
    index = tmp_path / "index"
    write_manifest(index)
    generator = Generator()
    with Answerer(
        index,
        settings=settings_for(tmp_path, **answering),
        retriever=FakeRetriever(result),
        transport=generator.transport,
    ) as answerer:
        record = answerer.answer("Fictieve vraag?", selection_mode=selection_mode)
    return record, generator


# --- fused ----------------------------------------------------------------------------


def test_fused_takes_the_first_fused_k_candidates_with_their_ranks(tmp_path):
    chunks = numbered_chunks(6)
    result = search_result(
        dense=[(item, 0.9 - n / 100) for n, item in enumerate(chunks)],
        sparse=[(chunks[5], 5.0), (chunks[3], 4.0)],
    )

    record, _ = _answer(tmp_path, result, fused_k=4)

    candidates = result.candidates
    assert [entry["chunk_id"] for entry in record.context] == [
        candidate["chunk_id"] for candidate in candidates[:4]
    ]
    for position, (entry, candidate) in enumerate(zip(record.context, candidates)):
        assert entry["prompt_number"] == position + 1
        assert entry["selected_by"] == "fused"
        assert (entry["rank"], entry["dense_rank"], entry["sparse_rank"]) == (
            candidate["rank"],
            candidate["dense_rank"],
            candidate["sparse_rank"],
        )
    assert record.dropped_for_budget == []
    assert record.path_hits is None


# --- per_path -------------------------------------------------------------------------


def _shared_lists():
    """Dense A B C D E, sparse B F A: the lists share A and B, and the sparse
    list is shorter than per_path_k 4. With candidate_count 3 the candidates
    are B (dense 2, sparse 1), A (dense 1, sparse 3) and F (sparse 2); C and
    D are only in the dense list."""
    a, b, c, d, e, f = (
        chunk(f"chunk-{letter}", f"Fictieve tekst {letter}.") for letter in "abcdef"
    )
    result = search_result(
        dense=[(a, 0.9), (b, 0.8), (c, 0.7), (d, 0.6), (e, 0.5)],
        sparse=[(b, 9.0), (f, 8.0), (a, 7.0)],
        candidate_count=3,
    )
    assert [candidate["chunk_id"] for candidate in result.candidates] == [
        "chunk-b",
        "chunk-a",
        "chunk-f",
    ]
    return result


# d1 A, s1 B, d2 B (skipped), s2 F, d3 C, s3 A (skipped), d4 D; the sparse
# list has run out, and E is dense rank 5, beyond per_path_k.
EXPECTED_PER_PATH = [
    # chunk_id, selected_by, rank, dense_rank, sparse_rank
    ("chunk-a", "dense", 2, 1, 3),
    ("chunk-b", "sparse", 1, 2, 1),
    ("chunk-f", "sparse", 3, None, 2),
    ("chunk-c", "dense", None, 3, None),
    ("chunk-d", "dense", None, 4, None),
]


def test_per_path_interleaves_the_two_lists_dense_first(tmp_path):
    record, _ = _answer(
        tmp_path, _shared_lists(), selection_mode="per_path", per_path_k=4
    )

    assert [
        (
            entry["chunk_id"],
            entry["selected_by"],
            entry["rank"],
            entry["dense_rank"],
            entry["sparse_rank"],
        )
        for entry in record.context
    ] == EXPECTED_PER_PATH
    assert [entry["prompt_number"] for entry in record.context] == [1, 2, 3, 4, 5]
    assert record.path_hits is not None


def test_per_path_selection_as_a_function():
    result = _shared_lists()

    selected = select_per_path(result.path_hits, result.candidates, 4)

    assert [
        (s.chunk_id, s.selected_by, s.rank, s.dense_rank, s.sparse_rank)
        for s in selected
    ] == EXPECTED_PER_PATH


def test_per_path_skips_a_list_that_runs_out():
    """The dense list is the shorter one here: its turns are skipped."""
    a, b, c = (chunk(f"chunk-{x}", f"Fictief {x}.") for x in "abc")
    result = search_result(dense=[(a, 0.9)], sparse=[(b, 3.0), (c, 2.0), (a, 1.0)])

    selected = select_per_path(result.path_hits, result.candidates, 10)

    assert [(s.chunk_id, s.selected_by) for s in selected] == [
        ("chunk-a", "dense"),
        ("chunk-b", "sparse"),
        ("chunk-c", "sparse"),
    ]


# --- The budget -----------------------------------------------------------------------


def _five_chunks():
    """Five chunks of one document on page 1; the third is long."""
    texts = [
        "Fictieve korte bepaling een.",
        "Fictieve korte bepaling twee.",
        "Fictieve lange bepaling drie. " * 10,
        "Fictieve korte bepaling vier.",
        "Fictieve korte bepaling vijf.",
    ]
    chunks = [chunk(f"c{n}", text) for n, text in enumerate(texts, start=1)]
    result = search_result(
        dense=[(item, 0.9 - n / 100) for n, item in enumerate(chunks)],
        sparse=[(item, 9.0 - n) for n, item in enumerate(chunks)],
    )
    blocks = [
        f"[{n}] (document D1, taal nld, pagina 1)\n{text}"
        for n, text in enumerate(texts, start=1)
    ]
    return result, texts, blocks


def test_the_selection_stops_at_the_first_block_that_does_not_fit(tmp_path):
    result, texts, blocks = _five_chunks()
    # The third block does not fit; the fourth would.
    budget = len(blocks[0]) + len(blocks[1]) + len(blocks[3]) + 5
    assert len(blocks[2]) > len(blocks[3]) + 5

    record, generator = _answer(tmp_path, result, max_context_chars=budget)

    assert [entry["chunk_id"] for entry in record.context] == ["c1", "c2"]
    assert record.dropped_for_budget == ["c3", "c4", "c5"]
    user = generator.bodies[0]["messages"][1]["content"]
    assert user == blocks[0] + "\n\n" + blocks[1] + "\n\nVraag: Fictieve vraag?"
    for text in texts[:2]:
        assert text in user


def test_a_block_that_exactly_fills_the_budget_fits(tmp_path):
    result, texts, blocks = _five_chunks()
    budget = len(blocks[0]) + len(blocks[1]) + len(blocks[2])

    record, generator = _answer(tmp_path, result, max_context_chars=budget)

    assert [entry["chunk_id"] for entry in record.context] == ["c1", "c2", "c3"]
    assert record.dropped_for_budget == ["c4", "c5"]
    user = generator.bodies[0]["messages"][1]["content"]
    for text in texts[:3]:
        assert text in user


def test_the_budget_drops_the_rest_of_the_per_path_selection(tmp_path):
    result = _shared_lists()
    first = "[1] (document D1, taal nld, pagina 1)\nFictieve tekst a."

    record, _ = _answer(
        tmp_path,
        result,
        selection_mode="per_path",
        per_path_k=4,
        max_context_chars=len(first),
    )

    assert [entry["chunk_id"] for entry in record.context] == ["chunk-a"]
    assert record.dropped_for_budget == [row[0] for row in EXPECTED_PER_PATH[1:]]


# --- The constraints (D3, R7) ---------------------------------------------------------


@pytest.mark.parametrize("mode", ["fused", "per_path"])
@pytest.mark.parametrize(
    ("retrieval", "answering", "key"),
    [
        ({"candidate_count": 5}, {"fused_k": 6}, "fused_k"),
        ({"dense_limit": 4}, {"per_path_k": 5}, "per_path_k"),
        ({"sparse_limit": 4}, {"per_path_k": 5}, "per_path_k"),
    ],
    ids=[
        "fused_k-above-candidate_count",
        "per_path_k-above-dense_limit",
        "per_path_k-above-sparse_limit",
    ],
)
def test_the_constraints_hold_whatever_the_mode(mode, retrieval, answering, key):
    with pytest.raises(SettingsError) as raised:
        answering_settings(
            {"retrieval": retrieval, "answering": {"selection_mode": mode, **answering}}
        )

    assert str(raised.value).startswith(f"answering.{key}: ")
    assert (raised.value.section, raised.value.key) == ("answering", key)


def test_the_constraints_allow_equality():
    config = answering_settings(
        {
            "retrieval": {"candidate_count": 5, "dense_limit": 4, "sparse_limit": 6},
            "answering": {"fused_k": 5, "per_path_k": 4},
        }
    )

    assert (config.fused_k, config.per_path_k) == (5, 4)
