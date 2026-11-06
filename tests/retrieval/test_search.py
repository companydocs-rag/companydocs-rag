"""``Retriever`` (spec 006): A4, A12 through Retriever, A14, R14, and the
``retrieval`` settings (R12, A15).

Indexes are built in ``tmp_path`` with the stub embedder from hand-written
chunk files (``hand_record``) and searched with the stub injected. Characters
outside ASCII are built with ``chr``.
"""

from __future__ import annotations

import pytest
import yaml
from qdrant_client import QdrantClient

from companydocs_rag.embeddings.dense import SettingsError
from companydocs_rag.embeddings.sparse import sparse_terms
from companydocs_rag.ingestion.settings import find_project_root
from companydocs_rag.retrieval import search
from companydocs_rag.retrieval.search import Retriever, retrieval_settings
from tests.embeddings._helpers import StubEmbedder
from tests.vectorstore._helpers import hand_record, run_index, write_chunk_file

# A5's colliding pair: different terms, one id.
COLLIDING = ("term121635", "term172646")


def _index(tmp_path, *records):
    """An index of ``records``, one chunk file each."""
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    for position, record in enumerate(records):
        write_chunk_file(chunks, f"document-{position}.json", record)
    assert run_index(chunks, index) == 0
    return index


def _search(index, query, **retrieval):
    with Retriever(
        index, embedder=StubEmbedder(), settings={"retrieval": retrieval}
    ) as retriever:
        return retriever.search(query)


def _by_text(result):
    return {candidate["text"]: candidate for candidate in result.candidates}


# --- D5 symmetry (A4) -----------------------------------------------------------------


def test_a_query_variant_that_hard_normalize_equalizes_finds_the_same_chunk(tmp_path):
    index = _index(
        tmp_path,
        hand_record(
            "Fictief: de huur's vervaldag is 31-12-2024.",
            "Een andere bepaling zonder datum.",
            "Nog een bepaling over het pand.",
            sha256="11" * 32,
        ),
    )
    plain = "huur's 31-12-2024 vervaldag"
    fullwidth = {digit: chr(0xFF10 + int(digit)) for digit in "0123456789"}
    en_dash, curly_apostrophe, no_break_space = chr(0x2013), chr(0x2019), chr(0x00A0)
    variant = (
        "HUUR"
        + curly_apostrophe
        + "S "
        + "".join(fullwidth[digit] for digit in "31")
        + en_dash
        + "".join(fullwidth[digit] for digit in "12")
        + en_dash
        + "".join(fullwidth[digit] for digit in "2024")
        + no_break_space
        + "VERVALDAG"
    )

    found_plain = _search(index, plain)
    found_variant = _search(index, variant)

    assert (
        found_variant.query_terms
        == found_plain.query_terms
        == [
            "huur",
            "s",
            "31",
            "12",
            "2024",
            "31122024",
            "vervaldag",
        ]
    )
    assert found_variant.query == variant
    (first,) = [c for c in found_variant.candidates if c["sparse_rank"] == 1]
    assert first["text"] == "Fictief: de huur's vervaldag is 31-12-2024."
    assert found_variant.leg_counts["sparse"] == 1


# --- The legs through Retriever (A12, D8) ---------------------------------------------


def _five(tmp_path):
    return _index(
        tmp_path,
        hand_record(
            *[f"Fictieve bepaling {word} over de huur." for word in "abcde"],
            sha256="22" * 32,
        ),
    )


def test_a_query_without_terms_skips_the_sparse_leg(tmp_path, monkeypatch):
    index = _five(tmp_path)

    def must_not_run(*args, **kwargs):
        raise AssertionError("the sparse leg was queried")

    monkeypatch.setattr(search, "query_sparse", must_not_run)

    result = _search(index, "?! ... -- //")

    assert result.query_terms == []
    assert result.leg_counts == {"dense": 5, "sparse": 0}
    assert len(result.candidates) == 5
    assert all(candidate["sparse_rank"] is None for candidate in result.candidates)
    assert all(candidate["sparse_score"] is None for candidate in result.candidates)
    assert all(candidate["dense_rank"] is not None for candidate in result.candidates)


def test_the_limits_bound_the_legs(tmp_path):
    index = _five(tmp_path)

    result = _search(index, "huur", dense_limit=2, sparse_limit=1, candidate_count=3)

    assert result.leg_counts == {"dense": 2, "sparse": 1}
    assert {c["dense_rank"] for c in result.candidates} <= {1, 2, None}
    assert {c["sparse_rank"] for c in result.candidates} <= {1, None}
    assert len(result.candidates) <= 3
    assert result.settings == {
        "rrf_k": 60,
        "dense_limit": 2,
        "sparse_limit": 1,
        "candidate_count": 3,
    }


# --- Matched terms (A14, D21) ---------------------------------------------------------


def test_matched_terms_are_sorted_distinct_and_include_the_clause_label(tmp_path):
    index = _index(
        tmp_path,
        hand_record(
            (
                "Artikel 4 Onderhoud",
                "4",
                "De verhuurder zorgt voor het dak en het dak.",
            ),
            "Een bepaling zonder gedeelde woorden.",
            sha256="33" * 32,
        ),
    )

    result = _search(index, "Dak onderhoud dak")

    labelled = _by_text(result)["De verhuurder zorgt voor het dak en het dak."]
    assert "onderhoud" not in sparse_terms(labelled["text"])
    assert labelled["sparse_matched_terms"] == ["dak", "onderhoud"]
    assert labelled["sparse_rank"] == 1
    other = _by_text(result)["Een bepaling zonder gedeelde woorden."]
    assert (other["sparse_rank"], other["sparse_matched_terms"]) == (None, [])


def test_a_dense_only_candidate_has_its_matched_terms_too(tmp_path):
    """Two chunks share "dak" with the query, but the sparse leg returns one."""
    index = _index(
        tmp_path,
        hand_record(
            "Het dak van het pand.",
            "Een dak, een goot en een muur van het pand.",
            "Iets heel anders.",
            sha256="44" * 32,
        ),
    )

    result = _search(index, "dak", sparse_limit=1)

    shared = [
        candidate
        for candidate in result.candidates
        if candidate["sparse_matched_terms"] == ["dak"]
    ]
    assert len(shared) == 2
    assert sorted(c["sparse_rank"] is None for c in shared) == [False, True]
    assert _by_text(result)["Iets heel anders."]["sparse_matched_terms"] == []


def test_the_sparse_leg_can_return_a_chunk_through_a_colliding_id_alone(tmp_path):
    index = _index(
        tmp_path,
        hand_record(
            f"Fictief {COLLIDING[1]} gevonden.",
            "Een bepaling zonder dat woord.",
            sha256="55" * 32,
        ),
    )

    result = _search(index, COLLIDING[0])

    found = _by_text(result)[f"Fictief {COLLIDING[1]} gevonden."]
    assert result.leg_counts["sparse"] == 1
    assert found["sparse_rank"] == 1
    assert found["sparse_matched_terms"] == []


# --- The library (R14) ----------------------------------------------------------------


def test_an_empty_query_is_a_value_error(tmp_path):
    index = _five(tmp_path)

    with Retriever(index, embedder=StubEmbedder()) as retriever:
        for query in ("", "   ", "\t\n"):
            with pytest.raises(ValueError, match="Empty query"):
                retriever.search(query)


def test_the_query_is_used_as_given(tmp_path):
    index = _five(tmp_path)
    stub = StubEmbedder()

    with Retriever(index, embedder=stub) as retriever:
        result = retriever.search("  de huur  ")

    assert stub.query_calls == ["  de huur  "]
    assert result.query == "  de huur  "


def test_closing_releases_the_index(tmp_path):
    index = _five(tmp_path)
    retriever = Retriever(index, embedder=StubEmbedder())
    retriever.close()
    retriever.close()

    with Retriever(index, embedder=StubEmbedder()):
        pass
    client = QdrantClient(path=str(index))
    client.close()


def test_to_dict_is_the_d22_object(tmp_path):
    index = _five(tmp_path)

    result = _search(index, "huur")
    as_dict = result.to_dict()

    assert list(as_dict) == [
        "query",
        "query_terms",
        "query_instruction",
        "index",
        "settings",
        "leg_counts",
        "candidates",
    ]
    assert as_dict["index"] == {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "model_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "embedding_text_version": 1,
        "tokenizer_version": 1,
    }
    as_dict["candidates"].clear()
    assert result.candidates  # to_dict hands out a copy


# --- Settings (R12, A15) --------------------------------------------------------------

RETRIEVAL_KEYS = [
    ("bm25_k1", "BM25_K1"),
    ("bm25_b", "BM25_B"),
    ("rrf_k", "RRF_K"),
    ("dense_limit", "DENSE_LIMIT"),
    ("sparse_limit", "SPARSE_LIMIT"),
    ("candidate_count", "CANDIDATE_COUNT"),
    ("reranker", "RERANKER"),
    ("judge_model_dir", "JUDGE_MODEL_DIR"),
    ("judge_adapter", "JUDGE_ADAPTER"),
    ("judge_device", "JUDGE_DEVICE"),
    ("judge_documents", "JUDGE_DOCUMENTS"),
    ("judge_doc_chunks", "JUDGE_DOC_CHUNKS"),
    ("judge_window_chars", "JUDGE_WINDOW_CHARS"),
]


def test_the_defaults_are_the_spec_values():
    assert (search.BM25_K1, search.BM25_B, search.RRF_K) == (1.2, 0.75, 60)
    assert (search.DENSE_LIMIT, search.SPARSE_LIMIT, search.CANDIDATE_COUNT) == (
        50,
        50,
        20,
    )
    config = retrieval_settings({})
    for key, constant in RETRIEVAL_KEYS:
        assert getattr(config, key) == getattr(search, constant)


def test_integer_bm25_values_are_read_as_floats():
    config = retrieval_settings({"retrieval": {"bm25_k1": 2, "bm25_b": 1}})

    assert (config.bm25_k1, config.bm25_b) == (2.0, 1.0)
    assert isinstance(config.bm25_k1, float) and isinstance(config.bm25_b, float)
    assert retrieval_settings({"retrieval": {"bm25_b": 0}}).bm25_b == 0.0


@pytest.mark.parametrize(
    ("values", "key"),
    [
        *[
            ({"bm25_k1": value}, "bm25_k1")
            for value in ("1.2", 0, -0.5, float("inf"), float("nan"), None, True)
        ],
        *[
            ({"bm25_b": value}, "bm25_b")
            for value in ("0.75", -0.01, 1.01, float("nan"), None, False)
        ],
        *[
            ({key: value}, key)
            for key in ("rrf_k", "dense_limit", "sparse_limit", "candidate_count")
            for value in ("5", 5.0, 0, None, True)
        ],
        (
            {"dense_limit": 5, "sparse_limit": 6, "candidate_count": 12},
            "candidate_count",
        ),
        ({"rrf_k": 0, "candidate_count": 0}, "rrf_k"),
    ],
)
def test_an_invalid_value_names_its_key(values, key):
    with pytest.raises(SettingsError) as raised:
        retrieval_settings({"retrieval": values})

    assert str(raised.value).startswith(f"retrieval.{key}: ")


def test_candidate_count_may_equal_the_two_limits():
    config = retrieval_settings(
        {"retrieval": {"dense_limit": 5, "sparse_limit": 6, "candidate_count": 11}}
    )

    assert config.candidate_count == 11


def _settings_yaml_text():
    return (find_project_root() / "config" / "settings.yaml").read_text(
        encoding="utf-8"
    )


def test_the_yaml_section_holds_the_python_defaults():
    section = yaml.safe_load(_settings_yaml_text())["retrieval"]

    assert list(section) == [key for key, _ in RETRIEVAL_KEYS]
    for key, constant in RETRIEVAL_KEYS:
        assert section[key] == getattr(search, constant)


def test_the_yaml_section_carries_the_settings_table_comments():
    text = _settings_yaml_text()
    lines = text[text.index("\nretrieval:") + 1 :].splitlines()[1:]
    comment = ""
    found = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            comment += stripped.lstrip("#").strip() + " "
        elif ":" in stripped:
            found[stripped.split(":")[0]] = comment.strip()
            comment = ""

    assert found == {
        "bm25_k1": "Standard value, not measured.",
        "bm25_b": "Standard value, not measured.",
        "rrf_k": "Standard value, not measured.",
        "dense_limit": "Not measured.",
        "sparse_limit": "Not measured.",
        "candidate_count": "Not measured.",
        "reranker": (
            "The judge route (Rutger's experiment, retrieval/judge.py): "
            "none (the baseline) or judge."
        ),
        "judge_model_dir": (
            "The downloaded d3-lite folder; empty while the judge route is off."
        ),
        "judge_adapter": (
            "The judge file (LoRA, readout rows, weight edits); "
            "empty is the base d3-lite."
        ),
        "judge_device": "cpu or cuda:0.",
        "judge_documents": (
            "Not measured on this corpus; the benchmark lab's flat-top 2."
        ),
        "judge_doc_chunks": "Not measured; bounds the judgments per question.",
        "judge_window_chars": (
            "The text length the judge was fine-tuned on (characters)."
        ),
    }
