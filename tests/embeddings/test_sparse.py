"""The sparse side (spec 006): terms (A3), term ids and values (A5).

Fabricated text only; characters outside ASCII are built with ``chr``.
"""

from __future__ import annotations

import hashlib

import pytest

from companydocs_rag.embeddings import sparse
from companydocs_rag.embeddings.sparse import (
    SparseValues,
    bm25_vector,
    document_sparse_vector,
    query_sparse_vector,
    sparse_terms,
    term_id,
    term_id_collisions,
    tokenize,
)
from companydocs_rag.ingestion.normalization import hard_normalize

# Two fabricated terms whose real ids collide (D7, A5). Found outside the
# suite by a birthday search over "term<n>": both have id 306320197.
COLLIDING = ("term121635", "term172646")
COLLIDING_ID = 306320197

E_DIAERESIS = chr(0x00EB)  # U+00EB LATIN SMALL LETTER E WITH DIAERESIS
L_STROKE = chr(0x0142)  # U+0142 LATIN SMALL LETTER L WITH STROKE
COMBINING_DIAERESIS = chr(0x0308)  # U+0308 COMBINING DIAERESIS


# --- tokenize (A3, D6, R1-R3) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        # the four D6 examples
        ("1.1", ["1", "1", "11"]),
        ("31-12-2024", ["31", "12", "2024", "31122024"]),
        ("12.500,00", ["12", "500", "00", "1250000"]),
        ("12 345 678", ["12", "345", "678"]),
        # R1-R3
        ("a1.2.3", ["a1", "2", "3"]),
        ("1.2.3b", ["1", "2", "3b"]),
        ("1..2", ["1", "2"]),
        ("1.2.", ["1", "2", "12"]),
        ("x 1.1 y", ["x", "1", "1", "11", "y"]),
        ("1/2", ["1", "2", "12"]),
        ("a_b", ["a", "b"]),
    ],
)
def test_tokenize_gives_exactly_these_tokens(text, tokens):
    assert tokenize(text) == tokens


def test_composites_follow_their_sequence_in_text_order():
    assert tokenize("zie 8.9 en 31-12-2024, of 1/2") == [
        "zie",
        "8",
        "9",
        "89",
        "en",
        "31",
        "12",
        "2024",
        "31122024",
        "of",
        "1",
        "2",
        "12",
    ]


def test_a_dutch_sentence_keeps_its_one_letter_words_and_stopwords():
    text = "u en ik zien de huur van het pand op 1 mei"

    assert tokenize(text) == text.split()


def test_a_word_with_diacritics_stays_one_token():
    word = "financi" + E_DIAERESIS + "le"
    polish = "op" + L_STROKE + "ata"  # a fabricated word

    assert tokenize(f"de {word} {polish}") == ["de", word, polish]
    # hard_normalize composes a decomposed diacritic, so it stays one term too
    assert sparse_terms("financie" + COMBINING_DIAERESIS + "le") == [word]


@pytest.mark.parametrize(
    "text",
    [
        "ARTIKEL 8.9",
        "Huur" + chr(0x2019) + "s 31" + chr(0x2013) + "12" + chr(0x2013) + "2024",
        "  " + chr(0xFF11) + chr(0xFF12) + chr(0x00A0) + "VAN  ",
    ],
)
def test_sparse_terms_is_tokenize_of_hard_normalize(text):
    assert sparse_terms(text) == tokenize(hard_normalize(text))


def test_sparse_terms_of_an_article_reference():
    assert sparse_terms("ARTIKEL 8.9") == ["artikel", "8", "9", "89"]


def test_the_version_constants():
    assert sparse.TOKENIZER_VERSION == 1
    assert sparse.TERM_HASH == "sha256-u32"


# --- Term ids (A5, D7) ----------------------------------------------------------------


@pytest.mark.parametrize("term", ["huurprijs", "financi" + E_DIAERESIS + "le"])
def test_term_id_is_the_first_four_bytes_of_the_sha256_big_endian(term):
    digest = hashlib.sha256(term.encode("utf-8")).digest()

    assert term_id(term) == int.from_bytes(digest[:4], byteorder="big")
    assert 0 <= term_id(term) < 2**32


def test_the_colliding_pair_shares_one_id():
    assert [sparse_terms(term) for term in COLLIDING] == [
        [COLLIDING[0]],
        [COLLIDING[1]],
    ]
    assert term_id(COLLIDING[0]) == term_id(COLLIDING[1]) == COLLIDING_ID


# --- Document and query values (A5, D8, R4, R5) ---------------------------------------


def test_document_values_on_a_hand_computed_fixture():
    """ "Huur 8.9 huur" has the terms huur, 8, 9, 89, huur: dl is 5, the
    composite included, and tf is 2 for huur and 1 for the others. With
    avgdl 4, k1 1.2 and b 0.75: 1 - b + b*dl/avgdl = 0.25 + 0.75*5/4 = 1.1875,
    and k1 times that is 1.425. huur: 2*2.2/(2 + 1.425) = 4.4/3.425; each
    other term: 1*2.2/(1 + 1.425) = 2.2/2.425."""
    vector = document_sparse_vector("Huur 8.9 huur", avgdl=4.0, k1=1.2, b=0.75)

    expected = {
        term_id("huur"): 4.4 / 3.425,
        term_id("8"): 2.2 / 2.425,
        term_id("9"): 2.2 / 2.425,
        term_id("89"): 2.2 / 2.425,
    }
    assert vector.indices == sorted(expected)
    assert vector.values == pytest.approx(
        [expected[index] for index in vector.indices], rel=1e-12
    )


def test_document_values_depend_on_dl_against_avgdl():
    """One term, tf 1, dl 1: with avgdl 1 the norm is 1 and the value
    2.2/2.2 = 1; with avgdl 2 it is 0.625 and the value 2.2/(1 + 0.75)."""
    assert document_sparse_vector("huur", avgdl=1.0, k1=1.2, b=0.75).values == [
        pytest.approx(1.0)
    ]
    assert document_sparse_vector("huur", avgdl=2.0, k1=1.2, b=0.75).values == [
        pytest.approx(2.2 / 1.75)
    ]


def test_a_term_twice_in_a_query_gets_one_once():
    vector = query_sparse_vector("Huur huur HUUR van")

    assert vector == SparseValues(sorted([term_id("huur"), term_id("van")]), [1.0, 1.0])


def test_colliding_terms_in_one_text_give_one_index_with_the_sum():
    """Both terms tf 1, dl 2, avgdl 2: each is worth 2.2/(1 + 1.2) = 1."""
    text = " ".join(COLLIDING)

    document = document_sparse_vector(text, avgdl=2.0, k1=1.2, b=0.75)
    query = query_sparse_vector(text)

    assert document.indices == [COLLIDING_ID]
    assert document.values == [pytest.approx(2.0)]
    assert query == SparseValues([COLLIDING_ID], [2.0])


def test_colliding_terms_in_a_corpus_count_two():
    assert term_id_collisions([*COLLIDING, "huur", "huur", "van"]) == 2
    assert term_id_collisions(["huur", "van"]) == 0


@pytest.mark.parametrize("text", ["", "--- ... ///", "!?" + chr(0x2026)])
def test_a_text_without_letters_or_digits_gives_an_empty_vector(text):
    empty = SparseValues([], [])

    assert sparse_terms(text) == []
    assert document_sparse_vector(text, avgdl=3.0, k1=1.2, b=0.75) == empty
    assert bm25_vector([], avgdl=None, k1=1.2, b=0.75) == empty
    assert query_sparse_vector(text) == empty
