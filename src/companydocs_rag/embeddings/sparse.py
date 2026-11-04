"""The sparse side of the index (spec 006): terms, term ids and BM25 values.

Terms (D5, D6, R1-R3)
---------------------
:func:`sparse_terms` is ``tokenize(hard_normalize(text))`` and the single
function through which indexing and querying produce terms, so a query and a
chunk that ``hard_normalize`` makes equal give the same terms.

:func:`tokenize` yields base tokens: the maximal runs of characters for which
``str.isalnum()`` is true, in text order (so the underscore separates). On
top of those, a composite token for numbers written with separators: a digit
run is a maximal run of ``0`` to ``9``, and a maximal sequence of two or more
digit runs, each joined to the next by exactly one of ``.`` ``,`` ``-``
``/``, yields one extra token, its digits concatenated, directly after the
last base token of the sequence. A sequence directly preceded or followed by
a letter or digit yields none. So "31-12-2024" gives 31, 12, 2024, 31122024,
and "a1.2.3" gives a1, 2, 3. There are no stopwords, no stemming and no
minimum length, and diacritics stay as ``hard_normalize`` leaves them.
:data:`TOKENIZER_VERSION` is bumped on any change to ``tokenize``,
``sparse_terms`` or ``hard_normalize``.

Term ids and values (D7, D8, R4, R5)
------------------------------------
A term's id is the first four bytes of the sha256 of its UTF-8 bytes, read as
a big-endian unsigned integer (:data:`TERM_HASH`); Python's ``hash()`` is
never used. A document holds, per distinct term,
``tf*(k1+1)/(tf + k1*(1 - b + b*dl/avgdl))``, with ``dl`` the number of terms
(composites included) of its text; a query holds 1.0 per distinct term. IDF
is Qdrant's (``Modifier.IDF``). Distinct terms that share an id add their
values, so each index occurs once, and indices are sorted ascending. A text
without terms gives an empty vector.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections import Counter
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence

from companydocs_rag.ingestion.normalization import hard_normalize

TOKENIZER_VERSION = 1
TERM_HASH = "sha256-u32"

# R1: a digit run is 0-9 only; two or more of them, each joined to the next by
# exactly one separator. A match of this pattern is a maximal sequence: it
# starts at the start of a digit run and takes every run that joins.
_DIGITS = "0123456789"
_SEQUENCE = re.compile(r"[0-9]+(?:[.,/-][0-9]+)+")
_TERM_ID_BYTES = 4


class SparseValues(NamedTuple):
    """A sparse vector: indices ascending, one value per index."""

    indices: List[int]
    values: List[float]


def tokenize(text: str) -> List[str]:
    """The base tokens of ``text`` with the composite number tokens, in text
    order (D6, R1-R3)."""
    # Each composite goes directly after the base token that ends where its
    # sequence ends: no letter or digit follows the sequence, so one does.
    composites: Dict[int, str] = {}
    for match in _SEQUENCE.finditer(text):
        start, end = match.span()
        if start > 0 and text[start - 1].isalnum():
            continue
        if end < len(text) and text[end].isalnum():
            continue
        composites[end] = "".join(char for char in match.group() if char in _DIGITS)

    tokens: List[str] = []
    offset = 0
    for is_token, characters in itertools.groupby(text, key=str.isalnum):
        run = "".join(characters)
        offset += len(run)
        if is_token:
            tokens.append(run)
            if offset in composites:
                tokens.append(composites[offset])
    return tokens


def sparse_terms(text: str) -> List[str]:
    """``tokenize(hard_normalize(text))``: the terms of a chunk's composed
    text and of a query alike (D5)."""
    return tokenize(hard_normalize(text))


def term_id(term: str) -> int:
    """The first four bytes of the sha256 of ``term``'s UTF-8, big-endian
    (D7)."""
    digest = hashlib.sha256(term.encode("utf-8")).digest()
    return int.from_bytes(digest[:_TERM_ID_BYTES], "big")


def _sorted_vector(by_index: Dict[int, float]) -> SparseValues:
    indices = sorted(by_index)
    return SparseValues(indices, [by_index[index] for index in indices])


def bm25_vector(
    terms: Sequence[str], *, avgdl: Optional[float], k1: float, b: float
) -> SparseValues:
    """The document values of ``terms`` (D8): per distinct term
    ``tf*(k1+1)/(tf + k1*(1 - b + b*dl/avgdl))``, colliding ids added (R4).
    No terms give an empty vector, whatever ``avgdl`` is (R5)."""
    dl = len(terms)
    by_index: Dict[int, float] = {}
    for term, tf in Counter(terms).items():
        value = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
        index = term_id(term)
        by_index[index] = by_index.get(index, 0.0) + value
    return _sorted_vector(by_index)


def document_sparse_vector(
    text: str, *, avgdl: Optional[float], k1: float, b: float
) -> SparseValues:
    """The sparse vector of a chunk's composed text (D8), given the corpus's
    ``avgdl`` and the BM25 settings."""
    return bm25_vector(sparse_terms(text), avgdl=avgdl, k1=k1, b=b)


def query_sparse_vector(text: str) -> SparseValues:
    """The sparse vector of a query (D8): 1.0 per distinct term, colliding ids
    added (R4)."""
    by_index: Dict[int, float] = {}
    for term in dict.fromkeys(sparse_terms(text)):
        index = term_id(term)
        by_index[index] = by_index.get(index, 0.0) + 1.0
    return _sorted_vector(by_index)


def term_id_collisions(vocabulary: Iterable[str]) -> int:
    """The number of distinct terms that share their id with at least one
    other distinct term: two terms on one id count 2 (R4)."""
    per_id = Counter(term_id(term) for term in set(vocabulary))
    return sum(count for count in per_id.values() if count > 1)
