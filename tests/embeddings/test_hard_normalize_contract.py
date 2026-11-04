"""Contract test: hard_normalize's output, pinned for the sparse index (D9).

``sparse_terms`` is ``tokenize(hard_normalize(text))``, and an index holds the
terms of the chunks as they were normalized when it was built. A change to
``hard_normalize`` changes the terms of new queries but not those of an
existing index, so changing ``hard_normalize`` requires bumping
``TOKENIZER_VERSION`` in ``companydocs_rag.embeddings.sparse``: the search
then refuses an index built with the old version (D17) until it is rebuilt.
If this test fails after such a change, bump the version and update the
expected values here.

The strings are fabricated. Characters outside ASCII are written with
``chr``, named in the comments.
"""

from __future__ import annotations

import pytest

from companydocs_rag.embeddings.sparse import TOKENIZER_VERSION
from companydocs_rag.ingestion.normalization import hard_normalize

CASES = [
    # diacritics stay; case folding lowers the capital: U+00C9 LATIN CAPITAL
    # LETTER E WITH ACUTE, U+00E9 LATIN SMALL LETTER E WITH ACUTE, U+00EB
    # LATIN SMALL LETTER E WITH DIAERESIS
    (
        chr(0x00C9) + "en financi" + chr(0x00EB) + "le",
        chr(0x00E9) + "en financi" + chr(0x00EB) + "le",
    ),
    # U+0141 LATIN CAPITAL LETTER L WITH STROKE, U+0142 LATIN SMALL LETTER L
    # WITH STROKE, U+017D and U+017E LATIN CAPITAL and SMALL LETTER Z WITH CARON
    (
        chr(0x0141) + "AD " + chr(0x017D) + "ELEZ",
        chr(0x0142) + "ad " + chr(0x017E) + "elez",
    ),
    # curly quotes: U+2018 and U+2019 LEFT and RIGHT SINGLE QUOTATION MARK,
    # U+201C and U+201D LEFT and RIGHT DOUBLE QUOTATION MARK, U+201E DOUBLE
    # LOW-9 QUOTATION MARK, U+00AB and U+00BB LEFT- and RIGHT-POINTING DOUBLE
    # ANGLE QUOTATION MARK
    (
        chr(0x2018) + "huur" + chr(0x2019) + "s",
        "'huur's",
    ),
    (
        chr(0x201E) + "a" + chr(0x201C) + " " + chr(0x00AB) + "b" + chr(0x00BB),
        '"a" "b"',
    ),
    # dash variants: U+2010 HYPHEN, U+2011 NON-BREAKING HYPHEN, U+2013 EN DASH,
    # U+2014 EM DASH, U+2015 HORIZONTAL BAR, U+2212 MINUS SIGN
    (
        "1"
        + chr(0x2010)
        + "2"
        + chr(0x2011)
        + "3"
        + chr(0x2013)
        + "4"
        + chr(0x2014)
        + "5"
        + chr(0x2015)
        + "6"
        + chr(0x2212)
        + "7",
        "1-2-3-4-5-6-7",
    ),
    # U+00A0 NO-BREAK SPACE, and whitespace runs collapsed and stripped
    ("  12" + chr(0x00A0) + "500 \t euro\n ", "12 500 euro"),
    # full-width digits: U+FF13 FULLWIDTH DIGIT THREE, U+FF11 FULLWIDTH DIGIT
    # ONE, U+FF0E FULLWIDTH FULL STOP
    (chr(0xFF13) + chr(0xFF11) + chr(0xFF0E) + "12", "31.12"),
    # a ligature: U+FB01 LATIN SMALL LIGATURE FI
    (chr(0xFB01) + "nale", "finale"),
    # case folding: U+00DF LATIN SMALL LETTER SHARP S folds to "ss"
    ("STRA" + chr(0x00DF) + "E Huurder", "strasse huurder"),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_hard_normalize_output_is_pinned(text, expected):
    assert hard_normalize(text) == expected


def test_the_pinned_output_belongs_to_tokenizer_version_1():
    """Bump TOKENIZER_VERSION, and this number with it, when the cases above
    have to change."""
    assert TOKENIZER_VERSION == 1
