"""Formatting-consistency tests: each generic structural pattern in isolation.

Reminder of what is being tested (see structure.py): this is a PRESENCE check
for recognisable structure, an interim proxy for formatting consistency in the
fuller sense. The tests check that each pattern fires on the shape it is meant
to catch and does not fire on the others.
"""

from __future__ import annotations

import pytest

from companydocs_rag.ingestion.structure import analyse_structure, describe_pattern
from tests.ingestion._helpers import make_pages

# --- (a) Numbered clauses, in their several variants ------------------------

ARABIC_ARTICLES = """
Artikel 1 Definities
In deze overeenkomst wordt verstaan onder het gehuurde de kantoorruimte.
Artikel 2 Duur en verlenging
De overeenkomst wordt aangegaan voor de duur van vijf jaar.
Artikel 3 Betaling
De betaling geschiedt maandelijks bij vooruitbetaling.
"""

DECIMAL_NUMBERING = """
1.1 De huurprijs wordt jaarlijks aangepast aan de index.
1.2 De waarborgsom bedraagt drie maandtermijnen.
2.1 Oplevering vindt plaats in de staat waarin het zich bevindt.
"""

ROMAN_ARTICLES = """
Article I Definitions
The parties agree on the definitions set out below.
Article II Term
The lease term is five years from the commencement date.
Article III Rent
Rent is payable monthly in advance.
"""

LETTERED_SUBCLAUSES = """
(a) de huurder draagt zorg voor het dagelijks onderhoud;
(b) de verhuurder verzorgt het groot onderhoud;
(c) partijen treden jaarlijks in overleg over de staat van het gehuurde.
"""

NESTED_LETTERED_SUBCLAUSES = """
Conform 2.1(a) is de huurder aansprakelijk voor de schade.
Zie ook 2.1(b) van de bij deze overeenkomst behorende bepalingen.
"""

# --- (b) Fixed field-label/value pairs --------------------------------------

COLON_SEPARATED_FIELDS = """
KvK-nummer: 12345678
Statutaire naam: Voorbeeld Holding B.V.
Rechtsvorm: Besloten Vennootschap
Statutaire zetel: Amsterdam
Bezoekadres: Voorbeeldweg 1
Datum van oprichting: 12-03-1998
"""

# NOT how pdfplumber renders the two-column KvK Handelsregister layout (the
# original comment here claimed it was; that was factually wrong -- see
# structure.py's known-broken marker on the column-aligned pattern, and
# test_column_aligned_field_labels_as_pdfplumber_actually_renders_them below
# for what pdfplumber's real output looks like). This is a hand-fabricated,
# wide-gap layout used only to exercise that regex in isolation, on its own
# terms, independent of whether real extracted text can ever produce it.
COLUMN_ALIGNED_FIELDS = """
KvK-nummer            12345678
Statutaire naam       Voorbeeld Holding B.V.
Rechtsvorm            Besloten Vennootschap
Bezoekadres           Voorbeeldweg 1
"""

# What a two-column KvK row actually looks like once extracted through this
# codebase's extraction.py (`Page.extract_text()`, no kwargs -> pdfplumber's
# `layout=False` path): a single space, no colon. See the known-broken marker
# on the column-aligned pattern in structure.py for why neither field-label
# pattern fires on this.
REAL_COLUMN_ALIGNED_FIELDS_AS_EXTRACTED = """
KvK-nummer 12345678
Statutaire naam Voorbeeld Holding B.V.
Rechtsvorm Besloten Vennootschap
Bezoekadres Voorbeeldweg 1
"""

# --- (c) Heading-driven sections without numbering --------------------------

ALL_CAPS_HEADINGS = """
ALGEMENE BEPALINGEN
De verhuurder en de huurder komen het volgende met elkaar overeen.
OPLEVERING VAN HET GEHUURDE
Het gehuurde wordt bezemschoon en ontruimd opgeleverd.
ONDERHOUD EN HERSTELLINGEN
Partijen dragen zorg voor het dagelijks onderhoud.
"""

TITLE_CASE_HEADINGS = """
Duur En Verlenging
De overeenkomst loopt door voor de duur van vijf jaar.
Betaling Van De Huurprijs
Betaling geschiedt maandelijks bij vooruitbetaling.
Oplevering En Onderhoud
Het gehuurde wordt bezemschoon opgeleverd aan de verhuurder.
"""

# --- No recognisable structure ----------------------------------------------

UNSTRUCTURED_PROSE = """
Wij hebben uw bericht in goede orde ontvangen en danken u hartelijk daarvoor.
Naar aanleiding daarvan willen wij graag op korte termijn een afspraak maken.
Wij nemen binnenkort telefonisch contact met u op om dit verder te bespreken.
"""


def test_arabic_article_numbering():
    result = analyse_structure(make_pages(ARABIC_ARTICLES))
    assert result.matched_pattern == "numbered_clauses"
    assert result.first_page == 1


def test_decimal_nested_numbering():
    result = analyse_structure(make_pages(DECIMAL_NUMBERING))
    assert result.matched_pattern == "numbered_clauses"


def test_roman_article_numbering():
    result = analyse_structure(make_pages(ROMAN_ARTICLES))
    assert result.matched_pattern == "numbered_clauses"


def test_lettered_subclauses():
    result = analyse_structure(make_pages(LETTERED_SUBCLAUSES))
    assert result.matched_pattern == "numbered_clauses"


def test_lettered_subclauses_nested_in_decimal_numbering():
    result = analyse_structure(make_pages(NESTED_LETTERED_SUBCLAUSES))
    assert result.matched_pattern == "numbered_clauses"


def test_colon_separated_field_labels():
    result = analyse_structure(make_pages(COLON_SEPARATED_FIELDS))
    assert result.matched_pattern == "field_label_pairs"
    assert result.first_page == 1


def test_column_aligned_field_labels():
    """Exercises the column-aligned regex in isolation, on its own terms.

    This is NOT extraction-realistic input -- see the fixture's comment and
    the skipped test below for the real gap.
    """
    result = analyse_structure(make_pages(COLUMN_ALIGNED_FIELDS))
    assert result.matched_pattern == "field_label_pairs"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN-BROKEN (spec 001), not fixed here: extraction.py's "
        "Page.extract_text() (no kwargs) routes to pdfplumber's "
        "layout=False path, which collapses inter-word gaps to a single "
        "space and produces no colon for a two-column KvK row -- "
        "'KvK-nummer 12345678'. Neither field-label pattern matches that "
        "shape, so this is the real gap the (never-matching) column-aligned "
        "regex was meant to cover. See structure.py's known-broken marker "
        "on that pattern for the full reasoning; rewriting the regex against "
        "a second untested assumption is deferred until real documents "
        "exist to check it against."
    ),
)
def test_column_aligned_field_labels_as_pdfplumber_actually_renders_them():
    result = analyse_structure(make_pages(REAL_COLUMN_ALIGNED_FIELDS_AS_EXTRACTED))
    assert result.matched_pattern == "field_label_pairs"


def test_all_caps_headings_without_numbering():
    result = analyse_structure(make_pages(ALL_CAPS_HEADINGS))
    assert result.matched_pattern == "headings"


def test_title_case_headings_without_numbering():
    result = analyse_structure(make_pages(TITLE_CASE_HEADINGS))
    assert result.matched_pattern == "headings"


# --- Negative and edge cases ------------------------------------------------


def test_plain_prose_has_no_structural_match():
    result = analyse_structure(make_pages(UNSTRUCTURED_PROSE))
    assert result.matched_pattern is None
    assert result.score == 0.0
    assert result.first_page is None


def test_empty_document_does_not_crash():
    result = analyse_structure(make_pages())
    assert result.matched_pattern is None
    assert result.score == 0.0


def test_a_single_numbered_line_is_below_the_minimum():
    """One stray numbered line is not a structural pattern."""
    result = analyse_structure(
        make_pages("Artikel 1 Definities\nGewone lopende tekst hier.")
    )
    assert result.matched_pattern is None


def test_each_line_is_credited_to_only_the_most_specific_pattern():
    """A numbered heading is numbering, not a heading or a label pair."""
    result = analyse_structure(make_pages(ARABIC_ARTICLES))
    by_name = {pattern.name: pattern.hits for pattern in result.patterns}
    assert by_name["numbered_clauses"] == 3
    assert by_name["headings"] == 0
    assert by_name["field_label_pairs"] == 0


def test_best_match_wins_when_several_patterns_are_present():
    mixed = ARABIC_ARTICLES + ARABIC_ARTICLES + "\nBezoekadres: Voorbeeldweg 1\n"
    result = analyse_structure(make_pages(mixed))
    assert result.matched_pattern == "numbered_clauses"


def test_first_page_of_the_winning_pattern_is_recorded():
    pages = make_pages("Voorblad bij de overeenkomst", ARABIC_ARTICLES)
    result = analyse_structure(pages)
    assert result.matched_pattern == "numbered_clauses"
    assert result.first_page == 2


def test_all_patterns_are_reported_for_later_recalibration():
    result = analyse_structure(make_pages(ARABIC_ARTICLES))
    assert {pattern.name for pattern in result.patterns} == {
        "numbered_clauses",
        "field_label_pairs",
        "headings",
    }


def test_saturation_can_be_overridden_through_settings():
    lenient = {
        "ingestion": {"parse_pass": {"structure": {"saturation": {"headings": 1}}}}
    }
    result = analyse_structure(make_pages(ALL_CAPS_HEADINGS), settings=lenient)
    assert result.score == 1.0


def test_describe_pattern_is_readable_in_both_directions():
    matched = analyse_structure(make_pages(ARABIC_ARTICLES))
    assert (
        describe_pattern(matched) == "article/clause numbering detected starting page 1"
    )

    unmatched = analyse_structure(make_pages(UNSTRUCTURED_PROSE))
    assert describe_pattern(unmatched) == "no clear structural pattern"
