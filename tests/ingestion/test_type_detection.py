"""Type detection tests, on short inline text fragments.

The fragments below are hand-written approximations of the vocabulary each
document type uses, not excerpts from real documents. They test the matching
and scoring logic, not the completeness of the keyword lists -- that only gets
settled by error analysis against the real corpus.
"""

from __future__ import annotations

from companydocs_rag.ingestion.type_detection import (
    UNKNOWN_TYPE,
    detect_type,
    summarise_evidence,
)
from tests.ingestion._helpers import make_document

# --- Inline fragments, one per type -----------------------------------------

DUTCH_LEASE = """
HUUROVEREENKOMST KANTOORRUIMTE

De ondergetekenden, hierna te noemen verhuurder, en de huurder komen overeen
dat het gehuurde met ingang van 1 januari wordt opgeleverd. De huurprijs
bedraagt EUR 4.500 per maand. De huurder stelt een waarborgsom.
"""

DUTCH_ARTICLES_OF_ASSOCIATION = """
STATUTEN

Blijkens deze akte van oprichting wordt het bestuur benoemd door de algemene
vergadering. Elke aandeelhouder is bevoegd de vergadering bij te wonen.
De vennootschap wordt bestuurd met inachtneming van artikel 2:242 BW.
"""

DUTCH_KVK_EXTRACT = """
Uittreksel Handelsregister Kamer van Koophandel

KvK-nummer: 12345678
Rechtsvorm: Besloten Vennootschap
Statutaire zetel: Amsterdam
Bezoekadres: Voorbeeldweg 1
SBI-code: 68201
"""

DUTCH_PERMIT = """
OMGEVINGSVERGUNNING

Het bevoegd gezag heeft de aanvraag beoordeeld en de vergunning verleend aan
de vergunninghouder. De vergunningvoorschriften zijn opgenomen in bijlage A.
"""

ENGLISH_LEASE = """
LEASE AGREEMENT

The lessor and the lessee agree that the leased premises are handed over on
1 January. The rent is payable monthly for the full lease term.
"""

GERMAN_LEASE = """
MIETVERTRAG

Der Vermieter und der Mieter vereinbaren, dass das Mietobjekt uebergeben wird.
Der Mietzins ist monatlich zu zahlen. Die Kuendigung erfolgt schriftlich.
"""

# Deliberately carries none of the four types' vocabulary.
UNRELATED_DOCUMENT = """
FACTUUR 2024-0031

Wij danken u voor uw bestelling. Het totaalbedrag is binnen dertig dagen
te voldoen op het onderstaande rekeningnummer. Vermeld daarbij het
factuurnummer.
"""


def test_detects_dutch_lease_agreement():
    result = detect_type(make_document(DUTCH_LEASE))
    assert result.detected_type == "lease_agreement"
    assert result.confidence > 0.5


def test_detects_dutch_articles_of_association():
    result = detect_type(make_document(DUTCH_ARTICLES_OF_ASSOCIATION))
    assert result.detected_type == "articles_of_association"
    assert result.confidence > 0.5


def test_detects_kvk_extract():
    result = detect_type(make_document(DUTCH_KVK_EXTRACT))
    assert result.detected_type == "chamber_of_commerce_extract"
    assert result.confidence > 0.5


def test_detects_permit():
    result = detect_type(make_document(DUTCH_PERMIT))
    assert result.detected_type == "permit"
    assert result.confidence > 0.5


def test_detects_english_lease_via_english_keywords():
    """A minority of documents are English; the union of languages is matched."""
    result = detect_type(make_document(ENGLISH_LEASE))
    assert result.detected_type == "lease_agreement"
    assert {item.language for item in result.evidence} == {"en"}


def test_detects_german_lease_via_german_keywords():
    result = detect_type(make_document(GERMAN_LEASE))
    assert result.detected_type == "lease_agreement"
    assert {item.language for item in result.evidence} == {"de"}


def test_bw_book2_article_reference_is_recorded_as_a_strong_regex_signal():
    result = detect_type(make_document(DUTCH_ARTICLES_OF_ASSOCIATION))
    strong = [item for item in result.evidence if item.strong]
    assert [item.term for item in strong] == ["bw_book2_article_reference"]
    assert strong[0].kind == "regex"


# --- Fallback to unknown/other ----------------------------------------------


def test_unrelated_document_falls_back_to_unknown():
    result = detect_type(make_document(UNRELATED_DOCUMENT))
    assert result.detected_type == UNKNOWN_TYPE
    assert result.confidence == 0.0
    assert result.fallback_reason


def test_empty_document_does_not_crash_and_returns_unknown():
    result = detect_type(make_document("", ""))
    assert result.detected_type == UNKNOWN_TYPE
    assert result.confidence == 0.0


def test_document_with_no_pages_returns_unknown():
    result = detect_type(make_document())
    assert result.detected_type == UNKNOWN_TYPE


def test_single_matching_term_is_not_enough_for_a_type():
    """One incidental keyword must not force a guess."""
    result = detect_type(make_document("De aanvraag is ontvangen op 3 maart."))
    assert result.detected_type == UNKNOWN_TYPE
    assert "distinct term" in (result.fallback_reason or "")


# --- Scoring behaviour ------------------------------------------------------


def test_diversity_outweighs_repetition():
    """Five different terms beat one term repeated five times."""
    repeated = make_document("huurprijs " * 8)
    diverse = make_document("verhuurder huurder het gehuurde huurprijs waarborgsom")

    repeated_result = detect_type(repeated)
    diverse_result = detect_type(diverse)

    assert (
        diverse_result.scores["lease_agreement"]
        > repeated_result.scores["lease_agreement"]
    )


def test_evidence_records_the_page_of_every_match():
    document = make_document(
        "STATUTEN van de vennootschap",  # page 1
        "Het bestuur wordt benoemd door de algemene vergadering.",  # page 2
    )
    result = detect_type(document)

    pages_by_term = {item.term: item.page for item in result.evidence}
    assert pages_by_term["statuten"] == 1
    assert pages_by_term["vennootschap"] == 1
    assert pages_by_term["bestuur"] == 2
    assert pages_by_term["algemene vergadering"] == 2


def test_multi_word_keyword_matches_across_a_line_break():
    """PDF extraction routinely wraps a phrase mid-way."""
    document = make_document(
        "Deze algemene\nbepalingen maken deel uit van de huurovereenkomst."
    )
    result = detect_type(document)
    assert "algemene bepalingen" in {item.term for item in result.evidence}


def test_keyword_matching_respects_word_boundaries():
    """'verhuurder' must not also register as a match for 'huurder'."""
    result = detect_type(make_document("De verhuurder verklaart het volgende."))
    assert {item.term for item in result.evidence} == {"verhuurder"}


def test_confidence_is_suppressed_when_two_types_score_similarly():
    """Ambiguity between types must lower confidence, not be hidden."""
    clear = detect_type(make_document(DUTCH_LEASE))
    ambiguous = detect_type(make_document(DUTCH_LEASE + DUTCH_ARTICLES_OF_ASSOCIATION))
    assert ambiguous.confidence < clear.confidence
    assert ambiguous.runner_up is not None


def test_all_candidate_scores_are_returned_for_recalibration():
    result = detect_type(make_document(DUTCH_LEASE))
    assert set(result.scores) == {
        "lease_agreement",
        "articles_of_association",
        "chamber_of_commerce_extract",
        "permit",
    }


def test_missing_type_configuration_yields_unknown_rather_than_an_error():
    result = detect_type(make_document(DUTCH_LEASE), settings={})
    assert result.detected_type == UNKNOWN_TYPE
    assert "settings.yaml" in (result.fallback_reason or "")


def test_thresholds_can_be_overridden_through_settings():
    """A stricter minimum pushes an otherwise-detected document to unknown."""
    strict = {
        "ingestion": {
            "parse_pass": {
                "type_detection": {
                    "min_distinct_terms": 99,
                    "types": {
                        "lease_agreement": {
                            "label": "Lease/rental/hire agreement",
                            "keywords": {"nl": ["verhuurder", "huurder", "huurprijs"]},
                        }
                    },
                }
            }
        }
    }
    result = detect_type(make_document(DUTCH_LEASE), settings=strict)
    assert result.detected_type == UNKNOWN_TYPE


def test_summarise_evidence_is_readable_and_empty_safe():
    result = detect_type(make_document(DUTCH_LEASE))
    summary = summarise_evidence(result.evidence)
    assert "(p1)" in summary
    assert summarise_evidence([]) == "no type keywords matched"
