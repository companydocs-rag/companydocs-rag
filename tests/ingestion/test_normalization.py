"""Text normalization (spec 004): the soft operations, the document output and
hard normalization.

Inline fabricated page texts and store records only (``_helpers.py``).
Characters outside ASCII are built with ``unicodedata.lookup`` or ``chr``.
The label families and conjunction lists come from ``config/settings.yaml``;
tests that override a setting pass a deep copy of it as a settings mapping.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import unicodedata

import pytest

from companydocs_rag.ingestion.normalization import (
    DEHYPHENATION_COUNTERS,
    NormalizationConfigError,
    StoreRecordError,
    clean_characters,
    collapse_filler_runs,
    dehyphenate_text,
    hard_normalize,
    normalization_settings,
    normalize_document,
    normalize_spaces,
    validate_store_record,
)
from companydocs_rag.ingestion.settings import load_settings
from tests.ingestion._helpers import store_page, store_record

LOOKUP = unicodedata.lookup

ZERO_WIDTH_SPACE = LOOKUP("ZERO WIDTH SPACE")
ZERO_WIDTH_NON_JOINER = LOOKUP("ZERO WIDTH NON-JOINER")
ZERO_WIDTH_JOINER = LOOKUP("ZERO WIDTH JOINER")
WORD_JOINER = LOOKUP("WORD JOINER")
BYTE_ORDER_MARK = LOOKUP("ZERO WIDTH NO-BREAK SPACE")
SOFT_HYPHEN = LOOKUP("SOFT HYPHEN")
NO_BREAK_SPACE = LOOKUP("NO-BREAK SPACE")
NARROW_NO_BREAK_SPACE = LOOKUP("NARROW NO-BREAK SPACE")
COMBINING_ACUTE = LOOKUP("COMBINING ACUTE ACCENT")
ELLIPSIS = LOOKUP("HORIZONTAL ELLIPSIS")
PRIVATE_USE = chr(0xE000)  # the first private-use code point; it has no name


def _letters(*names: str) -> str:
    return "".join(LOOKUP(name) for name in names)


E_ACUTE = _letters("LATIN SMALL LETTER E WITH ACUTE")
A_GRAVE = _letters("LATIN SMALL LETTER A WITH GRAVE")

# One word with a diacritic per language, from all seven (A13).
DIACRITIC_WORDS = [
    "financi" + _letters("LATIN SMALL LETTER E WITH DIAERESIS") + "le",  # nld
    "na" + _letters("LATIN SMALL LETTER I WITH DIAERESIS") + "ve",  # eng
    _letters("LATIN CAPITAL LETTER U WITH DIAERESIS")
    + "bergabe Stra"
    + _letters("LATIN SMALL LETTER SHARP S")
    + "e",  # deu
    "fa" + _letters("LATIN SMALL LETTER C WITH CEDILLA") + "ade " + A_GRAVE,  # fra
    "zobowi"
    + _letters("LATIN SMALL LETTER A WITH OGONEK")
    + "zanie "
    + _letters(
        "LATIN SMALL LETTER L WITH STROKE", "LATIN SMALL LETTER Z WITH DOT ABOVE"
    )
    + "ko",  # pol
    "p" + _letters("LATIN SMALL LETTER R WITH CARON") + "edm" + E_ACUTE + "t",  # ces
    "ve"
    + _letters("LATIN SMALL LETTER L WITH CARON")
    + "k"
    + _letters("LATIN SMALL LETTER O WITH CIRCUMFLEX"),  # slk
]


def _texts(document):
    return [page["text"] for page in document["pages"]]


def _settings(**changes):
    """A deep copy of the real settings, with ``ingestion.normalization``
    values replaced: keys are paths below it, joined by "__"."""
    settings = copy.deepcopy(load_settings())
    section = settings["ingestion"]["normalization"]
    for path, value in changes.items():
        *parents, key = path.split("__")
        node = section
        for parent in parents:
            node = node[parent]
        node[key] = value
    return settings


# --- Operation 1: characters (A13, A14) ---------------------------------------------


def test_zero_width_characters_and_soft_hyphens_go_and_no_break_spaces_become_spaces():
    text = (
        f"{BYTE_ORDER_MARK}huur{ZERO_WIDTH_SPACE}der{ZERO_WIDTH_NON_JOINER} "
        f"ver{ZERO_WIDTH_JOINER}huur{WORD_JOINER}der ak{SOFT_HYPHEN}koord"
        f"{NO_BREAK_SPACE}en{NARROW_NO_BREAK_SPACE}prima"
    )

    cleaned, counts = clean_characters(text)

    assert cleaned == "huurder verhuurder akkoord en prima"
    assert counts == {
        "nfc_changed": False,
        "zero_width_removed": 5,
        "soft_hyphens_removed": 1,
        "nbsp_replaced": 2,
    }


def test_nfc_runs_after_the_removals():
    """A zero-width character between a letter and its combining mark (A14):
    removed first, then NFC composes the two."""
    cleaned, counts = clean_characters(f"cafe{ZERO_WIDTH_SPACE}{COMBINING_ACUTE}")

    assert cleaned == "caf" + E_ACUTE
    assert counts["nfc_changed"] is True
    assert counts["zero_width_removed"] == 1


def test_operation_1_is_idempotent():
    text = (
        f"cafe{ZERO_WIDTH_SPACE}{COMBINING_ACUTE}{NO_BREAK_SPACE}"
        f"ak{SOFT_HYPHEN}koord{BYTE_ORDER_MARK}\nregel twee{NARROW_NO_BREAK_SPACE}"
    )
    once, _ = clean_characters(text)
    twice, counts = clean_characters(once)

    assert twice == once
    assert counts == {
        "nfc_changed": False,
        "zero_width_removed": 0,
        "soft_hyphens_removed": 0,
        "nbsp_replaced": 0,
    }


# --- Operation 2: spaces (A13, A14) ---------------------------------------------------


def test_space_runs_collapse_and_trailing_whitespace_goes_line_by_line():
    text = "de   huurder  betaalt\nde huur \t \n   \n  twee spaties vooraan\nslot  "

    spaced, counts = normalize_spaces(text)

    assert spaced == "de huurder betaalt\nde huur\n\n twee spaties vooraan\nslot"
    # A trailing run counts only as stripped whitespace (I8).
    assert counts == {"space_runs_collapsed": 3, "trailing_whitespace_stripped": 3}


def test_operations_1_and_2_keep_the_number_of_lines_and_a_trailing_newline():
    text = f"een{NO_BREAK_SPACE} \n\n  \ntwee{ZERO_WIDTH_SPACE}\n"
    spaced, _ = normalize_spaces(clean_characters(text)[0])

    assert spaced == "een\n\n\ntwee\n"
    assert len(spaced.split("\n")) == len(text.split("\n"))


def test_operation_2_is_idempotent():
    once, _ = normalize_spaces("a   b \t\n   \nc  d  ")
    twice, counts = normalize_spaces(once)

    assert twice == once == "a b\n\nc d"
    assert counts == {"space_runs_collapsed": 0, "trailing_whitespace_stripped": 0}


# --- A page that only operations 1 and 2 change (A13) ---------------------------------

LEFT_DOUBLE = LOOKUP("LEFT DOUBLE QUOTATION MARK")
RIGHT_DOUBLE = LOOKUP("RIGHT DOUBLE QUOTATION MARK")
LEFT_SINGLE = LOOKUP("LEFT SINGLE QUOTATION MARK")
RIGHT_SINGLE = LOOKUP("RIGHT SINGLE QUOTATION MARK")
LEFT_GUILLEMET = LOOKUP("LEFT-POINTING DOUBLE ANGLE QUOTATION MARK")
RIGHT_GUILLEMET = LOOKUP("RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK")
EN_DASH = LOOKUP("EN DASH")
EM_DASH = LOOKUP("EM DASH")
SECTION_SIGN = LOOKUP("SECTION SIGN")
EURO_SIGN = LOOKUP("EURO SIGN")
SUPERSCRIPT_TWO = LOOKUP("SUPERSCRIPT TWO")

KEPT_LINES = [
    f"De huurder zegt {LEFT_DOUBLE}akkoord{RIGHT_DOUBLE} en {LEFT_SINGLE}prima"
    f"{RIGHT_SINGLE}; zie {LEFT_GUILLEMET}bijlage{RIGHT_GUILLEMET}.",
    f"Looptijd 2024{EN_DASH}2025 {EM_DASH} daarna stilzwijgend verlengd.",
    "Diakrieten: " + " ".join(DIACRITIC_WORDS),
    f"Bedragen: {SECTION_SIGN} 3, $ 5, {EURO_SIGN} 10 per m{SUPERSCRIPT_TWO}, "
    f"enzovoort{ELLIPSIS}{ELLIPSIS}",
    f"Het teken {PRIVATE_USE} blijft staan.",
    "HOOFDLETTERS en kleine letters",
    "",
    "Na een lege regel.",
    "",
    "",
    "Na twee lege regels " + "caf" + E_ACUTE + ".",
]

NOISY_LINES = [
    f"{BYTE_ORDER_MARK}De huur{ZERO_WIDTH_SPACE}der zegt {LEFT_DOUBLE}ak{SOFT_HYPHEN}"
    f"koord{RIGHT_DOUBLE} en{NO_BREAK_SPACE}{LEFT_SINGLE}prima{RIGHT_SINGLE}; "
    f"zie{NARROW_NO_BREAK_SPACE}{LEFT_GUILLEMET}bijlage{RIGHT_GUILLEMET}.",
    f"Looptijd 2024{EN_DASH}2025 {EM_DASH} daarna stil{ZERO_WIDTH_NON_JOINER}"
    f"zwijgend verlengd.",
    "Diakrieten: " + " ".join(DIACRITIC_WORDS),
    f"Bedragen: {SECTION_SIGN} 3, $ 5, {EURO_SIGN} 10 per m{SUPERSCRIPT_TWO}, "
    f"enzovoort{ELLIPSIS}{ELLIPSIS}",
    f"Het teken {PRIVATE_USE} blijft{WORD_JOINER} staan.",
    "HOOFDLETTERS   en kleine  letters",
    "   ",
    "Na een lege regel. \t",
    "",
    "",
    f"Na twee lege regels cafe{COMBINING_ACUTE}{ZERO_WIDTH_JOINER}.",
]


def test_a_page_comes_out_unchanged_apart_from_the_character_and_space_operations():
    document = normalize_document(store_record("\n".join(NOISY_LINES)))

    (page,) = document["pages"]
    assert page["text"] == "\n".join(KEPT_LINES)
    assert page["removed_lines"] == []
    assert page["joins"] == []
    assert page["counts"] == {
        "nfc_changed": True,
        "zero_width_removed": 5,
        "soft_hyphens_removed": 1,
        "nbsp_replaced": 2,
        "space_runs_collapsed": 2,
        "trailing_whitespace_stripped": 2,
        "page_label_lines_removed": 0,
        "page_number_run_lines_removed": 0,
        "dehyphenation": {name: 0 for name in DEHYPHENATION_COUNTERS},
        "filler_runs_collapsed": 0,
        "filler_chars_removed": 0,
    }


# --- Operation 5: filler runs (A14, A19) ----------------------------------------------


def test_runs_of_four_or_more_filler_characters_collapse_to_three():
    text = (
        f"Naam: ______ datum: ........ tot----- en{ELLIPSIS * 5}\n"
        f"kort: ___ ... -- {ELLIPSIS * 3} gemengd: ._._._"
    )

    collapsed, counts = collapse_filler_runs(text)

    assert collapsed == (
        f"Naam: ___ datum: ... tot--- en{ELLIPSIS * 3}\n"
        f"kort: ___ ... -- {ELLIPSIS * 3} gemengd: ._._._"
    )
    assert counts == {"filler_runs_collapsed": 4, "filler_chars_removed": 12}


def test_operation_5_is_idempotent():
    once, _ = collapse_filler_runs(f"a........b ____ c{ELLIPSIS * 7}")
    twice, counts = collapse_filler_runs(once)

    assert twice == once
    assert counts == {"filler_runs_collapsed": 0, "filler_chars_removed": 0}


def test_the_filler_minimum_is_read_from_the_settings_at_call_time():
    record = store_record("Naam: ....")

    assert _texts(normalize_document(record)) == ["Naam: ..."]
    assert _texts(normalize_document(record, _settings(filler__min_run=5))) == [
        "Naam: ...."
    ]


# --- Step 3: page labels (D8; A16, A17) -----------------------------------------------

FAMILY_LABELS = [
    ("dash_number", "- 2 -"),
    ("pagina_van", "Pagina 2 van 5"),
    ("slash_total", "2/5"),
    ("seite_von", "Seite 2 von 5"),
    ("strona_z", "Strona 2 z 5"),
    ("strana_z", "Strana 2 z 5"),
    ("strana", "Strana 2"),
    ("page_of", "Page 2 of 5"),
]


@pytest.mark.parametrize(
    ("family", "label"), FAMILY_LABELS, ids=[family for family, _ in FAMILY_LABELS]
)
def test_each_label_family_is_removed_at_either_page_edge(family, label):
    page_two = "\n".join([label.upper(), "Tekst van pagina twee.", "Slot.", label])
    document = normalize_document(store_record("Tekst van pagina een.", page_two))

    page = document["pages"][1]
    assert page["text"] == "Tekst van pagina twee.\nSlot."
    assert page["removed_lines"] == [
        {
            "line_index": 0,
            "text": label.upper(),
            "edge": "top",
            "reason": "page_label",
            "family": family,
            "run_offset": None,
        },
        {
            "line_index": 3,
            "text": label,
            "edge": "bottom",
            "reason": "page_label",
            "family": family,
            "run_offset": None,
        },
    ]
    assert page["counts"]["page_label_lines_removed"] == 2


@pytest.mark.parametrize(
    ("family", "label"), FAMILY_LABELS, ids=[family for family, _ in FAMILY_LABELS]
)
def test_a_label_inside_a_longer_line_or_beyond_the_edge_lines_survives(family, label):
    inside = f"Zie {label} voor de tekening."
    lines = [inside, "Een.", "Twee.", label, "Drie.", "Vier.", "Vijf.", inside]
    document = normalize_document(store_record("\n".join(lines)))

    assert _texts(document) == ["\n".join(lines)]
    assert document["pages"][0]["removed_lines"] == []


def test_the_edge_is_counted_in_non_empty_lines():
    """The label is the second of three non-empty lines: a tie, so bottom,
    although blank lines put it much nearer the top in raw line indices (I9)."""
    lines = ["Kop", "- 2 -", "", "", "", "", "", "Slot"]
    document = normalize_document(store_record("Pagina een.", "\n".join(lines)))

    (removed,) = document["pages"][1]["removed_lines"]
    assert (removed["line_index"], removed["edge"]) == (1, "bottom")


def test_a_bare_number_run_that_tracks_the_page_numbers_is_removed():
    document = normalize_document(
        store_record("Tekst een.\n1", "Tekst twee.\n2", "Tekst drie.\n3")
    )

    assert _texts(document) == ["Tekst een.", "Tekst twee.", "Tekst drie."]
    assert document["page_labels"]["runs"] == [{"offset": 0, "pages": [1, 2, 3]}]
    assert [page["removed_lines"] for page in document["pages"]] == [
        [
            {
                "line_index": 1,
                "text": str(number),
                "edge": "bottom",
                "reason": "page_number_run",
                "family": None,
                "run_offset": 0,
            }
        ]
        for number in (1, 2, 3)
    ]
    assert all(
        page["counts"]["page_number_run_lines_removed"] == 1
        for page in document["pages"]
    )


def test_a_run_with_a_constant_non_zero_offset_is_removed():
    document = normalize_document(
        store_record("11\nTekst een.", "12\nTekst twee.", "13\nTekst drie.")
    )

    assert _texts(document) == ["Tekst een.", "Tekst twee.", "Tekst drie."]
    assert document["page_labels"]["runs"] == [{"offset": 10, "pages": [1, 2, 3]}]
    assert {page["removed_lines"][0]["edge"] for page in document["pages"]} == {"top"}


def test_one_line_per_page_per_run_the_one_closest_to_an_edge():
    """Page 2 holds two "2" lines, each one non-empty line from an edge: the
    tie goes to the bottom one. In raw line indices the top one is nearer
    (I9)."""
    page_two = "\n".join(["Kop", "2", "Tekst", "2", "", "", "Voet"])
    document = normalize_document(
        store_record("Tekst een.\n1", page_two, "Tekst drie.\n3")
    )

    page = document["pages"][1]
    assert page["text"] == "Kop\n2\nTekst\n\n\nVoet"
    assert [(item["line_index"], item["edge"]) for item in page["removed_lines"]] == [
        (3, "bottom")
    ]


def test_runs_are_sorted_by_offset_with_their_pages_ascending():
    pages = [
        f"{number + 10}\nTekst a.\nTekst b.\n{number - 1}\n{number}"
        for number in (1, 2, 3)
    ]
    document = normalize_document(store_record(*pages))

    assert document["page_labels"]["runs"] == [
        {"offset": -1, "pages": [1, 2, 3]},
        {"offset": 0, "pages": [1, 2, 3]},
        {"offset": 10, "pages": [1, 2, 3]},
    ]
    assert _texts(document) == ["Tekst a.\nTekst b."] * 3


def test_clause_numbers_at_the_page_edges_survive():
    pages = ["2.1\nDe huurder betaalt.\n2.2", "3.1\nDe verhuurder levert.\n3.2"]
    document = normalize_document(store_record(*pages, "4.1\nSlot.\n4.2"))

    assert _texts(document) == [*pages, "4.1\nSlot.\n4.2"]
    assert document["page_labels"]["runs"] == []


def test_edge_numbers_that_form_no_run_survive():
    pages = ["Tekst een.\n7", "Tekst twee.\n42", "Tekst drie.\n5"]
    too_short = ["Tekst een.\n1", "Tekst twee.\n2", "Tekst drie zonder nummer."]

    for texts in (pages, too_short):
        document = normalize_document(store_record(*texts))
        assert _texts(document) == texts
        assert document["page_labels"]["runs"] == []


def test_a_one_page_document_loses_its_label_and_keeps_its_bare_number():
    document = normalize_document(store_record("Pagina 1 van 1\nTekst.\n1"))

    assert _texts(document) == ["Tekst.\n1"]
    assert [item["family"] for item in document["pages"][0]["removed_lines"]] == [
        "pagina_van"
    ]


def test_the_edge_line_count_and_the_run_minimum_are_read_at_call_time():
    lines = ["Een.", "Twee.", "Drie.", "- 1 -", "Vier.", "Vijf.", "Zes.", "Zeven."]
    one_page = store_record("\n".join(lines))
    two_pages = store_record("Tekst een.\n1", "Tekst twee.\n2")

    assert "- 1 -" in _texts(normalize_document(one_page))[0]
    widened = normalize_document(one_page, _settings(page_labels__edge_lines=4))
    assert "- 1 -" not in _texts(widened)[0]

    assert _texts(normalize_document(two_pages)) == ["Tekst een.\n1", "Tekst twee.\n2"]
    lowered = normalize_document(two_pages, _settings(page_labels__min_run_pages=2))
    assert _texts(lowered) == ["Tekst een.", "Tekst twee."]


def test_removed_lines_keep_the_store_text_at_their_line_index():
    store_text = f"Pagina{NO_BREAK_SPACE}2 van 3 \t\nTekst."
    document = normalize_document(store_record("Tekst.", store_text, "Slot."))

    (removed,) = document["pages"][1]["removed_lines"]
    assert removed["text"] == store_text.split("\n")[removed["line_index"]]
    assert document["pages"][1]["text"] == "Tekst."


def test_the_stated_totals_can_be_read_back_from_the_output():
    """Probe section B8, from the output alone: the removed lines, the family
    regexes recorded in page_labels and the page numbers (A17)."""
    labelled = store_record(
        "Pagina 1 van 4\nTekst.",
        "Tekst.\npagina 2 van 4",
        "PAGINA 3 VAN 4\nTekst.",
        path="invoer/fictief-a.pdf",
    )
    matching = store_record(
        "Tekst.\nPagina 1 van 2", "Tekst.\nPagina 2 van 2", path="invoer/fictief-b.pdf"
    )

    table = []
    for document in (normalize_document(labelled), normalize_document(matching)):
        regex = document["page_labels"]["constants"]["families"]["pagina_van"]
        for page in document["pages"]:
            for removed in page["removed_lines"]:
                assert removed["family"] == "pagina_van"
                match = re.fullmatch(regex, removed["text"].strip(), re.IGNORECASE)
                table.append(
                    (
                        document["file"]["path"],
                        int(match["page"]) == page["page_number"],
                        int(match["total"]),
                        document["page_count"],
                    )
                )

    assert table == [
        ("invoer/fictief-a.pdf", True, 4, 3),
        ("invoer/fictief-a.pdf", True, 4, 3),
        ("invoer/fictief-a.pdf", True, 4, 3),
        ("invoer/fictief-b.pdf", True, 2, 2),
        ("invoer/fictief-b.pdf", True, 2, 2),
    ]


def test_pages_considered_count_pages_with_a_non_empty_line_before_and_after_labels():
    document = normalize_document(store_record("Tekst.", "", "- 3 -"))

    assert document["page_labels"]["pages_considered"] == 2
    assert document["header_footer"]["pages_considered"] == 1


# --- Step 4: dehyphenation (D7; A15) --------------------------------------------------


def _counts(**nonzero):
    counts = {name: 0 for name in DEHYPHENATION_COUNTERS}
    counts.update(nonzero)
    return counts


def test_rule_a_a_next_line_that_does_not_start_lowercase_is_not_joined():
    text = "De huur-\nOvereenkomst.\nDe betalings-\n2024 geldt.\nHet onder-\n(houd)"

    joined, counts, joins = dehyphenate_text(text)

    assert joined == text
    assert counts == _counts(a_not_lowercase=3)
    assert joins == []


@pytest.mark.parametrize(
    "first",
    [LOOKUP("CIRCLED LATIN SMALL LETTER A"), LOOKUP("SMALL ROMAN NUMERAL ONE")],
    ids=["U+24D0", "U+2170"],
)
def test_rule_a_holds_when_the_next_line_starts_lowercase_but_not_with_a_letter(first):
    """C5: a lowercase letter is a letter (str.isalpha) that is lowercase
    (str.islower). U+24D0 and U+2170 are lowercase but not letters, so rule a
    holds and nothing is joined (review finding F3)."""
    assert first.islower() and not first.isalpha()
    text = f"De huur-\n{first} volgt."

    joined, counts, joins = dehyphenate_text(text)

    assert joined == text
    assert counts == _counts(a_not_lowercase=1)
    assert joins == []


NLD_EN = ("nld", "Het kantoor-", "en bedrijfspand.")
CONJUNCTION_CASES = [
    NLD_EN,
    ("eng", "the land-", "and buildings."),
    ("deu", "die Miet-", "und Nebenkosten."),
    ("fra", "les frais pre-", "et post."),
    ("fra", "les frais pre-", f"{A_GRAVE} titre."),
    ("pol", "koszty przed-", "i po."),
    ("ces", "naklady pred-", "a po."),
    ("slk", "naklady pred-", "aj po."),
]


@pytest.mark.parametrize(
    ("language", "line", "following"),
    CONJUNCTION_CASES,
    ids=["nld", "eng", "deu", "fra-et", "fra-a-grave", "pol", "ces", "slk"],
)
def test_rule_b_a_conjunction_of_the_page_language_keeps_the_hyphen(
    language, line, following
):
    text = f"{line}\n{following}"
    document = normalize_document(
        store_record(store_page(1, text, language=language), language=language)
    )

    (page,) = document["pages"]
    assert page["text"] == text
    assert page["counts"]["dehyphenation"] == _counts(b_conjunction=1)


def test_rule_b_uses_the_page_language_only():
    """ "en" is a Dutch conjunction but not an English one."""
    text = "\n".join(NLD_EN[1:])
    english = normalize_document(
        store_record(store_page(1, text, language="eng"), language="eng")
    )

    assert _texts(english) == ["Het kantooren bedrijfspand."]
    assert english["pages"][0]["counts"]["dehyphenation"] == _counts(e_hyphen_removed=1)


def test_rule_c_an_acronym_keeps_its_hyphen():
    text = "Het bedrijf heeft een KvK-\nnummer."

    joined, counts, joins = dehyphenate_text(text)

    assert joined == text
    assert counts == _counts(c_inner_uppercase=1)
    assert joins == []


def test_rule_d_a_hyphenated_form_elsewhere_in_the_document_is_joined_with_its_hyphen():
    document = normalize_document(
        store_record("Een e-mailadres volstaat.", "Het e-\nmailadres staat hier.")
    )

    page = document["pages"][1]
    assert page["text"] == "Het e-mailadres staat hier."
    assert page["counts"]["dehyphenation"] == _counts(d_hyphen_kept=1)
    assert page["joins"] == [
        {"line_index": 0, "rule": "d", "fragment": "e", "first_word": "mailadres"}
    ]


def test_rule_e_joins_without_the_hyphen_and_records_the_joined_form_separately():
    without = "De huur-\novereenkomst geldt."
    with_form = "De huur-\novereenkomst geldt.\nDeze huurovereenkomst eindigt."
    longer_word_only = "De huur-\novereenkomst geldt.\nDe huurovereenkomsten eindigen."

    results = [
        dehyphenate_text(text) for text in (without, with_form, longer_word_only)
    ]

    assert [joined.split("\n")[0] for joined, _, _ in results] == [
        "De huurovereenkomst geldt."
    ] * 3
    assert [counts for _, counts, _ in results] == [
        _counts(e_hyphen_removed=1),
        _counts(e_hyphen_removed=1, e_joined_form_in_document=1),
        _counts(e_hyphen_removed=1),
    ]
    assert all(
        joins
        == [
            {
                "line_index": 0,
                "rule": "e",
                "fragment": "huur",
                "first_word": "overeenkomst",
            }
        ]
        for _, _, joins in results
    )


def test_one_blank_line_between_is_joined_across_and_two_are_not():
    one_blank = "De betalings-\n\ntermijn is kort."
    two_blanks = "De betalings-\n\n\ntermijn is kort."

    joined, counts, joins = dehyphenate_text(one_blank)
    assert joined == "De betalingstermijn is kort."
    assert counts == _counts(e_hyphen_removed=1)
    assert joins[0]["line_index"] == 0

    joined, counts, joins = dehyphenate_text(two_blanks)
    assert joined == two_blanks
    assert counts == _counts(no_next_line=1)
    assert joins == []


def test_a_candidate_on_the_last_line_of_a_page_is_not_joined_with_the_next_page():
    document = normalize_document(
        store_record("Het onder-", "houd van het pand.", "Tekst.")
    )

    assert _texts(document)[:2] == ["Het onder-", "houd van het pand."]
    assert document["pages"][0]["counts"]["dehyphenation"] == _counts(no_next_line=1)
    assert document["pages"][1]["joins"] == []


def test_leading_spaces_of_the_next_line_are_skipped_and_not_carried_into_the_join():
    joined, counts, _ = dehyphenate_text("De huur-\n   overeenkomst geldt.")
    assert joined == "De huurovereenkomst geldt."
    assert counts == _counts(e_hyphen_removed=1)

    kept, counts, _ = dehyphenate_text(
        "Het kantoor-\n  en bedrijfspand.", conjunctions=["en"]
    )
    assert kept == "Het kantoor-\n  en bedrijfspand."
    assert counts == _counts(b_conjunction=1)

    kept, counts, _ = dehyphenate_text("Het kantoor-\n  En verder.")
    assert counts == _counts(a_not_lowercase=1)


def test_a_word_split_over_three_lines_is_joined_twice_in_one_pass():
    """C4: the joined line is examined again. Each join records the store line
    that carries the hyphen it joins (I5)."""
    joined, counts, joins = dehyphenate_text("Het ver-\nhuur-\nder is bekend.")

    assert joined == "Het verhuurder is bekend."
    assert counts == _counts(e_hyphen_removed=2)
    assert joins == [
        {"line_index": 0, "rule": "e", "fragment": "ver", "first_word": "huur"},
        {"line_index": 1, "rule": "e", "fragment": "verhuur", "first_word": "der"},
    ]


def test_every_candidate_ends_in_one_rule_counter_and_every_join_is_recorded():
    lines = [
        "De huur-",  # 0: rule e, joined form elsewhere
        "overeenkomst geldt.",
        "Het KvK-",  # 2: rule c
        "nummer volgt.",
        "Het kantoor-",  # 4: rule b
        "en bedrijfspand.",
        "De betalings-",  # 6: rule e across one blank line
        "",
        "termijn is kort.",
        "Het e-",  # 9: rule d
        "mailadres staat hier.",
        "Een e-mailadres en een huurovereenkomst.",
        "Zie Artikel-",  # 12: rule a
        "Drie.",
        "Het ver-",  # 14: rule e, then the joined line again: rule e
        "huur-",
        "der betaalt.",
        "Slot van de bouw-",  # 17: no next line within reach
        "",
        "",
        "werk.",
    ]
    document = normalize_document(store_record("\n".join(lines)))

    (page,) = document["pages"]
    counts = page["counts"]["dehyphenation"]
    assert counts == _counts(
        a_not_lowercase=1,
        b_conjunction=1,
        c_inner_uppercase=1,
        d_hyphen_kept=1,
        e_hyphen_removed=4,
        e_joined_form_in_document=1,
        no_next_line=1,
    )
    # Nine examinations: eight of the nine candidate lines as they come, and
    # the joined "Het verhuur-" once more; its second line is never examined
    # on its own.
    examinations = 9
    rule_counters = [
        name for name in DEHYPHENATION_COUNTERS if name != "e_joined_form_in_document"
    ]
    assert sum(counts[name] for name in rule_counters) == examinations
    assert len(page["joins"]) == counts["d_hyphen_kept"] + counts["e_hyphen_removed"]
    assert [(join["line_index"], join["rule"]) for join in page["joins"]] == [
        (0, "e"),
        (6, "e"),
        (9, "d"),
        (14, "e"),
        (15, "e"),
    ]
    assert [(join["fragment"], join["first_word"]) for join in page["joins"]] == [
        ("huur", "overeenkomst"),
        ("betalings", "termijn"),
        ("e", "mailadres"),
        ("ver", "huur"),
        ("verhuur", "der"),
    ]


def test_operation_4_is_idempotent():
    text = "\n".join(
        [
            "Het ver-",
            "huur-",
            "der is bekend.",
            "Onder KvK-",
            "nummer.",
            "Het kantoor-",
            "en bedrijfspand.",
            "De huur-",
            "",
            "overeenkomst.",
            "Slot bouw-",
            "",
            "",
            "werk.",
        ]
    )
    nld = normalization_settings().conjunctions["nld"]

    once, _, _ = dehyphenate_text(text, conjunctions=nld)
    twice, counts, joins = dehyphenate_text(once, conjunctions=nld)

    assert twice == once
    assert counts["d_hyphen_kept"] == counts["e_hyphen_removed"] == 0
    assert joins == []


def test_operation_4_is_not_idempotent_after_a_conjunction_line_joined_by_rule_e():
    """The accepted exception (section 5): "bouw-" stays by rule b because its next
    line starts with "en"; that line is then joined by rule e into "enzo", so
    a second pass joins "bouw-" too."""
    nld = normalization_settings().conjunctions["nld"]

    once, counts, _ = dehyphenate_text(
        "Dit is bouw-\nen-\nzo verder.", conjunctions=nld
    )
    assert once == "Dit is bouw-\nenzo verder."
    assert counts == _counts(b_conjunction=1, e_hyphen_removed=1)

    twice, counts, _ = dehyphenate_text(once, conjunctions=nld)
    assert twice == "Dit is bouwenzo verder."
    assert counts == _counts(e_hyphen_removed=1)


def test_a_page_language_without_conjunctions_warns_once_per_run(caplog):
    record = store_record(
        store_page(1, "Het kantoor-\nen bedrijfspand.", language="ita"),
        store_page(2, "Tekst.", language="ita"),
        language="ita",
    )
    warned = set()

    with caplog.at_level(
        logging.WARNING, logger="companydocs_rag.ingestion.normalization"
    ):
        document = normalize_document(record, warned_languages=warned)
        normalize_document(record, warned_languages=warned)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "'ita'" in warnings[0]
    assert warned == {"ita"}
    assert document["pages"][0]["counts"]["dehyphenation"] == _counts(
        e_hyphen_removed=1
    )


# --- The document (D5) ----------------------------------------------------------------


def test_a_trailing_newline_survives_and_page_hashes_cover_the_store_and_output_text():
    store_text = f"De huur-\novereenkomst{NO_BREAK_SPACE}geldt.\n"
    document = normalize_document(store_record(store_text))

    (page,) = document["pages"]
    assert page["text"] == "De huurovereenkomst geldt.\n"
    assert (
        page["input_sha256"] == hashlib.sha256(store_text.encode("utf-8")).hexdigest()
    )
    assert (
        page["normalized_sha256"]
        == hashlib.sha256(page["text"].encode("utf-8")).hexdigest()
    )


def test_the_document_hash_is_that_of_the_page_number_and_hash_pairs():
    document = normalize_document(store_record("Een.", "Twee.", "Drie."))

    pairs = [
        [page["page_number"], page["normalized_sha256"]] for page in document["pages"]
    ]
    expected = hashlib.sha256(
        json.dumps(pairs, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    assert document["normalized_sha256"] == expected


def test_a_failed_store_document_is_written_without_pages():
    document = normalize_document(store_record(status="failed", page_count=0))

    assert document["status"] == "failed"
    assert document["pages"] == []
    assert document["page_labels"]["status"] == "not_applicable"
    assert document["page_labels"]["pages_considered"] == 0
    assert document["page_labels"]["runs"] == []
    assert document["header_footer"] == {
        "status": "not_applicable",
        "pages_considered": 0,
        "constants": {},
    }
    assert document["normalized_sha256"] == hashlib.sha256(b"[]").hexdigest()


def test_the_output_carries_unicode_and_normalization_versions():
    document = normalize_document(store_record("Tekst."))

    assert document["schema_version"] == 1
    assert document["normalization_version"] == 1
    assert document["unicode_version"] == unicodedata.unidata_version


# --- Settings (D11; A19) --------------------------------------------------------------


def test_the_yaml_carries_the_normalization_settings():
    config = normalization_settings()

    assert (config.edge_lines, config.min_run_pages, config.bare_number_max_digits) == (
        3,
        3,
        3,
    )
    assert [name for name, _ in config.families] == [
        family for family, _ in FAMILY_LABELS
    ]
    assert sorted(config.conjunctions) == [
        "ces",
        "deu",
        "eng",
        "fra",
        "nld",
        "pol",
        "slk",
    ]
    assert config.conjunctions["nld"] >= {"en", "of", "tot", "noch"}
    assert A_GRAVE in config.conjunctions["fra"]
    assert config.filler_characters == "._-" + ELLIPSIS
    assert (config.filler_min_run, config.filler_collapse_to) == (4, 3)


def test_numeric_settings_fall_back_to_the_python_defaults():
    settings = _settings()
    section = settings["ingestion"]["normalization"]
    section["page_labels"] = {"families": section["page_labels"]["families"]}
    del section["filler"]

    config = normalization_settings(settings)

    assert (config.edge_lines, config.min_run_pages, config.bare_number_max_digits) == (
        3,
        3,
        3,
    )
    assert config.filler_characters == "._-" + ELLIPSIS
    assert (config.filler_min_run, config.filler_collapse_to) == (4, 3)


@pytest.mark.parametrize("breakage", ["families", "conjunctions", "regex"])
def test_missing_lists_or_a_regex_that_does_not_compile_are_config_errors(breakage):
    settings = _settings()
    section = settings["ingestion"]["normalization"]
    if breakage == "families":
        del section["page_labels"]["families"]
    elif breakage == "conjunctions":
        del section["dehyphenation"]["conjunctions"]
    else:
        section["page_labels"]["families"]["pagina_van"] = "pagina (?P<page>[0-9"

    with pytest.raises(NormalizationConfigError):
        normalization_settings(settings)
    with pytest.raises(NormalizationConfigError):
        normalize_document(store_record("Tekst."), settings)


# --- The store record (I12) -----------------------------------------------------------


def _broken(change):
    record = store_record("Tekst.")
    change(record)
    return record


@pytest.mark.parametrize(
    "record",
    [
        [],
        _broken(lambda record: record.pop("file")),
        _broken(lambda record: record.update(status="klaar")),
        _broken(lambda record: record.update(page_count="3")),
        _broken(lambda record: record.update(pages={})),
        _broken(lambda record: record["pages"][0].update(text=None)),
        _broken(lambda record: record["pages"][0].update(text_origin="scan")),
        _broken(lambda record: record["pages"][0].pop("tier")),
        _broken(lambda record: record["pages"][0].update(ocr_mean_confidence=True)),
    ],
    ids=[
        "not-an-object",
        "no-file",
        "unknown-status",
        "page-count-text",
        "pages-not-a-list",
        "text-null",
        "unknown-text-origin",
        "no-tier",
        "confidence-bool",
    ],
)
def test_a_record_that_is_not_a_valid_store_record_is_rejected(record):
    with pytest.raises(StoreRecordError):
        validate_store_record(record)


def test_the_valid_records_pass_the_check():
    validate_store_record(
        store_record("Tekst.", store_page(2, "OCR.", text_origin="ocr"))
    )
    validate_store_record(store_record(status="failed", page_count=0))


# --- Hard normalization (D10; A20) ----------------------------------------------------


def test_nfkc_expands_the_ellipsis_and_the_superscript_two():
    assert hard_normalize(f"zie bijlage{ELLIPSIS}") == "zie bijlage..."
    assert hard_normalize(f"120 m{SUPERSCRIPT_TWO}") == "120 m2"


def test_casefolding_lowers_case_and_folds_sharp_s():
    sharp_s = LOOKUP("LATIN SMALL LETTER SHARP S")
    assert hard_normalize(f"HUUROVEREENKOMST Stra{sharp_s}e") == (
        "huurovereenkomst strasse"
    )


def test_nfkc_runs_again_after_casefolding():
    j_caron = LOOKUP("LATIN SMALL LETTER J WITH CARON")
    assert j_caron.casefold() == "j" + LOOKUP("COMBINING CARON")
    assert hard_normalize(j_caron) == j_caron


SINGLE_QUOTES = [
    "LEFT SINGLE QUOTATION MARK",
    "RIGHT SINGLE QUOTATION MARK",
    "SINGLE LOW-9 QUOTATION MARK",
    "SINGLE HIGH-REVERSED-9 QUOTATION MARK",
    "SINGLE LEFT-POINTING ANGLE QUOTATION MARK",
    "SINGLE RIGHT-POINTING ANGLE QUOTATION MARK",
]
DOUBLE_QUOTES = [
    "LEFT DOUBLE QUOTATION MARK",
    "RIGHT DOUBLE QUOTATION MARK",
    "DOUBLE LOW-9 QUOTATION MARK",
    "DOUBLE HIGH-REVERSED-9 QUOTATION MARK",
    "LEFT-POINTING DOUBLE ANGLE QUOTATION MARK",
    "RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK",
]
DASHES = [
    "HYPHEN",
    "NON-BREAKING HYPHEN",
    "FIGURE DASH",
    "EN DASH",
    "EM DASH",
    "HORIZONTAL BAR",
    "MINUS SIGN",
]


@pytest.mark.parametrize("name", SINGLE_QUOTES)
def test_single_quote_characters_become_an_apostrophe(name):
    quote = LOOKUP(name)
    assert hard_normalize(f"{quote}huurder{quote}") == "'huurder'"


@pytest.mark.parametrize("name", DOUBLE_QUOTES)
def test_double_quote_characters_become_a_straight_double_quote(name):
    quote = LOOKUP(name)
    assert hard_normalize(f"{quote}verhuurder{quote}") == '"verhuurder"'


def test_primes_and_the_backtick_do_not_become_quotes():
    prime = LOOKUP("PRIME")
    double_prime = LOOKUP("DOUBLE PRIME")
    # NFKC decomposes the double prime into two primes; neither is a quote.
    assert hard_normalize(f"5{prime} 6{double_prime} `code`") == (
        f"5{prime} 6{prime}{prime} `code`"
    )


@pytest.mark.parametrize("name", DASHES)
def test_dashes_and_the_minus_sign_become_a_hyphen_minus(name):
    dash = LOOKUP(name)
    assert hard_normalize(f"1{dash}2 huur{dash}prijs") == "1-2 huur-prijs"


def test_private_use_characters_are_removed():
    assert hard_normalize(f"{chr(0xF0B7)} artikel{PRIVATE_USE} een") == "artikel een"


def test_tabs_newlines_and_space_runs_collapse_to_one_space_and_are_stripped():
    text = f"  de\thuurder\n\nbetaalt   de\r\nhuur {NO_BREAK_SPACE} "
    assert hard_normalize(text) == "de huurder betaalt de huur"


DIACRITICS = [
    "LATIN SMALL LETTER E WITH ACUTE",
    "LATIN SMALL LETTER C WITH CARON",
    "LATIN SMALL LETTER L WITH STROKE",
    "LATIN SMALL LETTER R WITH CARON",
    "LATIN SMALL LETTER Z WITH DOT ABOVE",
]


@pytest.mark.parametrize("name", DIACRITICS)
def test_diacritics_are_kept_also_when_casefolded(name):
    lower = LOOKUP(name)
    upper = LOOKUP(name.replace("SMALL", "CAPITAL"))
    assert hard_normalize(f"{lower} {upper}") == f"{lower} {lower}"
