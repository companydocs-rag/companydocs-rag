"""Contract test for preocr's private text-quality function.

``quality._page_quality`` calls ``preocr.core.signals._compute_text_quality_signals``
directly rather than reimplementing it, and its output decides the page's
Clean/Decent/Garbage tier through ``quality._tier_for_page``. That function is
private (no leading-underscore API is a public contract), so any preocr release
-- patch releases included -- could rename it, reshape its output or change its
values without any version-range check noticing. This module pins the
dependency deliberately and visibly: the exact import path and callability, the
exact keys and values it returns on a fixed fabricated input, and its
10-character cut-off, so a change to any of them fails a named test instead of
silently shifting page tiers.

The values pinned below are preocr **1.11.0**'s (see ``preocr==1.11.0`` in
``pyproject.toml``, reflected in ``uv.lock``). Upgrading the pin means
re-deriving them deliberately -- by running preocr from the project ``.venv``
against the fixture strings below -- not carrying the old numbers forward. See
the Outcome of spec 002 for the exact command used to derive them.
"""

from __future__ import annotations

from companydocs_rag.ingestion import quality

# preocr.core.signals._compute_text_quality_signals returns {} for any text
# shorter than this many characters (preocr/core/signals.py:16-17: `if not
# text or len(text) < 10: return {}`). Below this, `quality._page_quality`'s
# `_compute_text_quality_signals(page.text) or {}` fallback turns the missing
# signals into three `None`s, and `quality._tier_for_page`'s `is None` checks
# on its noise average and on `avg_word_length` count a missing signal as
# passing rather than failing -- a page under this length would therefore be
# tiered on character count alone. That cannot happen today, because a page
# only reaches this function's output as `native` (tiered) once it clears
# `MIN_CHARS_NATIVE_PAGE` (50, see the threshold guard test below); it becomes
# possible only if that threshold ever drops below this cut-off, or if
# `source` stops being decided by character count -- both out of scope for
# this test, which can only guard the first.
TEXT_QUALITY_MIN_CHARS = 10


def test_private_text_quality_function_is_importable_and_callable():
    """Pins the import path that `quality._page_quality`'s lazy import relies on.

    A rename anywhere along `preocr.core.signals._compute_text_quality_signals`
    must fail here, loudly and by name, rather than surface later as an
    ImportError deep inside a folder run.
    """
    from preocr.core import signals as preocr_signals

    assert callable(preocr_signals._compute_text_quality_signals)


def test_exact_keys_and_values_on_a_fabricated_dutch_text():
    """Pins preocr 1.11.0's output for one fixed, fabricated Dutch string.

    The fixture is fabricated Dutch running text (never taken from a real
    document) chosen so all three signals are non-zero and a change to any one
    of them shows up:

    - one C0 control character that is not whitespace (``\\x07``, BEL) for
      ``non_printable_ratio`` (preocr/core/signals.py:19-20);
    - one U+FFFD replacement character for ``unicode_noise_ratio``
      (preocr/core/signals.py:22-26);
    - the word "reeel" spelled with the Dutch diacritic "e-trema" (``reëel``)
      for ``avg_word_length``, so a change to preocr's word regex (``\\b\\w+\\b``,
      Unicode-aware today, preocr/core/signals.py:28) changes the word count or
      the word boundaries and fails this test.

    A single ``==`` against a dict literal checks the key set (exactly
    ``non_printable_ratio``, ``unicode_noise_ratio``, ``avg_word_length``) and
    the values together. Not ``pytest.approx``: preocr rounds its output (4, 4
    and 2 decimals, preocr/core/signals.py:30-34), so an approximate comparison
    would hide a change in that rounding.

    Values derived by running preocr 1.11.0 from the project ``.venv`` directly
    against this exact string -- see spec 002's Outcome for the command.
    """
    from preocr.core.signals import _compute_text_quality_signals

    fixture = (
        "Dit is een fictieve, volledig verzonnen Nederlandse tekst ter "
        "illustratie: de resultaten blijven reëel en verifieerbaar, ook al "
        "bevat dit fragment wat ruis \x07 en een storend teken � aan het "
        "einde van de zin."
    )
    assert len(fixture) >= TEXT_QUALITY_MIN_CHARS

    result = _compute_text_quality_signals(fixture)

    assert result == {
        "non_printable_ratio": 0.0049,
        "unicode_noise_ratio": 0.0049,
        "avg_word_length": 5.03,
    }


def test_below_the_cutoff_returns_an_empty_dict():
    """One character short of the cut-off returns `{}`.

    See preocr/core/signals.py:16-17.
    """
    from preocr.core.signals import _compute_text_quality_signals

    nine_chars = "a" * (TEXT_QUALITY_MIN_CHARS - 1)
    assert len(nine_chars) == 9

    assert _compute_text_quality_signals(nine_chars) == {}


def test_at_the_cutoff_returns_the_three_keys():
    """Exactly at the cut-off, the three signal keys are present.

    Pairs with the test above: without this second case, preocr raising the
    cut-off (e.g. from 10 to 20 characters) would not fail any test in this
    module, because the below-cutoff case alone stays true either way.
    """
    from preocr.core.signals import _compute_text_quality_signals

    ten_chars = "a" * TEXT_QUALITY_MIN_CHARS
    assert len(ten_chars) == TEXT_QUALITY_MIN_CHARS

    result = _compute_text_quality_signals(ten_chars)

    assert set(result.keys()) == {
        "non_printable_ratio",
        "unicode_noise_ratio",
        "avg_word_length",
    }


def test_min_chars_native_page_threshold_is_at_or_above_the_cutoff(monkeypatch):
    """Turns the cut-off's precondition into a failing test, not just a comment.

    A page only reaches ``_compute_text_quality_signals`` as ``native`` once it
    clears ``min_chars_native_page`` (see `quality.py`'s Axis 2). As long as
    that threshold is at least :data:`TEXT_QUALITY_MIN_CHARS`, no tiered page
    can ever be short enough to hit the empty-dict branch above. This reads the
    real, unpatched ``config/settings.yaml`` -- with ``COMPANYDOCS_RAG_SETTINGS``
    removed and the settings cache cleared before and after -- rather than an
    inline override, so a future edit to that file is caught here too. The
    second half of the precondition -- that `source` keeps being decided by
    character count at all -- cannot be expressed as a test and stays
    documented above, at :data:`TEXT_QUALITY_MIN_CHARS`.
    """
    from companydocs_rag.ingestion.settings import load_settings

    monkeypatch.delenv("COMPANYDOCS_RAG_SETTINGS", raising=False)
    load_settings.cache_clear()
    try:
        min_chars_native_page = quality._thresholds()["min_chars_native_page"]
        assert min_chars_native_page >= TEXT_QUALITY_MIN_CHARS
    finally:
        load_settings.cache_clear()
