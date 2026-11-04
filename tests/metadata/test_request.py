"""The request (spec 008, D3, D7, R5, R10, R11, A6)."""

from __future__ import annotations

import logging

import pytest

from companydocs_rag.ingestion.settings import find_project_root
from tests.metadata._helpers import (
    LEASE,
    LOGGER,
    SHA_A,
    Generator,
    chunk,
    chunk_record,
    extract,
    log_records,
    reply,
    use_settings,
    write_json_file,
    write_overrides,
)

ROOT = find_project_root()
KEY_VARIABLE = "COMPANYDOCS_FICTIEVE_METADATA_SLEUTEL"
KEY_VALUE = "fictief-geheim-8R3yW"

# "The prompt file"'s text, written out.
PROMPT_TEXT = (
    "You extract dates and terms from one agreement. Use only the numbered "
    "sources in the user message. Each source is text read from pages of "
    "the document, often by OCR, and may contain reading errors.\n"
    "\n"
    "Rules:\n"
    "1. Extract only. Never calculate: do not add a period to a date, do "
    "not derive an end date from a start date and a duration, and do not "
    "count days.\n"
    "2. For every candidate give value, value_kind, as_written, quote and "
    "source. value is the value in the form rule 8 gives, with a date "
    "converted to YYYY-MM-DD. as_written is the words of the source that "
    "state the value, exactly as written. quote is the sentence or phrase "
    "that states the value, copied character for character from one single "
    "source: do not join text from two sources, do not correct reading "
    "errors and do not translate.\n"
    "3. source is the number n of the source [n] the quote is copied from. "
    "Refer to a source only by its number.\n"
    "4. List every candidate. When the sources give different values for a "
    "field, list each value as a separate candidate.\n"
    "5. When a field is not stated in the sources, return an empty list "
    "for it. Do not guess.\n"
    "6. Use only the dates of this agreement itself. Do not use dates from "
    "annexes, guarantees, register extracts, e-mails or copies of identity "
    "documents.\n"
    "7. notice_period is only the notice period for ordinary termination "
    "at the end of a term period. Break options, early termination on "
    "grounds, termination for cause and limits on one party's right to "
    "give notice go in other_termination_provisions, never in "
    "notice_period.\n"
    "8. The fields:\n"
    "   - signing_date: the date on which the agreement was signed or "
    "concluded.\n"
    "   - commencement_date: the date on which the agreement takes effect.\n"
    "   - initial_term_end: the last day of the first term.\n"
    '     For these three, value_kind is "date" for a calendar date, with '
    'the value as YYYY-MM-DD; "conditional" when the date depends on an '
    'event or a condition, which as_written names; "blank" when the '
    'document has an empty template field for the date; "illegible" when '
    'the date is there but cannot be read. For "conditional", "blank" and '
    '"illegible" the value is null.\n'
    "   - initial_term: the length of the first term. value_kind "
    '"duration", with the value as {"amount": <whole number>, "unit": '
    '"days", "weeks", "months" or "years"}; or "indefinite" when the '
    "agreement runs for an indefinite period, with the value null.\n"
    "   - renewal: what happens when the first term ends. value_kind "
    '"none" when the agreement then ends without renewal or continuation; '
    '"fixed_period" for a renewal or continuation for a fixed period, with '
    'that period as the value, written as for initial_term; "indefinite" '
    'for a continuation for an indefinite period. For "none" and '
    '"indefinite" the value is null.\n'
    '   - notice_period: value_kind "duration", with the period as the '
    "value, written as for initial_term.\n"
    "9. other_termination_provisions: one item per provision, with party, "
    'as_written, quote and source. party is "tenant", "landlord", "both" '
    'or "unspecified": the party the provision gives a right to, or whose '
    "right it limits.\n"
    "Answer with JSON only.\n"
)

# "The response schema", written out.
STRING_OR_NULL = {"anyOf": [{"type": "string"}, {"type": "null"}]}
DURATION = {
    "type": "object",
    "properties": {
        "amount": {"type": "number"},
        "unit": {"type": "string", "enum": ["days", "weeks", "months", "years"]},
    },
    "required": ["amount", "unit"],
    "additionalProperties": False,
}
DURATION_OR_NULL = {"anyOf": [DURATION, {"type": "null"}]}
DATE_KINDS = ["date", "conditional", "blank", "illegible"]


def candidate_schema(kinds, value):
    """``C(kinds, V)``."""
    return {
        "type": "object",
        "properties": {
            "value": value,
            "value_kind": {"type": "string", "enum": kinds},
            "as_written": {"type": "string"},
            "quote": {"type": "string"},
            "source": {"type": "integer"},
        },
        "required": ["value", "value_kind", "as_written", "quote", "source"],
        "additionalProperties": False,
    }


PROVISION = {
    "type": "object",
    "properties": {
        "party": {
            "type": "string",
            "enum": ["tenant", "landlord", "both", "unspecified"],
        },
        "as_written": {"type": "string"},
        "quote": {"type": "string"},
        "source": {"type": "integer"},
    },
    "required": ["party", "as_written", "quote", "source"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "metadata_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "signing_date": {
                    "type": "array",
                    "items": candidate_schema(DATE_KINDS, STRING_OR_NULL),
                },
                "commencement_date": {
                    "type": "array",
                    "items": candidate_schema(DATE_KINDS, STRING_OR_NULL),
                },
                "initial_term": {
                    "type": "array",
                    "items": candidate_schema(
                        ["duration", "indefinite"], DURATION_OR_NULL
                    ),
                },
                "initial_term_end": {
                    "type": "array",
                    "items": candidate_schema(DATE_KINDS, STRING_OR_NULL),
                },
                "renewal": {
                    "type": "array",
                    "items": candidate_schema(
                        ["none", "fixed_period", "indefinite"], DURATION_OR_NULL
                    ),
                },
                "notice_period": {
                    "type": "array",
                    "items": candidate_schema(["duration"], DURATION),
                },
                "other_termination_provisions": {"type": "array", "items": PROVISION},
            },
            "required": [
                "signing_date",
                "commencement_date",
                "initial_term",
                "initial_term_end",
                "renewal",
                "notice_period",
                "other_termination_provisions",
            ],
            "additionalProperties": False,
        },
    },
}
FIRST_TEXT = "De fictieve huur gaat in op 1 april 2030.\nOpzegging per kwartaal."
SECOND_TEXT = "Fictieve looptijd van vijf jaar,\nwaarna verlenging volgt."


def _lease(tmp_path, *chunks):
    """One document of the lease family, through the override file."""
    write_json_file(
        tmp_path / "chunks",
        "huur.json",
        chunk_record(tmp_path / "los" / "fictief-huur.pdf", SHA_A, *chunks),
    )
    return write_overrides(
        tmp_path / "handmatig" / "types.csv", [("fictief-huur.pdf", LEASE)]
    )


def _two_sources(tmp_path):
    return _lease(
        tmp_path,
        chunk(SHA_A, 0, FIRST_TEXT, page_start=2),
        chunk(SHA_A, 1, SECOND_TEXT, page_start=3, page_end=4),
    )


def _errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


# --- The body (D3, D7, R5, R11) -------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    ["http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1/"],
    ids=["plain", "trailing-slash"],
)
def test_one_post_per_batch_with_the_body_in_order(tmp_path, monkeypatch, base_url):
    use_settings(
        tmp_path,
        monkeypatch,
        base_url=base_url,
        model="fictief-model",
        temperature=0.25,
        max_tokens=321,
        seed=7,
        reasoning_effort="low",
        extra_body={"keep_alive": "5m"},
        selection_budget_chars=len(FIRST_TEXT),
    )
    overrides = _two_sources(tmp_path)
    generator = Generator()

    assert extract(tmp_path, overrides=overrides, generator=generator) == 0

    assert len(generator.requests) == 2
    for request in generator.requests:
        assert request.method == "POST"
        assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
    for body in generator.bodies:
        assert list(body) == [
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "seed",
            "stream",
            "reasoning_effort",
            "keep_alive",
            "response_format",
        ]
        assert body["model"] == "fictief-model"
        assert body["temperature"] == 0.25
        assert body["max_tokens"] == 321
        assert body["seed"] == 7
        assert body["stream"] is False
        assert body["reasoning_effort"] == "low"
        assert body["keep_alive"] == "5m"
        assert body["response_format"] == RESPONSE_FORMAT


def test_a_null_reasoning_effort_leaves_the_key_out(tmp_path, monkeypatch):
    use_settings(tmp_path, monkeypatch, reasoning_effort=None)
    overrides = _two_sources(tmp_path)
    generator = Generator()

    assert extract(tmp_path, overrides=overrides, generator=generator) == 0

    (body,) = generator.bodies
    assert "reasoning_effort" not in body
    assert list(body)[-2:] == ["keep_alive", "response_format"]


def test_the_messages_are_the_prompt_file_and_the_sources(tmp_path):
    overrides = _two_sources(tmp_path)
    generator = Generator()

    assert extract(tmp_path, overrides=overrides, generator=generator) == 0

    (body,) = generator.bodies
    system, user = body["messages"]
    assert system == {"role": "system", "content": PROMPT_TEXT}
    assert user == {
        "role": "user",
        "content": (
            "source [1], page 2\n"
            "De fictieve huur gaat in op 1 april 2030.\n"
            "Opzegging per kwartaal.\n"
            "\n"
            "source [2], pages 3-4\n"
            "Fictieve looptijd van vijf jaar,\n"
            "waarna verlenging volgt."
        ),
    }


# --- The prompt file (R10) ------------------------------------------------------------


def test_the_committed_prompt_file_holds_the_specs_text():
    raw = (ROOT / "config" / "prompts" / "metadata_extract_system.txt").read_bytes()

    assert raw.isascii()
    assert b"\r" not in raw
    assert raw.endswith(b".\n") and not raw.endswith(b"\n\n")
    assert raw.decode("ascii") == PROMPT_TEXT


@pytest.mark.parametrize("kind", ["missing", "not-utf-8"])
def test_an_unusable_prompt_file_cannot_start(tmp_path, monkeypatch, caplog, kind):
    prompt = tmp_path / "prompt.txt"
    if kind == "not-utf-8":
        prompt.write_bytes(b"Fictieve prompt \xff\n")
    use_settings(tmp_path, monkeypatch, prompt_file=str(prompt))
    overrides = _two_sources(tmp_path)
    generator = Generator()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(tmp_path, overrides=overrides, generator=generator)

    assert code == 2
    (error,) = _errors(caplog)
    assert error.startswith("Invalid settings: metadata.prompt_file: ")
    assert generator.requests == []
    assert not (tmp_path / "extracted").exists()


# --- The API key (R5) -----------------------------------------------------------------


def test_no_key_variable_sends_no_authorization(tmp_path):
    overrides = _two_sources(tmp_path)
    generator = Generator()

    assert extract(tmp_path, overrides=overrides, generator=generator) == 0

    assert "authorization" not in generator.requests[0].headers


def test_a_set_key_variable_is_sent_and_never_shown(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(KEY_VARIABLE, KEY_VALUE)
    use_settings(tmp_path, monkeypatch, api_key_env=KEY_VARIABLE, warmup=True)
    overrides = _two_sources(tmp_path)
    generator = Generator()

    with caplog.at_level(logging.DEBUG):
        code = extract(tmp_path, overrides=overrides, generator=generator)

    assert code == 0
    assert [request.headers["authorization"] for request in generator.requests] == [
        f"Bearer {KEY_VALUE}"
    ] * 2
    assert KEY_VALUE not in caplog.text
    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert any(path.suffix == ".jsonl" for path in written)
    for path in written:
        assert KEY_VALUE.encode("utf-8") not in path.read_bytes(), path


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_an_unset_or_empty_key_variable_cannot_start(
    tmp_path, monkeypatch, caplog, value
):
    if value is None:
        monkeypatch.delenv(KEY_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(KEY_VARIABLE, value)
    use_settings(tmp_path, monkeypatch, api_key_env=KEY_VARIABLE)
    overrides = _two_sources(tmp_path)
    generator = Generator()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = extract(tmp_path, overrides=overrides, generator=generator)

    assert code == 2
    (error,) = _errors(caplog)
    assert error.startswith("Invalid settings: metadata.api_key_env: ")
    assert KEY_VARIABLE in error
    assert generator.requests == []


# --- The warm-up (R5) -----------------------------------------------------------------


def test_the_warm_up_goes_first_and_is_not_logged(tmp_path, monkeypatch):
    use_settings(tmp_path, monkeypatch, warmup=True)
    overrides = _two_sources(tmp_path)
    generator = Generator(reply("fictief", completion_tokens=1, finish_reason="length"))

    assert extract(tmp_path, overrides=overrides, generator=generator) == 0

    warm_up, extraction = generator.bodies
    assert warm_up["messages"] == [{"role": "user", "content": "ok"}]
    assert warm_up["max_tokens"] == 1
    assert warm_up["response_format"] == RESPONSE_FORMAT
    assert warm_up["reasoning_effort"] == "high"
    assert extraction["max_tokens"] == 4000
    (record,) = log_records(tmp_path / "logs")
    assert record["request"] == extraction


def test_without_warm_up_only_the_extraction_requests_are_sent(tmp_path):
    overrides = _two_sources(tmp_path)
    generator = Generator()

    assert extract(tmp_path, overrides=overrides, generator=generator) == 0

    (body,) = generator.bodies
    assert body["max_tokens"] == 4000
    assert body["messages"][0]["role"] == "system"
