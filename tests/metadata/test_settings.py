"""The ``metadata`` settings (spec 008, D2, R3, R4, A1)."""

from __future__ import annotations

import datetime
import logging

import pytest
import yaml

from companydocs_rag.embeddings.dense import SettingsError
from companydocs_rag.ingestion.settings import find_project_root
from companydocs_rag.metadata import settings
from companydocs_rag.metadata.selection import vocabulary
from companydocs_rag.metadata.settings import MetadataSettings, metadata_settings
from tests.metadata._helpers import (
    LEASE,
    Generator,
    chunk_record,
    extract,
    extraction_record,
    lease_text,
    pdf,
    report,
    use_settings,
    write_json_file,
)

# D2: the keys in order, with their constants.
KEYS = [
    ("base_url", "BASE_URL"),
    ("model", "MODEL"),
    ("api_key_env", "API_KEY_ENV"),
    ("context_window", "CONTEXT_WINDOW"),
    ("max_tokens", "MAX_TOKENS"),
    ("temperature", "TEMPERATURE"),
    ("seed", "SEED"),
    ("reasoning_effort", "REASONING_EFFORT"),
    ("extra_body", "EXTRA_BODY"),
    ("connect_timeout_s", "CONNECT_TIMEOUT_S"),
    ("read_timeout_s", "READ_TIMEOUT_S"),
    ("warmup", "WARMUP"),
    ("max_chars_per_prompt_token", "MAX_CHARS_PER_PROMPT_TOKEN"),
    ("prompt_file", "PROMPT_FILE"),
    ("log_dir", "LOG_DIR"),
    ("selection_budget_chars", "SELECTION_BUDGET_CHARS"),
    ("max_batches_per_document", "MAX_BATCHES_PER_DOCUMENT"),
    ("lease_family_types", "LEASE_FAMILY_TYPES"),
    ("amendment_types", "AMENDMENT_TYPES"),
    ("keywords", "KEYWORDS"),
    ("month_names", "MONTH_NAMES"),
]
NLD_INFIX = [
    "opzeg",
    "opgezegd",
    "termijn",
    "ingang",
    "ingaan",
    "ingaande",
    "aanvang",
    "duur",
    "looptijd",
    "verleng",
    "voortgezet",
    "onbepaalde",
    "aldus",
    "einde",
]
ENG_PREFIX = [
    "notice",
    "terminat",
    "commence",
    "term",
    "extend",
    "extension",
    "expir",
    "handover",
    "concluded",
    "prolong",
]
D2_DEFAULTS = {
    "base_url": "http://127.0.0.1:11434/v1",
    "model": "companydocs-gemma4-e4b-8k",
    "api_key_env": None,
    "context_window": 8192,
    "max_tokens": 4000,
    "temperature": 0.0,
    "seed": 0,
    "reasoning_effort": "high",
    "extra_body": {"keep_alive": "60m"},
    "connect_timeout_s": 5.0,
    "read_timeout_s": 600.0,
    "warmup": True,
    "max_chars_per_prompt_token": 4.6,
    "prompt_file": "config/prompts/metadata_extract_system.txt",
    "log_dir": "data/processed/metadata_logs",
    "selection_budget_chars": 9000,
    "max_batches_per_document": 3,
    "lease_family_types": [
        "Huurovereenkomst",
        "Allonge",
        "Lease- / exploitatieovereenkomst",
    ],
    "amendment_types": ["Allonge"],
    "keywords": {
        "nld": {"infix": NLD_INFIX, "whole_token": ["getekend", "ondertekend"]},
        "eng": {"prefix": ENG_PREFIX},
    },
    "month_names": {
        "nld": {
            "full": [
                "januari",
                "februari",
                "maart",
                "april",
                "mei",
                "juni",
                "juli",
                "augustus",
                "september",
                "oktober",
                "november",
                "december",
            ],
            "abbreviated": [
                "jan",
                "feb",
                "mrt",
                "apr",
                "jun",
                "jul",
                "aug",
                "sep",
                "sept",
                "okt",
                "nov",
                "dec",
            ],
        },
        "eng": {
            "full": [
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            ],
            "abbreviated": [
                "jan",
                "feb",
                "mar",
                "apr",
                "jun",
                "jul",
                "aug",
                "sep",
                "sept",
                "oct",
                "nov",
                "dec",
            ],
        },
    },
}
NAN, INF = float("nan"), float("inf")


def _yaml_text():
    return (find_project_root() / "config" / "settings.yaml").read_text(
        encoding="utf-8"
    )


# --- The defaults and the YAML section ------------------------------------------------


def test_the_defaults_are_d2s():
    assert list(MetadataSettings.__dataclass_fields__) == [key for key, _ in KEYS]
    for key, constant in KEYS:
        assert getattr(settings, constant) == D2_DEFAULTS[key], key
    config = metadata_settings({})
    for key, _ in KEYS:
        assert getattr(config, key) == D2_DEFAULTS[key], key
        assert type(getattr(config, key)) is type(D2_DEFAULTS[key]), key


def test_the_section_sits_between_answering_and_retrieval():
    sections = list(yaml.safe_load(_yaml_text()))

    position = sections.index("metadata")
    assert sections[position - 1 : position + 2] == [
        "answering",
        "metadata",
        "retrieval",
    ]


def test_the_yaml_section_holds_the_python_defaults_in_order():
    section = yaml.safe_load(_yaml_text())["metadata"]

    assert list(section) == [key for key, _ in KEYS]
    for key, constant in KEYS:
        assert section[key] == getattr(settings, constant), key
        assert type(section[key]) is type(getattr(settings, constant)), key


def _comments():
    """The comment lines above each key of the YAML section."""
    text = _yaml_text()
    lines = text[text.index("\nmetadata:") + 1 :].splitlines()[1:]
    comments, found = [], {}
    for line in lines:
        if not line.strip():
            continue
        if not line.startswith("  "):
            break
        stripped = line.strip()
        if stripped.startswith("#"):
            comments.append(stripped.lstrip("#").strip())
        elif line[2] != " " and ":" in stripped:
            found[stripped.split(":")[0]] = comments
            comments = []
    return found


def test_every_key_carries_the_settings_tables_comment():
    comments = _comments()

    assert list(comments) == [key for key, _ in KEYS]
    joined = {key: " ".join(lines) for key, lines in comments.items()}
    expected = {
        "max_tokens": (
            "Probe 008 Phase B: 1,058-3,673 completion tokens per request with "
            'reasoning_effort "high", reasoning included.'
        ),
        "reasoning_effort": (
            'Probe 008 Phase B ran every batch once with "none" and twice with "high".'
        ),
        "extra_body": (
            "keep_alive in the request body persists (probe 007 A6); the code adds "
            "response_format."
        ),
        "read_timeout_s": (
            "Probe 008 Phase B: the longest request took 237 s with "
            'reasoning_effort "high".'
        ),
        "max_chars_per_prompt_token": (
            "Probe 008 Phase B: 2.94-4.36 characters per prompt token; above this "
            "value the prompt was probably cut (spec 007 D6)."
        ),
        "selection_budget_chars": (
            "Probe 008 A6: at 9,000 characters, batch 1 held 43 of 51 "
            "ground-truth-draft passages and three batches 47."
        ),
        "keywords": (
            "Probe 008 A6.1: stems by document language; infix matches inside a "
            "word, prefix at its start, whole_token the whole word."
        ),
    }
    for key, comment in expected.items():
        assert joined[key] == comment, key
    for key, lines in comments.items():
        if key not in expected:
            assert len(lines) == 1, key


def test_the_section_header_names_the_spec_the_subcommands_and_the_package():
    text = _yaml_text()
    header = text[: text.index("\nmetadata:")].rsplit("\n\n", 1)[-1]

    for part in ("spec 008", "extract", "report", "src/companydocs_rag/metadata/"):
        assert part in header, part
    assert "probe 008" in header


# --- Reading the settings -------------------------------------------------------------


def test_a_yaml_value_replaces_its_default_at_call_time(tmp_path, monkeypatch):
    found = []
    for budget in (1234, 4321):
        use_settings(tmp_path, monkeypatch, selection_budget_chars=budget)
        found.append(metadata_settings().selection_budget_chars)

    assert found == [1234, 4321]


def test_an_absent_key_takes_its_default():
    config = metadata_settings({"metadata": {"max_tokens": 64}})

    assert config.max_tokens == 64
    assert config.context_window == 8192
    assert config.extra_body == {"keep_alive": "60m"}
    assert config.extra_body is not settings.EXTRA_BODY
    assert config.keywords is not settings.KEYWORDS


@pytest.mark.parametrize(
    ("key", "value", "resolved"),
    [
        ("api_key_env", None, None),
        ("reasoning_effort", None, None),
        ("temperature", 0, 0.0),
        ("read_timeout_s", 600, 600.0),
        ("max_chars_per_prompt_token", 5, 5.0),
        ("seed", 0, 0),
        ("warmup", False, False),
        ("extra_body", {}, {}),
        ("lease_family_types", [], []),
        ("keywords", {}, {}),
        ("month_names", {}, {}),
        (
            "base_url",
            "https://llm.voorbeeld.invalid/v1",
            "https://llm.voorbeeld.invalid/v1",
        ),
    ],
)
def test_valid_values(key, value, resolved):
    found = getattr(metadata_settings({"metadata": {key: value}}), key)

    assert found == resolved
    assert type(found) is type(resolved)


def test_a_yaml_keywords_with_only_nld_leaves_english_without_stems():
    config = metadata_settings(
        {"metadata": {"keywords": {"nld": {"infix": ["opzeg"]}}}}
    )

    assert config.keywords == {"nld": {"infix": ["opzeg"]}}
    assert vocabulary("eng", config.keywords, config.month_names).stems == ()


# --- Invalid values (R3, R4) ----------------------------------------------------------

STRING_KEYS = ["model", "prompt_file", "log_dir"]
INTEGER_KEYS = [
    "context_window",
    "max_tokens",
    "selection_budget_chars",
    "max_batches_per_document",
]
ABOVE_ZERO_KEYS = ["connect_timeout_s", "read_timeout_s", "max_chars_per_prompt_token"]


def _cases(keys, values):
    return [(key, value) for key in keys for value in values]


INVALID = [
    *_cases(
        ["base_url"], [5, "", "ftp://127.0.0.1/v1", "127.0.0.1:11434/v1", None, True]
    ),
    *_cases(STRING_KEYS, [5, "", None, True]),
    *_cases(["api_key_env", "reasoning_effort"], [5, "", ["x"], True]),
    *_cases(INTEGER_KEYS, ["9000", 2.5, 0, -3, None, True]),
    *_cases(["temperature"], ["0", -0.5, INF, NAN, None, True]),
    *_cases(["seed"], ["0", 1.5, -1, None, True]),
    *_cases(
        ["extra_body"],
        [
            "keep_alive",
            ["x"],
            {5: "x"},
            {"x": NAN},
            {"x": datetime.date(2031, 1, 2)},
            {"response_format": {"type": "json_object"}},
            {"model": "x"},
            {"messages": []},
            {"temperature": 1},
            {"max_tokens": 1},
            {"seed": 1},
            {"stream": True},
            {"reasoning_effort": "laag"},
            None,
            True,
        ],
    ),
    *_cases(ABOVE_ZERO_KEYS, ["5", 0, -1.0, INF, NAN, None, True]),
    *_cases(["warmup"], ["true", 1, 0, None]),
    *_cases(
        ["lease_family_types", "amendment_types"],
        ["Allonge", [""], [5], [None], {"Allonge": 1}, None, True],
    ),
    *_cases(
        ["keywords"],
        [
            ["opzeg"],
            {"nld": ["opzeg"]},
            {"nld": {"midden": ["opzeg"]}},
            {"nld": {"infix": "opzeg"}},
            {"nld": {"infix": [""]}},
            {"nld": {"infix": [5]}},
            {"nld": {"infix": ["opzeg2"]}},
            {"nld": {"infix": ["op zeg"]}},
            {"nld": {"infix": ["Opzeg"]}},
            None,
            True,
        ],
    ),
    *_cases(
        ["month_names"],
        [
            ["mei"],
            {"nld": {"full": ["mei"], "abbreviated": [], "anders": []}},
            {"nld": {"full": "mei", "abbreviated": []}},
            {"nld": {"full": [""], "abbreviated": []}},
            {"nld": {"full": [5], "abbreviated": []}},
            {"nld": {"full": ["mei2"], "abbreviated": []}},
            {"nld": {"full": ["m ei"], "abbreviated": []}},
            {"nld": {"full": ["Mei"], "abbreviated": []}},
            None,
            True,
        ],
    ),
]


def _case_id(case):
    key, value = case
    return f"{key}={value!r}"


@pytest.mark.parametrize(("key", "value"), INVALID, ids=[_case_id(c) for c in INVALID])
def test_an_invalid_value_is_a_settings_error(key, value):
    with pytest.raises(SettingsError) as raised:
        metadata_settings({"metadata": {key: value}})

    assert (raised.value.section, raised.value.key) == ("metadata", key)
    assert str(raised.value).startswith(f"metadata.{key}: ")


def test_the_first_invalid_key_in_d2_order_is_reported():
    with pytest.raises(SettingsError) as raised:
        metadata_settings({"metadata": {"keywords": 5, "seed": -1, "model": 5}})

    assert raised.value.key == "model"


def _errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.parametrize(("key", "value"), INVALID, ids=[_case_id(c) for c in INVALID])
def test_an_invalid_value_makes_both_subcommands_exit_2(
    tmp_path, monkeypatch, caplog, key, value
):
    use_settings(tmp_path, monkeypatch, **{key: value})
    write_json_file(
        tmp_path / "chunks",
        "fictief-huur.json",
        chunk_record(pdf(tmp_path, "fictief-huur.pdf"), "a1" * 32, lease_text()),
    )
    write_json_file(
        tmp_path / "extracted",
        "fictief-huur.json",
        extraction_record("bron/fictief-huur.pdf", "a1" * 32, document_type=LEASE),
    )
    generator = Generator()

    with caplog.at_level(logging.INFO, logger="companydocs_rag.metadata"):
        extract_code = extract(
            tmp_path, generator=generator, extracted_dir=tmp_path / "uit"
        )
        report_code = report(tmp_path)

    assert (extract_code, report_code) == (2, 2)
    errors = _errors(caplog)
    assert len(errors) == 2
    for error in errors:
        assert error.startswith(f"Invalid settings: metadata.{key}: ")
    assert generator.requests == []
    assert not (tmp_path / "uit").exists()
    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "rapport").exists()
