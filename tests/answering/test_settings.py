"""The ``answering`` settings (spec 007, D12, R20, Q8, A13)."""

from __future__ import annotations

import datetime
import logging

import pytest
import yaml

from companydocs_rag.answering import cli, settings
from companydocs_rag.answering.settings import AnsweringSettings, answering_settings
from companydocs_rag.embeddings.dense import SettingsError
from companydocs_rag.ingestion.settings import find_project_root
from tests.answering._helpers import (
    FakeRetriever,
    Generator,
    simple_result,
    use_settings,
    write_manifest,
)

# D12: the keys in order, with their constants.
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
    ("prompt_file", "PROMPT_FILE"),
    ("selection_mode", "SELECTION_MODE"),
    ("fused_k", "FUSED_K"),
    ("per_path_k", "PER_PATH_K"),
    ("max_context_chars", "MAX_CONTEXT_CHARS"),
    ("max_chars_per_prompt_token", "MAX_CHARS_PER_PROMPT_TOKEN"),
    ("min_dense_score", "MIN_DENSE_SCORE"),
    ("log_dir", "LOG_DIR"),
]
D12_DEFAULTS = {
    "base_url": "http://127.0.0.1:11434/v1",
    "model": "companydocs-gemma4-e4b-8k",
    "api_key_env": None,
    "context_window": 8192,
    "max_tokens": 800,
    "temperature": 0.0,
    "seed": 0,
    "reasoning_effort": "none",
    "extra_body": {"keep_alive": "60m"},
    "connect_timeout_s": 5.0,
    "read_timeout_s": 180.0,
    "warmup": True,
    "prompt_file": "config/prompts/answer_system_nl.txt",
    "selection_mode": "fused",
    "fused_k": 20,
    "per_path_k": 10,
    "max_context_chars": 15000,
    "max_chars_per_prompt_token": 4.6,
    "min_dense_score": None,
    "log_dir": "data/processed/answer_logs",
}
NAN, INF = float("nan"), float("inf")


def _yaml_text():
    return (find_project_root() / "config" / "settings.yaml").read_text(
        encoding="utf-8"
    )


def test_the_defaults_are_d12s():
    assert [field for field in AnsweringSettings.__dataclass_fields__] == [
        key for key, _ in KEYS
    ]
    for key, constant in KEYS:
        assert getattr(settings, constant) == D12_DEFAULTS[key], key
    config = answering_settings({})
    for key, _ in KEYS:
        assert getattr(config, key) == D12_DEFAULTS[key], key


def test_the_yaml_section_holds_the_python_defaults_in_order():
    section = yaml.safe_load(_yaml_text())["answering"]

    assert list(section) == [key for key, _ in KEYS]
    for key, constant in KEYS:
        assert section[key] == getattr(settings, constant), key
        assert type(section[key]) is type(getattr(settings, constant)), key


def _comments():
    """The comment lines above each key of the YAML section, joined."""
    text = _yaml_text()
    lines = text[text.index("\nanswering:") + 1 :].splitlines()[1:]
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


def test_the_yaml_section_carries_r20s_comments():
    comments = _comments()

    assert list(comments) == [key for key, _ in KEYS]
    joined = {key: " ".join(lines).lower() for key, lines in comments.items()}
    expected = {
        "max_context_chars": (
            "characters per prompt token: nld 3.20, deu 3.62, eng 4.25, pol 2.86, "
            "ces 2.49, slk 2.47 (probe 007 a4); the budget assumes the worst."
        ),
        "model": (
            "base tags run with context 4096 and truncate silently (probe 007 a2)."
        ),
        "reasoning_effort": (
            'reasoning is on by default; reasoning_effort "none" turns it off '
            "(probe 007 a5)."
        ),
        "extra_body": "keep_alive in the request body persists (probe 007 a6).",
        "min_dense_score": (
            "spec-006 run, no-answer query 0.4505; three answered numeric queries "
            "0.4402-0.4444; the other six 0.5037-0.7919; ten queries, no calibration."
        ),
    }
    for key, comment in expected.items():
        assert joined[key] == comment, key
    assert "max_context_chars" in joined["max_chars_per_prompt_token"]
    for key, lines in comments.items():
        if key not in expected:
            assert len(lines) == 1, key


# --- Invalid values (R20) -------------------------------------------------------------

STRING_KEYS = ["model", "prompt_file", "log_dir"]
INTEGER_KEYS = ["context_window", "max_tokens", "max_context_chars"]
TIMEOUT_KEYS = ["connect_timeout_s", "read_timeout_s", "max_chars_per_prompt_token"]


def _cases(keys, values):
    return [(key, value) for key in keys for value in values]


INVALID = [
    *_cases(
        ["base_url"], [5, "", "ftp://127.0.0.1/v1", "127.0.0.1:11434/v1", None, True]
    ),
    *_cases(STRING_KEYS, [5, "", None, True]),
    *_cases(["api_key_env", "reasoning_effort"], [5, "", ["x"], True]),
    *_cases(INTEGER_KEYS, ["8192", 2.5, 0, -3, None, True]),
    *_cases(["temperature"], ["0", -0.5, INF, NAN, None, True]),
    *_cases(["seed"], ["0", 1.5, -1, None, True]),
    *_cases(
        ["extra_body"],
        [
            "keep_alive",
            ["x"],
            {5: "x"},
            {"x": NAN},
            {"x": datetime.date(2026, 1, 2)},
            {"model": "x"},
            {"reasoning_effort": "laag"},
            None,
            True,
        ],
    ),
    *_cases(TIMEOUT_KEYS, ["5", 0, -1.0, INF, NAN, None, True]),
    *_cases(["warmup"], ["true", 1, 0, None]),
    *_cases(["selection_mode"], [5, "hybride", "", "Fused", None, True]),
    *_cases(["fused_k"], ["20", 2.5, 0, 21, None, True]),
    *_cases(["per_path_k"], ["10", 2.5, 0, 51, None, True]),
    *_cases(["min_dense_score"], ["0.5", -1.5, 1.5, NAN, INF, True]),
]


def _case_id(case):
    key, value = case
    return f"{key}={value!r}"


@pytest.mark.parametrize(("key", "value"), INVALID, ids=[_case_id(c) for c in INVALID])
def test_an_invalid_value_is_a_settings_error(key, value):
    with pytest.raises(SettingsError) as raised:
        answering_settings({"answering": {key: value}})

    assert (raised.value.section, raised.value.key) == ("answering", key)
    assert str(raised.value).startswith(f"answering.{key}: ")


@pytest.mark.parametrize(("key", "value"), INVALID, ids=[_case_id(c) for c in INVALID])
def test_an_invalid_value_makes_the_command_exit_2(
    tmp_path, monkeypatch, caplog, capsys, key, value
):
    use_settings(
        tmp_path,
        monkeypatch,
        {"answering": {"log_dir": str(tmp_path / "logs"), "warmup": False, key: value}},
    )
    write_manifest(tmp_path / "index")
    retriever, generator = FakeRetriever(simple_result()), Generator()

    with caplog.at_level(logging.WARNING, logger="companydocs_rag.answering.cli"):
        code = cli.main(
            ["Fictieve vraag?", "--index-dir", str(tmp_path / "index")],
            retriever=retriever,
            transport=generator.transport,
        )

    assert code == 2
    (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert error.startswith(f"Invalid settings: answering.{key}: ")
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "logs").exists()
    assert (retriever.calls, generator.requests) == ([], [])


def test_the_first_invalid_key_in_d12_order_is_reported():
    with pytest.raises(SettingsError) as raised:
        answering_settings({"answering": {"log_dir": "", "seed": -1, "model": 5}})

    assert raised.value.key == "model"


@pytest.mark.parametrize(
    ("key", "value", "resolved"),
    [
        ("api_key_env", None, None),
        ("reasoning_effort", None, None),
        ("min_dense_score", None, None),
        ("min_dense_score", -1, -1.0),
        ("min_dense_score", 1, 1.0),
        ("temperature", 0, 0.0),
        ("temperature", 1.5, 1.5),
        ("seed", 0, 0),
        ("warmup", False, False),
        ("extra_body", {}, {}),
        ("selection_mode", "per_path", "per_path"),
        ("connect_timeout_s", 1, 1.0),
        (
            "base_url",
            "https://llm.voorbeeld.invalid:8443/v1",
            "https://llm.voorbeeld.invalid:8443/v1",
        ),
        ("base_url", "http://192.0.2.10:11434/v1/", "http://192.0.2.10:11434/v1/"),
    ],
)
def test_valid_values(key, value, resolved):
    found = getattr(answering_settings({"answering": {key: value}}), key)

    assert found == resolved
    assert type(found) is type(resolved)


def test_an_absent_key_takes_its_default():
    config = answering_settings({"answering": {"max_tokens": 64}})

    assert config.max_tokens == 64
    assert config.context_window == 8192
    assert config.extra_body == {"keep_alive": "60m"}
    assert config.extra_body is not settings.EXTRA_BODY


def test_the_settings_are_read_at_call_time(tmp_path, monkeypatch, capsys):
    write_manifest(tmp_path / "index")
    found = []
    for max_tokens in (111, 222):
        use_settings(
            tmp_path,
            monkeypatch,
            {
                "answering": {
                    "log_dir": str(tmp_path / "logs"),
                    "warmup": False,
                    "max_tokens": max_tokens,
                }
            },
        )
        generator = Generator()
        code = cli.main(
            ["Fictieve vraag?", "--index-dir", str(tmp_path / "index")],
            retriever=FakeRetriever(simple_result()),
            transport=generator.transport,
        )
        assert code == 0
        found.append(generator.bodies[0]["max_tokens"])

    assert found == [111, 222]
