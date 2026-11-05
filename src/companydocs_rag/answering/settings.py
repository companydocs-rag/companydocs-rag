"""The ``answering`` settings (spec 007, D12, R20).

The top-level ``answering`` section of ``config/settings.yaml``, read at call
time with ``get_section("answering")`` by :func:`answering_settings`, YAML
over the defaults below; ``COMPANYDOCS_RAG_SETTINGS`` replaces the file, as
for every section. A key that is absent takes its default. Null is invalid
except for ``api_key_env``, ``reasoning_effort`` and ``min_dense_score``. A
bool is not an integer or a number, and an integer counts as a number (read
as a float). Keys are checked in the order below, and the first invalid one
raises :class:`companydocs_rag.embeddings.dense.SettingsError` with section
``answering``.

:func:`answering_settings` resolves the ``retrieval`` section too, for D3's
constraints, which are checked whatever ``selection_mode`` is: ``fused_k`` is
at most ``retrieval.candidate_count``, and ``per_path_k`` at most
``retrieval.dense_limit`` and ``retrieval.sparse_limit``. No key of
``extra_body`` may be one of the request's own fields
(:data:`REQUEST_FIELDS`), ``reasoning_effort`` included when that setting is
null (D5, R10).

``prompt_file`` and ``log_dir`` are relative to the project root (the folder
that holds ``pyproject.toml``, else the working directory) unless absolute
(:func:`resolve_project_path`). The prompt file and the API key variable are
checked when the answerer opens, not here (R8, R10).
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from companydocs_rag.embeddings.dense import (
    SettingsError,
    is_finite_above_zero,
    setting_integer,
    setting_number,
    setting_string,
)
from companydocs_rag.ingestion.settings import find_project_root, get_section
from companydocs_rag.retrieval.search import retrieval_settings

# --- Settings defaults (answering), read at call time ---------------------------------
BASE_URL = "http://127.0.0.1:11434/v1"
MODEL = "companydocs-gemma4-e4b-8k"
API_KEY_ENV: Optional[str] = None
CONTEXT_WINDOW = 8192
MAX_TOKENS = 800
TEMPERATURE = 0.0
SEED = 0
REASONING_EFFORT: Optional[str] = "none"
EXTRA_BODY: Dict[str, Any] = {"keep_alive": "60m"}
CONNECT_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 180.0
WARMUP = True
PROMPT_FILE = "config/prompts/answer_system_nl.txt"
SELECTION_MODE = "fused"
FUSED_K = 20
PER_PATH_K = 10
MAX_CONTEXT_CHARS = 15000
MAX_CHARS_PER_PROMPT_TOKEN = 4.6
MIN_DENSE_SCORE: Optional[float] = None
LOG_DIR = "data/processed/answer_logs"

# --- Python constants, not settings ---------------------------------------------------
SELECTION_FUSED = "fused"
SELECTION_PER_PATH = "per_path"
SELECTION_MODES = (SELECTION_FUSED, SELECTION_PER_PATH)
# D5: the request's own fields; extra_body may not hold any of them (R10).
REQUEST_FIELDS = (
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "seed",
    "stream",
    "reasoning_effort",
)

_SECTION = "answering"
_URL_SCHEMES = ("http://", "https://")


@dataclass(frozen=True)
class AnsweringSettings:
    """The resolved ``answering`` section: the 20 keys in D12 order."""

    base_url: str
    model: str
    api_key_env: Optional[str]
    context_window: int
    max_tokens: int
    temperature: float
    seed: int
    reasoning_effort: Optional[str]
    extra_body: Dict[str, Any]
    connect_timeout_s: float
    read_timeout_s: float
    warmup: bool
    prompt_file: str
    selection_mode: str
    fused_k: int
    per_path_k: int
    max_context_chars: int
    max_chars_per_prompt_token: float
    min_dense_score: Optional[float]
    log_dir: str


def _shown(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _fail(key: str, requirement: str, value: Any) -> SettingsError:
    return SettingsError(_SECTION, key, f"must be {requirement}, not {_shown(value)}")


def _base_url(values: Mapping[str, Any]) -> str:
    value = values.get("base_url", BASE_URL)
    if not isinstance(value, str) or not value.startswith(_URL_SCHEMES):
        raise _fail("base_url", "a string that starts with http:// or https://", value)
    return value


def _optional_string(
    values: Mapping[str, Any], key: str, default: Optional[str]
) -> Optional[str]:
    value = values.get(key, default)
    if value is not None and (not isinstance(value, str) or value == ""):
        raise _fail(key, "null or a non-empty string", value)
    return value


def _extra_body(values: Mapping[str, Any]) -> Dict[str, Any]:
    key = "extra_body"
    value = values.get(key, EXTRA_BODY)
    requirement = "a mapping with string keys and values that JSON can encode"
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise _fail(key, requirement, value)
    try:
        json.dumps(dict(value), allow_nan=False)
    except (TypeError, ValueError):
        raise _fail(key, requirement, value) from None
    for field in REQUEST_FIELDS:
        if field in value:
            raise SettingsError(
                _SECTION,
                key,
                f"must not hold {_shown(field)}, a field the request sets itself",
            )
    return copy.deepcopy(dict(value))


def _boolean(values: Mapping[str, Any], key: str, default: bool) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise _fail(key, "true or false", value)
    return value


def _choice(
    values: Mapping[str, Any], key: str, default: str, choices: Sequence[str]
) -> str:
    value = values.get(key, default)
    if not isinstance(value, str) or value not in choices:
        raise _fail(key, " or ".join(choices), value)
    return value


def _optional_score(
    values: Mapping[str, Any], key: str, default: Optional[float]
) -> Optional[float]:
    value = values.get(key, default)
    if value is None:
        return None
    requirement = "null, or a number from -1 to 1"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(key, requirement, value)
    number = float(value)
    if not -1.0 <= number <= 1.0:
        raise _fail(key, requirement, value)
    return number


def _at_most(key: str, value: int, limit_name: str, limit: int) -> None:
    if value > limit:
        raise SettingsError(
            _SECTION,
            key,
            f"must be at most retrieval.{limit_name} ({limit}), not {value}",
        )


def answering_settings(
    settings: Optional[Mapping[str, Any]] = None,
) -> AnsweringSettings:
    """Resolve the ``answering`` section at call time, YAML over the defaults
    above; ``settings`` replaces ``config/settings.yaml``. The ``retrieval``
    section is resolved first, for D3's constraints, and an invalid one raises
    its own :class:`SettingsError`. Then the keys are checked in D12 order,
    and the first invalid one raises."""
    retrieval = retrieval_settings(settings)
    section = _SECTION
    values = get_section(section, settings=settings)

    base_url = _base_url(values)
    model = setting_string(section, values, "model", MODEL, empty_ok=False)
    api_key_env = _optional_string(values, "api_key_env", API_KEY_ENV)
    context_window = setting_integer(
        section, values, "context_window", CONTEXT_WINDOW, minimum=1
    )
    max_tokens = setting_integer(section, values, "max_tokens", MAX_TOKENS, minimum=1)
    temperature = setting_number(
        section,
        values,
        "temperature",
        TEMPERATURE,
        requirement="a finite number of at least 0",
        valid=lambda number: math.isfinite(number) and number >= 0,
    )
    seed = setting_integer(section, values, "seed", SEED, minimum=0)
    reasoning_effort = _optional_string(values, "reasoning_effort", REASONING_EFFORT)
    extra_body = _extra_body(values)
    connect_timeout_s = setting_number(
        section,
        values,
        "connect_timeout_s",
        CONNECT_TIMEOUT_S,
        requirement="a finite number above 0",
        valid=is_finite_above_zero,
    )
    read_timeout_s = setting_number(
        section,
        values,
        "read_timeout_s",
        READ_TIMEOUT_S,
        requirement="a finite number above 0",
        valid=is_finite_above_zero,
    )
    warmup = _boolean(values, "warmup", WARMUP)
    prompt_file = setting_string(
        section, values, "prompt_file", PROMPT_FILE, empty_ok=False
    )
    selection_mode = _choice(values, "selection_mode", SELECTION_MODE, SELECTION_MODES)
    fused_k = setting_integer(section, values, "fused_k", FUSED_K, minimum=1)
    _at_most("fused_k", fused_k, "candidate_count", retrieval.candidate_count)
    per_path_k = setting_integer(section, values, "per_path_k", PER_PATH_K, minimum=1)
    _at_most("per_path_k", per_path_k, "dense_limit", retrieval.dense_limit)
    _at_most("per_path_k", per_path_k, "sparse_limit", retrieval.sparse_limit)
    max_context_chars = setting_integer(
        section, values, "max_context_chars", MAX_CONTEXT_CHARS, minimum=1
    )
    max_chars_per_prompt_token = setting_number(
        section,
        values,
        "max_chars_per_prompt_token",
        MAX_CHARS_PER_PROMPT_TOKEN,
        requirement="a finite number above 0",
        valid=is_finite_above_zero,
    )
    min_dense_score = _optional_score(values, "min_dense_score", MIN_DENSE_SCORE)
    log_dir = setting_string(section, values, "log_dir", LOG_DIR, empty_ok=False)

    return AnsweringSettings(
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        context_window=context_window,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body,
        connect_timeout_s=connect_timeout_s,
        read_timeout_s=read_timeout_s,
        warmup=warmup,
        prompt_file=prompt_file,
        selection_mode=selection_mode,
        fused_k=fused_k,
        per_path_k=per_path_k,
        max_context_chars=max_context_chars,
        max_chars_per_prompt_token=max_chars_per_prompt_token,
        min_dense_score=min_dense_score,
        log_dir=log_dir,
    )


def settings_snapshot(
    config: AnsweringSettings, *, selection_mode: str
) -> Dict[str, Any]:
    """The record's ``settings.answering``: the 20 resolved keys in D12 order,
    with ``selection_mode`` the mode the answer used (Q6). ``api_key_env`` is
    the variable's name, never its value (R10)."""
    snapshot = {
        field.name: copy.deepcopy(getattr(config, field.name))
        for field in fields(config)
    }
    snapshot["selection_mode"] = selection_mode
    return snapshot


def _project_base() -> Path:
    root = find_project_root()
    return root if root is not None else Path.cwd()


def resolve_project_path(value: str) -> Path:
    """``value`` as a path: as is when absolute, else under the project root
    (the folder that holds ``pyproject.toml``, else the working directory)."""
    path = Path(value)
    return path if path.is_absolute() else _project_base() / path
