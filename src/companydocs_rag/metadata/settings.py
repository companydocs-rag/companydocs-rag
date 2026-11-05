"""The ``metadata`` settings (spec 008, D2, R3, R4).

The top-level ``metadata`` section of ``config/settings.yaml``, read at call
time with ``get_section("metadata")`` by :func:`metadata_settings`, YAML over
the defaults below; ``COMPANYDOCS_RAG_SETTINGS`` replaces the file, as for
every section. A key that is absent takes its default. Null is invalid except
for ``api_key_env`` and ``reasoning_effort``. A bool is not an integer or a
number, and an integer counts as a number (read as a float). Keys are checked
in D2 order, and the first invalid one raises
:class:`companydocs_rag.embeddings.dense.SettingsError` with section
``metadata``. No other section is resolved here.

``extra_body`` may hold neither ``response_format``, which the code adds, nor
any of the request's own fields (:data:`REQUEST_FIELDS`). ``keywords`` and
``month_names`` (R4) are mappings from a language code to a mapping of lists:
``infix``, ``prefix`` and ``whole_token`` (each optional) for the stems,
``full`` and ``abbreviated`` for the month names. Every stem and name is a
non-empty run of letters that equals its own ``hard_normalize``, so it can
match a token. A YAML value replaces the whole default mapping; nothing is
merged.

``prompt_file`` and ``log_dir`` are relative to the project root (the folder
that holds ``pyproject.toml``, else the working directory) unless absolute
(:func:`resolve_project_path`). The prompt file and the API key variable are
checked when ``extract`` starts, not here (R5, R10).
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from companydocs_rag.embeddings.dense import (
    SettingsError,
    is_finite_above_zero,
    setting_integer,
    setting_number,
    setting_string,
)
from companydocs_rag.ingestion.normalization import hard_normalize
from companydocs_rag.ingestion.settings import find_project_root, get_section

# --- Settings defaults (metadata), read at call time ----------------------------------
BASE_URL = "http://127.0.0.1:11434/v1"
MODEL = "companydocs-gemma4-e4b-8k"
API_KEY_ENV: Optional[str] = None
CONTEXT_WINDOW = 8192
MAX_TOKENS = 4000
TEMPERATURE = 0.0
SEED = 0
REASONING_EFFORT: Optional[str] = "high"
EXTRA_BODY: Dict[str, Any] = {"keep_alive": "60m"}
CONNECT_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 600.0
WARMUP = True
MAX_CHARS_PER_PROMPT_TOKEN = 4.6
PROMPT_FILE = "config/prompts/metadata_extract_system.txt"
LOG_DIR = "data/processed/metadata_logs"
SELECTION_BUDGET_CHARS = 9000
MAX_BATCHES_PER_DOCUMENT = 3
LEASE_FAMILY_TYPES: List[str] = [
    "Huurovereenkomst",
    "Allonge",
    "Lease- / exploitatieovereenkomst",
]
AMENDMENT_TYPES: List[str] = ["Allonge"]
# D6: the stems by document language, each list under its match mode.
KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "nld": {
        "infix": [
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
        ],
        "whole_token": ["getekend", "ondertekend"],
    },
    "eng": {
        "prefix": [
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
        ],
    },
}
# R4: probe 008's month names, full and abbreviated.
MONTH_NAMES: Dict[str, Dict[str, List[str]]] = {
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
}

# --- Python constants, not settings ---------------------------------------------------
# The request's own fields (``request_body`` in spec 007's generator); extra_body
# may not hold any of them (R3). Restated here: no module that ``report`` uses
# imports from ``companydocs_rag.answering`` (R1).
REQUEST_FIELDS = (
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "seed",
    "stream",
    "reasoning_effort",
)
# D2: the code adds it to extra_body.
RESPONSE_FORMAT_KEY = "response_format"
KEYWORD_INFIX = "infix"
KEYWORD_PREFIX = "prefix"
KEYWORD_WHOLE_TOKEN = "whole_token"
KEYWORD_MODES = (KEYWORD_INFIX, KEYWORD_PREFIX, KEYWORD_WHOLE_TOKEN)
MONTHS_FULL = "full"
MONTHS_ABBREVIATED = "abbreviated"
MONTH_NAME_KINDS = (MONTHS_FULL, MONTHS_ABBREVIATED)

_SECTION = "metadata"
_URL_SCHEMES = ("http://", "https://")
_LETTERS = re.compile(r"[^\W\d_]+")


@dataclass(frozen=True)
class MetadataSettings:
    """The resolved ``metadata`` section: the 21 keys in D2 order."""

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
    max_chars_per_prompt_token: float
    prompt_file: str
    log_dir: str
    selection_budget_chars: int
    max_batches_per_document: int
    lease_family_types: List[str]
    amendment_types: List[str]
    keywords: Dict[str, Dict[str, List[str]]]
    month_names: Dict[str, Dict[str, List[str]]]


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
    if RESPONSE_FORMAT_KEY in value:
        raise SettingsError(
            _SECTION,
            key,
            f"must not hold {_shown(RESPONSE_FORMAT_KEY)}, which the code adds",
        )
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


def _string_list(values: Mapping[str, Any], key: str, default: List[str]) -> List[str]:
    value = values.get(key, default)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item != "" for item in value
    ):
        raise _fail(key, "a list of non-empty strings", value)
    return list(value)


def is_matchable_word(value: Any) -> bool:
    """R4: a non-empty string of letters only that equals its own
    ``hard_normalize``, so it can match a token of a hard-normalized line."""
    return (
        isinstance(value, str)
        and _LETTERS.fullmatch(value) is not None
        and hard_normalize(value) == value
    )


def _word_lists(
    values: Mapping[str, Any],
    key: str,
    default: Mapping[str, Mapping[str, List[str]]],
    sub_keys: Sequence[str],
    *,
    all_required: bool,
) -> Dict[str, Dict[str, List[str]]]:
    """``keywords`` or ``month_names`` (R4): language code -> sub-key -> words."""
    value = values.get(key, default)
    sub_keys_shown = ", ".join(sub_keys)
    requirement = (
        "a mapping from a language code to a mapping "
        + ("with the keys " if all_required else "with the optional keys ")
        + f"{sub_keys_shown}, each a list of lowercase words of letters only"
    )
    if not isinstance(value, Mapping):
        raise _fail(key, requirement, value)
    resolved: Dict[str, Dict[str, List[str]]] = {}
    for language, lists in value.items():
        if not isinstance(language, str) or language == "":
            raise _fail(key, requirement, value)
        if not isinstance(lists, Mapping):
            raise _fail(key, requirement, value)
        unknown = [name for name in lists if name not in sub_keys]
        if unknown:
            raise SettingsError(
                _SECTION,
                key,
                f"{language}: {_shown(unknown[0])} is not one of {sub_keys_shown}",
            )
        if all_required:
            missing = [name for name in sub_keys if name not in lists]
            if missing:
                raise SettingsError(
                    _SECTION, key, f"{language}: lacks {_shown(missing[0])}"
                )
        resolved[language] = {}
        for name, words in lists.items():
            if not isinstance(words, list):
                raise SettingsError(
                    _SECTION,
                    key,
                    f"{language}.{name} must be a list, not {_shown(words)}",
                )
            for word in words:
                if not is_matchable_word(word):
                    raise SettingsError(
                        _SECTION,
                        key,
                        f"{language}.{name}: {_shown(word)} is not a non-empty "
                        "lowercase word of letters only that equals its own "
                        "hard_normalize",
                    )
            resolved[language][name] = list(words)
    return resolved


def metadata_settings(
    settings: Optional[Mapping[str, Any]] = None,
) -> MetadataSettings:
    """Resolve the ``metadata`` section at call time, YAML over the defaults
    above; ``settings`` replaces ``config/settings.yaml``. Keys are checked in
    D2 order, and the first invalid one raises :class:`SettingsError`."""
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
    max_chars_per_prompt_token = setting_number(
        section,
        values,
        "max_chars_per_prompt_token",
        MAX_CHARS_PER_PROMPT_TOKEN,
        requirement="a finite number above 0",
        valid=is_finite_above_zero,
    )
    prompt_file = setting_string(
        section, values, "prompt_file", PROMPT_FILE, empty_ok=False
    )
    log_dir = setting_string(section, values, "log_dir", LOG_DIR, empty_ok=False)
    selection_budget_chars = setting_integer(
        section, values, "selection_budget_chars", SELECTION_BUDGET_CHARS, minimum=1
    )
    max_batches_per_document = setting_integer(
        section,
        values,
        "max_batches_per_document",
        MAX_BATCHES_PER_DOCUMENT,
        minimum=1,
    )
    lease_family_types = _string_list(values, "lease_family_types", LEASE_FAMILY_TYPES)
    amendment_types = _string_list(values, "amendment_types", AMENDMENT_TYPES)
    keywords = _word_lists(
        values, "keywords", KEYWORDS, KEYWORD_MODES, all_required=False
    )
    month_names = _word_lists(
        values, "month_names", MONTH_NAMES, MONTH_NAME_KINDS, all_required=True
    )

    return MetadataSettings(
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
        max_chars_per_prompt_token=max_chars_per_prompt_token,
        prompt_file=prompt_file,
        log_dir=log_dir,
        selection_budget_chars=selection_budget_chars,
        max_batches_per_document=max_batches_per_document,
        lease_family_types=lease_family_types,
        amendment_types=amendment_types,
        keywords=keywords,
        month_names=month_names,
    )


# --- Project paths --------------------------------------------------------------------


def project_base() -> Path:
    """The project root (the folder that holds ``pyproject.toml``), else the
    working directory. Tests replace it."""
    root = find_project_root()
    return root if root is not None else Path.cwd()


def resolve_project_path(value: Union[str, Path]) -> Path:
    """``value`` as a path: as is when absolute, else under :func:`project_base`
    (the one place the package resolves a path against the project root)."""
    path = Path(value)
    return path if path.is_absolute() else project_base() / path
