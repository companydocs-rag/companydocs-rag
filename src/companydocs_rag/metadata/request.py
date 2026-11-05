"""The request (spec 008, D3, D7, R5, R10, R11).

One request per batch. The system message is the prompt file, read as bytes
and decoded as UTF-8, unchanged; its sha256 is that of the bytes. The user
message holds the batch's sources only: per source the header
``source [n], page P`` (or ``pages P-Q`` when the chunk spans pages), a line
feed and the chunk text unchanged; the blocks are joined by a blank line.

``response_format`` carries :data:`RESPONSE_SCHEMA`, fixed here: per field a
list of candidates, and ``other_termination_provisions`` a list of items.
Every object lists all its keys under ``required`` and has
``additionalProperties`` false.

:class:`GeneratorConfig` is what spec 007's ``GeneratorClient`` and
``request_body`` read (D3): exactly their eight attributes, in that order,
its ``extra_body`` the settings' value in YAML order with ``response_format``
appended last. This module does not import the client; :mod:`.extract` does.
"""

from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from companydocs_rag.embeddings.dense import SettingsError

from .fields import (
    DATE_KINDS,
    FIELD_KINDS,
    FIELDS,
    INITIAL_TERM,
    NOTICE_PERIOD,
    PARTIES,
    PROVISIONS,
    RENEWAL,
    UNITS,
)
from .settings import RESPONSE_FORMAT_KEY, MetadataSettings, resolve_project_path

SCHEMA_NAME = "metadata_extraction"
ROLE_SYSTEM = "system"
ROLE_USER = "user"
BLOCK_SEPARATOR = "\n\n"

_SECTION = "metadata"

# --- The response schema (D7, R11) ----------------------------------------------------

STRING_OR_NULL: Dict[str, Any] = {"anyOf": [{"type": "string"}, {"type": "null"}]}
DURATION: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "amount": {"type": "number"},
        "unit": {"type": "string", "enum": list(UNITS)},
    },
    "required": ["amount", "unit"],
    "additionalProperties": False,
}
DURATION_OR_NULL: Dict[str, Any] = {"anyOf": [DURATION, {"type": "null"}]}
PROVISION: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "party": {"type": "string", "enum": list(PARTIES)},
        "as_written": {"type": "string"},
        "quote": {"type": "string"},
        "source": {"type": "integer"},
    },
    "required": ["party", "as_written", "quote", "source"],
    "additionalProperties": False,
}


def _candidate(kinds: Sequence[str], value: Mapping[str, Any]) -> Dict[str, Any]:
    """``C(kinds, V)``: one candidate object of the schema."""
    return {
        "type": "object",
        "properties": {
            "value": copy.deepcopy(dict(value)),
            "value_kind": {"type": "string", "enum": list(kinds)},
            "as_written": {"type": "string"},
            "quote": {"type": "string"},
            "source": {"type": "integer"},
        },
        "required": ["value", "value_kind", "as_written", "quote", "source"],
        "additionalProperties": False,
    }


def _field_value(field: str) -> Mapping[str, Any]:
    if FIELD_KINDS[field] == DATE_KINDS:
        return STRING_OR_NULL
    if field == NOTICE_PERIOD:
        return DURATION
    if field in (INITIAL_TERM, RENEWAL):
        return DURATION_OR_NULL
    raise ValueError(f"no schema value for {field!r}")


RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        **{
            field: {
                "type": "array",
                "items": _candidate(FIELD_KINDS[field], _field_value(field)),
            }
            for field in FIELDS
        },
        PROVISIONS: {"type": "array", "items": copy.deepcopy(PROVISION)},
    },
    "required": [*FIELDS, PROVISIONS],
    "additionalProperties": False,
}


def response_format() -> Dict[str, Any]:
    """The request's ``response_format`` (D7), a fresh copy."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": SCHEMA_NAME,
            "strict": True,
            "schema": copy.deepcopy(RESPONSE_SCHEMA),
        },
    }


# --- The generator configuration (D3, R5) ---------------------------------------------


@dataclass(frozen=True)
class GeneratorConfig:
    """Exactly the attributes ``GeneratorClient`` and ``request_body`` read
    (D3), in that order."""

    base_url: str
    connect_timeout_s: float
    read_timeout_s: float
    model: str
    temperature: float
    seed: int
    reasoning_effort: Optional[str]
    extra_body: Dict[str, Any]


def generator_config(config: MetadataSettings) -> GeneratorConfig:
    """The client's configuration: the settings' values, ``extra_body`` with
    ``response_format`` appended last (R5)."""
    extra_body = copy.deepcopy(config.extra_body)
    extra_body[RESPONSE_FORMAT_KEY] = response_format()
    return GeneratorConfig(
        base_url=config.base_url,
        connect_timeout_s=config.connect_timeout_s,
        read_timeout_s=config.read_timeout_s,
        model=config.model,
        temperature=config.temperature,
        seed=config.seed,
        reasoning_effort=config.reasoning_effort,
        extra_body=extra_body,
    )


# --- The prompt file and the API key (R5, R10) ----------------------------------------


@dataclass(frozen=True)
class SystemPrompt:
    """The prompt file's text and the sha256 of its bytes (R10)."""

    text: str
    sha256: str


def load_system_prompt(config: MetadataSettings) -> SystemPrompt:
    """Read the prompt file once, as bytes. A missing or unreadable file, or
    one that is not UTF-8, raises ``SettingsError`` for
    ``metadata.prompt_file``."""
    path = resolve_project_path(config.prompt_file)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SettingsError(
            _SECTION,
            "prompt_file",
            f"could not read {path}: {type(error).__name__}: {error}",
        ) from None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SettingsError(
            _SECTION, "prompt_file", f"{path} is not UTF-8: {error}"
        ) from None
    return SystemPrompt(text=text, sha256=hashlib.sha256(raw).hexdigest())


def read_api_key(config: MetadataSettings) -> Optional[str]:
    """The API key: the value of the variable ``api_key_env`` names, or None
    when it names none. An unset or empty variable raises ``SettingsError``
    for ``metadata.api_key_env``, naming the variable only (R5)."""
    name = config.api_key_env
    if name is None:
        return None
    value = os.environ.get(name)
    if value is None:
        raise SettingsError(
            _SECTION, "api_key_env", f"the environment variable {name} is not set"
        )
    if value == "":
        raise SettingsError(
            _SECTION, "api_key_env", f"the environment variable {name} is empty"
        )
    return value


# --- The messages (D7, R10) -----------------------------------------------------------


def source_header(number: int, page_start: int, page_end: int) -> str:
    """``source [n], page P``, or ``source [n], pages P-Q`` (R10)."""
    if page_start == page_end:
        return f"source [{number}], page {page_start}"
    return f"source [{number}], pages {page_start}-{page_end}"


def user_message(sources: Sequence[Mapping[str, Any]]) -> str:
    """The batch's sources, numbered from 1 in the given order (R10)."""
    return BLOCK_SEPARATOR.join(
        source_header(number, chunk["page_start"], chunk["page_end"])
        + "\n"
        + chunk["text"]
        for number, chunk in enumerate(sources, start=1)
    )


def messages(system: str, user: str) -> List[Dict[str, str]]:
    """``[system, user]``."""
    return [
        {"role": ROLE_SYSTEM, "content": system},
        {"role": ROLE_USER, "content": user},
    ]
