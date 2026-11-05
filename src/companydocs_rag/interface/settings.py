"""The ``interface`` settings (spec 009, D14, R15).

The top-level ``interface`` section of ``config/settings.yaml``, read at call
time with ``get_section("interface")`` by :func:`interface_settings`, YAML
over the defaults below; ``COMPANYDOCS_RAG_SETTINGS`` replaces the file, as
for every section. A key that is absent takes its default; null is invalid,
and a bool is not an integer. Keys are checked in the order below, and the
first invalid one raises
:class:`companydocs_rag.embeddings.dense.SettingsError` with section
``interface``: through ``setting_integer`` for the type and the lower bound,
and with the same form for the upper bound.

- ``port``: an integer from 1 to 65535; ``--port`` replaces it for a run.
- ``render_dpi``: an integer from 50 to 300, the resolution of a page view.
- ``max_question_chars``: an integer of at least 1, the longest question in
  Python str characters (code points).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from companydocs_rag.embeddings.dense import SettingsError, setting_integer
from companydocs_rag.ingestion.settings import get_section

# --- Settings defaults (interface), read at call time ---------------------------------
PORT = 8765
RENDER_DPI = 150
MAX_QUESTION_CHARS = 1000

# --- Python constants, not settings ---------------------------------------------------
PORT_MIN = 1
PORT_MAX = 65535
RENDER_DPI_MIN = 50
RENDER_DPI_MAX = 300
MAX_QUESTION_CHARS_MIN = 1

_SECTION = "interface"


@dataclass(frozen=True)
class InterfaceSettings:
    """The resolved ``interface`` section: the three keys in R15's order."""

    port: int
    render_dpi: int
    max_question_chars: int


def _integer_in_range(
    values: Mapping[str, Any], key: str, default: int, minimum: int, maximum: int
) -> int:
    value = setting_integer(_SECTION, values, key, default, minimum=minimum)
    if value > maximum:
        raise SettingsError(
            _SECTION,
            key,
            f"must be an integer of at most {maximum}, not {json.dumps(value)}",
        )
    return value


def interface_settings(
    settings: Optional[Mapping[str, Any]] = None,
) -> InterfaceSettings:
    """Resolve the ``interface`` section at call time, YAML over the defaults
    above; ``settings`` replaces ``config/settings.yaml``. The first invalid
    key raises :class:`SettingsError`."""
    values = get_section(_SECTION, settings=settings)
    port = _integer_in_range(values, "port", PORT, PORT_MIN, PORT_MAX)
    render_dpi = _integer_in_range(
        values, "render_dpi", RENDER_DPI, RENDER_DPI_MIN, RENDER_DPI_MAX
    )
    max_question_chars = setting_integer(
        _SECTION,
        values,
        "max_question_chars",
        MAX_QUESTION_CHARS,
        minimum=MAX_QUESTION_CHARS_MIN,
    )
    return InterfaceSettings(
        port=port, render_dpi=render_dpi, max_question_chars=max_question_chars
    )
