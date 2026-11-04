"""Loading of ``config/settings.yaml``.

The project keeps tuning data (thresholds, keyword lists, weights) in
``config/settings.yaml`` and program logic in Python.

Numeric thresholds and weights all have Python-side defaults, so the YAML file
only needs to carry the values actually being overridden. Two kinds of data
are the exception and live only in the YAML file, with no Python copy to drift
out of sync. The document type definitions (keyword lists, labels and regex
signals): without them, type detection reports "unknown/other" and says so,
and the parse pass still runs. And the
page-label regexes and conjunction lists under ``ingestion.normalization``
(spec 004): without them, the normalize command stops before it reads any
file.

The settings path can be overridden with the ``COMPANYDOCS_RAG_SETTINGS``
environment variable (defined in ``.env``), which is useful for experimenting
with alternative keyword lists during calibration without touching the
committed configuration.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

_SETTINGS_ENV_VAR = "COMPANYDOCS_RAG_SETTINGS"
_DEFAULT_RELATIVE_PATH = Path("config") / "settings.yaml"


def find_project_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk upwards from ``start`` looking for the project root.

    The root is recognised by containing ``pyproject.toml``. Returns ``None``
    when no such directory is found, which keeps the library importable from
    contexts where the source tree is not laid out as expected (e.g. an
    installed wheel).
    """
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def settings_path() -> Optional[Path]:
    """Resolve the settings file location, or ``None`` if there is none."""
    override = os.environ.get(_SETTINGS_ENV_VAR)
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None

    root = find_project_root()
    if root is None:
        return None
    path = root / _DEFAULT_RELATIVE_PATH
    return path if path.is_file() else None


@lru_cache(maxsize=1)
def load_settings() -> Dict[str, Any]:
    """Return the parsed settings file, or an empty mapping if unavailable.

    Cached, because it is read once per process and consulted from several
    places. Call ``load_settings.cache_clear()`` after editing the file inside
    a long-running session (a notebook, for instance).
    """
    path = settings_path()
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def get_section(
    *keys: str, settings: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Return a nested settings section, or an empty mapping if absent.

    ``get_section("ingestion", "parse_pass", "quality")`` returns the quality
    thresholds, and never raises for a missing or partially written file.
    """
    node: Any = load_settings() if settings is None else settings
    for key in keys:
        if not isinstance(node, Mapping):
            return {}
        node = node.get(key)
    return dict(node) if isinstance(node, Mapping) else {}


def parse_pass_section(
    name: str, settings: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Shorthand for the ``ingestion.parse_pass.<name>`` section."""
    return get_section("ingestion", "parse_pass", name, settings=settings)
