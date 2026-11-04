"""Checking a response (spec 008, D8, D9, R12).

After the request, the tripwire and ``finish_reason`` (:mod:`.extract`), the
content is parsed as JSON (:func:`parse_content`): a null, empty or
whitespace-only content, or one that ``json.loads`` raises any exception for
(a ``RecursionError`` for deep nesting included; review F3), is
``invalid_json``. Then
its structure (:func:`check_structure`): an object with exactly the seven
keys of the schema, each value a list; each candidate an object with exactly
``value``, ``value_kind``, ``as_written``, ``quote`` and ``source``, the
middle three strings; each provision item an object with exactly ``party``,
``as_written``, ``quote`` and ``source``, the first three strings and
``party`` one of its four values. Anything else is ``schema_mismatch``.

``value`` and ``source`` are checked per candidate, so one bad candidate does
not void a batch (:func:`candidate_record`, :func:`provision_record`). A
``source`` lies within the batch when it is an integer, not a bool, from 1 to
the number of sources; outside, the quote does not verify and the candidate
has ``chunk_id`` null and ``page_spans`` []. A quote verifies when its
``hard_normalize`` is not empty and occurs in the ``hard_normalize`` of its
source chunk's text. A candidate counts when its value is valid and its quote
verifies.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from companydocs_rag.ingestion.normalization import hard_normalize

from .fields import FIELDS, PARTIES, PROVISIONS, stored_value, value_fits

CANDIDATE_KEYS = ("value", "value_kind", "as_written", "quote", "source")
PROVISION_KEYS = ("party", "as_written", "quote", "source")
RESPONSE_KEYS = (*FIELDS, PROVISIONS)


class InvalidContent(ValueError):
    """The content is not a JSON value (``invalid_json``)."""


class SchemaMismatch(ValueError):
    """The content does not have the schema's structure (``schema_mismatch``)."""


def parse_content(content: Optional[str]) -> Any:
    """The JSON value of a response's content (R12). Any exception while it
    is parsed is :class:`InvalidContent` (review F3)."""
    if content is None:
        raise InvalidContent("the content is null")
    if not content.strip():
        raise InvalidContent("the content is empty or whitespace only")
    try:
        return json.loads(content)
    except Exception as error:  # noqa: BLE001 - review F3: RecursionError included
        raise InvalidContent(f"the content is not JSON: {error}") from None


def _exact_keys(item: Any, keys: Sequence[str], where: str) -> None:
    if not isinstance(item, Mapping):
        raise SchemaMismatch(f"{where} is not an object")
    if set(item) != set(keys):
        missing = [key for key in keys if key not in item]
        extra = sorted(str(key) for key in item if key not in keys)
        problems = []
        if missing:
            problems.append("lacks " + ", ".join(missing))
        if extra:
            problems.append("has the extra key(s) " + ", ".join(extra))
        raise SchemaMismatch(f"{where} " + " and ".join(problems))


def _strings(item: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    for key in keys:
        if not isinstance(item[key], str):
            raise SchemaMismatch(f"{where}.{key} is not a string")


def check_structure(data: Any) -> None:
    """Raise :class:`SchemaMismatch` unless ``data`` has the structure of the
    response schema (R12)."""
    _exact_keys(data, RESPONSE_KEYS, "the content")
    for key in RESPONSE_KEYS:
        if not isinstance(data[key], list):
            raise SchemaMismatch(f"{key} is not a list")
    for field in FIELDS:
        for position, candidate in enumerate(data[field]):
            where = f"{field}[{position}]"
            _exact_keys(candidate, CANDIDATE_KEYS, where)
            _strings(candidate, ("value_kind", "as_written", "quote"), where)
    for position, item in enumerate(data[PROVISIONS]):
        where = f"{PROVISIONS}[{position}]"
        _exact_keys(item, PROVISION_KEYS, where)
        _strings(item, ("party", "as_written", "quote"), where)
        if item["party"] not in PARTIES:
            raise SchemaMismatch(
                f"{where}.party is {json.dumps(item['party'], ensure_ascii=False)}, "
                "not " + ", ".join(PARTIES)
            )


def source_chunk(
    source: Any, sources: Sequence[Mapping[str, Any]]
) -> Optional[Mapping[str, Any]]:
    """The chunk ``source`` refers to, or None when it lies outside the
    batch (R12)."""
    if type(source) is not int or not 1 <= source <= len(sources):
        return None
    return sources[source - 1]


def quote_verifies(quote: Any, chunk: Optional[Mapping[str, Any]]) -> bool:
    """R12: a non-empty hard-normalized quote that occurs in the
    hard-normalized text of its source chunk."""
    if chunk is None or not isinstance(quote, str):
        return False
    normalized = hard_normalize(quote)
    return normalized != "" and normalized in hard_normalize(chunk["text"])


def _provenance(chunk: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if chunk is None:
        return {"chunk_id": None, "page_spans": []}
    return {
        "chunk_id": chunk["chunk_id"],
        "page_spans": copy.deepcopy(chunk["page_spans"]),
    }


def candidate_record(
    field: str,
    raw: Mapping[str, Any],
    batch_index: int,
    sources: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """One candidate of the extraction file (D10), keys in order."""
    chunk = source_chunk(raw["source"], sources)
    kind = raw["value_kind"]
    value_valid = value_fits(field, kind, raw["value"])
    quote_verified = quote_verifies(raw["quote"], chunk)
    provenance = _provenance(chunk)
    return {
        "value": copy.deepcopy(stored_value(field, kind, raw["value"])),
        "value_kind": kind,
        "as_written": raw["as_written"],
        "quote": raw["quote"],
        "quote_verified": quote_verified,
        "value_valid": value_valid,
        "counted": value_valid and quote_verified,
        "batch_index": batch_index,
        "chunk_id": provenance["chunk_id"],
        "page_spans": provenance["page_spans"],
    }


def provision_record(
    raw: Mapping[str, Any],
    batch_index: int,
    sources: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """One item of ``other_termination_provisions`` (D10), keys in order."""
    chunk = source_chunk(raw["source"], sources)
    provenance = _provenance(chunk)
    return {
        "party": raw["party"],
        "as_written": raw["as_written"],
        "quote": raw["quote"],
        "quote_verified": quote_verifies(raw["quote"], chunk),
        "batch_index": batch_index,
        "chunk_id": provenance["chunk_id"],
        "page_spans": provenance["page_spans"],
    }


def batch_items(
    data: Mapping[str, Any],
    batch_index: int,
    sources: Sequence[Mapping[str, Any]],
) -> tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """The candidates per field and the provision items of one ok batch, in
    response order."""
    candidates = {
        field: [
            candidate_record(field, raw, batch_index, sources) for raw in data[field]
        ]
        for field in FIELDS
    }
    provisions = [
        provision_record(raw, batch_index, sources) for raw in data[PROVISIONS]
    ]
    return candidates, provisions
