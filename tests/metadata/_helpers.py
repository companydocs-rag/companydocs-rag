"""Shared helpers for the spec 008 tests: fabricated chunk files, client
indexes, override and relations files, a recording MockTransport handler and
fabricated extraction files.

Every path a test writes is under ``tmp_path``, and every run passes all of
the command's folder and file flags, so no default under ``data/`` is read or
written; the log's folder is set under ``tmp_path`` too (``conftest.py``).
Text, dates and file names are fabricated, never taken from a contract.
This module imports neither ``companydocs_rag.answering`` nor
``qdrant_client``, so the import checks of ``report`` (A15) can use it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import httpx

from tests.embeddings._helpers import write_settings

LOGGER = "companydocs_rag.metadata"
SHA_A = "a1" * 32
SHA_B = "b2" * 32
SHA_C = "c3" * 32
SHA_D = "d4" * 32
LEASE = "Huurovereenkomst"
AMENDMENT = "Allonge"
OTHER_TYPE = "Statuten"
UTF8_BOM = b"\xef\xbb\xbf"
RESPONSE_KEYS = (
    "signing_date",
    "commencement_date",
    "initial_term",
    "initial_term_end",
    "renewal",
    "notice_period",
    "other_termination_provisions",
)


# --- Settings -------------------------------------------------------------------------


def use_settings(tmp_path: Path, monkeypatch: Any, **metadata: Any) -> Path:
    """Point ``COMPANYDOCS_RAG_SETTINGS`` at a file whose ``metadata`` section
    holds no warm-up and the log under ``tmp_path``, unless overridden."""
    values: Dict[str, Any] = {"warmup": False, "log_dir": str(tmp_path / "logs")}
    values.update(metadata)
    return write_settings(tmp_path, monkeypatch, {"metadata": values})


# --- Chunk files ----------------------------------------------------------------------


def span(page: int, *, origin: str = "native", length: int = 10) -> Dict[str, Any]:
    """One ``page_spans`` entry, as the chunker writes it."""
    return {
        "page_number": page,
        "char_start": 0,
        "char_end": length,
        "text_origin": origin,
        "tier": "Clean" if origin == "native" else None,
        "ocr_mean_confidence": None if origin == "native" else 80.0,
    }


def chunk_id(file_sha256: Optional[str], index: int, text: str) -> str:
    return hashlib.sha256(f"{file_sha256}:{index}:{text}".encode("utf-8")).hexdigest()[
        :20
    ]


def chunk(
    file_sha256: Optional[str],
    index: int,
    text: str,
    *,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    origin: str = "native",
) -> Dict[str, Any]:
    """A chunk in the shape ``companydocs-chunk`` writes; its pages default
    to page ``index + 1``."""
    start = index + 1 if page_start is None else page_start
    end = start if page_end is None else page_end
    spans = [span(page, origin=origin) for page in range(start, end + 1)]
    return {
        "chunk_id": chunk_id(file_sha256, index, text),
        "chunk_index": index,
        "file_sha256": file_sha256,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "char_start": 0,
        "char_end": len(text),
        "page_start": start,
        "page_end": end,
        "text_origins": [origin],
        "page_spans": spans,
        "chunk_method": "generic",
        "clause_label": None,
        "clause_number": None,
        "numbering_run": None,
        "heading_family": None,
        "part": None,
        "part_count": None,
        "exceeds_max_chars": False,
    }


def chunk_record(
    path: Union[str, Path],
    sha256: Optional[str],
    *items: Union[str, Dict[str, Any]],
    language: Optional[str] = "nld",
    language_source: Optional[str] = "index",
    status: str = "processed",
    origin: str = "native",
) -> Dict[str, Any]:
    """A chunk file's record. Each item is a chunk, or a text that becomes
    the next chunk on its own page."""
    chunks = [
        item if isinstance(item, dict) else chunk(sha256, position, item, origin=origin)
        for position, item in enumerate(items)
    ]
    return {
        "schema_version": 1,
        "chunker_version": 1,
        "file": {"path": str(path), "sha256": sha256},
        "status": status,
        "normalized_hash": "0" * 64,
        "language": language,
        "language_source": language_source,
        "structure_hint": "unknown",
        "route": "generic",
        "empty_pages": [],
        "heading_evidence": {
            "family": None,
            "candidate_counts": {"keyword": 0, "decimal": 0},
            "runs": [],
        },
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def write_json_file(folder: Path, relative: str, content: Any) -> Path:
    """Write ``content`` as JSON (a string as it is) at ``folder/relative``."""
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        content
        if isinstance(content, str)
        else json.dumps(content, indent=2, ensure_ascii=False)
    )
    path.write_text(text, encoding="utf-8")
    return path


def pdf(tmp_path: Path, *parts: str) -> Path:
    """A fabricated PDF path under ``tmp_path / "bron"``; nothing is written."""
    return tmp_path.joinpath("bron", *parts)


def lease_text(index: int = 1) -> str:
    """A fabricated chunk text that scores: stems and a date."""
    return (
        f"Artikel {index}. De fictieve huur gaat in op 1 april 2030 en de "
        "looptijd is vijf jaar.\nOpzegging geschiedt met een termijn van drie "
        "maanden."
    )


# --- CSV files ------------------------------------------------------------------------


def write_csv(path: Path, rows: Sequence[Sequence[str]], *, bom: bool = True) -> Path:
    """A ``;``-separated file, LF line ends, UTF-8 with or without BOM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(";".join(row) + "\n" for row in rows)
    path.write_bytes((UTF8_BOM if bom else b"") + text.encode("utf-8"))
    return path


def write_index(folder: Path, rows: Sequence[Sequence[str]], **kwargs: Any) -> Path:
    """A client index ``00_INDEX.csv`` in ``folder``: (Map, Bestandsnaam,
    Documenttype) rows."""
    return write_csv(
        folder / "00_INDEX.csv",
        [("Map", "Bestandsnaam", "Documenttype"), *rows],
        **kwargs,
    )


def write_overrides(path: Path, rows: Sequence[Sequence[str]], **kwargs: Any) -> Path:
    return write_csv(path, [("Bestandsnaam", "Documenttype"), *rows], **kwargs)


def write_relations(path: Path, rows: Sequence[Sequence[str]], **kwargs: Any) -> Path:
    return write_csv(
        path, [("Bestandsnaam", "Relatie", "Doelbestand", "Detail"), *rows], **kwargs
    )


# --- The generator: a recording MockTransport handler ---------------------------------


def candidate(
    value: Any,
    kind: str,
    quote: str,
    *,
    source: Any = 1,
    as_written: str = "fictief",
) -> Dict[str, Any]:
    """One candidate as the model returns it."""
    return {
        "value": value,
        "value_kind": kind,
        "as_written": as_written,
        "quote": quote,
        "source": source,
    }


def provision(
    party: str, quote: str, *, source: Any = 1, as_written: str = "fictief"
) -> Dict[str, Any]:
    """One provision item as the model returns it."""
    return {"party": party, "as_written": as_written, "quote": quote, "source": source}


def years(amount: Any) -> Dict[str, Any]:
    return {"amount": amount, "unit": "years"}


def months(amount: Any) -> Dict[str, Any]:
    return {"amount": amount, "unit": "months"}


def answer(**fields: Any) -> str:
    """A response content with the seven keys, each [] unless given."""
    return json.dumps({key: fields.get(key, []) for key in RESPONSE_KEYS})


def prompt_characters(body: Mapping[str, Any]) -> int:
    return sum(len(message["content"]) for message in body["messages"])


def completion_body(
    content: Optional[str],
    *,
    prompt_tokens: int,
    finish_reason: Optional[str] = "stop",
    completion_tokens: Optional[int] = 12,
    reasoning: Optional[str] = None,
) -> Dict[str, Any]:
    """A chat-completions response body as Ollama's ``/v1`` writes it."""
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "id": "chatcmpl-fictief",
        "object": "chat.completion",
        "model": "fictief-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + (completion_tokens or 0),
        },
    }


Reply = Union[httpx.Response, BaseException, Callable[[httpx.Request], httpx.Response]]


def reply(
    content: Optional[str] = None,
    *,
    prompt_tokens: Optional[int] = None,
    empty: bool = False,
    **fields: Any,
) -> Callable[[httpx.Request], httpx.Response]:
    """A 200 response with ``content`` (an empty answer unless given; None
    with ``empty``). Unless given, ``prompt_tokens`` is the request's
    characters divided by 3, rounded up: well below the tripwire."""
    text = None if empty else (answer() if content is None else content)

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        tokens = (
            prompt_tokens
            if prompt_tokens is not None
            else math.ceil(prompt_characters(body) / 3)
        )
        return httpx.Response(
            200, json=completion_body(text, prompt_tokens=tokens, **fields)
        )

    return respond


class Generator:
    """The handler of an ``httpx.MockTransport``. Each request takes the next
    reply: a response, an exception to raise, or a function of the request;
    when the script runs out, ``default`` answers. Records every request."""

    def __init__(self, *replies: Reply, default: Optional[Reply] = None) -> None:
        self.replies = list(replies)
        self.default = default if default is not None else reply()
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answered = self.replies.pop(0) if self.replies else self.default
        if isinstance(answered, BaseException):
            raise answered
        if isinstance(answered, httpx.Response):
            return answered
        return answered(request)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @property
    def bodies(self) -> List[Dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def user_messages(self) -> List[str]:
        return [
            body["messages"][-1]["content"]
            for body in self.bodies
            if body["max_tokens"] != 1
        ]


# --- Running the command --------------------------------------------------------------


def extract(
    tmp_path: Path,
    *extra: str,
    generator: Optional[Generator] = None,
    chunks_dir: Optional[Path] = None,
    extracted_dir: Optional[Path] = None,
    overrides: Optional[Path] = None,
    relations: Optional[Path] = None,
) -> int:
    """``companydocs-metadata extract`` with every folder and file under
    ``tmp_path``."""
    from companydocs_rag.metadata import cli

    return cli.main(
        [
            "extract",
            "--chunks-dir",
            str(chunks_dir or tmp_path / "chunks"),
            "--extracted-dir",
            str(extracted_dir or tmp_path / "extracted"),
            "--type-overrides",
            str(overrides or tmp_path / "handmatig" / "types.csv"),
            "--relations",
            str(relations or tmp_path / "handmatig" / "relaties.csv"),
            *extra,
        ],
        transport=(generator if generator is not None else Generator()).transport,
    )


def report(
    tmp_path: Path,
    reference: str = "2031-09-01",
    *,
    extracted_dir: Optional[Path] = None,
    report_dir: Optional[Path] = None,
) -> int:
    """``companydocs-metadata report`` with both folders under ``tmp_path``."""
    from companydocs_rag.metadata import cli

    return cli.main(
        [
            "report",
            "--reference-date",
            reference,
            "--extracted-dir",
            str(extracted_dir or tmp_path / "extracted"),
            "--report-dir",
            str(report_dir or tmp_path / "rapport"),
        ]
    )


def extraction_files(folder: Path) -> Dict[str, Dict[str, Any]]:
    """Every extraction file under ``folder``, by forward-slash relative
    path."""
    return {
        path.relative_to(folder).as_posix(): json.loads(
            path.read_text(encoding="utf-8")
        )
        for path in sorted(folder.rglob("*.json"))
    }


def log_records(folder: Path) -> List[Dict[str, Any]]:
    """Every record of the one log file in ``folder``."""
    (path,) = sorted(folder.glob("extract-*.jsonl"))
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").splitlines()]


def messages(caplog: Any, level: int) -> List[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == level]


# --- Extraction files, for the report -------------------------------------------------


def spans(page_start: int, page_end: Optional[int] = None, *, origin: str = "ocr"):
    end = page_start if page_end is None else page_end
    return [span(page, origin=origin) for page in range(page_start, end + 1)]


def counted(
    value: Any,
    kind: str,
    *,
    pages: Sequence[int] = (2, 2),
    quote: str = "fictief citaat",
    origin: str = "ocr",
    batch_index: int = 1,
) -> Dict[str, Any]:
    """A counted candidate as ``extract`` stores it."""
    return {
        "value": value,
        "value_kind": kind,
        "as_written": "fictief",
        "quote": quote,
        "quote_verified": True,
        "value_valid": True,
        "counted": True,
        "batch_index": batch_index,
        "chunk_id": "0" * 20,
        "page_spans": spans(pages[0], pages[1], origin=origin),
    }


def uncounted(value: Any, kind: str, *, quote: str = "elders") -> Dict[str, Any]:
    """A candidate whose quote did not verify."""
    return {
        **counted(value, kind, quote=quote),
        "quote_verified": False,
        "counted": False,
        "chunk_id": None,
        "page_spans": [],
    }


def field(status: str, *candidates: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": status, "candidates": list(candidates)}


def found(value: Any, kind: str, **kwargs: Any) -> Dict[str, Any]:
    return field("found", counted(value, kind, **kwargs))


def not_found() -> Dict[str, Any]:
    return field("not_found")


def provision_item(
    party: str,
    quote: str,
    *,
    pages: Sequence[int] = (3, 3),
    verified: bool = True,
    origin: str = "ocr",
) -> Dict[str, Any]:
    return {
        "party": party,
        "as_written": "fictief",
        "quote": quote,
        "quote_verified": verified,
        "batch_index": 1,
        "chunk_id": "0" * 20 if verified else None,
        "page_spans": spans(pages[0], pages[1], origin=origin) if verified else [],
    }


def batch_entries(*statuses: str) -> List[Dict[str, Any]]:
    return [
        {
            "batch_index": position,
            "chunk_ids": ["0" * 20],
            "chars": 100,
            "status": status,
            "error_type": None if status == "ok" else "invalid_json",
            "prompt_tokens": 50,
            "completion_tokens": 12,
            "finish_reason": "stop",
            "duration_s": 1.5,
        }
        for position, status in enumerate(statuses, start=1)
    ]


def extraction_record(
    path: str,
    sha256: Optional[str],
    *,
    document_type: Optional[str] = LEASE,
    status: str = "extracted",
    fields: Optional[Mapping[str, Dict[str, Any]]] = None,
    provisions: Sequence[Dict[str, Any]] = (),
    relations: Sequence[Dict[str, Any]] = (),
    batches: Optional[List[Dict[str, Any]]] = None,
    language: Optional[str] = "nld",
) -> Dict[str, Any]:
    """An extraction file with D10's keys; every field ``not_found`` unless
    given."""
    values = {name: not_found() for name in RESPONSE_KEYS[:-1]}
    values.update(fields or {})
    return {
        "schema_version": 1,
        "extractor_version": 1,
        "file": {"path": path, "sha256": sha256},
        "language": language,
        "language_source": "index",
        "index_type": document_type,
        "document_type": document_type,
        "document_type_source": "index",
        "relations": list(relations),
        "extraction_status": status,
        "model": "companydocs-gemma4-e4b-8k",
        "reasoning_effort": "high",
        "batches": batch_entries("ok") if batches is None else batches,
        "fields": values,
        "other_termination_provisions": list(provisions),
    }


def relation(
    kind: str, target: Optional[str], detail: str = "", *, resolved: bool = True
) -> Dict[str, Any]:
    return {
        "relation": kind,
        "target_file_sha256": target if resolved else None,
        "detail": detail,
        "status": "resolved" if resolved else "target_not_found",
    }
