"""Shared helpers for the spec 007 tests: the fake Retriever, fabricated
search results, a fabricated ``manifest.json``, a recording MockTransport
handler and settings writers.

No test reaches a server or loads the embedding model (Constraints).
:class:`FakeRetriever` stands in for ``Retriever``: it returns fabricated
:class:`SearchResult` objects, built the way ``Retriever.search`` builds them
(:func:`search_result`, with the real fusion), and records its calls.
:class:`Generator` is the handler of an ``httpx.MockTransport``: it records
every request and answers from a script. Every path a test writes is under
``tmp_path``; the default ``log_dir`` of these tests is set there by
``conftest.py``. Text is fabricated, never taken from a contract.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import httpx

from companydocs_rag.retrieval.fusion import fuse, rank_leg
from companydocs_rag.retrieval.search import SearchResult
from companydocs_rag.vectorstore.store import PAYLOAD_KEYS
from tests.embeddings._helpers import write_settings

INDEX = {
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "model_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "embedding_text_version": 1,
    "tokenizer_version": 1,
}
QUERY_INSTRUCTION = "Instruct: fictief\nQuery:"
SHA_A = "a1" * 32
SHA_B = "b2" * 32
SHA_C = "c3" * 32


# --- Fabricated chunks and search results ---------------------------------------------


def span(
    page: int,
    *,
    origin: str = "native",
    tier: Optional[str] = "Clean",
    confidence: Optional[float] = None,
    length: int = 10,
) -> Dict[str, Any]:
    """One ``page_spans`` entry, as the chunker writes it."""
    return {
        "page_number": page,
        "char_start": 0,
        "char_end": length,
        "text_origin": origin,
        "tier": tier,
        "ocr_mean_confidence": confidence,
    }


def chunk(
    chunk_id: str,
    text: str,
    *,
    file_sha256: str = SHA_A,
    file_path: str = "invoer/fictief-huur.pdf",
    language: Optional[str] = "nld",
    page_start: int = 1,
    page_end: Optional[int] = None,
    page_spans: Optional[List[Dict[str, Any]]] = None,
    clause_label: Optional[str] = None,
) -> Dict[str, Any]:
    """A fabricated payload: ``chunk_id`` and the payload keys in D13 order."""
    end = page_start if page_end is None else page_end
    spans = (
        page_spans
        if page_spans is not None
        else [span(page) for page in range(page_start, end + 1)]
    )
    origins = sorted({entry["text_origin"] for entry in spans})
    labelled = clause_label is not None
    values = {
        "chunk_id": chunk_id,
        "chunk_index": 0,
        "file_sha256": file_sha256,
        "file_path": file_path,
        "language": language,
        "language_source": "index",
        "route": "structured" if labelled else "generic",
        "text": text,
        "page_start": page_start,
        "page_end": end,
        "text_origins": origins,
        "page_spans": spans,
        "chunk_method": "heading_section" if labelled else "generic",
        "clause_label": clause_label,
        "clause_number": "1" if labelled else None,
        "part": None,
        "part_count": None,
    }
    assert list(values) == list(PAYLOAD_KEYS)
    return values


def numbered_chunks(count: int, *, prefix: str = "c", **kwargs: Any) -> List[Dict]:
    """``count`` fabricated chunks with ids ``c01``, ``c02``, ..."""
    return [
        chunk(
            f"{prefix}{number:02d}",
            f"Fictieve bepaling {number} over de huur.",
            **kwargs,
        )
        for number in range(1, count + 1)
    ]


Leg = Sequence[Tuple[Mapping[str, Any], float]]


def search_result(
    dense: Leg = (),
    sparse: Leg = (),
    *,
    query: str = "Wat is de fictieve huurprijs?",
    rrf_k: int = 60,
    dense_limit: int = 50,
    sparse_limit: int = 50,
    candidate_count: int = 20,
) -> SearchResult:
    """A result as ``Retriever.search(..., include_path_hits=True)`` builds
    it, from fabricated legs: (payload, score) pairs."""
    payloads = {item["chunk_id"]: item for item, _ in [*dense, *sparse]}
    candidates = []
    for fused in fuse(
        [(item["chunk_id"], score) for item, score in dense],
        [(item["chunk_id"], score) for item, score in sparse],
        rrf_k=rrf_k,
        candidate_count=candidate_count,
    ):
        candidate = {
            "rank": fused.rank,
            "chunk_id": fused.chunk_id,
            "rrf_score": fused.rrf_score,
            "dense_rank": fused.dense_rank,
            "dense_score": fused.dense_score,
            "sparse_rank": fused.sparse_rank,
            "sparse_score": fused.sparse_score,
            "sparse_matched_terms": [],
        }
        for key in PAYLOAD_KEYS[1:]:
            candidate[key] = payloads[fused.chunk_id][key]
        candidates.append(candidate)

    def path_list(leg: Leg) -> List[Dict[str, Any]]:
        ranked = rank_leg([(item["chunk_id"], score) for item, score in leg])
        entries = []
        for chunk_id, (rank, score) in sorted(ranked.items(), key=lambda i: i[1][0]):
            entry = {"rank": rank, "chunk_id": chunk_id, "score": score}
            for key in PAYLOAD_KEYS[1:]:
                entry[key] = payloads[chunk_id][key]
            entries.append(entry)
        return entries

    return SearchResult(
        query=query,
        query_terms=["fictief"],
        query_instruction=QUERY_INSTRUCTION,
        index=dict(INDEX),
        settings={
            "rrf_k": rrf_k,
            "dense_limit": dense_limit,
            "sparse_limit": sparse_limit,
            "candidate_count": candidate_count,
        },
        leg_counts={"dense": len(dense), "sparse": len(sparse)},
        candidates=candidates,
        path_hits={"dense": path_list(dense), "sparse": path_list(sparse)},
    )


def simple_result(count: int = 3, **kwargs: Any) -> SearchResult:
    """``count`` chunks that both legs return, in the same order."""
    chunks = numbered_chunks(count)
    return search_result(
        dense=[(item, 0.9 - position / 100) for position, item in enumerate(chunks)],
        sparse=[(item, 9.0 - position) for position, item in enumerate(chunks)],
        **kwargs,
    )


class FakeRetriever:
    """Stands in for ``Retriever``: ``search`` returns ``result`` (or what
    ``result(query)`` returns), with ``path_hits`` only when asked for, as
    the real one does. It records every call and every ``close``."""

    def __init__(
        self, result: Union[SearchResult, Callable[[str], SearchResult]]
    ) -> None:
        self.result = result
        self.calls: List[Tuple[str, bool]] = []
        self.closed = 0

    def search(self, query: str, *, include_path_hits: bool = False) -> SearchResult:
        self.calls.append((query, include_path_hits))
        found = self.result(query) if callable(self.result) else self.result
        found = dataclasses.replace(found, query=query)
        return (
            found if include_path_hits else dataclasses.replace(found, path_hits=None)
        )

    def close(self) -> None:
        self.closed += 1


def write_manifest(index_dir: Path) -> bytes:
    """A fabricated ``manifest.json`` in ``index_dir``; returns its bytes."""
    index_dir.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        {"manifest_version": 1, "fictief": "manifest voor tests"}, indent=2
    ).encode("utf-8")
    (index_dir / "manifest.json").write_bytes(raw)
    return raw


# --- The generator: a recording MockTransport handler ---------------------------------

Reply = Union[httpx.Response, BaseException, Callable[[httpx.Request], httpx.Response]]


def prompt_characters(body: Mapping[str, Any]) -> int:
    return sum(len(message["content"]) for message in body["messages"])


def completion_body(
    content: Optional[str],
    *,
    prompt_tokens: int,
    finish_reason: Optional[str] = "stop",
    completion_tokens: Optional[int] = 12,
    cached_tokens: Optional[int] = None,
    reasoning: Optional[str] = None,
) -> Dict[str, Any]:
    """A chat-completions response body as Ollama's ``/v1`` writes it."""
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    usage: Dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + (completion_tokens or 0),
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return {
        "id": "chatcmpl-fictief",
        "object": "chat.completion",
        "model": "fictief-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


def reply(
    content: Optional[str],
    *,
    prompt_tokens: Optional[int] = None,
    **fields: Any,
) -> Callable[[httpx.Request], httpx.Response]:
    """A 200 response with ``content``. Unless given, ``prompt_tokens`` is
    the request's characters divided by 3, rounded up: well below the
    tripwire."""

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        tokens = (
            prompt_tokens
            if prompt_tokens is not None
            else math.ceil(prompt_characters(body) / 3)
        )
        return httpx.Response(
            200, json=completion_body(content, prompt_tokens=tokens, **fields)
        )

    return respond


class Generator:
    """The handler of an ``httpx.MockTransport``. Each request takes the next
    reply: a response, an exception to raise, or a function of the request;
    when the script runs out, ``default`` answers. Records every request."""

    def __init__(self, *replies: Reply, default: Optional[Reply] = None) -> None:
        self.replies = list(replies)
        self.default = default if default is not None else reply("Fictief [1].")
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.replies.pop(0) if self.replies else self.default
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        return answer(request)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @property
    def bodies(self) -> List[Dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


def _usage(prompt_tokens: Any = 10) -> Dict[str, Any]:
    return {"prompt_tokens": prompt_tokens, "completion_tokens": 1}


def _message(content: Any = "ok") -> Dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def json_response(body: Any) -> Callable[[], httpx.Response]:
    """A factory of 200 responses whose body is ``body`` as JSON."""
    return lambda: httpx.Response(200, content=json.dumps(body).encode("utf-8"))


# Each malformed response of R4, as a factory of fresh responses.
MALFORMED = {
    "not-json": lambda: httpx.Response(200, content=b"geen json {"),
    "not-an-object": json_response([1, 2]),
    "choices-missing": json_response({"usage": _usage()}),
    "choices-empty": json_response({"choices": [], "usage": _usage()}),
    "choices-not-a-list": json_response(
        {"choices": {"message": {}}, "usage": _usage()}
    ),
    "message-missing": json_response({"choices": [{"index": 0}], "usage": _usage()}),
    "message-not-an-object": json_response(
        {"choices": [{"message": "ok"}], "usage": _usage()}
    ),
    "choice-not-an-object": json_response({"choices": ["ok"], "usage": _usage()}),
    "content-a-number": json_response({**_message(5), "usage": _usage()}),
    "content-a-list": json_response({**_message(["ok"]), "usage": _usage()}),
    "usage-missing": json_response(_message()),
    "usage-not-an-object": json_response({**_message(), "usage": [10]}),
    "prompt_tokens-missing": json_response(
        {**_message(), "usage": {"completion_tokens": 1}}
    ),
    "prompt_tokens-0": json_response({**_message(), "usage": _usage(0)}),
    "prompt_tokens-a-float": json_response({**_message(), "usage": _usage(10.0)}),
    "prompt_tokens-a-string": json_response({**_message(), "usage": _usage("10")}),
    "prompt_tokens-a-bool": json_response({**_message(), "usage": _usage(True)}),
}

# --- Settings -------------------------------------------------------------------------


def settings_for(tmp_path: Path, **answering: Any) -> Dict[str, Any]:
    """A ``settings`` mapping: no warm-up and the log under ``tmp_path``,
    unless overridden."""
    values: Dict[str, Any] = {
        "warmup": False,
        "log_dir": str(tmp_path / "logs"),
    }
    values.update(answering)
    return {"answering": values}


def use_settings(tmp_path: Path, monkeypatch: Any, settings: Mapping[str, Any]) -> Path:
    """Point ``COMPANYDOCS_RAG_SETTINGS`` at a file holding ``settings``."""
    return write_settings(tmp_path, monkeypatch, settings)


def log_records(folder: Path) -> List[Dict[str, Any]]:
    """Every record of the one log file in ``folder``."""
    (path,) = sorted(folder.glob("answers-*.jsonl"))
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").splitlines()]
