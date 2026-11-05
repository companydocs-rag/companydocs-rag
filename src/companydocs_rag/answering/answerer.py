"""The answer layer's API (spec 007): :class:`Answerer` and :class:`AnswerResult`.

Opening (R2), in this order: the resolved settings (``answering``, which
resolves ``retrieval`` too); the prompt file and its sha256; the API key
variable, when one is configured; the Retriever, which takes the index lock
and can raise :class:`companydocs_rag.retrieval.search.IndexUnusableError`;
the sha256 of ``manifest.json``; the httpx client; and, when ``warmup`` is
true, one request with the user message "ok" and ``max_tokens`` 1, whose
failure raises :class:`GeneratorUnavailableError` or :class:`GeneratorError`
from the constructor. A failure while opening closes the Retriever and the
client first; an injected Retriever belongs to the answerer from the start,
so a failure at the settings, the prompt file or the API key closes it too.
Settings are resolved once, at open.

Per question (:meth:`Answerer.answer`, R11), in this order: retrieval;
zero candidates (abstain, ``no_candidates``, no request); the score gate
(``min_dense_score`` against the dense leg's rank-1 score, ``score_gate``, no
request); selection and the budget (nothing fits: status ``error``,
``context_budget_exceeded``, no request); the prompt; the request; the
truncation tripwire (characters of the system and user message per
``usage.prompt_tokens`` above ``max_chars_per_prompt_token``: ``error``,
``prompt_truncation_suspected``); empty content; the marker; the citations.
``truncated`` (``finish_reason`` length) and ``context_nearly_full``
(``prompt_tokens + max_tokens > context_window``) are computed for every
response. Every answer is written to the answerer's log (:mod:`.log`) before
:meth:`Answerer.answer` returns. An unavailable generator writes a record in
status ``error`` (``generator_unavailable``) and then raises
:class:`GeneratorUnavailableError`; any other failed request is a record in
status ``error`` with the failure's error type.

Records without a request (zero candidates, the score gate, the budget) have
``prompt``, ``generation`` and ``answer_text`` null, ``sources`` [] and no
citations. After a failed request, ``prompt`` holds what was sent and
``generation`` only ``model``, ``base_url`` and ``generation_ms``.
"""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import httpx

from companydocs_rag.retrieval.search import Retriever, SearchResult
from companydocs_rag.vectorstore.manifest import MANIFEST_FILE_NAME

from .generator import Completion, GeneratorClient, RequestFailure, read_api_key
from .log import AnswerLog, format_timestamp
from .parsing import (
    ABSTENTION_NO_CANDIDATES,
    ABSTENTION_SCORE_GATE,
    STATUS_ANSWERED,
    STATUS_ERROR,
    STATUS_INSUFFICIENT_EVIDENCE,
    parse_answer,
)
from .prompt import (
    Budgeted,
    PromptBlock,
    fit_budget,
    load_system_prompt,
    select_fused,
    select_per_path,
    user_message,
)
from .provenance import source_entry
from .settings import (
    SELECTION_FUSED,
    SELECTION_MODES,
    SELECTION_PER_PATH,
    AnsweringSettings,
    answering_settings,
    resolve_project_path,
    settings_snapshot,
)

SCHEMA_VERSION = 1
ERROR_PROMPT_TRUNCATION_SUSPECTED = "prompt_truncation_suspected"
ERROR_CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"
FINISH_REASON_LENGTH = "length"
WARMUP_MESSAGE = "ok"
WARMUP_MAX_TOKENS = 1

_ROLE_SYSTEM = "system"
_ROLE_USER = "user"


class GeneratorUnavailableError(Exception):
    """The generator cannot be reached: a refused or unresolvable connection,
    a connect timeout, or HTTP 404 (an unknown model). Raised from the
    constructor at warm-up, and from :meth:`Answerer.answer` after the
    question's record was written; ``result`` is the :class:`AnswerResult`
    whose record was written before the raise, None at warm-up."""

    def __init__(self, cause: str, *, result: Optional["AnswerResult"] = None) -> None:
        super().__init__(cause)
        self.cause = cause
        self.result = result


class GeneratorError(Exception):
    """Any other failed request: another HTTP status outside 2xx, a read
    timeout, another httpx transport error, or a malformed response
    (``error_type``). Raised from the constructor at warm-up only; during
    :meth:`Answerer.answer` it becomes a record in status ``error``."""

    def __init__(self, cause: str, *, error_type: str) -> None:
        super().__init__(cause)
        self.cause = cause
        self.error_type = error_type


@dataclass(frozen=True)
class AnswerResult:
    """One answer: one field per key of the log record, in D10's order.
    :meth:`to_dict` is the record."""

    schema_version: int
    timestamp_utc: str
    question_id: Optional[str]
    question: str
    status: str
    abstention_source: Optional[str]
    answer_text: Optional[str]
    sources: List[Dict[str, Any]]
    citations: Dict[str, List[int]]
    context: List[Dict[str, Any]]
    dropped_for_budget: List[str]
    candidates: List[Dict[str, Any]]
    path_hits: Optional[Dict[str, List[Dict[str, Any]]]]
    prompt: Optional[Dict[str, Any]]
    generation: Optional[Dict[str, Any]]
    retrieval_ms: float
    settings: Dict[str, Any]
    index: Dict[str, Any]
    error: Optional[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        """The log record (D10): a deep copy, keys in D10's order."""
        return copy.deepcopy(
            {field.name: getattr(self, field.name) for field in fields(self)}
        )


def _utc_now() -> datetime:
    """The current UTC time; tests fix it."""
    return datetime.now(timezone.utc)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _public_error(failure: RequestFailure) -> Exception:
    if failure.unavailable:
        return GeneratorUnavailableError(failure.cause)
    return GeneratorError(failure.cause, error_type=failure.error_type)


def _no_citations() -> Dict[str, List[int]]:
    return {"valid": [], "invalid": []}


def _candidate_entry(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(candidate[key])
        for key in (
            "rank",
            "chunk_id",
            "rrf_score",
            "dense_rank",
            "dense_score",
            "sparse_rank",
            "sparse_score",
            "file_sha256",
            "page_start",
            "page_end",
        )
    }


def _path_hits_entry(
    path_hits: Mapping[str, List[Mapping[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        path: [
            {
                "rank": entry["rank"],
                "chunk_id": entry["chunk_id"],
                "score": entry["score"],
            }
            for entry in path_hits[path]
        ]
        for path in ("dense", "sparse")
    }


def _context_entry(block: PromptBlock) -> Dict[str, Any]:
    selected = block.selected
    return {
        "prompt_number": block.prompt_number,
        "document_label": block.document_label,
        "chunk_id": selected.chunk_id,
        "file_sha256": selected.chunk["file_sha256"],
        "page_start": selected.chunk["page_start"],
        "page_end": selected.chunk["page_end"],
        "rank": selected.rank,
        "dense_rank": selected.dense_rank,
        "sparse_rank": selected.sparse_rank,
        "selected_by": selected.selected_by,
    }


def _dense_rank_one_score(
    path_hits: Optional[Mapping[str, List[Mapping[str, Any]]]],
) -> Optional[float]:
    dense = (path_hits or {}).get("dense") or []
    if not dense:
        return None
    return min(dense, key=lambda entry: entry["rank"])["score"]


class Answerer:
    """Answers questions over one index (D1, R1-R3).

    ``settings`` replaces ``config/settings.yaml``, as for
    :class:`~companydocs_rag.retrieval.search.Retriever`. ``retriever``, when
    given, is used instead of ``Retriever(index_dir, settings=settings)``; it
    needs ``search(query, *, include_path_hits=False)`` and ``close()``, and
    the answerer closes it, also when opening fails. ``index_dir`` is read
    for the sha256 of ``manifest.json`` either way. ``transport`` goes to the
    httpx client. Use it as a context manager, or call :meth:`close`, to
    release the index and the client.
    """

    def __init__(
        self,
        index_dir: Path,
        *,
        settings: Optional[Mapping[str, Any]] = None,
        retriever: Any = None,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self._started = _utc_now()
        self._closed = False
        # An injected Retriever belongs to the answerer from the start, so a
        # failure at any step of the opening closes it (R2).
        self._retriever: Any = retriever
        self._client: Optional[GeneratorClient] = None
        self._log: Optional[AnswerLog] = None
        index_dir = Path(index_dir)

        try:
            config = answering_settings(settings)
            self._system_prompt = load_system_prompt(config)
            api_key = read_api_key(config)
            self._config: AnsweringSettings = config
            if self._retriever is None:
                self._retriever = Retriever(index_dir, settings=settings)
            manifest = (index_dir / MANIFEST_FILE_NAME).read_bytes()
            self._manifest_sha256 = hashlib.sha256(manifest).hexdigest()
            self._client = GeneratorClient(config, api_key=api_key, transport=transport)
            if config.warmup:
                self._warm_up()
        except BaseException:
            self.close()
            raise
        self._log = AnswerLog(resolve_project_path(config.log_dir), self._started)

    def __enter__(self) -> "Answerer":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def log_path(self) -> Optional[Path]:
        """The log file, once the first record is written; else None."""
        return None if self._log is None else self._log.path

    def close(self) -> None:
        """Close the Retriever, injected or not, the client and the log;
        closing twice is harmless."""
        self._closed = True
        retriever, self._retriever = self._retriever, None
        client, self._client = self._client, None
        try:
            if retriever is not None:
                retriever.close()
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                if self._log is not None:
                    self._log.close()

    def _warm_up(self) -> None:
        try:
            self._client.complete(
                [{"role": _ROLE_USER, "content": WARMUP_MESSAGE}],
                max_tokens=WARMUP_MAX_TOKENS,
            )
        except RequestFailure as failure:
            raise _public_error(failure) from failure

    def answer(
        self,
        question: str,
        *,
        selection_mode: Optional[str] = None,
        question_id: Optional[str] = None,
    ) -> AnswerResult:
        """Answer ``question``, used as given (R9), and log the record.

        Raises ``ValueError("Empty question")`` for an empty or
        whitespace-only question, ``ValueError("The answerer is closed")``
        after :meth:`close`, and ``ValueError`` for a ``selection_mode``
        other than None, ``fused`` and ``per_path``. Raises
        :class:`GeneratorUnavailableError` after writing the question's
        record when the generator cannot be reached (R3)."""
        if not question.strip():
            raise ValueError("Empty question")
        if self._closed:
            raise ValueError("The answerer is closed")
        if selection_mode is not None and selection_mode not in SELECTION_MODES:
            raise ValueError(
                "selection_mode must be None, fused or per_path, "
                f"not {selection_mode!r}"
            )
        mode = self._config.selection_mode if selection_mode is None else selection_mode
        timestamp = format_timestamp(_utc_now())
        config = self._config

        include_path_hits = (
            mode == SELECTION_PER_PATH or config.min_dense_score is not None
        )
        started = time.perf_counter()
        result: SearchResult = self._retriever.search(
            question, include_path_hits=include_path_hits
        )
        retrieval_ms = _elapsed_ms(started)

        def record(**outcome: Any) -> AnswerResult:
            return AnswerResult(
                schema_version=SCHEMA_VERSION,
                timestamp_utc=timestamp,
                question_id=question_id,
                question=question,
                candidates=[_candidate_entry(c) for c in result.candidates],
                path_hits=(
                    _path_hits_entry(result.path_hits)
                    if mode == SELECTION_PER_PATH
                    else None
                ),
                retrieval_ms=retrieval_ms,
                settings={
                    "answering": settings_snapshot(config, selection_mode=mode),
                    "retrieval": copy.deepcopy(result.settings),
                },
                index={
                    **copy.deepcopy(result.index),
                    "manifest_sha256": self._manifest_sha256,
                },
                **outcome,
            )

        without_request = {
            "answer_text": None,
            "sources": [],
            "citations": _no_citations(),
            "prompt": None,
            "generation": None,
        }
        if not result.candidates:
            return self._logged(
                record(
                    status=STATUS_INSUFFICIENT_EVIDENCE,
                    abstention_source=ABSTENTION_NO_CANDIDATES,
                    context=[],
                    dropped_for_budget=[],
                    error=None,
                    **without_request,
                )
            )

        if config.min_dense_score is not None:
            score = _dense_rank_one_score(result.path_hits)
            if score is None or score < config.min_dense_score:
                return self._logged(
                    record(
                        status=STATUS_INSUFFICIENT_EVIDENCE,
                        abstention_source=ABSTENTION_SCORE_GATE,
                        context=[],
                        dropped_for_budget=[],
                        error=None,
                        **without_request,
                    )
                )

        if mode == SELECTION_FUSED:
            selection = select_fused(result.candidates, config.fused_k)
        else:
            selection = select_per_path(
                result.path_hits, result.candidates, config.per_path_k
            )
        budgeted: Budgeted = fit_budget(selection, config.max_context_chars)
        if not budgeted.blocks:
            return self._logged(
                record(
                    status=STATUS_ERROR,
                    abstention_source=None,
                    context=[],
                    dropped_for_budget=budgeted.dropped,
                    error={
                        "type": ERROR_CONTEXT_BUDGET_EXCEEDED,
                        "message": (
                            f"the first source block has {budgeted.rejected_chars} "
                            f"characters, more than max_context_chars "
                            f"({config.max_context_chars})"
                        ),
                    },
                    **without_request,
                )
            )

        system = self._system_prompt.text
        message = user_message(budgeted.blocks, question)
        prompt = {
            "system_file": config.prompt_file,
            "system_sha256": self._system_prompt.sha256,
            "user_message": message,
        }
        sent = {
            "context": [_context_entry(block) for block in budgeted.blocks],
            "dropped_for_budget": budgeted.dropped,
            "prompt": prompt,
        }

        started = time.perf_counter()
        try:
            completion = self._client.complete(
                [
                    {"role": _ROLE_SYSTEM, "content": system},
                    {"role": _ROLE_USER, "content": message},
                ],
                max_tokens=config.max_tokens,
            )
        except RequestFailure as failure:
            answer = self._logged(
                record(
                    status=STATUS_ERROR,
                    abstention_source=None,
                    answer_text=None,
                    sources=[],
                    citations=_no_citations(),
                    generation=self._generation(None, _elapsed_ms(started)),
                    error={"type": failure.error_type, "message": failure.cause},
                    **sent,
                )
            )
            if failure.unavailable:
                raise GeneratorUnavailableError(
                    failure.cause, result=answer
                ) from failure
            return answer
        generation = self._generation(completion, _elapsed_ms(started))

        content = completion.content
        parsed = parse_answer(content, len(budgeted.blocks))
        characters = len(system) + len(message)
        ratio = characters / completion.prompt_tokens
        if ratio > config.max_chars_per_prompt_token:
            status, abstention_source = STATUS_ERROR, None
            error: Optional[Dict[str, str]] = {
                "type": ERROR_PROMPT_TRUNCATION_SUSPECTED,
                "message": (
                    f"{characters} characters for {completion.prompt_tokens} "
                    f"prompt tokens is {ratio} per token, above "
                    f"max_chars_per_prompt_token ({config.max_chars_per_prompt_token})"
                ),
            }
        else:
            status, abstention_source = parsed.status, parsed.abstention_source
            error = (
                None
                if parsed.error_type is None
                else {
                    "type": parsed.error_type,
                    "message": "the content is null"
                    if content is None
                    else "the content is empty or whitespace only",
                }
            )
        blocks = budgeted.blocks
        return self._logged(
            record(
                status=status,
                abstention_source=abstention_source,
                answer_text=None if content is None else content.strip(),
                sources=(
                    [
                        source_entry(number, blocks[number - 1].selected.chunk)
                        for number in parsed.valid
                    ]
                    if status == STATUS_ANSWERED
                    else []
                ),
                citations={"valid": parsed.valid, "invalid": parsed.invalid},
                generation=generation,
                error=error,
                **sent,
            )
        )

    def _generation(
        self, completion: Optional[Completion], generation_ms: float
    ) -> Dict[str, Any]:
        """The record's ``generation``; after a failed request only
        ``model``, ``base_url`` and ``generation_ms`` are set (R11)."""
        config = self._config
        if completion is None:
            return {
                "model": config.model,
                "base_url": config.base_url,
                "finish_reason": None,
                "truncated": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "cached_tokens": None,
                "context_nearly_full": None,
                "reasoning": None,
                "generation_ms": generation_ms,
            }
        return {
            "model": config.model,
            "base_url": config.base_url,
            "finish_reason": completion.finish_reason,
            "truncated": completion.finish_reason == FINISH_REASON_LENGTH,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "cached_tokens": completion.cached_tokens,
            "context_nearly_full": (
                completion.prompt_tokens + config.max_tokens > config.context_window
            ),
            "reasoning": completion.reasoning,
            "generation_ms": generation_ms,
        }

    def _logged(self, answer: AnswerResult) -> AnswerResult:
        """Write ``answer`` to the log, then return it."""
        self._log.write(answer.to_dict())
        return answer
