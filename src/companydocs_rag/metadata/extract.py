"""The ``extract`` subcommand (spec 008, D3-D11, R5, R12-R18).

In this order (R18):

1. **Start.** The ``metadata`` settings, the prompt file, the API key
   variable, the type override file and the relations file. The first
   problem is one ERROR and exit 2, before ``--chunks-dir`` is looked at.
2. **Input.** ``--chunks-dir`` must be a folder (else exit 1), and every
   chunk file is read and checked before any document is processed
   (:mod:`.chunks`); any refused file is exit 1, with nothing written and no
   request. An empty folder is a WARNING and exit 0.
3. **Types and relations** of every chunk file (:mod:`.doctype`,
   :mod:`.relations`).
4. **Per chunk file**, in input order: out of scope (its type is not in
   ``lease_family_types``), a ``not_in_scope`` record; in scope with a chunk
   record whose ``status`` is ``failed``, an ``error`` record; otherwise the
   selection (:mod:`.selection`) and one request per batch, each with its
   checks (:mod:`.checks`) and one log record (:mod:`.log`). Then the field
   statuses, the ``extraction_status`` and the record, written at once under
   ``--extracted-dir`` at the chunk file's relative path, through a
   temporary file (:func:`.files.replace_file`). A document none of whose
   chunks scores above 0 is ``extracted`` without a request, with a WARNING.
   An unexpected exception while a document is processed or written is an
   ERROR; that document gets no new file (an older one stays as it was), the
   run goes on and ends with exit 4.
5. **The generator unreachable**, at the warm-up or at a request: the run
   stops with exit 3; the records written so far stay, the document in
   progress gets none.
6. **End.** Two summary lines, the log's path, then exit 4 when a document
   is ``partial`` or ``error``, else 0.

One client per run (spec 007's ``GeneratorClient``, unchanged), created
directly before the run's first request and closed when the run ends. When
``warmup`` is true, one request with the user message "ok" and
``max_tokens`` 1 goes through it directly before the run's first extraction
request; it is checked as every response is, writes no log record, and any
failure is exit 3. A run without extraction requests sends none.

This is the only module of the package that reads a clock: the wall clock
for the log's name and each record's ``timestamp_utc``, a monotonic clock for
``duration_s`` (Q2).
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from companydocs_rag.answering.generator import (
    Completion,
    GeneratorClient,
    RequestFailure,
    request_body,
)
from companydocs_rag.embeddings.dense import SettingsError
from companydocs_rag.ingestion.language import language_settings
from companydocs_rag.ingestion.parse_pass import STATUS_FAILED

from .checks import (
    InvalidContent,
    SchemaMismatch,
    batch_items,
    check_structure,
    parse_content,
)
from .chunks import ChunkFile, read_chunk_files
from .doctype import (
    ManualFileError,
    TypeResolution,
    TypeResolver,
    read_type_overrides,
)
from .fields import (
    BATCH_ERROR,
    BATCH_OK,
    EXTRACTION_ERROR,
    EXTRACTION_EXTRACTED,
    EXTRACTION_NOT_IN_SCOPE,
    EXTRACTION_PARTIAL,
    EXTRACTION_STATUSES,
    FIELDS,
    field_status,
)
from .files import file_name, write_json
from .log import ExtractLog, LogWriteError, format_timestamp
from .relations import read_relations, resolve_relations
from .request import (
    GeneratorConfig,
    SystemPrompt,
    generator_config,
    load_system_prompt,
    messages,
    read_api_key,
    user_message,
)
from .selection import batch_chars, plan_batches, scores_of, vocabulary
from .settings import MetadataSettings, metadata_settings, resolve_project_path

logger = logging.getLogger(__name__)

EXIT_DONE = 0
EXIT_INPUT_REFUSED = 1
EXIT_CANNOT_START = 2
EXIT_GENERATOR_UNAVAILABLE = 3
EXIT_DOCUMENT_ERRORS = 4

# The extraction file (D10): bumped on any change to its keys or their meaning,
# and on any change to what extract decides (selection, prompt, schema, checks).
SCHEMA_VERSION = 1
EXTRACTOR_VERSION = 1
LOG_SCHEMA_VERSION = 1

# D9's batch error types besides the GeneratorError types, and R15's.
ERROR_GENERATION_TRUNCATED = "generation_truncated"
ERROR_PROMPT_TRUNCATION_SUSPECTED = "prompt_truncation_suspected"
ERROR_INVALID_JSON = "invalid_json"
ERROR_SCHEMA_MISMATCH = "schema_mismatch"
ERROR_GENERATOR_UNAVAILABLE = "generator_unavailable"

FINISH_REASON_LENGTH = "length"
WARMUP_MESSAGE = "ok"
WARMUP_MAX_TOKENS = 1


def _utc_now() -> datetime:
    """The current UTC time; tests fix it."""
    return datetime.now(timezone.utc)


def _monotonic() -> float:
    return time.monotonic()


class GeneratorStopped(Exception):
    """The generator cannot be used: the run stops with exit 3. The text is
    the ERROR the run logs."""


@dataclass
class Totals:
    """What one run did, for its summary lines."""

    documents: Dict[str, int] = field(
        default_factory=lambda: {status: 0 for status in EXTRACTION_STATUSES}
    )
    # Documents that met an unexpected exception: no file, counted as error.
    failed: int = 0
    requests: int = 0
    batches_ok: int = 0
    batches_error: int = 0


@dataclass(frozen=True)
class BatchOutcome:
    """One batch: its entry in ``batches`` and, when ok, its items."""

    entry: Dict[str, Any]
    candidates: Dict[str, List[Dict[str, Any]]]
    provisions: List[Dict[str, Any]]

    @property
    def ok(self) -> bool:
        return self.entry["status"] == BATCH_OK


def _empty_fields() -> Dict[str, Dict[str, Any]]:
    return {name: {"status": field_status([]), "candidates": []} for name in FIELDS}


def _extraction_status(outcomes: Sequence[BatchOutcome]) -> str:
    """D9: ``extracted`` when every batch is ok (zero batches included),
    ``error`` when all failed, else ``partial``."""
    failed = sum(1 for outcome in outcomes if not outcome.ok)
    if failed == 0:
        return EXTRACTION_EXTRACTED
    if failed == len(outcomes):
        return EXTRACTION_ERROR
    return EXTRACTION_PARTIAL


class ExtractRun:
    """Step 4 to 6 of one run over the checked chunk files."""

    def __init__(
        self,
        config: MetadataSettings,
        prompt: SystemPrompt,
        api_key: Optional[str],
        extracted_dir: Path,
        started: datetime,
        transport: Optional[httpx.BaseTransport],
    ) -> None:
        self._config = config
        self._prompt = prompt
        self._api_key = api_key
        self._extracted_dir = extracted_dir
        self._transport = transport
        self._generator: GeneratorConfig = generator_config(config)
        self._client: Optional[GeneratorClient] = None
        self._warmed = False
        self._log = ExtractLog(resolve_project_path(config.log_dir), started)
        self.totals = Totals()

    # --- The run ----------------------------------------------------------------------

    def run(
        self,
        files: Sequence[ChunkFile],
        types: Sequence[TypeResolution],
        relations: Sequence[List[Dict[str, Any]]],
    ) -> int:
        stopped = False
        try:
            for position, (chunk_file, resolution, entries) in enumerate(
                zip(files, types, relations, strict=True), start=1
            ):
                label = f"[{position}/{len(files)}] {chunk_file.relative}"
                try:
                    record = self._document(label, chunk_file, resolution, entries)
                    write_json(self._extracted_dir / chunk_file.relative, record)
                except GeneratorStopped as stop:
                    logger.error("%s", stop)
                    stopped = True
                    break
                except LogWriteError:
                    raise
                except Exception as error:  # noqa: BLE001 - R14: the run goes on
                    logger.error(
                        "Could not extract %s: %s: %s",
                        chunk_file.relative,
                        type(error).__name__,
                        error,
                    )
                    self.totals.failed += 1
                    continue
                status = record["extraction_status"]
                self.totals.documents[status] += 1
                shown_type = record["document_type"]
                logger.info(
                    "%s: %s (%s)",
                    label,
                    "-" if shown_type is None else shown_type,
                    status,
                )
        finally:
            self.close()
        self._log_summary()
        if stopped:
            return EXIT_GENERATOR_UNAVAILABLE
        documents = self.totals.documents
        if (
            documents[EXTRACTION_PARTIAL]
            or documents[EXTRACTION_ERROR]
            or self.totals.failed
        ):
            return EXIT_DOCUMENT_ERRORS
        return EXIT_DONE

    def close(self) -> None:
        client, self._client = self._client, None
        try:
            if client is not None:
                client.close()
        finally:
            self._log.close()

    def _log_summary(self) -> None:
        documents = self.totals.documents
        logger.info(
            "Summary: documents extracted=%d, partial=%d, error=%d, not_in_scope=%d",
            documents[EXTRACTION_EXTRACTED],
            documents[EXTRACTION_PARTIAL],
            documents[EXTRACTION_ERROR] + self.totals.failed,
            documents[EXTRACTION_NOT_IN_SCOPE],
        )
        logger.info(
            "Summary: requests=%d, batches ok=%d, error=%d",
            self.totals.requests,
            self.totals.batches_ok,
            self.totals.batches_error,
        )
        if self._log.path is not None:
            logger.info("Log: %s", self._log.path)

    # --- One document -----------------------------------------------------------------

    def _record(
        self,
        chunk_file: ChunkFile,
        resolution: TypeResolution,
        relations: List[Dict[str, Any]],
        status: str,
        outcomes: Sequence[BatchOutcome],
    ) -> Dict[str, Any]:
        """The extraction file (D10, R16), keys in order."""
        record = chunk_file.record
        fields = _empty_fields()
        provisions: List[Dict[str, Any]] = []
        for outcome in outcomes:
            if not outcome.ok:
                continue
            for name in FIELDS:
                fields[name]["candidates"].extend(outcome.candidates[name])
            provisions.extend(outcome.provisions)
        for name in FIELDS:
            fields[name]["status"] = field_status(fields[name]["candidates"])
        return {
            "schema_version": SCHEMA_VERSION,
            "extractor_version": EXTRACTOR_VERSION,
            "file": {
                "path": record["file"]["path"],
                "sha256": record["file"]["sha256"],
            },
            "language": record["language"],
            "language_source": record["language_source"],
            "index_type": resolution.index_type,
            "document_type": resolution.document_type,
            "document_type_source": resolution.source,
            "relations": copy.deepcopy(relations),
            "extraction_status": status,
            "model": self._config.model,
            "reasoning_effort": self._config.reasoning_effort,
            "batches": [outcome.entry for outcome in outcomes],
            "fields": fields,
            "other_termination_provisions": provisions,
        }

    def _document(
        self,
        label: str,
        chunk_file: ChunkFile,
        resolution: TypeResolution,
        relations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        config = self._config
        record = chunk_file.record
        if resolution.document_type not in config.lease_family_types:
            return self._record(
                chunk_file, resolution, relations, EXTRACTION_NOT_IN_SCOPE, []
            )
        if record["status"] == STATUS_FAILED:
            return self._record(chunk_file, resolution, relations, EXTRACTION_ERROR, [])
        words = vocabulary(record["language"], config.keywords, config.month_names)
        chunks = record["chunks"]
        batches = plan_batches(
            chunks,
            scores_of(chunks, words),
            budget=config.selection_budget_chars,
            max_batches=config.max_batches_per_document,
        )
        if not batches:
            logger.warning("%s: no chunk scores above 0", chunk_file.relative)
        outcomes = [
            self._batch(label, chunk_file, index, len(batches), batch)
            for index, batch in enumerate(batches, start=1)
        ]
        return self._record(
            chunk_file, resolution, relations, _extraction_status(outcomes), outcomes
        )

    # --- The generator ----------------------------------------------------------------

    def _ready_client(self) -> GeneratorClient:
        """The run's client, created at the first request; the warm-up, when
        configured, goes first (R5)."""
        if self._client is None:
            self._client = GeneratorClient(
                self._generator, api_key=self._api_key, transport=self._transport
            )
        if self._config.warmup and not self._warmed:
            self._warmed = True
            try:
                self._client.complete(
                    [{"role": "user", "content": WARMUP_MESSAGE}],
                    max_tokens=WARMUP_MAX_TOKENS,
                )
            except RequestFailure as failure:
                if failure.unavailable:
                    raise GeneratorStopped(self._unavailable(failure)) from failure
                raise GeneratorStopped(
                    f"Generator failed at warm-up: {failure.cause}"
                ) from failure
        return self._client

    def _unavailable(self, failure: RequestFailure) -> str:
        return f"Generator unavailable at {self._config.base_url}: {failure.cause}"

    # --- One batch --------------------------------------------------------------------

    def _batch(
        self,
        label: str,
        chunk_file: ChunkFile,
        batch_index: int,
        batch_count: int,
        sources: Sequence[Dict[str, Any]],
    ) -> BatchOutcome:
        config = self._config
        client = self._ready_client()
        system = self._prompt.text
        user = user_message(sources)
        sent = messages(system, user)
        body = request_body(self._generator, sent, config.max_tokens)
        chunk_ids = [chunk["chunk_id"] for chunk in sources]
        timestamp = format_timestamp(_utc_now())
        started = _monotonic()
        self.totals.requests += 1
        completion: Optional[Completion] = None
        failure: Optional[RequestFailure] = None
        candidates: Dict[str, List[Dict[str, Any]]] = {}
        provisions: List[Dict[str, Any]] = []
        try:
            completion = client.complete(sent, max_tokens=config.max_tokens)
        except RequestFailure as raised:
            duration = _monotonic() - started
            failure = raised
            error_type = (
                ERROR_GENERATOR_UNAVAILABLE if raised.unavailable else raised.error_type
            )
            error: Optional[Tuple[str, str]] = (error_type, raised.cause)
        else:
            duration = _monotonic() - started
            error, candidates, provisions = self._judge(
                completion, len(system) + len(user), batch_index, sources
            )

        ratio: Optional[float] = None
        nearly_full: Optional[bool] = None
        if completion is not None:
            ratio = (len(system) + len(user)) / completion.prompt_tokens
            nearly_full = (
                completion.prompt_tokens + config.max_tokens > config.context_window
            )
        self._log.write(
            {
                "schema_version": LOG_SCHEMA_VERSION,
                "timestamp_utc": timestamp,
                "file": {
                    "path": chunk_file.record["file"]["path"],
                    "sha256": chunk_file.record["file"]["sha256"],
                },
                "batch_index": batch_index,
                "batch_count": batch_count,
                "chunk_ids": chunk_ids,
                "prompt": {
                    "system_file": config.prompt_file,
                    "system_sha256": self._prompt.sha256,
                },
                "request": body,
                "response": _response_entry(completion),
                "chars_per_prompt_token": ratio,
                "context_nearly_full": nearly_full,
                "duration_s": duration,
                "status": BATCH_OK if error is None else BATCH_ERROR,
                "error": None
                if error is None
                else {"type": error[0], "message": error[1]},
            }
        )
        status = BATCH_OK if error is None else BATCH_ERROR
        logger.info(
            "%s: batch %d/%d: %s, %.1f s",
            label,
            batch_index,
            batch_count,
            status,
            duration,
        )
        if failure is not None and failure.unavailable:
            raise GeneratorStopped(self._unavailable(failure)) from failure
        if error is None:
            self.totals.batches_ok += 1
        else:
            self.totals.batches_error += 1
            logger.warning(
                "%s: batch %d/%d: %s: %s",
                chunk_file.relative,
                batch_index,
                batch_count,
                error[0],
                error[1],
            )
        entry = {
            "batch_index": batch_index,
            "chunk_ids": chunk_ids,
            "chars": batch_chars(sources),
            "status": status,
            "error_type": None if error is None else error[0],
            "prompt_tokens": None if completion is None else completion.prompt_tokens,
            "completion_tokens": (
                None if completion is None else completion.completion_tokens
            ),
            "finish_reason": None if completion is None else completion.finish_reason,
            "duration_s": duration,
        }
        if error is not None:
            return BatchOutcome(entry=entry, candidates={}, provisions=[])
        return BatchOutcome(entry=entry, candidates=candidates, provisions=provisions)

    def _judge(
        self,
        completion: Completion,
        characters: int,
        batch_index: int,
        sources: Sequence[Dict[str, Any]],
    ) -> Tuple[
        Optional[Tuple[str, str]],
        Dict[str, List[Dict[str, Any]]],
        List[Dict[str, Any]],
    ]:
        """R12's checks after a response, in order: the tripwire,
        ``finish_reason``, the JSON, the structure; then the candidates."""
        config = self._config
        ratio = characters / completion.prompt_tokens
        if ratio > config.max_chars_per_prompt_token:
            return (
                (
                    ERROR_PROMPT_TRUNCATION_SUSPECTED,
                    f"{characters} characters for {completion.prompt_tokens} "
                    f"prompt tokens is {ratio} per token, above "
                    f"max_chars_per_prompt_token ({config.max_chars_per_prompt_token})",
                ),
                {},
                [],
            )
        if completion.finish_reason == FINISH_REASON_LENGTH:
            return (
                (
                    ERROR_GENERATION_TRUNCATED,
                    f"finish_reason is length: the generation reached max_tokens "
                    f"({config.max_tokens})",
                ),
                {},
                [],
            )
        try:
            data = parse_content(completion.content)
        except InvalidContent as invalid:
            return (ERROR_INVALID_JSON, str(invalid)), {}, []
        try:
            check_structure(data)
        except SchemaMismatch as mismatch:
            return (ERROR_SCHEMA_MISMATCH, str(mismatch)), {}, []
        candidates, provisions = batch_items(data, batch_index, sources)
        return None, candidates, provisions


def _response_entry(completion: Optional[Completion]) -> Optional[Dict[str, Any]]:
    """The log record's ``response`` (R17), or None after a failed request."""
    if completion is None:
        return None
    return {
        "content": completion.content,
        "reasoning": completion.reasoning,
        "finish_reason": completion.finish_reason,
        "usage": {
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "cached_tokens": completion.cached_tokens,
        },
    }


# --- The subcommand -------------------------------------------------------------------


def run_extract(
    chunks_dir: Path,
    extracted_dir: Path,
    type_overrides: Path,
    relations: Path,
    *,
    transport: Optional[httpx.BaseTransport] = None,
) -> int:
    """``companydocs-metadata extract``; returns the exit code (D11)."""
    started = _utc_now()
    try:
        config = metadata_settings()
        prompt = load_system_prompt(config)
        api_key = read_api_key(config)
    except SettingsError as error:
        logger.error("Invalid settings: %s", error)
        return EXIT_CANNOT_START
    try:
        overrides = read_type_overrides(type_overrides)
    except ManualFileError as error:
        logger.error("Invalid type overrides file %s", error.describe())
        return EXIT_CANNOT_START
    try:
        relations_file = read_relations(relations)
    except ManualFileError as error:
        logger.error("Invalid relations file %s", error.describe())
        return EXIT_CANNOT_START

    if not chunks_dir.is_dir():
        logger.error("Not a folder: %s", chunks_dir)
        return EXIT_INPUT_REFUSED
    files, refused = read_chunk_files(chunks_dir)
    if refused:
        logger.error("Refused %d chunk file(s); rerun companydocs-chunk", refused)
        return EXIT_INPUT_REFUSED
    if not files:
        logger.warning("No chunk files found in %s", chunks_dir)

    resolver = TypeResolver(overrides, language_settings()["index_filename"])
    types = [resolver.resolve(item.record["file"]["path"]) for item in files]
    resolver.unmatched_overrides(
        [file_name(item.record["file"]["path"]) for item in files]
    )
    entries = resolve_relations(relations_file, [item.record for item in files])

    run = ExtractRun(config, prompt, api_key, extracted_dir, started, transport)
    return run.run(files, types, entries)
