"""Selection, the budget and the prompt (spec 007, D3, D4, R6-R9).

Selection (D3, R6)
------------------
``fused`` takes the first ``fused_k`` candidates in fused rank order.
``per_path`` takes the top ``per_path_k`` of each leg's list
(``SearchResult.path_hits``) and interleaves them by path rank, dense first:
dense 1, sparse 1, dense 2, sparse 2, ...; a list that has run out loses its
turn, and a chunk already selected is skipped. ``selected_by`` is ``fused``,
or the path whose turn added the chunk. ``rank`` is the chunk's fused rank
when it is a candidate, else None; ``dense_rank`` and ``sparse_rank`` are its
ranks in the two lists.

The budget (D3, R7)
-------------------
Chunks are added in selection order while the rendered blocks fit in
``max_context_chars``: a block counts its header line, the line feed after it
and its text, in Python str characters, and fits when the total so far plus
the block is at most the budget. At the first chunk that does not fit the
selection stops, and that chunk and every later one go to
``dropped_for_budget``. A chunk is never cut.

The prompt (D4, R8, R9)
-----------------------
One block per chunk: ``[n] (document D<k>, taal <language>, pagina <p>)``,
or ``pagina <p1>-<p2>`` for a range, a line feed, and the chunk's text
unchanged. ``n`` runs 1..N in selection order; ``D<k>`` numbers the
documents (``file_sha256``) in order of first appearance; ``<language>`` is
the payload value as stored, ``onbekend`` when it is null. Blocks are joined
by a blank line, followed by a blank line and ``Vraag: <question>``, the
question as given. The prompt holds no file name, clause label, text origin
or chunk id. The system prompt is the prompt file's bytes decoded as UTF-8,
unchanged; its sha256 is that of the bytes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from companydocs_rag.embeddings.dense import SettingsError
from companydocs_rag.vectorstore.store import PAYLOAD_KEYS

from .settings import (
    SELECTION_FUSED,
    AnsweringSettings,
    resolve_project_path,
)

PATH_DENSE = "dense"
PATH_SPARSE = "sparse"
# R9: the language of a chunk whose payload has none.
UNKNOWN_LANGUAGE = "onbekend"
QUESTION_PREFIX = "Vraag: "
BLOCK_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class SelectedChunk:
    """One selected chunk: its payload (``chunk_id`` and the payload keys),
    its ranks and the path that selected it."""

    chunk: Dict[str, Any]
    rank: Optional[int]
    dense_rank: Optional[int]
    sparse_rank: Optional[int]
    selected_by: str

    @property
    def chunk_id(self) -> str:
        return self.chunk["chunk_id"]


@dataclass(frozen=True)
class PromptBlock:
    """A selected chunk that fits the budget, as rendered in the prompt."""

    prompt_number: int
    document_label: str
    selected: SelectedChunk
    text: str


@dataclass(frozen=True)
class Budgeted:
    """The blocks that fit, in order, and the ``chunk_id``s of the rest of
    the selection (``dropped_for_budget``). ``rejected_chars`` is the length
    of the first block that did not fit, None when every block fits."""

    blocks: List[PromptBlock]
    dropped: List[str]
    rejected_chars: Optional[int] = None


@dataclass(frozen=True)
class SystemPrompt:
    """The prompt file's text and the sha256 of its bytes (R8)."""

    text: str
    sha256: str


def _payload(entry: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: entry[key] for key in PAYLOAD_KEYS}


def select_fused(
    candidates: Sequence[Mapping[str, Any]], fused_k: int
) -> List[SelectedChunk]:
    """The first ``fused_k`` candidates in rank order, each with its own
    ranks (D3, R6)."""
    ordered = sorted(candidates, key=lambda candidate: candidate["rank"])
    return [
        SelectedChunk(
            chunk=_payload(candidate),
            rank=candidate["rank"],
            dense_rank=candidate["dense_rank"],
            sparse_rank=candidate["sparse_rank"],
            selected_by=SELECTION_FUSED,
        )
        for candidate in ordered[:fused_k]
    ]


def select_per_path(
    path_hits: Mapping[str, Sequence[Mapping[str, Any]]],
    candidates: Sequence[Mapping[str, Any]],
    per_path_k: int,
) -> List[SelectedChunk]:
    """The top ``per_path_k`` of each path list, interleaved by path rank,
    dense first, a chunk already selected skipped (D3, R6)."""
    lists = {
        path: sorted(path_hits[path], key=lambda entry: entry["rank"])
        for path in (PATH_DENSE, PATH_SPARSE)
    }
    ranks = {
        path: {entry["chunk_id"]: entry["rank"] for entry in entries}
        for path, entries in lists.items()
    }
    fused_ranks = {candidate["chunk_id"]: candidate["rank"] for candidate in candidates}
    tops = {path: entries[:per_path_k] for path, entries in lists.items()}

    selected: List[SelectedChunk] = []
    seen = set()
    for position in range(max(len(entries) for entries in tops.values())):
        for path in (PATH_DENSE, PATH_SPARSE):
            if position >= len(tops[path]):
                continue  # this list has run out: it loses its turn
            entry = tops[path][position]
            chunk_id = entry["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            selected.append(
                SelectedChunk(
                    chunk=_payload(entry),
                    rank=fused_ranks.get(chunk_id),
                    dense_rank=ranks[PATH_DENSE].get(chunk_id),
                    sparse_rank=ranks[PATH_SPARSE].get(chunk_id),
                    selected_by=path,
                )
            )
    return selected


def pages_text(page_start: int, page_end: int) -> str:
    """``<p>`` for one page, ``<p1>-<p2>`` for a range (R9)."""
    if page_start == page_end:
        return f"{page_start}"
    return f"{page_start}-{page_end}"


def block_header(
    prompt_number: int,
    document_label: str,
    language: Optional[str],
    page_start: int,
    page_end: int,
) -> str:
    """A block's header line (D4, R9)."""
    shown = UNKNOWN_LANGUAGE if language is None else language
    return (
        f"[{prompt_number}] (document {document_label}, taal {shown}, "
        f"pagina {pages_text(page_start, page_end)})"
    )


def fit_budget(selection: Sequence[SelectedChunk], max_context_chars: int) -> Budgeted:
    """The blocks that fit ``max_context_chars``, in selection order, and the
    ``chunk_id``s of the rest (D3, R7)."""
    documents: Dict[str, str] = {}
    blocks: List[PromptBlock] = []
    total = 0
    for position, selected in enumerate(selection):
        chunk = selected.chunk
        label = documents.get(chunk["file_sha256"], f"D{len(documents) + 1}")
        header = block_header(
            position + 1,
            label,
            chunk["language"],
            chunk["page_start"],
            chunk["page_end"],
        )
        text = header + "\n" + chunk["text"]
        if total + len(text) > max_context_chars:
            return Budgeted(
                blocks=blocks,
                dropped=[item.chunk_id for item in selection[position:]],
                rejected_chars=len(text),
            )
        total += len(text)
        documents.setdefault(chunk["file_sha256"], label)
        blocks.append(PromptBlock(position + 1, label, selected, text))
    return Budgeted(blocks=blocks, dropped=[])


def user_message(blocks: Sequence[PromptBlock], question: str) -> str:
    """The blocks joined by a blank line, then a blank line and the question
    as given (D4, R9)."""
    return (
        BLOCK_SEPARATOR.join(block.text for block in blocks)
        + BLOCK_SEPARATOR
        + QUESTION_PREFIX
        + question
    )


def load_system_prompt(config: AnsweringSettings) -> SystemPrompt:
    """Read the prompt file once, as bytes (R8). A missing or unreadable file,
    or one that is not UTF-8, raises ``SettingsError`` for
    ``answering.prompt_file``."""
    path = resolve_project_path(config.prompt_file)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SettingsError(
            "answering",
            "prompt_file",
            f"could not read {path}: {type(error).__name__}: {error}",
        ) from None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SettingsError(
            "answering", "prompt_file", f"{path} is not UTF-8: {error}"
        ) from None
    return SystemPrompt(text=text, sha256=hashlib.sha256(raw).hexdigest())
