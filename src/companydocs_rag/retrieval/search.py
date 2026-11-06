"""Hybrid search over the index (spec 006): :class:`Retriever`.

Spec 007's answer layer calls this in-process; ``companydocs-search`` is a thin
wrapper around it. A :class:`Retriever` opens the index once (R14), in this
order: ``manifest.json`` must be present and readable; its eight D17 keys
must match the current settings and constants; Qdrant is opened, which takes
its lock (D24); and the embedder is loaded from local files only (D3), unless
one is injected. Each of these failures raises :class:`IndexUnusableError`
with the messages as its ``reasons``; any other error propagates.

Per query (D5, D8, D20, D21): the query's terms; the dense leg, with the
query instruction as prompt; the sparse leg, skipped when the query has no
terms; fusion (:mod:`.fusion`); and per candidate both legs' ranks and
scores, the terms it shares with the query, and its payload. There is no
threshold and no abstention (D23): zero candidates is a normal result.

Path hits (spec 007, D2, R5)
----------------------------
``search(query, include_path_hits=True)`` also returns each leg's own list
as :attr:`SearchResult.path_hits`, ``{"dense": [...], "sparse": [...]}``:
every point the leg returned, ranked as fusion ranks a leg (score
descending, then ``chunk_id``), so a candidate's rank and score there equal
its ``dense_rank``/``dense_score`` and ``sparse_rank``/``sparse_score``.
Each entry has ``rank``, ``chunk_id`` and ``score``, then the payload keys
after ``chunk_id``; a skipped sparse leg gives ``[]``. The answer layer
selects chunks from these lists. By default ``path_hits`` is None and
:meth:`SearchResult.to_dict` leaves the key out, so ``companydocs-search``
prints what it printed before.

Settings (R12)
--------------
The top-level ``retrieval`` section, read at call time by
:func:`retrieval_settings`, with the defaults below. ``bm25_k1`` and
``bm25_b`` are the index's BM25 parameters; the search compares them with the
manifest (D17). The others are reported in every result and never compared.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from companydocs_rag.embeddings.dense import (
    Embedder,
    SentenceTransformerEmbedder,
    SettingsError,
    compose_embedding_text,
    embeddings_settings,
    is_finite_above_zero,
    setting_integer,
    setting_number,
    setting_string,
)
from companydocs_rag.embeddings.sparse import query_sparse_vector, sparse_terms
from companydocs_rag.ingestion.settings import get_section
from companydocs_rag.vectorstore.manifest import (
    MANIFEST_FILE_NAME,
    compare_manifest,
    current_values,
    read_manifest,
)
from companydocs_rag.vectorstore.store import (
    PAYLOAD_KEYS,
    Hit,
    IndexInUseError,
    all_payloads,
    open_client,
    query_dense,
    query_dense_in_file,
    query_sparse,
)

from .fusion import fuse, rank_leg

# --- Settings defaults (retrieval), read at call time ---------------------------------
BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60
DENSE_LIMIT = 50
SPARSE_LIMIT = 50
CANDIDATE_COUNT = 20
# The judge route (Rutger's experiment, retrieval/judge.py); "none" is the baseline.
RERANKER = "none"
RERANKERS = ("none", "judge")
JUDGE_MODEL_DIR = ""
JUDGE_ADAPTER = ""
JUDGE_DEVICE = "cpu"
JUDGE_DOCUMENTS = 2
JUDGE_DOC_CHUNKS = 300
JUDGE_WINDOW_CHARS = 1600

_SECTION = "retrieval"
# D22: the manifest keys a result reports under "index".
_INDEX_KEYS = (
    ("model_name", "embedding"),
    ("model_revision", "embedding"),
    ("embedding_text_version", "embedding"),
    ("tokenizer_version", "sparse"),
)


@dataclass(frozen=True)
class RetrievalSettings:
    """The resolved ``retrieval`` section."""

    bm25_k1: float
    bm25_b: float
    rrf_k: int
    dense_limit: int
    sparse_limit: int
    candidate_count: int
    reranker: str = RERANKER
    judge_model_dir: str = JUDGE_MODEL_DIR
    judge_adapter: str = JUDGE_ADAPTER
    judge_device: str = JUDGE_DEVICE
    judge_documents: int = JUDGE_DOCUMENTS
    judge_doc_chunks: int = JUDGE_DOC_CHUNKS
    judge_window_chars: int = JUDGE_WINDOW_CHARS


def retrieval_settings(
    settings: Optional[Mapping[str, Any]] = None,
) -> RetrievalSettings:
    """Resolve the ``retrieval`` section at call time, YAML over the defaults
    above; ``settings`` replaces ``config/settings.yaml``. Keys are checked in
    the order of the Settings table, and the first invalid one raises
    :class:`companydocs_rag.embeddings.dense.SettingsError`."""
    section = _SECTION
    values = get_section(section, settings=settings)
    bm25_k1 = setting_number(
        section,
        values,
        "bm25_k1",
        BM25_K1,
        requirement="a finite number above 0",
        valid=is_finite_above_zero,
    )
    bm25_b = setting_number(
        section,
        values,
        "bm25_b",
        BM25_B,
        requirement="a number from 0 to 1",
        valid=lambda number: 0.0 <= number <= 1.0,
    )
    rrf_k = setting_integer(section, values, "rrf_k", RRF_K, minimum=1)
    dense_limit = setting_integer(
        section, values, "dense_limit", DENSE_LIMIT, minimum=1
    )
    sparse_limit = setting_integer(
        section, values, "sparse_limit", SPARSE_LIMIT, minimum=1
    )
    candidate_count = setting_integer(
        section, values, "candidate_count", CANDIDATE_COUNT, minimum=1
    )
    if candidate_count > dense_limit + sparse_limit:
        raise SettingsError(
            section,
            "candidate_count",
            f"must be at most dense_limit + sparse_limit "
            f"({dense_limit + sparse_limit}), not {candidate_count}",
        )
    reranker = setting_string(section, values, "reranker", RERANKER, empty_ok=False)
    if reranker not in RERANKERS:
        raise SettingsError(
            section,
            "reranker",
            f"must be one of {', '.join(RERANKERS)}, not {reranker!r}",
        )
    judge_model_dir = setting_string(
        section, values, "judge_model_dir", JUDGE_MODEL_DIR, empty_ok=True
    )
    judge_adapter = setting_string(
        section, values, "judge_adapter", JUDGE_ADAPTER, empty_ok=True
    )
    judge_device = setting_string(
        section, values, "judge_device", JUDGE_DEVICE, empty_ok=False
    )
    judge_documents = setting_integer(
        section, values, "judge_documents", JUDGE_DOCUMENTS, minimum=1
    )
    judge_doc_chunks = setting_integer(
        section, values, "judge_doc_chunks", JUDGE_DOC_CHUNKS, minimum=1
    )
    judge_window_chars = setting_integer(
        section, values, "judge_window_chars", JUDGE_WINDOW_CHARS, minimum=200
    )
    if reranker == "judge" and not judge_model_dir:
        raise SettingsError(
            section,
            "judge_model_dir",
            "must name the d3-lite folder when reranker is judge",
        )
    return RetrievalSettings(
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        rrf_k=rrf_k,
        dense_limit=dense_limit,
        sparse_limit=sparse_limit,
        candidate_count=candidate_count,
        reranker=reranker,
        judge_model_dir=judge_model_dir,
        judge_adapter=judge_adapter,
        judge_device=judge_device,
        judge_documents=judge_documents,
        judge_doc_chunks=judge_doc_chunks,
        judge_window_chars=judge_window_chars,
    )


class IndexUnusableError(Exception):
    """The index cannot be searched (R14). ``reasons`` are the ERROR texts
    ``companydocs-search`` logs, one per D17 key that differs."""

    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons: List[str] = list(reasons)


@dataclass(frozen=True)
class SearchResult:
    """One search's result; :meth:`to_dict` is the D22 object.

    ``path_hits`` is each leg's own ranked list when the search asked for it
    (``include_path_hits=True``), else None (spec 007, D2, R5)."""

    query: str
    query_terms: List[str]
    query_instruction: str
    index: Dict[str, Any]
    settings: Dict[str, int]
    leg_counts: Dict[str, int]
    candidates: List[Dict[str, Any]]
    path_hits: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """The D22 object that ``companydocs-search --json`` prints, keys in
        order; ``path_hits`` follows ``candidates`` when it is not None."""
        result: Dict[str, Any] = {
            "query": self.query,
            "query_terms": self.query_terms,
            "query_instruction": self.query_instruction,
            "index": self.index,
            "settings": self.settings,
            "leg_counts": self.leg_counts,
            "candidates": self.candidates,
        }
        if self.path_hits is not None:
            result["path_hits"] = self.path_hits
        return copy.deepcopy(result)


class Retriever:
    """Hybrid search over one index, opened once (D19, R14).

    ``settings`` replaces ``config/settings.yaml``, as for the chunker. An
    injected ``embedder`` replaces the model. Use it as a context manager, or
    call :meth:`close`, to release the index's lock.
    """

    def __init__(
        self,
        index_dir: Path,
        *,
        embedder: Optional[Embedder] = None,
        settings: Optional[Mapping[str, Any]] = None,
        judge: Optional[Any] = None,
    ) -> None:
        """``judge`` (a score function, see retrieval/judge.py) replaces the
        decision model of the judge route; it is used only when
        ``retrieval.reranker`` is ``judge``."""
        self._client: Any = None
        self._judge: Any = None
        self._documents: Dict[str, Dict[str, Any]] = {}
        index_dir = Path(index_dir)
        embeddings_config = embeddings_settings(settings)
        self._settings = retrieval_settings(settings)
        self._query_instruction = embeddings_config.query_instruction

        manifest_path = index_dir / MANIFEST_FILE_NAME
        if not manifest_path.exists():
            raise IndexUnusableError(
                [f"No index at {index_dir}: manifest.json missing"]
            )
        try:
            manifest = read_manifest(manifest_path)
        except (OSError, ValueError) as error:
            raise IndexUnusableError(
                [f"Could not read {manifest_path}: {type(error).__name__}: {error}"]
            ) from None
        mismatches = compare_manifest(
            manifest,
            current_values(
                model_name=embeddings_config.model_name,
                model_revision=embeddings_config.model_revision,
                bm25_k1=self._settings.bm25_k1,
                bm25_b=self._settings.bm25_b,
            ),
        )
        if mismatches:
            raise IndexUnusableError(mismatches)
        self._index = {key: manifest[where][key] for key, where in _INDEX_KEYS}

        try:
            self._client = open_client(index_dir)
        except IndexInUseError:
            raise IndexUnusableError(
                [f"Index is in use by another process: {index_dir}"]
            ) from None

        if embedder is not None:
            self._embedder: Embedder = embedder
            self._setup_judge(judge)
            return
        try:
            self._embedder = SentenceTransformerEmbedder(
                embeddings_config, local_files_only=True
            )
        except Exception as error:  # noqa: BLE001 - any load failure is Q3's
            self.close()
            raise IndexUnusableError(
                [
                    f"Could not load embedding model {embeddings_config.model_name}"
                    f"@{embeddings_config.model_revision}: "
                    f"{type(error).__name__}: {error}"
                ]
            ) from error
        except BaseException:
            self.close()
            raise
        self._setup_judge(judge)

    def _setup_judge(self, judge: Optional[Any]) -> None:
        """The judge route's documents (title and opening per file hash, from
        the index) and its decision model; nothing when the route is off."""
        if self._settings.reranker != "judge":
            return
        from .judge import document_title

        first: Dict[str, Dict[str, Any]] = {}
        for hit in all_payloads(self._client):
            payload = hit.payload
            key = payload.get("file_sha256", "")
            if key not in first or payload.get("chunk_index", 0) < first[key].get(
                "chunk_index", 0
            ):
                first[key] = payload
        self._documents = {
            key: {"title": document_title(p), "opening": str(p.get("text", ""))[:400]}
            for key, p in first.items()
        }
        if judge is not None:
            self._judge = judge
            return
        from .judge import load_judge

        try:
            self._judge = load_judge(
                self._settings.judge_model_dir,
                self._settings.judge_adapter,
                self._settings.judge_device,
            )
        except Exception as error:  # noqa: BLE001 - any load failure makes the index unusable here
            self.close()
            raise IndexUnusableError(
                [
                    f"Could not load the judge from {self._settings.judge_model_dir}: "
                    f"{type(error).__name__}: {error}"
                ]
            ) from error

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release the index; closing twice is harmless."""
        client, self._client = self._client, None
        if client is not None:
            client.close()

    def search(self, query: str, *, include_path_hits: bool = False) -> SearchResult:
        """Search the index for ``query``, used as given (R18). Raises
        ``ValueError("Empty query")`` for an empty or whitespace-only query.
        With ``include_path_hits`` the result also holds each leg's ranked
        list (spec 007, D2)."""
        if not query.strip():
            raise ValueError("Empty query")
        if self._client is None:
            raise ValueError("The retriever is closed")
        config = self._settings
        query_terms = sparse_terms(query)

        query_vector = self._embedder.encode_query(query)
        dense_hits = query_dense(self._client, query_vector, config.dense_limit)
        sparse_hits: List[Hit] = []
        if query_terms:  # D8: a query without terms skips the sparse leg
            vector = query_sparse_vector(query)
            sparse_hits = query_sparse(
                self._client, vector.indices, vector.values, config.sparse_limit
            )

        payloads = {hit.chunk_id: hit.payload for hit in [*dense_hits, *sparse_hits]}
        wanted = set(query_terms)
        candidates: List[Dict[str, Any]] = []
        for fused in fuse(
            [(hit.chunk_id, hit.score) for hit in dense_hits],
            [(hit.chunk_id, hit.score) for hit in sparse_hits],
            rrf_k=config.rrf_k,
            candidate_count=config.candidate_count,
        ):
            payload = payloads[fused.chunk_id]
            candidate: Dict[str, Any] = {
                "rank": fused.rank,
                "chunk_id": fused.chunk_id,
                "rrf_score": fused.rrf_score,
                "dense_rank": fused.dense_rank,
                "dense_score": fused.dense_score,
                "sparse_rank": fused.sparse_rank,
                "sparse_score": fused.sparse_score,
                "sparse_matched_terms": sorted(
                    wanted & set(sparse_terms(compose_embedding_text(payload)))
                ),
            }
            for key in PAYLOAD_KEYS[1:]:
                candidate[key] = payload[key]
            candidates.append(candidate)

        if config.reranker == "judge":
            candidates = self._judge_route(query, query_vector, candidates, wanted)

        return SearchResult(
            query=query,
            query_terms=query_terms,
            query_instruction=self._query_instruction,
            index=dict(self._index),
            settings={
                "rrf_k": config.rrf_k,
                "dense_limit": config.dense_limit,
                "sparse_limit": config.sparse_limit,
                "candidate_count": config.candidate_count,
                **(
                    {
                        "reranker": config.reranker,
                        "judge_documents": config.judge_documents,
                        "judge_doc_chunks": config.judge_doc_chunks,
                        "judge_window_chars": config.judge_window_chars,
                    }
                    if config.reranker != "none"
                    else {}
                ),
            },
            leg_counts={"dense": len(dense_hits), "sparse": len(sparse_hits)},
            candidates=candidates,
            path_hits=(
                {
                    "dense": _path_list(dense_hits),
                    "sparse": _path_list(sparse_hits),
                }
                if include_path_hits
                else None
            ),
        )

    def _judge_route(
        self,
        query: str,
        query_vector: Sequence[float],
        hybrid: List[Dict[str, Any]],
        wanted: set,
    ) -> List[Dict[str, Any]]:
        """The judge route (retrieval/judge.py) on top of the hybrid candidates."""
        from .judge import judge_rank

        config = self._settings

        def doc_chunks(file_sha256: str) -> List[Dict[str, Any]]:
            found = []
            for hit in query_dense_in_file(
                self._client, query_vector, file_sha256, config.judge_doc_chunks
            ):
                candidate: Dict[str, Any] = {
                    "rank": None,
                    "chunk_id": hit.chunk_id,
                    "rrf_score": None,
                    "dense_rank": None,
                    "dense_score": hit.score,
                    "sparse_rank": None,
                    "sparse_score": None,
                    "sparse_matched_terms": sorted(
                        wanted & set(sparse_terms(compose_embedding_text(hit.payload)))
                    ),
                }
                for key in PAYLOAD_KEYS[1:]:
                    candidate[key] = hit.payload[key]
                found.append(candidate)
            return found

        return judge_rank(
            query,
            self._documents,
            hybrid,
            doc_chunks,
            self._judge,
            top_documents=config.judge_documents,
            window_chars=config.judge_window_chars,
            result_count=config.candidate_count,
        )


def _path_list(hits: Sequence[Hit]) -> List[Dict[str, Any]]:
    """One leg's hits in its rank order, as fusion ranks a leg (D20): each
    with ``rank``, ``chunk_id``, ``score`` and the payload keys after
    ``chunk_id`` (spec 007, R5)."""
    payloads = {hit.chunk_id: hit.payload for hit in hits}
    ranked = rank_leg([(hit.chunk_id, hit.score) for hit in hits])
    entries: List[Dict[str, Any]] = []
    for chunk_id, (rank, score) in sorted(ranked.items(), key=lambda item: item[1][0]):
        entry: Dict[str, Any] = {"rank": rank, "chunk_id": chunk_id, "score": score}
        for key in PAYLOAD_KEYS[1:]:
            entry[key] = payloads[chunk_id][key]
        entries.append(entry)
    return entries
