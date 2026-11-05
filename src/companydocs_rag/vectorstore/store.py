"""The Qdrant side of the index (spec 006): the only module that opens Qdrant.

Qdrant runs in local mode (``QdrantClient(path=...)``), its files directly
in the index folder (R9). The collection :data:`COLLECTION_NAME` has a named
dense vector :data:`DENSE_VECTOR_NAME` (Cosine) and a named sparse vector
:data:`SPARSE_VECTOR_NAME` with ``Modifier.IDF`` (D10). Each point is one
chunk: its id is the chunk id padded to a UUID (:func:`point_id_for`, D11,
R19) and its payload holds :data:`PAYLOAD_KEYS` in that order (D13).

Local mode locks its folder: a second client on it fails, in another process
or the same one (D24). :func:`open_client` raises :class:`IndexInUseError`
for exactly that error. Local mode raises it as a ``RuntimeError`` with a
fixed message while portalocker's ``LockException`` is handled; the message
is what tells it apart here, so no portalocker import is needed.
``tests/vectorstore/test_qdrant_capabilities.py`` pins it for the pinned
qdrant-client.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Sequence

from qdrant_client import QdrantClient, models

COLLECTION_NAME = "chunks"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# D13: the payload's keys, in this order.
PAYLOAD_KEYS = (
    "chunk_id",
    "chunk_index",
    "file_sha256",
    "file_path",
    "language",
    "language_source",
    "route",
    "text",
    "page_start",
    "page_end",
    "text_origins",
    "page_spans",
    "chunk_method",
    "clause_label",
    "clause_number",
    "part",
    "part_count",
)

_POINT_ID_PADDING = "0" * 12
_CHUNK_ID_HEX_CHARS = 20
# The fixed part of the message local mode raises for a folder another
# client holds.
_LOCK_MESSAGE = "is already accessed by another instance of Qdrant client"


class IndexInUseError(RuntimeError):
    """Another Qdrant client, in this process or another, holds the folder."""


class Hit(NamedTuple):
    """One point a query returned."""

    chunk_id: str
    score: float
    payload: Dict[str, Any]


class Point(NamedTuple):
    """One chunk to upload."""

    chunk_id: str
    dense: Sequence[float]
    sparse_indices: Sequence[int]
    sparse_values: Sequence[float]
    payload: Dict[str, Any]


def point_id_for(chunk_id: str) -> str:
    """The canonical UUID string, lowercase and hyphenated, of ``chunk_id``
    followed by twelve "0" (D11, R19)."""
    return str(uuid.UUID(chunk_id + _POINT_ID_PADDING))


def chunk_id_for(point_id: Any) -> str:
    """The chunk id of a point id: its first 20 hex characters. Raises
    ``ValueError`` when the last twelve are not all "0" (R19)."""
    hex_digits = uuid.UUID(str(point_id)).hex
    if hex_digits[_CHUNK_ID_HEX_CHARS:] != _POINT_ID_PADDING:
        raise ValueError(f"{point_id} is not a chunk's point id")
    return hex_digits[:_CHUNK_ID_HEX_CHARS]


def is_lock_error(error: BaseException) -> bool:
    """Whether ``error`` is local mode's error for a folder another client
    holds."""
    return isinstance(error, RuntimeError) and _LOCK_MESSAGE in str(error)


def open_client(path: Path) -> QdrantClient:
    """A local-mode client on ``path``, which takes its lock. Raises
    :class:`IndexInUseError` when another client holds it; any other error
    propagates."""
    try:
        return QdrantClient(path=str(path))
    except RuntimeError as error:
        if not is_lock_error(error):
            raise
    # Raised outside the handler, so nothing keeps the half-opened client's
    # files referenced.
    raise IndexInUseError(f"{path} is held by another client")


def create_collection(client: QdrantClient, dimension: int) -> None:
    """The collection with its two named vectors (D10)."""
    client.create_collection(
        COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=dimension, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )


def upload_points(
    client: QdrantClient, points: Iterable[Point], batch_size: int
) -> None:
    """Upsert ``points`` in batches of ``batch_size``, in order."""
    batch: List[models.PointStruct] = []
    for point in points:
        batch.append(
            models.PointStruct(
                id=point_id_for(point.chunk_id),
                vector={
                    DENSE_VECTOR_NAME: list(point.dense),
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=list(point.sparse_indices),
                        values=list(point.sparse_values),
                    ),
                },
                payload=point.payload,
            )
        )
        if len(batch) == batch_size:
            client.upsert(COLLECTION_NAME, points=batch, wait=True)
            batch = []
    if batch:
        client.upsert(COLLECTION_NAME, points=batch, wait=True)


def count_points(client: QdrantClient) -> int:
    return client.count(COLLECTION_NAME, exact=True).count


def _hits(points: Iterable[models.ScoredPoint]) -> List[Hit]:
    return [
        Hit(chunk_id_for(point.id), float(point.score), dict(point.payload or {}))
        for point in points
    ]


def query_dense(client: QdrantClient, vector: Sequence[float], limit: int) -> List[Hit]:
    """The dense leg: at most ``limit`` points by cosine score."""
    response = client.query_points(
        COLLECTION_NAME,
        query=list(vector),
        using=DENSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
    )
    return _hits(response.points)


def query_sparse(
    client: QdrantClient, indices: Sequence[int], values: Sequence[float], limit: int
) -> List[Hit]:
    """The sparse leg: at most ``limit`` points by IDF-weighted dot product,
    only points that share an index with the query."""
    response = client.query_points(
        COLLECTION_NAME,
        query=models.SparseVector(indices=list(indices), values=list(values)),
        using=SPARSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
    )
    return _hits(response.points)
