"""What the index needs from Qdrant local mode (spec 006, A1).

Written and run before any other code of spec 006: each test pins a capability
the spec's stop conditions name. Every client works in ``tmp_path`` and is
closed. Sparse indices and values are fabricated. The last test, which goes
through ``store.py``, was added once that module existed.
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient, models

from companydocs_rag.vectorstore import store

DENSE = "dense"
SPARSE = "sparse"
COLLECTION = "chunks"

FREQUENT = 1001  # a term in most points
RARE = 2002  # a term in one point
ELSEWHERE = 3003  # a term no query below holds


def _id(number: int) -> str:
    return str(uuid.UUID(int=number))


def _sparse(entries):
    pairs = sorted(entries.items())
    return models.SparseVector(
        indices=[index for index, _ in pairs], values=[value for _, value in pairs]
    )


def _create(client: QdrantClient, *, idf: bool = True) -> None:
    client.create_collection(
        COLLECTION,
        vectors_config={
            DENSE: models.VectorParams(size=4, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE: models.SparseVectorParams(
                modifier=models.Modifier.IDF if idf else None
            )
        },
    )


def _point(number: int, dense, sparse, payload=None) -> models.PointStruct:
    return models.PointStruct(
        id=_id(number),
        vector={DENSE: dense, SPARSE: _sparse(sparse)},
        payload=payload if payload is not None else {"number": number},
    )


def _dense_query(client: QdrantClient, vector, limit: int = 10):
    return client.query_points(
        COLLECTION, query=vector, using=DENSE, limit=limit, with_payload=True
    ).points


def _sparse_query(client: QdrantClient, entries, limit: int = 10):
    return client.query_points(
        COLLECTION, query=_sparse(entries), using=SPARSE, limit=limit, with_payload=True
    ).points


@pytest.fixture
def open_client(tmp_path):
    """Open local clients under ``tmp_path``; every one is closed at the end."""
    clients = []

    def make(folder="index"):
        client = QdrantClient(path=str(tmp_path / folder))
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.close()


def test_one_point_carries_a_named_dense_and_a_named_idf_sparse_vector(open_client):
    client = open_client()
    _create(client)
    payload = {"chunk_id": "ab" * 10, "text": "Fictieve tekst.", "page_start": 1}
    client.upsert(COLLECTION, [_point(1, [1.0, 2.0, 0.0, 0.0], {RARE: 0.5}, payload)])

    info = client.get_collection(COLLECTION)
    assert info.config.params.vectors[DENSE].distance == models.Distance.COSINE
    assert info.config.params.sparse_vectors[SPARSE].modifier == models.Modifier.IDF
    (dense_hit,) = _dense_query(client, [2.0, 4.0, 0.0, 0.0])
    (sparse_hit,) = _sparse_query(client, {RARE: 1.0})
    assert dense_hit.id == sparse_hit.id == _id(1)
    assert dense_hit.payload == sparse_hit.payload == payload
    assert list(dense_hit.payload) == ["chunk_id", "text", "page_start"]
    assert dense_hit.score == pytest.approx(1.0)  # cosine of parallel vectors
    (stored,) = client.retrieve(COLLECTION, [_id(1)], with_vectors=True)
    assert stored.vector[SPARSE].indices == [RARE]
    assert stored.vector[SPARSE].values == [0.5]


def test_the_sparse_query_does_not_return_a_point_that_shares_no_index(open_client):
    client = open_client()
    _create(client)
    client.upsert(
        COLLECTION,
        [
            _point(1, [1.0, 0.0, 0.0, 0.0], {RARE: 1.0}),
            _point(2, [0.0, 1.0, 0.0, 0.0], {ELSEWHERE: 9.0}),
            _point(3, [0.0, 0.0, 1.0, 0.0], {}),  # an empty sparse vector
        ],
    )

    hits = _sparse_query(client, {RARE: 1.0, FREQUENT: 1.0})

    assert [hit.id for hit in hits] == [_id(1)]
    assert len(_dense_query(client, [1.0, 1.0, 1.0, 1.0])) == 3


def _idf_fixture(client: QdrantClient) -> None:
    """Six points. The frequent term is in five of them, the rare one in one.
    The frequent-term point holds its term at document value 2.0, the
    rare-term point its term at 1.0: by dot product alone the frequent-term
    point scores 2.0 against 1.0."""
    client.upsert(
        COLLECTION,
        [
            _point(1, [1.0, 0.0, 0.0, 0.0], {FREQUENT: 2.0}),  # the frequent-term point
            _point(2, [1.0, 0.0, 0.0, 0.0], {RARE: 1.0}),  # the rare-term point
            *[
                _point(number, [0.0, 1.0, 0.0, 0.0], {FREQUENT: 0.1, ELSEWHERE: 1.0})
                for number in range(3, 7)
            ],
        ],
    )


def test_with_idf_a_rare_query_term_outranks_a_frequent_one(open_client):
    client = open_client()
    _create(client)
    _idf_fixture(client)

    hits = _sparse_query(client, {FREQUENT: 1.0, RARE: 1.0})

    assert [hit.id for hit in hits[:2]] == [_id(2), _id(1)]
    assert len(hits) == 6


def test_without_idf_the_same_fixture_ranks_the_frequent_term_point_first(
    open_client,
):
    """The control for the test above: only the modifier differs."""
    client = open_client()
    _create(client, idf=False)
    _idf_fixture(client)

    hits = _sparse_query(client, {FREQUENT: 1.0, RARE: 1.0})

    assert [hit.id for hit in hits[:2]] == [_id(1), _id(2)]
    assert [hit.score for hit in hits[:2]] == pytest.approx([2.0, 1.0])


def test_a_closed_index_folder_can_be_renamed_and_reopened(tmp_path, open_client):
    client = open_client("staging")
    _create(client)
    _idf_fixture(client)
    before = sorted(
        (point.id, point.payload["number"])
        for point in client.scroll(COLLECTION, limit=100)[0]
    )
    client.close()

    (tmp_path / "staging").rename(tmp_path / "index")
    reopened = open_client("index")

    after = sorted(
        (point.id, point.payload["number"])
        for point in reopened.scroll(COLLECTION, limit=100)[0]
    )
    assert after == before
    assert len(after) == 6
    assert [hit.id for hit in _sparse_query(reopened, {RARE: 1.0})] == [_id(2)]


def test_a_second_client_on_a_held_folder_raises_a_recognisable_error(
    tmp_path, open_client
):
    """The lock error of local mode: a RuntimeError raised while portalocker's
    LockException is handled, with a fixed message."""
    open_client("index")

    with pytest.raises(RuntimeError) as raised:
        QdrantClient(path=str(tmp_path / "index"))

    assert "is already accessed by another instance of Qdrant client" in str(
        raised.value
    )
    context = type(raised.value.__context__)
    assert "LockException" in [cls.__name__ for cls in context.__mro__]
    assert store.is_lock_error(raised.value)


def test_store_tells_the_lock_error_apart_from_other_errors(tmp_path, open_client):
    open_client("index")
    not_a_folder = tmp_path / "bestand"
    not_a_folder.write_text("geen index", encoding="utf-8")

    with pytest.raises(store.IndexInUseError):
        store.open_client(tmp_path / "index")
    with pytest.raises(OSError) as other:
        store.open_client(not_a_folder)

    assert not isinstance(other.value, store.IndexInUseError)
    assert not store.is_lock_error(other.value)
    assert not store.is_lock_error(RuntimeError("something else went wrong"))
