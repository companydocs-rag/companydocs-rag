"""Point ids (spec 006, D11, R19) and the store's names (D10)."""

from __future__ import annotations

import uuid

import pytest

from companydocs_rag.vectorstore import store
from companydocs_rag.vectorstore.store import chunk_id_for, point_id_for

CHUNK_ID = "0123456789abcdef0123"


def test_the_point_id_is_the_padded_chunk_id_as_a_canonical_uuid():
    point_id = point_id_for(CHUNK_ID)

    assert point_id == "01234567-89ab-cdef-0123-000000000000"
    assert point_id == str(uuid.UUID(point_id))
    assert point_id.endswith("0" * 12)


def test_chunk_id_for_gives_the_chunk_id_back():
    assert chunk_id_for(point_id_for(CHUNK_ID)) == CHUNK_ID
    assert chunk_id_for(uuid.UUID(point_id_for(CHUNK_ID))) == CHUNK_ID


@pytest.mark.parametrize(
    "point_id",
    [str(uuid.UUID(int=1)), "01234567-89ab-cdef-0123-00000000000f"],
)
def test_chunk_id_for_refuses_a_uuid_without_the_twelve_zeros(point_id):
    with pytest.raises(ValueError):
        chunk_id_for(point_id)


def test_the_store_names():
    assert store.COLLECTION_NAME == "chunks"
    assert store.DENSE_VECTOR_NAME == "dense"
    assert store.SPARSE_VECTOR_NAME == "sparse"
