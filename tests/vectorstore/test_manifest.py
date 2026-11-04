"""``manifest.json`` (spec 006): D16, D27, R10 (A10), and the D17 comparison
and reading (R11) at function level.

The input is written by hand (``hand_record``) so that it holds what A10
needs: two chunks tied on the largest token count, the colliding term pair,
a non-ASCII file name, and two paths whose forward-slash order differs from
``Path`` order.
"""

from __future__ import annotations

import copy
import hashlib
import json
from importlib import metadata
from pathlib import Path

import pytest

from companydocs_rag.chunking import cli as chunk_cli
from companydocs_rag.embeddings.dense import compose_embedding_text
from companydocs_rag.embeddings.sparse import sparse_terms
from companydocs_rag.ingestion.settings import load_settings
from companydocs_rag.vectorstore import manifest as manifest_module
from companydocs_rag.vectorstore.manifest import (
    compare_manifest,
    current_values,
    read_manifest,
)
from tests.embeddings._helpers import StubEmbedder, stub_token_count, write_settings
from tests.vectorstore._helpers import (
    hand_record,
    run_index,
    stored_points,
    write_chunk_file,
)

R_CARON = chr(0x0159)  # U+0159 LATIN SMALL LETTER R WITH CARON
FAILED_NAME = f"d/fictief-{R_CARON}.json"

TOP_KEYS = [
    "manifest_version",
    "embedding",
    "sparse",
    "collection",
    "inputs",
    "libraries",
]
EMBEDDING_KEYS = [
    "model_name",
    "model_revision",
    "dimension",
    "distance",
    "embedding_text_version",
    "max_tokens_limit",
    "max_tokens_measured",
    "max_tokens_chunk_id",
]
SPARSE_KEYS = [
    "tokenizer_version",
    "term_hash",
    "bm25_k1",
    "bm25_b",
    "avgdl",
    "chunk_count",
    "vocabulary_size",
    "term_id_collisions",
]
COLLECTION_KEYS = ["name", "dense_vector", "sparse_vector"]
INPUTS_KEYS = [
    "chunks_schema_version",
    "documents",
    "document_count",
    "documents_without_chunks",
]
DOCUMENT_KEYS = [
    "chunk_file",
    "chunk_file_sha256",
    "file_sha256",
    "status",
    "chunk_count",
]
LIBRARY_KEYS = ["qdrant_client", "sentence_transformers", "transformers", "torch"]


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    def use(settings):
        return write_settings(tmp_path, monkeypatch, settings)

    yield use
    load_settings.cache_clear()


def _a10_input(chunks: Path):
    """Three chunk files; returns their records by relative path. The first
    chunk of each processed file has ten words, the largest count."""
    records = {
        "a-b/x.json": hand_record(
            "een twee drie vier vijf zes zeven acht negen tien",
            ("Artikel 2 Huur", "2", "Artikel 2 Huur\nterm121635 huur"),
            sha256="d4" * 32,
            path="invoer/x.pdf",
        ),
        "a/y.json": hand_record(
            "elf twaalf dertien veertien vijftien zestien zeventien achttien "
            "negentien twintig",
            "term172646 pand 8.9",
            sha256="e5" * 32,
            path="invoer/y.pdf",
        ),
        FAILED_NAME: hand_record(sha256=None, path="invoer/z.pdf", status="failed"),
    }
    for relative, record in records.items():
        write_chunk_file(chunks, relative, record)
    return records


def test_the_manifest_holds_the_d16_keys_in_order_with_their_values(
    tmp_path, settings_file
):
    settings_file(
        {"embeddings": {"model_name": "fictief/model", "model_revision": "abc"}}
    )
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    records = _a10_input(chunks)
    stub = StubEmbedder(dimension=5, max_seq_length=300)

    assert run_index(chunks, index, embedder=stub) == 0

    text = (index / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(text)
    assert text == json.dumps(manifest, indent=2, ensure_ascii=False)
    assert R_CARON in text
    chunk_cli._write_json(tmp_path / "zo-schrijft-chunk.json", manifest)
    assert (index / "manifest.json").read_bytes() == (
        tmp_path / "zo-schrijft-chunk.json"
    ).read_bytes()

    assert list(manifest) == TOP_KEYS
    assert list(manifest["embedding"]) == EMBEDDING_KEYS
    assert list(manifest["sparse"]) == SPARSE_KEYS
    assert list(manifest["collection"]) == COLLECTION_KEYS
    assert list(manifest["inputs"]) == INPUTS_KEYS
    assert [list(document) for document in manifest["inputs"]["documents"]] == [
        DOCUMENT_KEYS
    ] * 3
    assert list(manifest["libraries"]) == LIBRARY_KEYS

    # R10: forward-slash order, which here differs from Path order.
    order = ["a-b/x.json", "a/y.json", FAILED_NAME]
    assert sorted(records) == order
    assert sorted(Path(name) for name in order[:2])[0] == Path("a/y.json")
    chunk_list = [chunk for name in order for chunk in records[name]["chunks"]]
    composed = [compose_embedding_text(chunk) for chunk in chunk_list]
    counts = [stub_token_count(text) for text in composed]
    assert counts[0] == counts[2] == max(counts) == 12  # a tie: the first wins
    terms = [sparse_terms(text) for text in composed]

    assert manifest["manifest_version"] == 1
    assert manifest["embedding"] == {
        "model_name": "fictief/model",
        "model_revision": "abc",
        "dimension": 5,
        "distance": "Cosine",
        "embedding_text_version": 1,
        "max_tokens_limit": 300,
        "max_tokens_measured": 12,
        "max_tokens_chunk_id": chunk_list[0]["chunk_id"],
    }
    assert manifest["sparse"] == {
        "tokenizer_version": 1,
        "term_hash": "sha256-u32",
        "bm25_k1": 1.2,
        "bm25_b": 0.75,
        "avgdl": sum(len(found) for found in terms) / len(terms),
        "chunk_count": 4,
        "vocabulary_size": len({term for found in terms for term in found}),
        "term_id_collisions": 2,
    }
    assert manifest["collection"] == {
        "name": "chunks",
        "dense_vector": "dense",
        "sparse_vector": "sparse",
    }
    assert manifest["inputs"] == {
        "chunks_schema_version": 1,
        "documents": [
            {
                "chunk_file": name,
                "chunk_file_sha256": hashlib.sha256(
                    (chunks / name).read_bytes()
                ).hexdigest(),
                "file_sha256": records[name]["file"]["sha256"],
                "status": records[name]["status"],
                "chunk_count": records[name]["chunk_count"],
            }
            for name in order
        ],
        "document_count": 3,
        "documents_without_chunks": 1,
    }
    assert manifest["libraries"] == {
        "qdrant_client": metadata.version("qdrant-client"),
        "sentence_transformers": metadata.version("sentence-transformers"),
        "transformers": metadata.version("transformers"),
        "torch": metadata.version("torch"),
    }


def _points(index: Path):
    return {
        point_id: (
            point.payload,
            point.vector["sparse"].indices,
            point.vector["sparse"].values,
            point.vector["dense"],
        )
        for point_id, point in stored_points(index).items()
    }


def test_two_runs_over_the_same_input_write_the_same_index(tmp_path):
    """D27: the manifest byte for byte; point ids, payloads, sparse and, with
    the stub, dense vectors identical."""
    chunks = tmp_path / "chunks"
    _a10_input(chunks)

    assert run_index(chunks, tmp_path / "index-1") == 0
    first_manifest = (tmp_path / "index-1" / "manifest.json").read_bytes()
    first_points = _points(tmp_path / "index-1")
    assert run_index(chunks, tmp_path / "index-1") == 0  # a rebuild in place
    assert run_index(chunks, tmp_path / "index-2") == 0

    for index in ("index-1", "index-2"):
        assert (tmp_path / index / "manifest.json").read_bytes() == first_manifest
        assert _points(tmp_path / index) == first_points
    assert len(first_points) == 4


# --- The D17 comparison (R11) ---------------------------------------------------------

CURRENT = current_values(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    bm25_k1=1.2,
    bm25_b=0.75,
)
MATCHING = {
    "manifest_version": 1,
    "embedding": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "model_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "embedding_text_version": 1,
    },
    "sparse": {
        "tokenizer_version": 1,
        "term_hash": "sha256-u32",
        "bm25_k1": 1.2,
        "bm25_b": 0.75,
    },
}
# Each D17 key: where it sits, another value, and how the message shows both.
ALTERED = [
    ("manifest_version", (), 2, "2", "1"),
    (
        "model_name",
        ("embedding",),
        "fictief/model",
        '"fictief/model"',
        '"Qwen/Qwen3-Embedding-0.6B"',
    ),
    (
        "model_revision",
        ("embedding",),
        "0" * 40,
        '"' + "0" * 40 + '"',
        '"97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"',
    ),
    ("embedding_text_version", ("embedding",), 2, "2", "1"),
    ("tokenizer_version", ("sparse",), 2, "2", "1"),
    ("term_hash", ("sparse",), "md5-u32", '"md5-u32"', '"sha256-u32"'),
    ("bm25_k1", ("sparse",), 1.5, "1.5", "1.2"),
    ("bm25_b", ("sparse",), 0.5, "0.5", "0.75"),
]


def _set(manifest, where, key, value):
    node = manifest
    for name in where:
        node = node[name]
    node[key] = value


def test_a_matching_manifest_gives_no_message():
    assert compare_manifest(MATCHING, CURRENT) == []


@pytest.mark.parametrize(
    ("key", "where", "value", "shown", "now"),
    ALTERED,
    ids=[case[0] for case in ALTERED],
)
def test_one_differing_key_gives_one_message(key, where, value, shown, now):
    manifest = copy.deepcopy(MATCHING)
    _set(manifest, where, key, value)

    assert compare_manifest(manifest, CURRENT) == [
        f"Index does not match settings: {key} is {shown} in the index, {now} now"
    ]


def test_every_key_differing_gives_eight_messages_in_d17_order():
    manifest = copy.deepcopy(MATCHING)
    for key, where, value, _, _ in ALTERED:
        _set(manifest, where, key, value)

    found = compare_manifest(manifest, CURRENT)

    assert [message.split(" is ")[0] for message in found] == [
        f"Index does not match settings: {key}" for key, *_ in ALTERED
    ]


def test_an_absent_key_reads_as_missing():
    manifest = copy.deepcopy(MATCHING)
    del manifest["sparse"]["term_hash"]
    manifest["embedding"] = ["geen object"]

    assert compare_manifest(manifest, CURRENT) == [
        "Index does not match settings: model_name is missing in the index, "
        '"Qwen/Qwen3-Embedding-0.6B" now',
        "Index does not match settings: model_revision is missing in the index, "
        '"97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3" now',
        "Index does not match settings: embedding_text_version is missing in the "
        "index, 1 now",
        "Index does not match settings: term_hash is missing in the index, "
        '"sha256-u32" now',
    ]


def test_numbers_compare_by_value_but_not_with_strings_or_bools():
    manifest = copy.deepcopy(MATCHING)
    manifest["sparse"]["tokenizer_version"] = 1.0
    manifest["sparse"]["bm25_b"] = 0.75
    assert (
        compare_manifest(
            manifest,
            current_values(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                model_revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                bm25_k1=1.2,
                bm25_b=0.75,
            ),
        )
        == []
    )

    manifest["manifest_version"] = "1"
    manifest["embedding"]["embedding_text_version"] = True
    assert compare_manifest(manifest, CURRENT) == [
        'Index does not match settings: manifest_version is "1" in the index, 1 now',
        "Index does not match settings: embedding_text_version is true in the "
        "index, 1 now",
    ]


def test_an_integer_k1_in_the_settings_matches_its_float_in_the_manifest():
    current = current_values(
        model_name=MATCHING["embedding"]["model_name"],
        model_revision=MATCHING["embedding"]["model_revision"],
        bm25_k1=2,
        bm25_b=1,
    )
    manifest = copy.deepcopy(MATCHING)
    manifest["sparse"]["bm25_k1"] = 2.0
    manifest["sparse"]["bm25_b"] = 1.0

    assert compare_manifest(manifest, current) == []


# --- Reading --------------------------------------------------------------------------


def test_reading_a_manifest(tmp_path):
    path = tmp_path / "manifest.json"

    with pytest.raises(FileNotFoundError):
        read_manifest(path)
    path.write_text("{geen json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_manifest(path)
    path.write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="the top level is not a JSON object"):
        read_manifest(path)
    path.write_text(json.dumps(MATCHING), encoding="utf-8")
    assert read_manifest(path) == MATCHING


def test_the_manifest_constants():
    assert manifest_module.MANIFEST_VERSION == 1
    assert manifest_module.MANIFEST_FILE_NAME == "manifest.json"
    assert manifest_module.STAGING_SUFFIX == ".building"
    assert manifest_module.PREVIOUS_SUFFIX == ".previous"
    assert manifest_module.REQUIRED_CHUNKS_SCHEMA_VERSION == 1
