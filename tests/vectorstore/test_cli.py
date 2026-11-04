"""The ``companydocs-index`` command (spec 006): A2, A7, A8, the command's
part of A15 and of A16.

Chunk files are fabricated in ``tmp_path``, most of them written by
``chunk_document`` (``_helpers.py``). The stub embedder is injected through
``load_embedder``; nothing under ``data/`` is read or written, and every
Qdrant client is closed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from companydocs_rag.embeddings.dense import compose_embedding_text
from companydocs_rag.embeddings.sparse import document_sparse_vector, sparse_terms
from companydocs_rag.ingestion.settings import find_project_root, load_settings
from companydocs_rag.vectorstore import cli
from companydocs_rag.vectorstore.store import chunk_id_for, point_id_for
from tests.embeddings._helpers import (
    INVALID_SETTINGS,
    StubEmbedder,
    StubLoader,
    install_fake_sentence_transformers,
    invalid_settings_id,
    stub_token_count,
    write_settings,
)
from tests.vectorstore._helpers import (
    LAST_ERROR,
    LOGGER,
    PAYLOAD_KEYS,
    chunk_record,
    messages,
    run_index,
    stored_points,
    structured_record,
    three_documents,
    tree,
    write_chunk_file,
)

_REPOSITORY = Path(__file__).resolve().parents[2]
_DELETE = object()


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """Point the settings at a YAML file written here, for one test."""

    def use(settings):
        return write_settings(tmp_path, monkeypatch, settings)

    yield use
    load_settings.cache_clear()


def _f32(value: float) -> float:
    """``value`` at the float32 precision Qdrant stores."""
    return struct.unpack("f", struct.pack("f", value))[0]


def _input_chunks(records):
    """The chunks of ``records``, keyed by relative path, in input order."""
    return [chunk for name in sorted(records) for chunk in records[name]["chunks"]]


def _read_manifest(index: Path):
    return json.loads((index / "manifest.json").read_text(encoding="utf-8"))


# --- A run over chunk files (A7) ------------------------------------------------------


def test_a_run_indexes_every_chunk_of_nested_chunk_files(tmp_path, caplog):
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    records = three_documents(chunks)
    loader = StubLoader()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert run_index(chunks, index, loader=loader) == 0

    expected = _input_chunks(records)
    points = stored_points(index)
    assert len(points) == len(expected) == 6
    assert sorted(points) == sorted(
        point_id_for(chunk["chunk_id"]) for chunk in expected
    )
    for point_id in points:
        assert point_id.endswith("-000000000000")
        assert point_id == point_id_for(chunk_id_for(point_id))

    document_of = {
        chunk["chunk_id"]: record
        for record in records.values()
        for chunk in record["chunks"]
    }
    by_chunk = {chunk_id_for(point_id): point for point_id, point in points.items()}
    for chunk in expected:
        payload = by_chunk[chunk["chunk_id"]].payload
        document = document_of[chunk["chunk_id"]]
        assert list(payload) == PAYLOAD_KEYS
        assert payload == {
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "file_sha256": chunk["file_sha256"],
            "file_path": document["file"]["path"],
            "language": document["language"],
            "language_source": document["language_source"],
            "route": document["route"],
            "text": chunk["text"],
            **{key: chunk[key] for key in PAYLOAD_KEYS[8:]},
        }
    assert {payload.payload["route"] for payload in points.values()} == {
        "structured",
        "generic",
    }

    composed = [compose_embedding_text(chunk) for chunk in expected]
    assert loader.embedder.documents == composed
    assert loader.embedder.query_calls == []
    assert len(loader.calls) == 1

    lengths = [len(sparse_terms(text)) for text in composed]
    avgdl = sum(lengths) / len(lengths)
    manifest = _read_manifest(index)
    assert manifest["sparse"]["avgdl"] == avgdl
    for chunk, text in zip(expected, composed):
        stored = by_chunk[chunk["chunk_id"]].vector["sparse"]
        wanted = document_sparse_vector(text, avgdl=avgdl, k1=1.2, b=0.75)
        assert stored.indices == wanted.indices
        assert [_f32(value) for value in stored.values] == [
            _f32(value) for value in wanted.values
        ]

    failed_bytes = (chunks / "b" / "fictief-mislukt.json").read_bytes()
    assert [
        document
        for document in manifest["inputs"]["documents"]
        if document["status"] == "failed"
    ] == [
        {
            "chunk_file": "b/fictief-mislukt.json",
            "chunk_file_sha256": hashlib.sha256(failed_bytes).hexdigest(),
            "file_sha256": None,
            "status": "failed",
            "chunk_count": 0,
        }
    ]
    assert manifest["inputs"]["documents_without_chunks"] == 1

    vocabulary = {term for text in composed for term in sparse_terms(text)}
    largest = max(stub_token_count(text) for text in composed)
    assert messages(caplog, logging.INFO) == [
        "Summary: documents=3, without_chunks=1, chunks=6",
        f"Summary: max_tokens={largest} (limit 512), vocabulary={len(vocabulary)}, "
        f"term_id_collisions=0, avgdl={json.dumps(avgdl)}",
    ]
    assert messages(caplog, logging.ERROR) == []


def test_chunk_files_are_unchanged_and_no_composed_text_is_stored(tmp_path):
    """A2: the composed text differs from the chunk's text only where the chunk
    has a clause label, so those are the texts that must appear nowhere."""
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    records = three_documents(chunks)
    before = tree(tmp_path)

    assert run_index(chunks, index) == 0

    after = tree(tmp_path)
    outside = {
        name: data
        for name, data in after.items()
        if name != "index" and not name.startswith("index/")
    }
    assert outside == before
    composed = [
        compose_embedding_text(chunk)
        for chunk in _input_chunks(records)
        if chunk["clause_label"] is not None
    ]
    assert len(composed) == 3
    for point in stored_points(index).values():
        for text in composed:
            assert text not in point.payload.values()


def test_file_names_starting_with_a_dot_are_skipped(tmp_path):
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    write_chunk_file(chunks, "fictief.json", structured_record())
    write_chunk_file(chunks, ".verborgen.json", "geen json")
    write_chunk_file(chunks, "notitie.txt", "geen chunkbestand")

    assert run_index(chunks, index) == 0
    assert [
        document["chunk_file"]
        for document in _read_manifest(index)["inputs"]["documents"]
    ] == ["fictief.json"]


def test_chunk_files_without_chunks_give_an_empty_index(tmp_path, caplog):
    """R5: avgdl and the token maximum are null; the model is still loaded."""
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    write_chunk_file(
        chunks,
        "mislukt.json",
        chunk_record(path="invoer/fictief.pdf", sha256=None, status="failed"),
    )
    loader = StubLoader()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert run_index(chunks, index, loader=loader) == 0

    manifest = _read_manifest(index)
    assert manifest["sparse"]["avgdl"] is None
    assert manifest["sparse"]["chunk_count"] == 0
    assert manifest["embedding"]["max_tokens_measured"] is None
    assert manifest["embedding"]["max_tokens_chunk_id"] is None
    assert stored_points(index) == {}
    assert loader.embedder.document_calls == []
    assert messages(caplog, logging.INFO) == [
        "Summary: documents=1, without_chunks=1, chunks=0",
        "Summary: max_tokens=null (limit 512), vocabulary=0, "
        "term_id_collisions=0, avgdl=null",
    ]


# --- Refused runs (A8) ----------------------------------------------------------------


@pytest.fixture
def previous_index(tmp_path):
    """A valid index in ``tmp_path/index``, built from its own chunk files.
    Returns its tree."""
    source = tmp_path / "eerder"
    write_chunk_file(
        source,
        "eerder.json",
        chunk_record("Een eerdere fictieve tekst.", sha256="c3" * 32),
    )
    assert run_index(source, tmp_path / "index") == 0
    return tree(tmp_path / "index")


def _altered(path, value):
    """A valid structured record with the key at ``path`` set to ``value``,
    or removed for :data:`_DELETE`."""
    record = structured_record()
    node = record
    for key in path[:-1]:
        node = node[key]
    if value is _DELETE:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return record


def _decode_error(text):
    try:
        json.loads(text)
    except json.JSONDecodeError as error:
        return f"JSONDecodeError: {error}"
    raise AssertionError("valid JSON")


# One invalid record per R6 rule: (path, value, the ChunkRecordError message).
R6_CASES = [
    (("file",), "invoer/fictief.pdf", "'file' is not an object"),
    (("file", "path"), None, "'file.path' is not a string"),
    (
        ("file", "sha256"),
        None,
        "'file.sha256' is not a string, and only a failed document may have none",
    ),
    (("file", "sha256"), _DELETE, "'file' lacks the key 'sha256'"),
    (("status",), "skipped", "'status' is \"skipped\", not processed or failed"),
    (("language",), 5, "'language' is not a string or null"),
    (("language_source",), ["index"], "'language_source' is not a string or null"),
    (("route",), "mixed", "'route' is \"mixed\", not structured or generic"),
    (("route",), _DELETE, "the record lacks the key 'route'"),
    (("chunks",), {}, "'chunks' is not a list"),
    (("status",), "failed", "'chunks' is not empty, and a failed document has none"),
    (("chunk_count",), True, "'chunk_count' is not an integer"),
    (("chunk_count",), 2, "'chunk_count' is 2, not the number of chunks (3)"),
    (("chunks", 0), "tekst", "chunks[0] is not an object"),
    (
        ("chunks", 1, "chunk_id"),
        "ABCDEF0123456789ABCD",
        "chunks[1].chunk_id is not 20 lowercase hex characters",
    ),
    (
        ("chunks", 1, "chunk_id"),
        "abcdef0123456789abc",
        "chunks[1].chunk_id is not 20 lowercase hex characters",
    ),
    (("chunks", 0, "chunk_index"), True, "chunks[0].chunk_index is not an integer"),
    (("chunks", 0, "page_start"), "1", "chunks[0].page_start is not an integer"),
    (("chunks", 0, "page_end"), 1.0, "chunks[0].page_end is not an integer"),
    (
        ("chunks", 0, "file_sha256"),
        "ef" * 32,
        "chunks[0].file_sha256 is not the document's file.sha256",
    ),
    (("chunks", 0, "text"), "", "chunks[0].text is not a non-empty string"),
    (("chunks", 0, "text"), None, "chunks[0].text is not a non-empty string"),
    (("chunks", 0, "text"), _DELETE, "chunks[0] lacks the key 'text'"),
    (
        ("chunks", 0, "text_origins"),
        ["native", 1],
        "chunks[0].text_origins is not a list of strings",
    ),
    (
        ("chunks", 0, "text_origins"),
        "native",
        "chunks[0].text_origins is not a list of strings",
    ),
    (("chunks", 0, "page_spans"), {}, "chunks[0].page_spans is not a list"),
    (("chunks", 0, "chunk_method"), None, "chunks[0].chunk_method is not a string"),
    (
        ("chunks", 0, "clause_label"),
        1,
        "chunks[0].clause_label is not a string or null",
    ),
    (
        ("chunks", 0, "clause_number"),
        2,
        "chunks[0].clause_number is not a string or null",
    ),
    (("chunks", 0, "part"), "1", "chunks[0].part is not an integer or null"),
    (
        ("chunks", 0, "part_count"),
        True,
        "chunks[0].part_count is not an integer or null",
    ),
]


def _refused_cases():
    """(id, the bad file's content, the ERRORs before the last one)."""
    cases = []
    for value, shown in [
        (2, "2"),
        (1.0, "1.0"),
        ("1", '"1"'),
        (True, "true"),
        (None, "null"),
        (_DELETE, "missing"),
    ]:
        cases.append(
            (
                f"schema-{shown}",
                json.dumps(_altered(("schema_version",), value)),
                [f"Refused bad.json: schema_version is {shown}, expected 1"],
            )
        )
    cases.append(
        (
            "not-json",
            "{geen json",
            [f"Could not read bad.json: {_decode_error('{geen json')}"],
        )
    )
    cases.append(
        (
            "not-an-object",
            "[1, 2]",
            [
                "Could not read bad.json: ChunkRecordError: "
                "the top level is not a JSON object"
            ],
        )
    )
    for position, (path, value, message) in enumerate(R6_CASES):
        cases.append(
            (
                f"r6-{position}-{'.'.join(map(str, path))}",
                json.dumps(_altered(path, value)),
                [f"Could not read bad.json: ChunkRecordError: {message}"],
            )
        )
    return cases


REFUSED_CASES = _refused_cases()


def _assert_refused(tmp_path, caplog, code, expected, *, previous):
    errors = messages(caplog, logging.ERROR)
    assert code == 1
    assert errors == [*expected, LAST_ERROR]
    assert not (tmp_path / "index.building").exists()
    if previous is None:
        assert not (tmp_path / "index").exists()
    else:
        assert tree(tmp_path / "index") == previous
    assert messages(caplog, logging.INFO) == []


@pytest.mark.parametrize(
    ("content", "expected"),
    [case[1:] for case in REFUSED_CASES],
    ids=[case[0] for case in REFUSED_CASES],
)
def test_a_refused_or_invalid_file_writes_no_index(
    tmp_path, caplog, previous_index, content, expected
):
    chunks = tmp_path / "chunks"
    write_chunk_file(chunks, "a/goed.json", structured_record())
    write_chunk_file(chunks, "bad.json", content)
    loader = StubLoader()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, tmp_path / "index", loader=loader)

    _assert_refused(tmp_path, caplog, code, expected, previous=previous_index)
    assert loader.calls == []


def test_a_file_that_is_not_utf8_cannot_be_read(tmp_path, caplog):
    chunks = tmp_path / "chunks"
    (chunks).mkdir()
    (chunks / "bad.json").write_bytes(b"\xff\xfe{}")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, tmp_path / "index")

    (first, last) = messages(caplog, logging.ERROR)
    assert first.startswith("Could not read bad.json: UnicodeDecodeError: ")
    assert (code, last) == (1, LAST_ERROR)
    assert not (tmp_path / "index").exists()


def test_two_bad_files_are_both_reported_and_without_a_previous_index_none_is_made(
    tmp_path, caplog
):
    chunks = tmp_path / "chunks"
    write_chunk_file(chunks, "a.json", "{geen json")
    write_chunk_file(chunks, "b/c.json", json.dumps(_altered(("schema_version",), 2)))
    write_chunk_file(chunks, "d.json", structured_record())
    loader = StubLoader()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, tmp_path / "index", loader=loader)

    _assert_refused(
        tmp_path,
        caplog,
        code,
        [
            f"Could not read a.json: {_decode_error('{geen json')}",
            f"Refused {Path('b', 'c.json')}: schema_version is 2, expected 1",
        ],
        previous=None,
    )
    assert loader.calls == []


def test_one_chunk_id_in_two_files_is_a_duplicate(tmp_path, caplog, previous_index):
    chunks = tmp_path / "chunks"
    record = structured_record()
    write_chunk_file(chunks, "a.json", record)
    write_chunk_file(chunks, "b/kopie.json", record)
    loader = StubLoader()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, tmp_path / "index", loader=loader)

    _assert_refused(
        tmp_path,
        caplog,
        code,
        [
            f"Duplicate chunk_id {chunk['chunk_id']} in a.json and "
            f"{Path('b', 'kopie.json')}"
            for chunk in record["chunks"]
        ],
        previous=previous_index,
    )
    assert loader.calls == []


def test_one_chunk_id_twice_in_one_file_is_a_duplicate(tmp_path, caplog):
    chunks = tmp_path / "chunks"
    record = structured_record()
    chunk_id = record["chunks"][0]["chunk_id"]
    record["chunks"][2]["chunk_id"] = chunk_id
    write_chunk_file(chunks, "a.json", record)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, tmp_path / "index")

    _assert_refused(
        tmp_path,
        caplog,
        code,
        [f"Duplicate chunk_id {chunk_id} in a.json and a.json"],
        previous=None,
    )


def _over_the_limit(record, limit):
    return [
        f"Embedding text of chunk {chunk['chunk_id']} has "
        f"{stub_token_count(compose_embedding_text(chunk))} tokens, limit {limit}"
        for chunk in record["chunks"]
        if stub_token_count(compose_embedding_text(chunk)) > limit
    ]


def test_a_text_over_max_tokens_writes_no_index(
    tmp_path, caplog, previous_index, settings_file
):
    settings_file({"embeddings": {"max_tokens": 75}})
    chunks = tmp_path / "chunks"
    record = structured_record()
    write_chunk_file(chunks, "a.json", record)
    loader = StubLoader()
    expected = _over_the_limit(record, 75)
    assert len(expected) == 1  # the fixture: one chunk of three is over

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, tmp_path / "index", loader=loader)

    _assert_refused(tmp_path, caplog, code, expected, previous=previous_index)
    assert len(loader.calls) == 1


def test_a_text_over_the_models_max_seq_length_writes_no_index(
    tmp_path, caplog, previous_index
):
    chunks = tmp_path / "chunks"
    record = structured_record()
    write_chunk_file(chunks, "a.json", record)
    loader = StubLoader(StubEmbedder(max_seq_length=75))

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, tmp_path / "index", loader=loader)

    _assert_refused(
        tmp_path, caplog, code, _over_the_limit(record, 75), previous=previous_index
    )
    assert len(loader.calls) == 1


def test_a_chunks_folder_without_chunk_files_writes_no_index(tmp_path, caplog):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "leeg.txt").write_text("geen chunkbestand", encoding="utf-8")
    loader = StubLoader()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, tmp_path / "index", loader=loader)

    _assert_refused(
        tmp_path, caplog, code, [f"No chunk files found in {chunks}"], previous=None
    )
    assert loader.calls == []


def test_a_folder_that_is_not_an_index_is_not_replaced(tmp_path, caplog):
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    three_documents(chunks)
    write_chunk_file(index, "notities.txt", "Eigen aantekeningen.")
    write_chunk_file(index, "map/meer.json", "{}")
    before = tree(index)
    loader = StubLoader()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, index, loader=loader)

    _assert_refused(
        tmp_path,
        caplog,
        code,
        [f"Not an index, not replaced: {index}"],
        previous=before,
    )
    assert loader.calls == []


def test_a_model_that_cannot_be_loaded_writes_no_index(
    tmp_path, caplog, previous_index, monkeypatch
):
    """Through the command's own loader, with a fake sentence-transformers
    whose constructor fails (Q3, R7)."""
    fake = install_fake_sentence_transformers(monkeypatch)
    fake.fail_with = OSError("geen netwerk")
    chunks = tmp_path / "chunks"
    three_documents(chunks)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = cli.main(
            ["--chunks-dir", str(chunks), "--index-dir", str(tmp_path / "index")]
        )

    _assert_refused(
        tmp_path,
        caplog,
        code,
        [
            "Could not load embedding model Qwen/Qwen3-Embedding-0.6B"
            "@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3: OSError: geen netwerk"
        ],
        previous=previous_index,
    )
    assert [attempt["local_files_only"] for attempt in fake.attempts] == [False]


def test_an_empty_folder_is_replaced(tmp_path):
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    three_documents(chunks)
    index.mkdir()

    assert run_index(chunks, index) == 0
    assert (index / "manifest.json").is_file()


# --- Cannot start (A15, R12) ----------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "values", "key"),
    INVALID_SETTINGS,
    ids=[invalid_settings_id(case) for case in INVALID_SETTINGS],
)
def test_invalid_settings_stop_the_command_before_it_reads_anything(
    tmp_path, caplog, capsys, settings_file, section, values, key
):
    settings_file({section: values})
    chunks = tmp_path / "chunks"
    three_documents(chunks)
    before = tree(tmp_path)
    loader = StubLoader()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        code = run_index(chunks, tmp_path / "index", loader=loader)

    assert code == 2
    (error,) = messages(caplog, logging.ERROR)
    assert error.startswith(f"Invalid settings: {section}.{key}: ")
    assert tree(tmp_path) == before
    assert loader.calls == []
    assert capsys.readouterr().out == ""


def test_a_missing_chunks_folder_exits_2_before_the_settings(
    tmp_path, caplog, settings_file
):
    settings_file({"retrieval": {"rrf_k": 0}})
    missing = tmp_path / "bestaat-niet"
    a_file = tmp_path / "bestand.json"
    a_file.write_text("{}", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert run_index(missing, tmp_path / "index") == 2
        assert run_index(a_file, tmp_path / "index") == 2

    assert messages(caplog, logging.ERROR) == [
        f"Not a folder: {missing}",
        f"Not a folder: {a_file}",
    ]
    assert not (tmp_path / "index").exists()


def test_settings_are_read_at_call_time(tmp_path, settings_file):
    chunks = tmp_path / "chunks"
    three_documents(chunks)
    settings_file({"retrieval": {"bm25_k1": 2}, "embeddings": {"max_tokens": 300}})

    assert run_index(chunks, tmp_path / "index") == 0

    manifest = _read_manifest(tmp_path / "index")
    assert manifest["sparse"]["bm25_k1"] == 2.0
    assert isinstance(manifest["sparse"]["bm25_k1"], float)
    assert manifest["embedding"]["max_tokens_limit"] == 300


# --- The command (A16) ----------------------------------------------------------------


def test_the_parser_has_the_two_folders_quiet_and_help_and_nothing_else():
    parser = cli.build_parser()

    options = sorted(
        option for action in parser._actions for option in action.option_strings
    )
    assert options == [
        "--chunks-dir",
        "--help",
        "--index-dir",
        "--quiet",
        "-h",
        "-q",
    ]
    assert [action for action in parser._actions if not action.option_strings] == []


def test_the_default_folders_are_under_the_project_root(monkeypatch, tmp_path):
    root = find_project_root()
    monkeypatch.chdir(tmp_path)

    args = cli.build_parser().parse_args([])

    assert args.chunks_dir == root / "data" / "interim" / "chunks"
    assert args.index_dir == root / "data" / "processed" / "index"
    assert cli.DEFAULT_INDEX_RELATIVE_PATH == Path("data", "processed", "index")
    assert (cli.EXIT_INDEX_WRITTEN, cli.EXIT_INDEX_NOT_WRITTEN) == (0, 1)
    assert cli.EXIT_CANNOT_START == 2


def test_the_log_lines_and_quiet(tmp_path):
    """Run in a fresh interpreter, where the logging set-up is the command's
    own: "LEVEL message" on stderr, nothing on stdout."""
    chunks = tmp_path / "chunks"
    three_documents(chunks)

    def run(index, *extra):
        code = (
            "import sys\n"
            "from tests.embeddings._helpers import StubLoader\n"
            "from companydocs_rag.vectorstore import cli\n"
            f"sys.exit(cli.main(['--chunks-dir', {str(chunks)!r}, '--index-dir', "
            f"{str(index)!r}, *{list(extra)!r}], load_embedder=StubLoader()))\n"
        )
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=_REPOSITORY,
        )

    loud = run(tmp_path / "index-1")
    quiet = run(tmp_path / "index-2", "-q")

    assert (loud.returncode, quiet.returncode) == (0, 0)
    lines = loud.stderr.splitlines()
    assert len(lines) == 2
    assert lines[0] == "INFO Summary: documents=3, without_chunks=1, chunks=6"
    assert lines[1].startswith("INFO Summary: max_tokens=")
    assert (loud.stdout, quiet.stdout, quiet.stderr) == ("", "", "")
