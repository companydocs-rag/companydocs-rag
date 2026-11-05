"""The ``companydocs-search`` command (spec 006): A11, A13 and the command's
part of A15.

Each index is built in ``tmp_path`` with the stub embedder; the searches get
the stub injected, except where a fake sentence-transformers stands in for a
model that fails to load. Nothing under ``data/`` is read or written.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from companydocs_rag.ingestion.settings import find_project_root, load_settings
from companydocs_rag.retrieval import cli
from companydocs_rag.retrieval.search import IndexUnusableError, Retriever
from tests.embeddings._helpers import (
    INVALID_SETTINGS,
    StubEmbedder,
    install_fake_sentence_transformers,
    invalid_settings_id,
    write_settings,
)
from tests.vectorstore._helpers import (
    chunk_record,
    hand_record,
    run_index,
    three_documents,
    write_chunk_file,
)

LOGGER = "companydocs_rag.retrieval.cli"
_REPOSITORY = Path(__file__).resolve().parents[2]

TOP_KEYS = [
    "query",
    "query_terms",
    "query_instruction",
    "index",
    "settings",
    "leg_counts",
    "candidates",
]
INDEX_KEYS = [
    "model_name",
    "model_revision",
    "embedding_text_version",
    "tokenizer_version",
]
SETTINGS_KEYS = ["rrf_k", "dense_limit", "sparse_limit", "candidate_count"]
LEG_COUNT_KEYS = ["dense", "sparse"]
CANDIDATE_KEYS = [
    "rank",
    "chunk_id",
    "rrf_score",
    "dense_rank",
    "dense_score",
    "sparse_rank",
    "sparse_score",
    "sparse_matched_terms",
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
]


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    def use(settings):
        return write_settings(tmp_path, monkeypatch, settings)

    yield use
    load_settings.cache_clear()


@pytest.fixture
def index(tmp_path):
    """An index of A7's three documents: six chunks, three with a label."""
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    three_documents(chunks)
    assert run_index(chunks, index) == 0
    return index


def _run(index, *args, embedder="stub"):
    stub = StubEmbedder() if embedder == "stub" else embedder
    return cli.main([*args, "--index-dir", str(index)], embedder=stub)


def _errors(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.ERROR and record.name == LOGGER
    ]


def _json(capsys):
    out = capsys.readouterr().out
    result = json.loads(out)
    assert out == json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    return result


# --- The index cannot be used (A11) ---------------------------------------------------


def _assert_unusable(index, caplog, capsys, expected, *, embedder="stub"):
    """The command exits 1, logs ``expected`` and prints nothing; Retriever
    raises with the same reasons."""
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        code = _run(index, "huur", embedder=embedder)

    assert code == 1
    assert _errors(caplog) == expected
    assert capsys.readouterr().out == ""
    stub = StubEmbedder() if embedder == "stub" else embedder
    with pytest.raises(IndexUnusableError) as raised:
        Retriever(index, embedder=stub)
    assert raised.value.reasons == expected


def test_a_missing_manifest(index, caplog, capsys):
    (index / "manifest.json").unlink()

    _assert_unusable(
        index, caplog, capsys, [f"No index at {index}: manifest.json missing"]
    )


@pytest.mark.parametrize("content", ["{geen json", "[1, 2]"], ids=["not-json", "list"])
def test_a_manifest_that_is_not_a_json_object(index, caplog, capsys, content):
    path = index / "manifest.json"
    path.write_text(content, encoding="utf-8")
    try:
        found = json.loads(content)
        assert not isinstance(found, dict)
        detail = "ValueError: the top level is not a JSON object"
    except json.JSONDecodeError as error:
        detail = f"JSONDecodeError: {error}"

    _assert_unusable(index, caplog, capsys, [f"Could not read {path}: {detail}"])


# Each D17 key: where it sits in the manifest, another value, and both as the
# message writes them.
D17_CASES = [
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
        "f" * 40,
        '"' + "f" * 40 + '"',
        '"97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"',
    ),
    ("embedding_text_version", ("embedding",), 2, "2", "1"),
    ("tokenizer_version", ("sparse",), 2, "2", "1"),
    ("term_hash", ("sparse",), "md5-u32", '"md5-u32"', '"sha256-u32"'),
    ("bm25_k1", ("sparse",), 1.5, "1.5", "1.2"),
    ("bm25_b", ("sparse",), 0.5, "0.5", "0.75"),
]


def _edit_manifest(index, changes):
    path = index / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key, where, value, _, _ in changes:
        node = manifest
        for name in where:
            node = node[name]
        node[key] = value
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _mismatch(key, shown, now):
    return f"Index does not match settings: {key} is {shown} in the index, {now} now"


@pytest.mark.parametrize(
    ("key", "where", "value", "shown", "now"),
    D17_CASES,
    ids=[case[0] for case in D17_CASES],
)
def test_one_key_that_differs(index, caplog, capsys, key, where, value, shown, now):
    _edit_manifest(index, [(key, where, value, shown, now)])

    _assert_unusable(index, caplog, capsys, [_mismatch(key, shown, now)])


def test_all_eight_keys_differing_give_eight_errors_in_order(index, caplog, capsys):
    _edit_manifest(index, D17_CASES)

    _assert_unusable(
        index,
        caplog,
        capsys,
        [_mismatch(key, shown, now) for key, _, _, shown, now in D17_CASES],
    )


def test_a_changed_bm25_setting_no_longer_matches_the_index(
    index, caplog, capsys, settings_file
):
    settings_file({"retrieval": {"bm25_b": 0.5}})

    _assert_unusable(index, caplog, capsys, [_mismatch("bm25_b", "0.75", "0.5")])


def test_an_index_held_by_another_client(index, caplog, capsys):
    holder = QdrantClient(path=str(index))
    try:
        _assert_unusable(
            index,
            caplog,
            capsys,
            [f"Index is in use by another process: {index}"],
        )
    finally:
        holder.close()


def test_a_model_that_cannot_be_loaded(index, caplog, capsys, monkeypatch):
    fake = install_fake_sentence_transformers(monkeypatch)
    fake.fail_with = OSError("model niet in de cache")

    _assert_unusable(
        index,
        caplog,
        capsys,
        [
            "Could not load embedding model Qwen/Qwen3-Embedding-0.6B"
            "@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3: "
            "OSError: model niet in de cache"
        ],
        embedder=None,
    )
    assert [attempt["local_files_only"] for attempt in fake.attempts] == [True, True]
    assert fake.instances == []
    QdrantClient(path=str(index)).close()  # the failed searches released it


def test_settings_that_are_not_compared_do_not_stop_a_search(
    index, capsys, settings_file
):
    settings_file(
        {
            "embeddings": {"query_instruction": "Zoek:"},
            "retrieval": {
                "rrf_k": 10,
                "dense_limit": 3,
                "sparse_limit": 2,
                "candidate_count": 4,
            },
        }
    )

    assert _run(index, "huurprijs", "--json") == 0

    result = _json(capsys)
    assert result["query_instruction"] == "Zoek:"
    assert result["settings"] == {
        "rrf_k": 10,
        "dense_limit": 3,
        "sparse_limit": 2,
        "candidate_count": 4,
    }
    assert result["leg_counts"]["dense"] == 3
    assert len(result["candidates"]) <= 4


# --- Output (A13, D22, R13) -----------------------------------------------------------


def test_json_prints_the_d22_object_with_its_keys_in_order(index, capsys):
    assert _run(index, "Artikel 2 huurprijs", "--json") == 0

    result = _json(capsys)
    assert list(result) == TOP_KEYS
    assert list(result["index"]) == INDEX_KEYS
    assert list(result["settings"]) == SETTINGS_KEYS
    assert list(result["leg_counts"]) == LEG_COUNT_KEYS
    assert result["candidates"]
    for candidate in result["candidates"]:
        assert list(candidate) == CANDIDATE_KEYS
    assert result["query"] == "Artikel 2 huurprijs"
    assert result["query_terms"] == ["artikel", "2", "huurprijs"]
    with Retriever(index, embedder=StubEmbedder()) as retriever:
        assert retriever.search("Artikel 2 huurprijs").to_dict() == result


def test_the_human_output(index, capsys):
    assert _run(index, "Artikel 2 huurprijs", "--json") == 0
    result = _json(capsys)
    assert _run(index, "Artikel 2 huurprijs") == 0
    out = capsys.readouterr().out

    lines = out.split("\n")
    assert lines[-1] == ""  # the output ends with a line feed
    lines = lines[:-1]
    candidates = result["candidates"]
    assert lines[0] == "Query terms: artikel 2 huurprijs"
    assert len(lines) == 1 + 2 * len(candidates)
    nulls = 0
    for position, candidate in enumerate(candidates):
        first, second = lines[1 + 2 * position], lines[2 + 2 * position]

        def leg(name):
            rank, score = candidate[f"{name}_rank"], candidate[f"{name}_score"]
            if rank is None:
                return f"{name} - -"
            return f"{name} {rank} {score:.4f}"

        assert first.split(" | ") == [
            f"#{candidate['rank']}",
            f"rrf {candidate['rrf_score']:.6f}",
            leg("dense"),
            leg("sparse"),
            candidate["file_sha256"][:8],
            f"p. {candidate['page_start']}-{candidate['page_end']}",
            ",".join(candidate["text_origins"]),
            candidate["clause_label"] or "-",
        ]
        nulls += candidate["sparse_rank"] is None
        assert second == candidate["text"][:160].replace("\n", " ")
        assert "\n" in candidate["text"][:160]
    assert nulls  # the fixture has dense-only candidates
    assert {candidate["clause_label"] is None for candidate in candidates} == {
        True,
        False,
    }
    for candidate in candidates:
        assert candidate["file_path"] not in out
        assert candidate["file_sha256"] not in out
    assert "invoer" not in out


def _non_ascii_index(tmp_path):
    """An index whose only chunk holds Czech and Polish letters: U+0159 LATIN
    SMALL LETTER R WITH CARON and U+0142 LATIN SMALL LETTER L WITH STROKE."""
    text = (
        "Fictieve tekst: " + chr(0x0159) + "e" + chr(0x0159) + " o" + chr(0x0142) + "a."
    )
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    write_chunk_file(chunks, "a.json", hand_record(text, sha256="66" * 32))
    assert run_index(chunks, index) == 0
    return index, text


def test_non_ascii_written_through_a_cp1252_stdout_arrives_as_utf8(
    tmp_path, monkeypatch
):
    index, text = _non_ascii_index(tmp_path)
    raw = io.BytesIO()
    stdout = io.TextIOWrapper(raw, encoding="cp1252", newline="\n")
    monkeypatch.setattr(sys, "stdout", stdout)

    assert _run(index, "fictieve tekst") == 0

    stdout.flush()
    assert text in raw.getvalue().decode("utf-8")


def test_non_ascii_through_a_cp1252_console_in_a_fresh_interpreter(tmp_path):
    index, text = _non_ascii_index(tmp_path)
    code = (
        "import sys\n"
        "from tests.embeddings._helpers import StubEmbedder\n"
        "from companydocs_rag.retrieval import cli\n"
        f"sys.exit(cli.main(['fictieve tekst', '--index-dir', {str(index)!r}, "
        "'--json'], embedder=StubEmbedder()))\n"
    )
    environment = {**os.environ, "PYTHONIOENCODING": "cp1252"}

    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=120,
        cwd=_REPOSITORY,
        env=environment,
    )

    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.decode("utf-8"))["candidates"][0]["text"] == text


def test_an_index_without_chunks_gives_zero_candidates(tmp_path, capsys):
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    write_chunk_file(chunks, "mislukt.json", chunk_record(sha256=None, status="failed"))
    assert run_index(chunks, index) == 0

    assert _run(index, "huur", "--json") == 0
    result = _json(capsys)
    assert _run(index, "huur") == 0

    assert result["candidates"] == []
    assert result["leg_counts"] == {"dense": 0, "sparse": 0}
    assert capsys.readouterr().out == "Query terms: huur\n"


# --- Cannot start (A13, A15, R12) -----------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\t"], ids=["empty", "spaces", "tab"])
def test_an_empty_query_exits_2_first(tmp_path, caplog, capsys, query):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _run(tmp_path, query) == 2
        assert _run(tmp_path / "bestaat-niet", query) == 2

    assert _errors(caplog) == ["Empty query", "Empty query"]
    assert capsys.readouterr().out == ""


def test_a_missing_index_folder_exits_2_before_the_settings(
    tmp_path, caplog, capsys, settings_file
):
    settings_file({"retrieval": {"rrf_k": 0}})
    missing = tmp_path / "bestaat-niet"
    a_file = tmp_path / "bestand.json"
    a_file.write_text("{}", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _run(missing, "huur") == 2
        assert _run(a_file, "huur") == 2

    assert _errors(caplog) == [f"Not a folder: {missing}", f"Not a folder: {a_file}"]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("section", "values", "key"),
    INVALID_SETTINGS,
    ids=[invalid_settings_id(case) for case in INVALID_SETTINGS],
)
def test_invalid_settings_exit_2(
    tmp_path, caplog, capsys, settings_file, section, values, key
):
    settings_file({section: values})
    folder = tmp_path / "index"
    folder.mkdir()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _run(folder, "huur") == 2

    (error,) = _errors(caplog)
    assert error.startswith(f"Invalid settings: {section}.{key}: ")
    assert capsys.readouterr().out == ""
    assert list(folder.iterdir()) == []


def test_settings_are_read_at_call_time(index, capsys, settings_file):
    found = []
    for count in (2, 4):
        settings_file({"retrieval": {"candidate_count": count}})
        assert _run(index, "huurprijs", "--json") == 0
        found.append(len(_json(capsys)["candidates"]))

    assert found == [2, 4]


def test_help_lists_query_index_dir_json_and_help_only():
    parser = cli.build_parser()

    options = sorted(
        option for action in parser._actions for option in action.option_strings
    )
    positionals = [action for action in parser._actions if not action.option_strings]
    assert options == ["--help", "--index-dir", "--json", "-h"]
    assert [(action.dest, action.metavar) for action in positionals] == [
        ("query", "QUERY")
    ]
    assert "QUERY" in parser.format_help()


def test_the_default_index_folder_and_the_exit_codes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    args = cli.build_parser().parse_args(["huur"])

    assert args.index_dir == find_project_root() / "data" / "processed" / "index"
    assert (args.query, args.json) == ("huur", False)
    assert (cli.EXIT_SEARCH_DONE, cli.EXIT_INDEX_UNUSABLE, cli.EXIT_CANNOT_START) == (
        0,
        1,
        2,
    )
