"""The dense side (spec 006): the embedding text (A2), the embedder (A6) and
the ``embeddings`` settings (R12, A15).

A fake ``sentence_transformers`` module replaces the real one and records its
arguments; the real model is never loaded (D25). The command-level checks of
``local_files_only`` build their index with the stub embedder in ``tmp_path``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest
import yaml

from companydocs_rag.embeddings import dense
from companydocs_rag.embeddings.dense import (
    QUERY_INSTRUCTION,
    SentenceTransformerEmbedder,
    SettingsError,
    compose_embedding_text,
    embeddings_settings,
)
from companydocs_rag.ingestion.settings import find_project_root
from companydocs_rag.retrieval import cli as search_cli
from companydocs_rag.retrieval.search import Retriever
from companydocs_rag.vectorstore import cli as index_cli
from tests.embeddings._helpers import (
    FAKE_DIMENSION,
    StubEmbedder,
    install_fake_sentence_transformers,
)
from tests.vectorstore._helpers import run_index, three_documents


def _norm(vector):
    return math.sqrt(sum(value * value for value in vector))


@pytest.fixture
def fake(monkeypatch):
    return install_fake_sentence_transformers(monkeypatch)


# --- The embedding text (A2, D4) ------------------------------------------------------


def test_a_label_and_the_text_joined_by_a_line_feed():
    chunk = {"clause_label": "Artikel 2 Huurprijs", "text": "Artikel 2 Huurprijs\nDe"}

    assert (
        compose_embedding_text(chunk) == "Artikel 2 Huurprijs\nArtikel 2 Huurprijs\nDe"
    )


def test_without_a_label_the_text_alone():
    assert compose_embedding_text({"clause_label": None, "text": "Tekst."}) == "Tekst."


def test_a_chunk_and_a_payload_compose_alike():
    """A payload has more keys than a chunk and lacks some (D13); only the
    label and the text count."""
    chunk = {"chunk_id": "ab" * 10, "clause_label": "Art. 4", "text": "Inhoud."}
    payload = {
        "chunk_id": "ab" * 10,
        "file_path": "invoer/fictief.pdf",
        "route": "structured",
        "text": "Inhoud.",
        "clause_label": "Art. 4",
    }

    assert compose_embedding_text(chunk) == compose_embedding_text(payload)
    assert compose_embedding_text(payload) == "Art. 4\nInhoud."


def test_the_embedding_text_version_is_1():
    assert dense.EMBEDDING_TEXT_VERSION == 1


# --- The embedder (A6, D1-D3, R16) ----------------------------------------------------


def test_construction_passes_exactly_the_r16_arguments(fake):
    config = embeddings_settings(
        {"embeddings": {"model_name": "fictief/model", "model_revision": "abc123"}}
    )

    embedder = SentenceTransformerEmbedder(config, local_files_only=False)

    (model,) = fake.instances
    assert model.args == ("fictief/model",)
    assert model.kwargs == {
        "device": "cpu",
        "trust_remote_code": False,
        "revision": "abc123",
        "local_files_only": False,
        "model_kwargs": {"dtype": "float32"},
    }
    assert "tokenizer_kwargs" not in model.kwargs
    assert (embedder.dimension, embedder.max_seq_length) == (FAKE_DIMENSION, 6)


def test_local_files_only_is_passed_through(fake):
    SentenceTransformerEmbedder(embeddings_settings({}), local_files_only=True)

    assert fake.instances[0].kwargs["local_files_only"] is True
    assert fake.instances[0].args == (dense.MODEL_NAME,)
    assert fake.instances[0].kwargs["revision"] == dense.MODEL_REVISION


def test_the_search_path_never_downloads_and_the_index_path_may(fake, tmp_path, caplog):
    """``Retriever`` and ``companydocs-search`` load with local_files_only
    True; ``companydocs-index`` with it False (D3)."""
    chunks, index = tmp_path / "chunks", tmp_path / "index"
    three_documents(chunks)
    assert run_index(chunks, index, embedder=StubEmbedder()) == 0

    with Retriever(index):
        pass
    assert search_cli.main(["huurprijs", "--index-dir", str(index)]) == 0
    fake.max_seq_length = 10_000
    with caplog.at_level(logging.INFO):
        code = index_cli.main(
            ["--chunks-dir", str(chunks), "--index-dir", str(tmp_path / "nieuw")]
        )

    assert code == 0
    assert [model.kwargs["local_files_only"] for model in fake.instances] == [
        True,
        True,
        False,
    ]


def test_encode_query_passes_the_instruction_as_the_prompt(fake):
    embedder = SentenceTransformerEmbedder(
        embeddings_settings({}), local_files_only=True
    )

    vector = embedder.encode_query("Wat is de huurprijs?")

    (call,) = fake.instances[0].encode_calls
    assert call["inputs"] == ["Wat is de huurprijs?"]
    assert call["prompt"] == QUERY_INSTRUCTION
    assert call["prompt"].endswith("question\nQuery:")
    assert "prompt_name" not in call
    assert call["normalize_embeddings"] is True
    assert abs(_norm(vector) - 1.0) < 1e-6


def test_an_empty_instruction_is_passed_as_an_empty_prompt(fake):
    config = embeddings_settings({"embeddings": {"query_instruction": ""}})
    embedder = SentenceTransformerEmbedder(config, local_files_only=True)

    embedder.encode_query("huurprijs")

    assert fake.instances[0].encode_calls[0]["prompt"] == ""


def test_encode_documents_passes_no_prompt_and_the_batch_size(fake):
    config = embeddings_settings({"embeddings": {"batch_size": 3}})
    embedder = SentenceTransformerEmbedder(config, local_files_only=False)
    texts = ["Artikel 1\nEen.", "Twee.", "Drie.", "Vier."]

    vectors = embedder.encode_documents(texts)

    (call,) = fake.instances[0].encode_calls
    assert call["inputs"] == texts
    assert "prompt" not in call and "prompt_name" not in call
    assert call["batch_size"] == 3
    assert len(vectors) == 4
    assert all(len(vector) == FAKE_DIMENSION for vector in vectors)
    assert all(abs(_norm(vector) - 1.0) < 1e-6 for vector in vectors)


def test_the_fake_returns_norm_2_unless_asked_to_normalize(fake):
    """So the norm-1 checks above show the embedder asks for normalization."""
    SentenceTransformerEmbedder(embeddings_settings({}), local_files_only=True)
    model = fake.instances[0]

    (raw,) = model.encode(["Tekst."])
    (normalized,) = model.encode(["Tekst."], normalize_embeddings=True)

    assert _norm(raw) == pytest.approx(2.0)
    assert _norm(normalized) == pytest.approx(1.0)


def test_count_tokens_counts_the_text_alone_with_special_tokens_untruncated(fake):
    """The fake tokenizer: one token per word, one special token, and a cut
    at max_seq_length (6) only when truncation is asked."""
    embedder = SentenceTransformerEmbedder(
        embeddings_settings({}), local_files_only=True
    )
    text = "een twee drie vier vijf zes zeven acht"

    count = embedder.count_tokens(text)

    assert count == 9 > embedder.max_seq_length
    (call,) = fake.instances[0].tokenizer.calls
    assert call["text"] == text
    assert call["add_special_tokens"] is True
    assert call["truncation"] is False


def test_a_failing_model_load_raises_from_the_constructor(fake):
    fake.fail_with = OSError("geen netwerk")

    with pytest.raises(OSError, match="geen netwerk"):
        SentenceTransformerEmbedder(embeddings_settings({}), local_files_only=True)


def test_importing_dense_does_not_import_sentence_transformers():
    """The import happens in the constructor; the subprocess checks in
    test_lazy_imports.py cover the module import itself."""
    source = Path(dense.__file__).read_text(encoding="utf-8")
    module_level = [
        line for line in source.splitlines() if line.startswith(("import", "from"))
    ]

    assert not any("sentence_transformers" in line for line in module_level)
    assert not any("torch" in line for line in module_level)


# --- Settings (R12, A15) --------------------------------------------------------------

EMBEDDINGS_KEYS = [
    ("model_name", "MODEL_NAME"),
    ("model_revision", "MODEL_REVISION"),
    ("query_instruction", "QUERY_INSTRUCTION"),
    ("max_tokens", "MAX_TOKENS"),
    ("batch_size", "BATCH_SIZE"),
]


def test_without_a_section_the_python_defaults_apply():
    config = embeddings_settings({})

    for key, constant in EMBEDDINGS_KEYS:
        assert getattr(config, key) == getattr(dense, constant)


def test_the_defaults_are_the_spec_values():
    assert dense.MODEL_NAME == "Qwen/Qwen3-Embedding-0.6B"
    assert dense.MODEL_REVISION == "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    assert dense.QUERY_INSTRUCTION == (
        "Instruct: Given a question about a contract, permit, register extract or "
        "other company document, retrieve the passages that answer the question"
        "\nQuery:"
    )
    assert (dense.MAX_TOKENS, dense.BATCH_SIZE) == (8192, 16)


def test_an_absent_key_takes_its_default_and_an_empty_instruction_is_valid():
    config = embeddings_settings(
        {"embeddings": {"query_instruction": "", "batch_size": 4}}
    )

    assert config.query_instruction == ""
    assert config.batch_size == 4
    assert config.max_tokens == dense.MAX_TOKENS


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (key, value)
        for key in ("model_name", "model_revision")
        for value in (5, "", None, True)
    ]
    + [("query_instruction", value) for value in (5, None, True)]
    + [
        (key, value)
        for key in ("max_tokens", "batch_size")
        for value in ("16", 2.5, 0, -1, None, True)
    ],
)
def test_an_invalid_value_names_its_key(key, value):
    with pytest.raises(SettingsError) as raised:
        embeddings_settings({"embeddings": {key: value}})

    assert str(raised.value).startswith(f"embeddings.{key}: ")
    assert (raised.value.section, raised.value.key) == ("embeddings", key)


def test_the_first_invalid_key_in_table_order_is_reported():
    with pytest.raises(SettingsError) as raised:
        embeddings_settings({"embeddings": {"batch_size": 0, "model_revision": ""}})

    assert raised.value.key == "model_revision"


def _settings_yaml_text():
    return (find_project_root() / "config" / "settings.yaml").read_text(
        encoding="utf-8"
    )


def test_the_yaml_section_holds_the_python_defaults():
    section = yaml.safe_load(_settings_yaml_text())["embeddings"]

    assert list(section) == [key for key, _ in EMBEDDINGS_KEYS]
    for key, constant in EMBEDDINGS_KEYS:
        assert section[key] == getattr(dense, constant)
    assert section["query_instruction"].count("\n") == 1


def test_the_yaml_section_carries_the_settings_table_comments():
    text = _settings_yaml_text()
    section = text[text.index("\nembeddings:") : text.index("\nretrieval:")]
    flat = " ".join(line.strip().lstrip("#").strip() for line in section.splitlines())

    for comment in (
        "the Hugging Face model id",
        "the pinned commit of the model repository",
        "passed as the prompt for queries only (D2)",
        "double-quoted, so \\n is a line feed",
        "equals the bge-m3 fallback limit, so a model switch never truncates silently",
        "documents per encoding batch",
    ):
        assert comment.lower() in flat.lower(), comment
    assert '  query_instruction: "Instruct: ' in section
