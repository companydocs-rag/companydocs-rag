"""Shared helpers for the spec 006 tests: the stub embedder, a fake
sentence-transformers module, settings files and the import checks.

No test loads the real model, imports sentence-transformers or downloads
anything (D25). :class:`StubEmbedder` is the fabricated embedder the
commands and ``Retriever`` get injected: a small fixed dimension,
deterministic vectors and a stub token counter.
:func:`install_fake_sentence_transformers` puts a fake
``sentence_transformers`` module in ``sys.modules``, so that
``SentenceTransformerEmbedder`` constructs :class:`FakeSentenceTransformer`,
which records its arguments.
"""

from __future__ import annotations

import hashlib
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from companydocs_rag.ingestion.settings import load_settings

STUB_DIMENSION = 8
STUB_MAX_SEQ_LENGTH = 512
# The stub counts whitespace-separated words plus this many special tokens.
STUB_SPECIAL_TOKENS = 2


def unit(raw: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


def hashed_vector(text: str, dimension: int = STUB_DIMENSION) -> List[float]:
    """A deterministic vector of norm 1 derived from ``text``."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return unit([digest[position] - 127.5 for position in range(dimension)])


class StubEmbedder:
    """A fabricated embedder (D25). A text in ``vectors`` gets that vector,
    normalized; any other text a hashed one. It records every call."""

    def __init__(
        self,
        *,
        dimension: int = STUB_DIMENSION,
        max_seq_length: int = STUB_MAX_SEQ_LENGTH,
        vectors: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> None:
        self.dimension = dimension
        self.max_seq_length = max_seq_length
        self.vectors = dict(vectors or {})
        self.document_calls: List[List[str]] = []
        self.query_calls: List[str] = []
        self.counted: List[str] = []

    def _vector(self, text: str) -> List[float]:
        if text in self.vectors:
            return unit(self.vectors[text])
        return hashed_vector(text, self.dimension)

    def encode_documents(self, texts: Sequence[str]) -> List[List[float]]:
        self.document_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def encode_query(self, text: str) -> List[float]:
        self.query_calls.append(text)
        return self._vector(text)

    def count_tokens(self, text: str) -> int:
        self.counted.append(text)
        return stub_token_count(text)

    @property
    def documents(self) -> List[str]:
        """Every text encoded as a document, in order."""
        return [text for call in self.document_calls for text in call]


def stub_token_count(text: str) -> int:
    return len(text.split()) + STUB_SPECIAL_TOKENS


class StubLoader:
    """A ``load_embedder`` for ``companydocs-index`` that returns ``embedder``
    and records the settings it was called with."""

    def __init__(self, embedder: Optional[StubEmbedder] = None) -> None:
        self.embedder = embedder if embedder is not None else StubEmbedder()
        self.calls: List[Any] = []

    def __call__(self, config: Any) -> StubEmbedder:
        self.calls.append(config)
        return self.embedder


# --- A fake sentence-transformers -----------------------------------------------------

FAKE_DIMENSION = STUB_DIMENSION
FAKE_MAX_SEQ_LENGTH = 6
FAKE_SPECIAL_TOKEN = 999


class FakeTokenizer:
    """Whitespace tokens, then one special token when ``add_special_tokens``;
    cut at ``max_length`` (else the model's ``max_seq_length``) only when
    ``truncation`` is true."""

    def __init__(self, owner: "FakeSentenceTransformer") -> None:
        self.owner = owner
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: Any = False,
        max_length: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, List[int]]:
        self.calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
                "max_length": max_length,
                **kwargs,
            }
        )
        ids = list(range(len(text.split())))
        if add_special_tokens:
            ids.append(FAKE_SPECIAL_TOKEN)
        if truncation:
            ids = ids[: max_length or self.owner.max_seq_length]
        return {"input_ids": ids}


class FakeSentenceTransformer:
    """Records its constructor and ``encode`` arguments. ``encode`` returns
    vectors of norm 2 unless asked to normalize, then of norm 1."""

    def __init__(self, model_name_or_path: str, *args: Any, **kwargs: Any) -> None:
        registry = type(self).registry
        registry.attempts.append(dict(kwargs))
        if registry.fail_with is not None:
            raise registry.fail_with
        self.args = (model_name_or_path, *args)
        self.kwargs = kwargs
        self.max_seq_length = registry.max_seq_length
        self.default_prompt_name = None
        self.tokenizer = FakeTokenizer(self)
        self.encode_calls: List[Dict[str, Any]] = []
        registry.instances.append(self)

    registry: "FakeRegistry"

    def get_embedding_dimension(self) -> int:
        return FAKE_DIMENSION

    def encode(self, inputs: Sequence[str], **kwargs: Any) -> List[List[float]]:
        self.encode_calls.append({"inputs": list(inputs), **kwargs})
        scale = 1.0 if kwargs.get("normalize_embeddings") else 2.0
        return [
            [scale * value for value in hashed_vector(text, FAKE_DIMENSION)]
            for text in inputs
        ]


class FakeRegistry:
    """What the fake module saw: the keyword arguments of every construction
    (``attempts``) and every constructed model. ``fail_with`` is an error the
    constructor raises when set; ``max_seq_length`` is the next model's."""

    def __init__(self) -> None:
        self.instances: List[FakeSentenceTransformer] = []
        self.attempts: List[Dict[str, Any]] = []
        self.fail_with: Optional[BaseException] = None
        self.max_seq_length = FAKE_MAX_SEQ_LENGTH


def install_fake_sentence_transformers(monkeypatch: Any) -> FakeRegistry:
    """Put a fake ``sentence_transformers`` module in ``sys.modules`` for
    one test, and return its registry."""
    registry = FakeRegistry()
    fake_class = type(
        "SentenceTransformer", (FakeSentenceTransformer,), {"registry": registry}
    )
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = fake_class
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return registry


# --- Settings (R12, A15) --------------------------------------------------------------


def write_settings(
    tmp_path: Path, monkeypatch: Any, settings: Mapping[str, Any]
) -> Path:
    """Point ``COMPANYDOCS_RAG_SETTINGS`` at a YAML file holding ``settings``,
    for one test. The caller clears ``load_settings``' cache at the end."""
    path = tmp_path / "instellingen" / "instellingen.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(settings), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("COMPANYDOCS_RAG_SETTINGS", str(path))
    load_settings.cache_clear()
    return path


def _cases(section: str, keys: Sequence[str], values: Sequence[Any]):
    return [(section, {key: value}, key) for key in keys for value in values]


# R12: (section, values, the key that is reported). Per key: the wrong type,
# out of bounds, null and a bool; and candidate_count above the two limits.
INVALID_SETTINGS: List[Tuple[str, Dict[str, Any], str]] = [
    *_cases("embeddings", ["model_name", "model_revision"], [5, "", None, True]),
    *_cases("embeddings", ["query_instruction"], [5, ["Query:"], None, True]),
    *_cases("embeddings", ["max_tokens", "batch_size"], ["16", 2.5, 0, None, True]),
    *_cases(
        "retrieval",
        ["bm25_k1"],
        ["1.2", 0, -1.0, float("inf"), float("nan"), None, True],
    ),
    *_cases("retrieval", ["bm25_b"], ["0.75", -0.1, 1.5, float("nan"), None, True]),
    *_cases(
        "retrieval",
        ["rrf_k", "dense_limit", "sparse_limit", "candidate_count"],
        ["5", 2.5, 0, None, True],
    ),
    (
        "retrieval",
        {"dense_limit": 5, "sparse_limit": 6, "candidate_count": 12},
        "candidate_count",
    ),
]


def invalid_settings_id(case: Tuple[str, Dict[str, Any], str]) -> str:
    section, values, _ = case
    return f"{section}-" + "-".join(f"{key}={value!r}" for key, value in values.items())


# --- Import checks (D26, R15) ---------------------------------------------------------

IMPORT_TIMEOUT_SECONDS = 120
# Never imported by the package, its new modules, or a run with the stub.
FORBIDDEN_MODULES = (
    "torch",
    "transformers",
    "sentence_transformers",
    "preocr",
    "magic",
)
_REPOSITORY = Path(__file__).resolve().parents[2]


def run_import_check(code: str) -> None:
    """Run ``code`` in a fresh interpreter, from the repository root, then
    assert that none of :data:`FORBIDDEN_MODULES` was imported."""
    import subprocess

    import pytest

    check = "\n".join(
        [
            "import sys",
            code,
            f"loaded = [name for name in {FORBIDDEN_MODULES!r} if name in sys.modules]",
            "assert loaded == [], loaded",
        ]
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", check],
            capture_output=True,
            text=True,
            timeout=IMPORT_TIMEOUT_SECONDS,
            cwd=_REPOSITORY,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"the check did not return within {IMPORT_TIMEOUT_SECONDS}s")
    assert result.returncode == 0, result.stderr
