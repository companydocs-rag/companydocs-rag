"""The dense side of the index (spec 006): embedding text, embedder, settings.

Embedding text (D4)
-------------------
:func:`compose_embedding_text` is the only composition function: a chunk's
clause label, a line feed and its text when the label is not null, else the
text. Both the dense and the sparse vector are computed from it. It holds no
document context, and it is never written anywhere: the index stores the
chunk's own text. :data:`EMBEDDING_TEXT_VERSION` is bumped on any change to
it.

The embedder (D1-D3, R16)
-------------------------
:class:`Embedder` is what the index and the search need from a model.
:class:`SentenceTransformerEmbedder` implements it with sentence-transformers
on CPU. It imports sentence-transformers, and so torch, only when it is
constructed. The model is loaded at the configured revision, with
``trust_remote_code=False``, ``device="cpu"`` and an explicit float32 dtype
as the only entry of ``model_kwargs``; ``local_files_only`` is the caller's
(the index may download the model, the search never does). Documents are
encoded without a prompt, queries with the query instruction as the
``prompt`` argument, and every vector is L2-normalized. Token counts include
the special tokens the tokenizer adds and are never truncated.

Settings (R12)
--------------
The top-level ``embeddings`` section, read at call time by
:func:`embeddings_settings`. A key that is absent takes the default below;
null is invalid. :class:`SettingsError` carries ``<section>.<key>: <reason>``
and also serves the ``retrieval`` section (:mod:`companydocs_rag.retrieval.search`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Protocol, Sequence

from companydocs_rag.ingestion.settings import get_section

# --- Settings defaults (embeddings), read at call time --------------------------------
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QUERY_INSTRUCTION = (
    "Instruct: Given a question about a contract, permit, register extract or "
    "other company document, retrieve the passages that answer the question\n"
    "Query:"
)
MAX_TOKENS = 8192
BATCH_SIZE = 16

# --- Python constants, not settings ---------------------------------------------------
# Bumped on any change to compose_embedding_text.
EMBEDDING_TEXT_VERSION = 1

_SECTION = "embeddings"
_LINE_FEED = "\n"

# R16: the model is loaded on CPU in float32, whatever the library's default.
_DEVICE = "cpu"
_DTYPE = "float32"


# --- Settings (R12) -------------------------------------------------------------------


class SettingsError(ValueError):
    """An invalid ``embeddings`` or ``retrieval`` setting (R12). The text is
    ``<section>.<key>: <reason>``; the commands log it after
    ``Invalid settings: ``."""

    def __init__(self, section: str, key: str, reason: str) -> None:
        super().__init__(f"{section}.{key}: {reason}")
        self.section = section
        self.key = key
        self.reason = reason


def _shown(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _fail(section: str, key: str, requirement: str, value: Any) -> SettingsError:
    return SettingsError(section, key, f"must be {requirement}, not {_shown(value)}")


def setting_string(
    section: str, values: Mapping[str, Any], key: str, default: str, *, empty_ok: bool
) -> str:
    """The string setting ``key``: absent takes ``default``; "" is valid only
    when ``empty_ok``."""
    value = values.get(key, default)
    if not isinstance(value, str) or (value == "" and not empty_ok):
        raise _fail(
            section, key, "a string" if empty_ok else "a non-empty string", value
        )
    return value


def setting_integer(
    section: str, values: Mapping[str, Any], key: str, default: int, *, minimum: int
) -> int:
    """The integer setting ``key``, at least ``minimum``; a bool is not an
    integer."""
    value = values.get(key, default)
    if type(value) is not int or value < minimum:
        raise _fail(section, key, f"an integer of at least {minimum}", value)
    return value


def setting_number(
    section: str,
    values: Mapping[str, Any],
    key: str,
    default: float,
    *,
    requirement: str,
    valid: Any,
) -> float:
    """The number setting ``key`` as a float: an integer is read as a float,
    a bool is not a number, and ``valid`` decides the bounds."""
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(section, key, requirement, value)
    number = float(value)
    if not valid(number):
        raise _fail(section, key, requirement, value)
    return number


def is_finite_above_zero(number: float) -> bool:
    return math.isfinite(number) and number > 0


@dataclass(frozen=True)
class EmbeddingsSettings:
    """The resolved ``embeddings`` section."""

    model_name: str
    model_revision: str
    query_instruction: str
    max_tokens: int
    batch_size: int


def embeddings_settings(
    settings: Optional[Mapping[str, Any]] = None,
) -> EmbeddingsSettings:
    """Resolve the ``embeddings`` section at call time, YAML over the defaults
    above; ``settings`` replaces ``config/settings.yaml``. Keys are checked in
    the order of the Settings table, and the first invalid one raises
    :class:`SettingsError`."""
    section = _SECTION
    values = get_section(section, settings=settings)
    return EmbeddingsSettings(
        model_name=setting_string(
            section, values, "model_name", MODEL_NAME, empty_ok=False
        ),
        model_revision=setting_string(
            section, values, "model_revision", MODEL_REVISION, empty_ok=False
        ),
        query_instruction=setting_string(
            section, values, "query_instruction", QUERY_INSTRUCTION, empty_ok=True
        ),
        max_tokens=setting_integer(
            section, values, "max_tokens", MAX_TOKENS, minimum=1
        ),
        batch_size=setting_integer(
            section, values, "batch_size", BATCH_SIZE, minimum=1
        ),
    )


# --- The embedding text (D4) ----------------------------------------------------------


def compose_embedding_text(chunk: Mapping[str, Any]) -> str:
    """The text both vectors are computed from: ``clause_label``, a line feed
    and ``text`` when the label is not null, else ``text``. Works on a chunk
    from a chunk file and on a payload alike."""
    label = chunk["clause_label"]
    text = chunk["text"]
    return text if label is None else label + _LINE_FEED + text


# --- The embedder (D1-D3, R16) --------------------------------------------------------


class Embedder(Protocol):
    """What the index and the search need from an embedding model."""

    @property
    def dimension(self) -> int:
        """The length of every vector."""
        ...

    @property
    def max_seq_length(self) -> int:
        """The most tokens the model encodes; longer input would be cut."""
        ...

    def encode_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """One L2-normalized vector per text, in order, without a prompt."""
        ...

    def encode_query(self, text: str) -> List[float]:
        """The L2-normalized vector of a query, with the query instruction."""
        ...

    def count_tokens(self, text: str) -> int:
        """The tokens of ``text`` as encoded for a document, special tokens
        included, without truncation."""
        ...


def _floats(vector: Any) -> List[float]:
    return [float(value) for value in vector]


class SentenceTransformerEmbedder:
    """:class:`Embedder` on a sentence-transformers model (D1-D3, R16).

    Constructing it loads the model, and any failure to do so is raised from
    here. ``local_files_only=True`` (the search) never downloads.
    """

    def __init__(self, config: EmbeddingsSettings, *, local_files_only: bool) -> None:
        # Imported here, not at module level: it imports torch (D26).
        from sentence_transformers import SentenceTransformer

        self._config = config
        self._model = SentenceTransformer(
            config.model_name,
            device=_DEVICE,
            trust_remote_code=False,
            revision=config.model_revision,
            local_files_only=local_files_only,
            model_kwargs={"dtype": _DTYPE},
        )
        self._dimension = int(self._model.get_embedding_dimension())
        self._max_seq_length = int(self._model.max_seq_length)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    def encode_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._config.batch_size,
            normalize_embeddings=True,
        )
        return [_floats(vector) for vector in vectors]

    def encode_query(self, text: str) -> List[float]:
        vectors = self._model.encode(
            [text],
            prompt=self._config.query_instruction,
            normalize_embeddings=True,
        )
        return _floats(vectors[0])

    def count_tokens(self, text: str) -> int:
        encoded = self._model.tokenizer(text, add_special_tokens=True, truncation=False)
        return len(encoded["input_ids"])
