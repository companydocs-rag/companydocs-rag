"""The ``companydocs-search`` command (spec 006): a thin wrapper around
:class:`.search.Retriever`.

    companydocs-search QUERY [--index-dir DIR] [--json]

``QUERY`` is one argument, used as given (R18); quote a query of several
words. The index is ``--index-dir`` (default ``data/processed/index`` under
the project root, where ``companydocs-index`` writes). The command reads the
index and its manifest only, never chunk files, and never downloads the model
(D3).

Output (D22, R13): stdout carries only the result, written as UTF-8 whatever
the console's code page; logs go to stderr. ``--json`` prints the D22 object
(``indent=2``, ``ensure_ascii=False``). Otherwise: a line with the query's
terms, then two lines per candidate: its rank, RRF score, dense and sparse
rank and score, the first 8 hex characters of its file hash (never the file
path), its pages, text origins and clause label ("-" for a null); then the
first 160 characters of its text, line feeds replaced by spaces.

Exit codes: 0 when the search ran, also with zero candidates (D23); 1 when
the index cannot be used (R14); 2 when the query is empty or whitespace
only, ``--index-dir`` is not a folder, or the settings are invalid, checked
in that order. One process at a time: Qdrant local mode locks the index
folder (D24).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from companydocs_rag.embeddings.dense import (
    Embedder,
    SettingsError,
    embeddings_settings,
)
from companydocs_rag.vectorstore.cli import default_index_dir

from .search import IndexUnusableError, Retriever, SearchResult, retrieval_settings

logger = logging.getLogger(__name__)

EXIT_SEARCH_DONE = 0
EXIT_INDEX_UNUSABLE = 1
EXIT_CANNOT_START = 2

_TEXT_LINE_CHARS = 160
_FILE_HASH_CHARS = 8
_NULL = "-"


def _rank(value: Optional[int]) -> str:
    return _NULL if value is None else str(value)


def _score(value: Optional[float]) -> str:
    return _NULL if value is None else f"{value:.4f}"


def _candidate_line(candidate: Mapping[str, Any]) -> str:
    label = candidate["clause_label"]
    return " | ".join(
        [
            f"#{candidate['rank']}",
            f"rrf {candidate['rrf_score']:.6f}",
            f"dense {_rank(candidate['dense_rank'])} "
            f"{_score(candidate['dense_score'])}",
            f"sparse {_rank(candidate['sparse_rank'])} "
            f"{_score(candidate['sparse_score'])}",
            candidate["file_sha256"][:_FILE_HASH_CHARS],
            f"p. {candidate['page_start']}-{candidate['page_end']}",
            ",".join(candidate["text_origins"]),
            _NULL if label is None else label,
        ]
    )


def render_human(result: SearchResult) -> str:
    """The human output: the query-terms line, then two lines per candidate
    (D22, R13)."""
    terms = " ".join(result.query_terms) if result.query_terms else _NULL
    lines = [f"Query terms: {terms}"]
    for candidate in result.candidates:
        lines.append(_candidate_line(candidate))
        lines.append(candidate["text"][:_TEXT_LINE_CHARS].replace("\n", " "))
    return "\n".join(lines) + "\n"


def render_json(result: SearchResult) -> str:
    """The ``--json`` output: the D22 object."""
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"


def _write_stdout(text: str) -> None:
    """Write ``text`` to stdout as UTF-8, whatever its code page (R13)."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")
    sys.stdout.write(text)
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companydocs-search",
        description=(
            "Search the index: a dense and a BM25 sparse query, fused with "
            "reciprocal rank fusion. Every candidate carries the rank and "
            "score from each leg. Reads the index only, never downloads."
        ),
    )
    parser.add_argument(
        "query",
        metavar="QUERY",
        help="The question or search terms, as one argument (quote several words)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=default_index_dir(),
        help=(
            "Folder of the index written by companydocs-index (default: "
            "data/processed/index under the project root -- git-ignored)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as one JSON object instead of the human lines",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None, *, embedder: Optional[Embedder] = None
) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if not args.query.strip():
        logger.error("Empty query")
        return EXIT_CANNOT_START
    if not args.index_dir.is_dir():
        logger.error("Not a folder: %s", args.index_dir)
        return EXIT_CANNOT_START
    try:
        embeddings_settings()
        retrieval_settings()
    except SettingsError as error:
        logger.error("Invalid settings: %s", error)
        return EXIT_CANNOT_START

    try:
        retriever = Retriever(args.index_dir, embedder=embedder)
    except IndexUnusableError as error:
        for reason in error.reasons:
            logger.error("%s", reason)
        return EXIT_INDEX_UNUSABLE
    with retriever:
        result = retriever.search(args.query)

    _write_stdout(render_json(result) if args.json else render_human(result))
    return EXIT_SEARCH_DONE


if __name__ == "__main__":
    sys.exit(main())
