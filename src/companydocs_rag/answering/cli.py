"""The ``companydocs-answer`` command (spec 007): a wrapper around
:class:`.answerer.Answerer`.

    companydocs-answer QUESTION [--selection-mode {fused,per_path}] [--json]
                      [--index-dir DIR]
    companydocs-answer --questions PATH [--selection-mode {fused,per_path}]
                      [--index-dir DIR]

``QUESTION`` is one argument, used as given; ``--questions`` is a CSV file
with the columns ``question_id`` and ``question`` (:mod:`.questions`),
answered in file order. Exactly one of the two is required, and ``--json``
goes with ``QUESTION`` only. The index is ``--index-dir`` (default
``data/processed/index`` under the project root, as for
``companydocs-search``). The generator has to run (Ollama on localhost by
default), and the command holds the index lock for its whole run. Every
answer goes to the run's log, ``answers-<time>.jsonl`` in the ``log_dir``
setting.

Before anything is opened, in this order, each failure an ERROR and exit 2:
the question (empty or whitespace only) or the questions file;
``--index-dir`` not a folder; the settings, ``embeddings``, ``retrieval``,
then ``answering``; the prompt file; the API key variable. Opening the index
can then exit 1, and the warm-up request 3.

Output (R18): stdout carries only results, as UTF-8 whatever the console's
code page; logs go to stderr at WARNING. A single question prints a status
line (the status, and the abstention source when set), the answer text when
there is one, then one line per source, ``[n] <file_path> pagina <p>[-<p2>]
(<provenance values>)`` and `` kop: <clause_label>`` when it has one; with
``--json`` the record instead. A questions file prints one line per status
with its count, then the log path. A record in status ``error`` logs one
WARNING. When the generator becomes unavailable the run stops (exit 3) and
prints the same lines for the records written.

Exit codes: 0 when every question has a record and none is in status
``error``; 1 when the index cannot be used; 2 when the command cannot start;
3 when the generator is unavailable, at warm-up or during the run; 4 when
the run finished with at least one record in status ``error``. 3 wins over
4, and 4 over 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import httpx

from companydocs_rag.embeddings.dense import SettingsError, embeddings_settings
from companydocs_rag.retrieval.search import IndexUnusableError, retrieval_settings
from companydocs_rag.vectorstore.cli import default_index_dir

from .answerer import (
    Answerer,
    AnswerResult,
    GeneratorError,
    GeneratorUnavailableError,
)
from .generator import read_api_key
from .parsing import STATUS_ERROR, STATUSES
from .prompt import load_system_prompt, pages_text
from .questions import Question, QuestionsFileError, read_questions
from .settings import SELECTION_MODES, answering_settings

logger = logging.getLogger(__name__)

EXIT_ANSWER_DONE = 0
EXIT_INDEX_UNUSABLE = 1
EXIT_CANNOT_START = 2
EXIT_GENERATOR_UNAVAILABLE = 3
EXIT_QUESTION_ERRORS = 4

_NULL_QUESTION_ID = "-"


def _distinct(values: Sequence[str]) -> List[str]:
    found: List[str] = []
    for value in values:
        if value not in found:
            found.append(value)
    return found


def source_line(source: Mapping[str, Any]) -> str:
    """``[n] <file_path> pagina <p>[-<p2>] (<provenance values>)``, then
    `` kop: <clause_label>`` when there is one (D11, R13)."""
    provenance = ", ".join(_distinct([page["provenance"] for page in source["pages"]]))
    line = (
        f"[{source['prompt_number']}] {source['file_path']} pagina "
        f"{pages_text(source['page_start'], source['page_end'])} ({provenance})"
    )
    if source["clause_label"] is not None:
        line += f" kop: {source['clause_label']}"
    return line


def status_line(result: AnswerResult) -> str:
    """The status, and the abstention source when set (R18)."""
    if result.abstention_source is None:
        return f"Status: {result.status}"
    return f"Status: {result.status} ({result.abstention_source})"


def render_text(result: AnswerResult) -> str:
    """A single question's text output: the status line, the answer text
    when it is not null, a blank line and the source lines when there are
    sources (R18)."""
    lines = [status_line(result)]
    if result.answer_text is not None:
        lines.append(result.answer_text)
    if result.sources:
        lines.append("")
        lines.extend(source_line(source) for source in result.sources)
    return "\n".join(lines) + "\n"


def render_json(result: AnswerResult) -> str:
    """``--json``: the record, ``indent=2``, ``ensure_ascii=False``."""
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"


def render_batch(results: Sequence[AnswerResult], log_path: Optional[Path]) -> str:
    """A questions file's output: one line per status in D7's order with its
    count, zeros included, then the log path (R18)."""
    lines = [
        f"{status}: {sum(result.status == status for result in results)}"
        for status in STATUSES
    ]
    lines.append(f"log: {log_path}")
    return "\n".join(lines) + "\n"


def _write_stdout(text: str) -> None:
    """Write ``text`` to stdout as UTF-8, whatever its code page."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")
    sys.stdout.write(text)
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companydocs-answer",
        description=(
            "Answer a question, or every question of a questions file, from "
            "the index: retrieval, then a local LLM behind an OpenAI-compatible "
            "endpoint (Ollama by default), which cites its sources. Every answer "
            "is logged as JSONL."
        ),
    )
    form = parser.add_mutually_exclusive_group(required=True)
    form.add_argument(
        "question",
        nargs="?",
        metavar="QUESTION",
        help="One question, as one argument (quote several words)",
    )
    form.add_argument(
        "--questions",
        type=Path,
        metavar="PATH",
        help=(
            "A CSV file with the columns question_id and question (comma or "
            "semicolon), answered in file order"
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default=None,
        help=(
            "How chunks are selected for the prompt, for every question of the "
            "run (default: the selection_mode setting)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Single question only: print the log record instead of the text",
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
    return parser


def _warn_if_error(result: AnswerResult) -> None:
    if result.status == STATUS_ERROR:
        question_id = (
            _NULL_QUESTION_ID if result.question_id is None else result.question_id
        )
        logger.warning(
            "Question %s: %s: %s",
            question_id,
            result.error["type"],
            result.error["message"],
        )


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    retriever: Any = None,
    transport: Optional[httpx.BaseTransport] = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.json and args.questions is not None:
        parser.error("argument --json: not allowed with argument --questions")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    batch = args.questions is not None
    if batch:
        try:
            questions = read_questions(args.questions)
        except QuestionsFileError as error:
            logger.error("%s", error)
            return EXIT_CANNOT_START
    else:
        if not args.question.strip():
            logger.error("Empty question")
            return EXIT_CANNOT_START
        questions = [Question(None, args.question, 0)]
    if not args.index_dir.is_dir():
        logger.error("Not a folder: %s", args.index_dir)
        return EXIT_CANNOT_START
    try:
        embeddings_settings()
        retrieval_settings()
        config = answering_settings()
        load_system_prompt(config)
        read_api_key(config)
    except SettingsError as error:
        logger.error("Invalid settings: %s", error)
        return EXIT_CANNOT_START

    try:
        answerer = Answerer(args.index_dir, retriever=retriever, transport=transport)
    except IndexUnusableError as error:
        for reason in error.reasons:
            logger.error("%s", reason)
        return EXIT_INDEX_UNUSABLE
    except SettingsError as error:
        logger.error("Invalid settings: %s", error)
        return EXIT_CANNOT_START
    except GeneratorUnavailableError as error:
        logger.error("Generator unavailable at %s: %s", config.base_url, error)
        return EXIT_GENERATOR_UNAVAILABLE
    except GeneratorError as error:
        logger.error("Generator failed at warm-up: %s", error)
        return EXIT_GENERATOR_UNAVAILABLE

    results: List[AnswerResult] = []
    code = EXIT_ANSWER_DONE
    with answerer:
        for question in questions:
            try:
                result = answerer.answer(
                    question.question,
                    selection_mode=args.selection_mode,
                    question_id=question.question_id,
                )
            except GeneratorUnavailableError as error:
                if error.result is not None:
                    _warn_if_error(error.result)
                    results.append(error.result)
                logger.error("Generator unavailable at %s: %s", config.base_url, error)
                code = EXIT_GENERATOR_UNAVAILABLE
                break
            _warn_if_error(result)
            results.append(result)
        log_path = answerer.log_path

    if code != EXIT_GENERATOR_UNAVAILABLE and any(
        result.status == STATUS_ERROR for result in results
    ):
        code = EXIT_QUESTION_ERRORS
    if batch:
        _write_stdout(render_batch(results, log_path))
    elif results:
        _write_stdout(render_json(results[0]) if args.json else render_text(results[0]))
    return code


if __name__ == "__main__":
    sys.exit(main())
