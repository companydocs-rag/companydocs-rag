"""The ``companydocs-demo`` command (spec 009, D1, D3, D6, D13, R2-R4, R14,
R19).

    companydocs-demo [--port PORT] [--index-dir DIR]

A local, single-user browser interface over spec 007's ``Answerer`` on
``http://127.0.0.1:<port>/``. The index is ``--index-dir`` (default
``data/processed/index`` under the project root, as for
``companydocs-answer``); the port is ``--port``, else the ``interface.port``
setting. The PDFs whose pages it shows are under ``data/raw`` of the project
root, and the document types come from spec 008's override file,
``data/manual/document_type_overrides.csv``, and the client index; neither
is a flag. The generator has to run (Ollama on localhost by default), and
the command holds the index lock for its whole run. It opens no browser.

Startup, in this order; each failure is one ERROR on stderr and an exit
code, and nothing later is opened (R2):

1. the arguments (2);
2. ``--index-dir`` must be a folder (2);
3. the settings ``embeddings``, ``retrieval``, ``answering``, ``interface``
   (2);
4. the prompt file and the API key variable (2);
5. the raw folder must be a folder (2);
6. the type override file (2);
7. the exclusive bind on 127.0.0.1 and the port (2);
8. the hash to file map and the document types (WARNINGs only);
9. the ``Answerer``: index unusable 1, settings 2, the warm-up 3;
10. the URL, the only line on stdout; then serving until Ctrl+C, which
    closes the server socket and the ``Answerer`` and exits 0.

Logging goes to stderr as ``LEVEL message`` at WARNING, plus one line per
request (:mod:`.server`). Startup messages may name paths and file names.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx

from companydocs_rag.answering.answerer import (
    Answerer,
    GeneratorError,
    GeneratorUnavailableError,
)
from companydocs_rag.answering.cli import (
    EXIT_CANNOT_START,
    EXIT_GENERATOR_UNAVAILABLE,
    EXIT_INDEX_UNUSABLE,
)
from companydocs_rag.answering.generator import read_api_key
from companydocs_rag.answering.prompt import load_system_prompt
from companydocs_rag.answering.settings import answering_settings, resolve_project_path
from companydocs_rag.embeddings.dense import SettingsError, embeddings_settings
from companydocs_rag.metadata.cli import DEFAULT_TYPE_OVERRIDES_RELATIVE_PATH
from companydocs_rag.metadata.doctype import ManualFileError, read_type_overrides
from companydocs_rag.retrieval.search import IndexUnusableError, retrieval_settings
from companydocs_rag.vectorstore.cli import default_index_dir

from .htmlpage import build_page
from .pageview import build_file_map, document_types
from .server import HOST, DemoApp, DemoServer, serve
from .settings import PORT_MAX, PORT_MIN, interface_settings

logger = logging.getLogger(__name__)

EXIT_STOPPED = 0
DEFAULT_RAW_RELATIVE_PATH = Path("data") / "raw"

_DIGITS = re.compile("[0-9]+")


def port_number(text: str) -> int:
    """``--port``: an integer from 1 to 65535."""
    if _DIGITS.fullmatch(text) is None or not PORT_MIN <= int(text) <= PORT_MAX:
        raise argparse.ArgumentTypeError(
            f"not a port number from {PORT_MIN} to {PORT_MAX}: {text}"
        )
    return int(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companydocs-demo",
        description=(
            "A local browser interface over the answer layer: ask a question, "
            "read the answer with its sources, and open each cited page of the "
            "PDF. Listens on 127.0.0.1 only; prints the URL and opens no browser."
        ),
    )
    parser.add_argument(
        "--port",
        type=port_number,
        default=None,
        help="The port on 127.0.0.1, 1-65535 (default: the interface.port setting)",
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


def _write_url(port: int) -> None:
    sys.stdout.write(f"http://{HOST}:{port}/\n")
    sys.stdout.flush()


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    raw_dir: Optional[Path] = None,
    type_overrides: Optional[Path] = None,
    retriever: Any = None,
    transport: Optional[httpx.BaseTransport] = None,
    stop: Optional[threading.Event] = None,
) -> int:
    """The command. The keyword-only seams (R19): ``raw_dir`` replaces
    ``data/raw``, ``type_overrides`` the override file, ``retriever`` and
    ``transport`` go to the ``Answerer`` (as for ``companydocs-answer``), and
    ``stop``, when set, ends the serving as Ctrl+C does."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if not args.index_dir.is_dir():
        logger.error("Not a folder: %s", args.index_dir)
        return EXIT_CANNOT_START
    try:
        embeddings_settings()
        retrieval_settings()
        config = answering_settings()
        interface = interface_settings()
        load_system_prompt(config)
        read_api_key(config)
    except SettingsError as error:
        logger.error("Invalid settings: %s", error)
        return EXIT_CANNOT_START

    raw = (
        raw_dir
        if raw_dir is not None
        else resolve_project_path(str(DEFAULT_RAW_RELATIVE_PATH))
    )
    if not raw.is_dir():
        logger.error("Not a folder: %s", raw)
        return EXIT_CANNOT_START
    overrides_path = (
        type_overrides
        if type_overrides is not None
        else resolve_project_path(str(DEFAULT_TYPE_OVERRIDES_RELATIVE_PATH))
    )
    try:
        overrides = read_type_overrides(overrides_path)
    except ManualFileError as error:
        logger.error("Invalid type overrides file %s", error.describe())
        return EXIT_CANNOT_START

    port = interface.port if args.port is None else args.port
    try:
        server = DemoServer(port)
    except OSError as error:
        logger.error(
            "Port %d on %s is not available: %s: %s",
            port,
            HOST,
            type(error).__name__,
            error,
        )
        return EXIT_CANNOT_START

    try:
        files = build_file_map(raw)
        types = document_types(files, overrides)
        try:
            answerer = Answerer(
                args.index_dir, retriever=retriever, transport=transport
            )
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

        with answerer:
            server.app = DemoApp(answerer, files, types, interface, build_page())
            _write_url(server.server_port)
            try:
                serve(server, stop if stop is not None else threading.Event())
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    finally:
        server.server_close()
    return EXIT_STOPPED


if __name__ == "__main__":
    sys.exit(main())
