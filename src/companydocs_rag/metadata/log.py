"""The extract log (spec 008, D10, R17): one JSONL file per ``extract`` run.

The file is ``extract-<YYYYMMDDTHHMMSSZ>.jsonl`` in ``log_dir``, named after
the UTC time the run started, to the second; when that name exists, a suffix
``-2``, ``-3``, ... goes before ``.jsonl``. Folders are created as needed.
The file is created, exclusively, when the first record is written, so a run
without extraction requests leaves none. Each record is one line of compact
JSON (``ensure_ascii=False``), UTF-8, ended by a line feed on every platform
and flushed when written. A failure to write raises :class:`LogWriteError`,
which the run does not catch.

This module reads no clock: the run passes the times in (:mod:`.extract`).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Mapping, Optional

FILE_PREFIX = "extract-"
FILE_SUFFIX = ".jsonl"
_NAME_TIME_FORMAT = "%Y%m%dT%H%M%SZ"


def format_timestamp(moment: datetime) -> str:
    """``YYYY-MM-DDTHH:MM:SS.sssZ`` for a UTC ``moment`` (R17)."""
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


class LogWriteError(Exception):
    """The log could not be written; it ends the run (R14)."""


class ExtractLog:
    """The log of one run. :attr:`path` is None until the first record is
    written."""

    def __init__(self, folder: Path, started: datetime) -> None:
        self._folder = Path(folder)
        self._stem = FILE_PREFIX + started.strftime(_NAME_TIME_FORMAT)
        self._handle: Optional[IO[bytes]] = None
        self.path: Optional[Path] = None

    def _open(self) -> IO[bytes]:
        self._folder.mkdir(parents=True, exist_ok=True)
        number = 1
        while True:
            suffix = "" if number == 1 else f"-{number}"
            path = self._folder / (self._stem + suffix + FILE_SUFFIX)
            try:
                handle = path.open("xb")
            except FileExistsError:
                number += 1
                continue
            self.path = path
            return handle

    def write(self, record: Mapping[str, Any]) -> None:
        """Append ``record`` as one line, creating the file first if needed."""
        try:
            if self._handle is None:
                self._handle = self._open()
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            self._handle.write((line + "\n").encode("utf-8"))
            self._handle.flush()
        except (OSError, TypeError, ValueError) as error:
            raise LogWriteError(
                f"could not write the log in {self._folder}: "
                f"{type(error).__name__}: {error}"
            ) from error

    def close(self) -> None:
        """Close the file; :attr:`path` keeps naming it."""
        handle, self._handle = self._handle, None
        if handle is not None:
            handle.close()
