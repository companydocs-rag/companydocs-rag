"""The highlight of a cited passage on its page (spec 011).

``GET /markering?sha256=<hash>&nummer=<n>&chunk=<id>`` gives the lines of
page ``n`` that hold the text of chunk ``id``: the passage the model read.
Each line is a rectangle ``[left, top, width, height]`` as fractions of the
page, so the page's script can draw it over the image at any zoom.

The words of a page and their boxes come from what the parse pass used:

- an OCR page (``text_origin`` ``ocr``): the word rows (level 5) of
  Tesseract's TSV in the OCR cache, divided by the page row's size (level
  1);
- a text-layer page (``native``): ``pdfplumber``'s ``extract_words``, divided
  by the page's size, only when the page has no rotation, since the words
  are in the unrotated page and the image is rotated.

The chunk's text on that page is the slice its ``page_spans`` entry names;
the spans' slices are the chunk text in order, joined by one fixed
separator, and when the lengths do not add up the whole chunk text stands
for each page. Words are compared case-folded, letters and digits only
(normalization rejoins hyphenated words and drops page labels, so some
differ). :class:`difflib.SequenceMatcher` aligns the passage's words with
the page's; blocks of fewer than :data:`MIN_BLOCK` words are left out for a
passage of :data:`SHORT_PASSAGE` words or more, and a passage of fewer than
:data:`MIN_PASSAGE_WORDS` words gets no highlight. The highlight runs from the
first to the last matched word, and only when at least
:data:`MIN_COVERAGE` of the passage's words were found and at least
:data:`MIN_DENSITY` of the words in that run matched. Otherwise, and for
anything that cannot be read, there is no highlight: none is better than a
wrong one.

At startup the chunk files are read once (only chunks of mapped files are
kept) and the OCR cache is indexed by the hash and page number in each
entry's key, read from the start of the file; when two entries share both,
the most recently written one is used. A page's words are read per request.
Nothing is written to disk.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from companydocs_rag.ingestion.quality import TEXT_ORIGIN_NATIVE, TEXT_ORIGIN_OCR

from .pageview import SHA256_PATTERN

logger = logging.getLogger(__name__)

CHUNK_ID_PATTERN = re.compile("[0-9a-f]{8,64}")
# The share of the passage's words that must be found on the page, in order.
MIN_COVERAGE = 0.6
# The share of the highlighted run's words that must be matched words.
MIN_DENSITY = 0.5
# Matching blocks shorter than this are ignored for a passage this long.
MIN_BLOCK = 2
SHORT_PASSAGE = 6
# A passage of fewer words is not highlighted: two words can match anywhere.
MIN_PASSAGE_WORDS = 3
# The start of an OCR cache entry that holds its key.
KEY_PREFIX_BYTES = 2048
TSV_WORD_LEVEL = "5"
TSV_PAGE_LEVEL = "1"

_KEY_SHA256 = re.compile(r'"sha256":\s*"([0-9a-f]{64})"')
_KEY_PAGE = re.compile(r'"page_number":\s*([0-9]+)')
_ROUND = 5


@dataclass(frozen=True)
class Word:
    """One word on a page: its comparison key, its line and its box as
    fractions of the page."""

    key: str
    line: Tuple[Any, ...]
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class Passage:
    """A chunk's text on one page and where that text came from."""

    text: str
    text_origin: Any


def word_key(text: str) -> str:
    """A word as compared: case-folded, letters and digits only."""
    return "".join(character for character in text.casefold() if character.isalnum())


def _keys(text: str) -> List[str]:
    return [key for key in (word_key(part) for part in text.split()) if key]


def passages(chunk: Mapping[str, Any]) -> Dict[int, Passage]:
    """The chunk's text per page number, from its ``page_spans``."""
    text = chunk.get("text")
    spans = chunk.get("page_spans")
    if not isinstance(text, str) or not isinstance(spans, list) or not spans:
        return {}
    lengths = []
    for span in spans:
        if not isinstance(span, Mapping):
            return {}
        start, end = span.get("char_start"), span.get("char_end")
        if type(start) is not int or type(end) is not int or end < start:
            return {}
        lengths.append(end - start)
    gap = len(text) - sum(lengths)
    joins = len(spans) - 1
    separator = gap // joins if joins and gap >= 0 and gap % joins == 0 else None
    exact = (joins == 0 and gap == 0) or separator is not None
    found: Dict[int, Passage] = {}
    position = 0
    for span, length in zip(spans, lengths):
        page = span.get("page_number")
        if type(page) is not int:
            continue
        piece = text[position : position + length] if exact else text
        found[page] = Passage(piece, span.get("text_origin"))
        position += length + (separator or 0)
    return found


def load_chunks(
    chunks_dir: Optional[Path], files: Mapping[str, Path]
) -> Dict[str, Tuple[str, Dict[int, Passage]]]:
    """Every chunk of a mapped file under ``chunks_dir``: its id to its
    file's hash and its passages."""
    found: Dict[str, Tuple[str, Dict[int, Passage]]] = {}
    if chunks_dir is None or not chunks_dir.is_dir():
        return found
    for path in sorted(chunks_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            logger.warning(
                "No highlights from %s: %s: %s", path, type(error).__name__, error
            )
            continue
        chunks = data.get("chunks") if isinstance(data, Mapping) else None
        for chunk in chunks if isinstance(chunks, list) else []:
            if not isinstance(chunk, Mapping):
                continue
            chunk_id, digest = chunk.get("chunk_id"), chunk.get("file_sha256")
            if (
                isinstance(chunk_id, str)
                and isinstance(digest, str)
                and digest in files
                and CHUNK_ID_PATTERN.fullmatch(chunk_id) is not None
            ):
                found[chunk_id] = (digest, passages(chunk))
    return found


def index_ocr_cache(cache_dir: Optional[Path]) -> Dict[Tuple[str, int], Path]:
    """The OCR cache's entries by the hash and page number of their key."""
    found: Dict[Tuple[str, int], Tuple[float, Path]] = {}
    if cache_dir is None or not cache_dir.is_dir():
        return {}
    for path in cache_dir.glob("*.json"):
        try:
            with path.open("rb") as handle:
                start = handle.read(KEY_PREFIX_BYTES).decode("utf-8", errors="replace")
            written = path.stat().st_mtime
        except OSError:
            continue
        digest, page = _KEY_SHA256.search(start), _KEY_PAGE.search(start)
        if digest is None or page is None:
            continue
        place = (digest.group(1), int(page.group(1)))
        if place not in found or written > found[place][0]:
            found[place] = (written, path)
    return {place: path for place, (_, path) in found.items()}


def ocr_words(path: Path) -> List[Word]:
    """The word rows of a cached TSV, in their order, as fractions of the
    page row's size."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tsv = data.get("tsv") if isinstance(data, Mapping) else None
    if not isinstance(tsv, str):
        return []
    width = height = None
    words: List[Word] = []
    reader = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        try:
            left, top = int(row["left"]), int(row["top"])
            right, bottom = left + int(row["width"]), top + int(row["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if row.get("level") == TSV_PAGE_LEVEL:
            width, height = right, bottom
            continue
        key = word_key(row.get("text") or "")
        if row.get("level") != TSV_WORD_LEVEL or not key or not width or not height:
            continue
        line = (row.get("block_num"), row.get("par_num"), row.get("line_num"))
        words.append(
            Word(key, line, left / width, top / height, right / width, bottom / height)
        )
    return words


def native_words(pdf_path: Path, page_number: int) -> List[Word]:
    """``pdfplumber``'s words of one page as fractions of its size; none for
    a rotated page."""
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        if not 1 <= page_number <= len(pdf.pages):
            return []
        page = pdf.pages[page_number - 1]
        if (getattr(page, "rotation", 0) or 0) % 360 != 0:
            return []
        width, height = float(page.width), float(page.height)
        if width <= 0 or height <= 0:
            return []
        words = []
        for item in page.extract_words():
            key = word_key(str(item.get("text", "")))
            if not key:
                continue
            words.append(
                Word(
                    key,
                    (),
                    float(item["x0"]) / width,
                    float(item["top"]) / height,
                    float(item["x1"]) / width,
                    float(item["bottom"]) / height,
                )
            )
        return words


def _native_lines(words: Sequence[Word]) -> List[Word]:
    """Native words with a line number: a new line where a word starts left
    of the previous one or below its box."""
    lines: List[Word] = []
    line = 0
    previous: Optional[Word] = None
    for word in words:
        if previous is not None and (
            word.left < previous.left or word.top >= previous.bottom
        ):
            line += 1
        lines.append(
            Word(word.key, (line,), word.left, word.top, word.right, word.bottom)
        )
        previous = word
    return lines


def matched_run(
    page_keys: Sequence[str], passage_keys: Sequence[str]
) -> Optional[Tuple[int, int]]:
    """The first and last index of the page words that hold the passage, or
    None when the alignment is too weak."""
    if not page_keys or len(passage_keys) < MIN_PASSAGE_WORDS:
        return None
    smallest = MIN_BLOCK if len(passage_keys) >= SHORT_PASSAGE else 1
    blocks = [
        block
        for block in SequenceMatcher(
            None, page_keys, passage_keys, autojunk=False
        ).get_matching_blocks()
        if block.size >= smallest
    ]
    matched = sum(block.size for block in blocks)
    if not blocks or matched < MIN_COVERAGE * len(passage_keys):
        return None
    first, last = blocks[0].a, blocks[-1].a + blocks[-1].size - 1
    if matched < MIN_DENSITY * (last - first + 1):
        return None
    return first, last


def line_boxes(words: Sequence[Word]) -> List[List[float]]:
    """One rectangle ``[left, top, width, height]`` per line of ``words``."""
    boxes: List[List[float]] = []
    current: Optional[Tuple[Any, ...]] = None
    for word in words:
        if word.line != current:
            boxes.append([word.left, word.top, word.right, word.bottom])
            current = word.line
            continue
        box = boxes[-1]
        box[0], box[1] = min(box[0], word.left), min(box[1], word.top)
        box[2], box[3] = max(box[2], word.right), max(box[3], word.bottom)
    return [
        [
            round(left, _ROUND),
            round(top, _ROUND),
            round(right - left, _ROUND),
            round(bottom - top, _ROUND),
        ]
        for left, top, right, bottom in boxes
    ]


class Highlighter:
    """The highlights of the chunks under ``chunks_dir`` on the pages of
    ``files``, with the OCR cache under ``ocr_cache_dir``."""

    def __init__(
        self,
        files: Mapping[str, Path],
        chunks_dir: Optional[Path] = None,
        ocr_cache_dir: Optional[Path] = None,
    ) -> None:
        self.files = files
        self.chunks = load_chunks(chunks_dir, files)
        self.ocr_cache = index_ocr_cache(ocr_cache_dir)

    def boxes(self, file_sha256: str, page: int, chunk_id: str) -> List[List[float]]:
        """The line rectangles of chunk ``chunk_id`` on page ``page`` of the
        file ``file_sha256``; empty when there is no reliable highlight."""
        if SHA256_PATTERN.fullmatch(file_sha256) is None:
            return []
        found = self.chunks.get(chunk_id)
        if found is None or found[0] != file_sha256 or file_sha256 not in self.files:
            return []
        passage = found[1].get(page)
        if passage is None:
            return []
        words = self._words(file_sha256, page, passage.text_origin)
        run = matched_run([word.key for word in words], _keys(passage.text))
        if run is None:
            return []
        return line_boxes(words[run[0] : run[1] + 1])

    def _words(self, file_sha256: str, page: int, text_origin: Any) -> List[Word]:
        try:
            if text_origin == TEXT_ORIGIN_OCR:
                path = self.ocr_cache.get((file_sha256, page))
                return [] if path is None else ocr_words(path)
            if text_origin == TEXT_ORIGIN_NATIVE:
                return _native_lines(native_words(self.files[file_sha256], page))
        except Exception as error:  # noqa: BLE001 - no highlight rather than an error
            logger.warning(
                "No highlight for %s page %d: %s",
                file_sha256,
                page,
                type(error).__name__,
            )
        return []
