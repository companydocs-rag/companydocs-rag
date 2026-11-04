"""Shared helpers for the chunking tests (spec 005).

No real document is read and nothing under ``data/`` is touched: normalized
records are built here from fabricated text, in the shape
``companydocs-normalize`` writes (``schema_version`` 1). The keys the chunker
does not read carry neutral values. :func:`check_chunks` checks what every
chunk of every test document must satisfy (A3, A4, A12), against the stream
rebuilt here, independently of the chunker.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

# The output keys, in order (External surface).
DOCUMENT_KEYS = [
    "schema_version",
    "chunker_version",
    "file",
    "status",
    "normalized_hash",
    "language",
    "language_source",
    "structure_hint",
    "route",
    "empty_pages",
    "heading_evidence",
    "chunk_count",
    "chunks",
]
HEADING_EVIDENCE_KEYS = ["family", "candidate_counts", "runs"]
CANDIDATE_COUNT_KEYS = ["keyword", "decimal"]
RUN_KEYS = [
    "run_index",
    "first_number",
    "length",
    "start_page",
    "median_section_chars",
    "accepted",
    "reject_reason",
]
CHUNK_KEYS = [
    "chunk_id",
    "chunk_index",
    "file_sha256",
    "text",
    "text_sha256",
    "char_start",
    "char_end",
    "page_start",
    "page_end",
    "text_origins",
    "page_spans",
    "chunk_method",
    "clause_label",
    "clause_number",
    "numbering_run",
    "heading_family",
    "part",
    "part_count",
    "exceeds_max_chars",
]
PAGE_SPAN_KEYS = [
    "page_number",
    "char_start",
    "char_end",
    "text_origin",
    "tier",
    "ocr_mean_confidence",
]
CLAUSE_KEYS = ["clause_label", "clause_number", "numbering_run", "heading_family"]

FILE_SHA256 = "cd" * 32

_FILLER = "Fictieve bepaling over een denkbeeldig pand, alleen bedoeld als testtekst. "


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def body(length: int) -> str:
    """A fabricated body line of exactly ``length`` characters, without a
    line feed. It starts with a capital letter, does not end in whitespace,
    and matches neither heading pattern."""
    assert length >= 1
    text = (_FILLER * (length // len(_FILLER) + 1))[:length]
    return text[:-1] + "." if text.endswith(" ") else text


def filler(length: int, line_length: int = 60) -> str:
    """Body lines, each with its line feed and at most ``line_length``
    characters long, exactly ``length`` characters in all (0, or 2 or
    more)."""
    assert length == 0 or length >= 2, length
    count = -(-length // line_length)
    text = ""
    for position in range(count):
        size = length // count + (1 if position < length % count else 0)
        text += body(size - 1) + "\n"
    assert len(text) == length
    return text


def block(
    heading: str, length: int, inner: Tuple[str, ...] = (), line_length: int = 60
) -> str:
    """``heading`` as a line, then the ``inner`` lines, then body lines, every
    line with its line feed: exactly ``length`` characters, so a section that
    starts here and ends at the next heading is ``length`` long."""
    text = heading + "\n" + "".join(line + "\n" for line in inner)
    return text + filler(length - len(text), line_length)


def lines(*parts: str) -> str:
    """One string per line, joined with a line feed."""
    return "\n".join(parts)


# The provenance normalize carries for a page of each text origin.
_PROVENANCE = {
    "native": ("native", "Clean", None, None),
    "ocr": ("no_text_layer", None, 90.0, None),
    "none": ("no_text_layer", None, None, "ocr_empty"),
}


def normalized_page(
    page_number: int,
    text: str = "",
    *,
    text_origin: Optional[str] = None,
    language: Optional[str] = "nld",
    language_source: Optional[str] = "index",
    **overrides: Any,
) -> Dict[str, Any]:
    """One fabricated normalized page. The text origin defaults to native,
    or none for an empty text; ``overrides`` replaces any key."""
    origin = text_origin or ("native" if text else "none")
    source, tier, confidence, reason = _PROVENANCE[origin]
    page = {
        "page_number": page_number,
        "source": source,
        "text_origin": origin,
        "no_text_reason": reason,
        "language": language,
        "language_source": language_source,
        "tier": tier,
        "ocr_mean_confidence": confidence,
        "input_sha256": sha256_hex(text),
        "text": text,
        "normalized_sha256": sha256_hex(text),
        "removed_lines": [],
        "joins": [],
        "counts": {},
    }
    page.update(overrides)
    return page


def normalized_record(
    *pages: Union[str, Dict[str, Any]],
    path: str = "invoer/fictief.pdf",
    sha256: Optional[str] = FILE_SHA256,
    status: str = "processed",
) -> Dict[str, Any]:
    """A fabricated normalized record (``schema_version`` 1). A string is a
    native page (none when it is ""), numbered by its position; a dict from
    :func:`normalized_page` is used as it is."""
    built = [
        normalized_page(number, item) if isinstance(item, str) else item
        for number, item in enumerate(pages, start=1)
    ]
    pairs = [[page["page_number"], page["normalized_sha256"]] for page in built]
    processed = status == "processed"
    return {
        "schema_version": 1,
        "normalization_version": 1,
        "unicode_version": "15.0.0",
        "file": {"path": path, "sha256": sha256},
        "status": status,
        "page_count": len(built),
        "page_labels": {
            "status": "applied" if processed else "not_applicable",
            "pages_considered": len(built),
            "constants": {},
            "runs": [],
        },
        "header_footer": {
            "status": "not_built" if processed else "not_applicable",
            "pages_considered": len(built),
            "constants": {},
        },
        "normalized_sha256": sha256_hex(
            json.dumps(pairs, separators=(",", ":"), ensure_ascii=True)
        ),
        "pages": built,
    }


# --- The stream, rebuilt (K2) ---------------------------------------------------------


def stream_layout(
    record: Mapping[str, Any],
) -> Tuple[str, List[Tuple[Optional[Mapping[str, Any]], int, int]]]:
    """K2, rebuilt here: the stream, and its stretches in order, each
    ``(page, start, end)`` for a page with text or ``(None, start, end)`` for
    the line feed between two pages."""
    pages = sorted(record["pages"], key=lambda page: page["page_number"])
    texts = [page for page in pages if page["text"] != ""]
    stretches: List[Tuple[Optional[Mapping[str, Any]], int, int]] = []
    offset = 0
    for page in texts:
        if stretches:
            stretches.append((None, offset, offset + 1))
            offset += 1
        stretches.append((page, offset, offset + len(page["text"])))
        offset += len(page["text"])
    return "\n".join(page["text"] for page in texts), stretches


def stream_of(record: Mapping[str, Any]) -> str:
    return stream_layout(record)[0]


def line_starts(stream: str) -> List[int]:
    """The offset of every line of ``stream``."""
    starts = [0] if stream else []
    starts.extend(
        index + 1
        for index, char in enumerate(stream)
        if char == "\n" and index + 1 < len(stream)
    )
    return starts


def check_chunks(
    record: Mapping[str, Any], document: Mapping[str, Any], *, max_chars: int
) -> None:
    """What every output document and every chunk in it must satisfy."""
    assert list(document) == DOCUMENT_KEYS
    evidence = document["heading_evidence"]
    assert list(evidence) == HEADING_EVIDENCE_KEYS
    assert list(evidence["candidate_counts"]) == CANDIDATE_COUNT_KEYS
    for run in evidence["runs"]:
        assert list(run) == RUN_KEYS

    stream, stretches = stream_layout(record)
    boundaries = set(line_starts(stream)) | {len(stream)}
    chunks = document["chunks"]
    assert document["chunk_count"] == len(chunks)
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    starts = [chunk["char_start"] for chunk in chunks]
    assert starts == sorted(set(starts))  # ascending and distinct
    covered = [False] * len(stream)
    file_sha256 = record["file"]["sha256"]

    for chunk in chunks:
        assert list(chunk) == CHUNK_KEYS
        start, end, text = chunk["char_start"], chunk["char_end"], chunk["text"]
        assert 0 <= start < end <= len(stream)
        assert text == stream[start:end]
        assert start in boundaries and end in boundaries
        assert not text.isspace()

        spans = chunk["page_spans"]
        assert spans, chunk
        for span in spans:
            assert list(span) == PAGE_SPAN_KEYS
        remaining = iter(spans)
        expected, rebuilt = [], []
        for page, low, high in stretches:
            first, last = max(start, low), min(end, high)
            if first >= last:
                continue
            if page is None:  # a page join inside the chunk
                rebuilt.append("\n")
                continue
            span = next(remaining)
            rebuilt.append(page["text"][span["char_start"] : span["char_end"]])
            expected.append(
                {
                    "page_number": page["page_number"],
                    "char_start": first - low,
                    "char_end": last - low,
                    "text_origin": page["text_origin"],
                    "tier": page["tier"],
                    "ocr_mean_confidence": page["ocr_mean_confidence"],
                }
            )
        assert next(remaining, None) is None
        assert spans == expected
        assert "".join(rebuilt) == text

        assert chunk["text_sha256"] == sha256_hex(text)
        assert chunk["file_sha256"] == file_sha256
        assert (
            chunk["chunk_id"]
            == (sha256_hex(f"{file_sha256}:{start}:{chunk['text_sha256']}")[:20])
        )
        assert chunk["page_start"] == spans[0]["page_number"]
        assert chunk["page_end"] == spans[-1]["page_number"]
        assert chunk["text_origins"] == sorted({span["text_origin"] for span in spans})
        assert chunk["exceeds_max_chars"] is (len(text) > max_chars)

        method = chunk["chunk_method"]
        clause = [chunk[key] for key in CLAUSE_KEYS]
        if method in ("heading_section", "heading_section_part"):
            assert None not in clause, chunk
        else:
            assert method in ("heading_section_overflow", "generic")
            assert clause == [None] * len(CLAUSE_KEYS)
        if method == "heading_section_part":
            assert 1 <= chunk["part"] <= chunk["part_count"]
        else:
            assert (chunk["part"], chunk["part_count"]) == (None, None)

        for position in range(start, end):
            covered[position] = True

    for position, char in enumerate(stream):
        if not char.isspace():
            assert covered[position], f"offset {position} lies in no chunk"
