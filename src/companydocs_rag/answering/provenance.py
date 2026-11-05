"""The provenance of a cited source (spec 007, D9, R13).

A source carries ``prompt_number``, ``chunk_id``, ``file_sha256``,
``file_path``, ``page_start``, ``page_end`` and ``clause_label`` (a hint,
passed through), then ``pages``: one entry per ``page_spans`` page whose
``text_origin`` is not ``none``, with ``page`` (the span's ``page_number``),
``text_origin``, ``tier``, ``ocr_mean_confidence`` and ``provenance``.
``provenance`` is ``ocr`` for OCR text, ``native_garbage`` for text-layer
text with tier Garbage, and ``native`` for text-layer text with tier Clean,
Decent or null. Display texts belong to the interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from companydocs_rag.ingestion.quality import (
    TEXT_ORIGIN_NONE,
    TEXT_ORIGIN_OCR,
    TIER_GARBAGE,
)

PROVENANCE_OCR = "ocr"
PROVENANCE_NATIVE = "native"
PROVENANCE_NATIVE_GARBAGE = "native_garbage"


def page_provenance(text_origin: str, tier: Any) -> str:
    """``ocr``, ``native`` or ``native_garbage`` (R13)."""
    if text_origin == TEXT_ORIGIN_OCR:
        return PROVENANCE_OCR
    if tier == TIER_GARBAGE:
        return PROVENANCE_NATIVE_GARBAGE
    return PROVENANCE_NATIVE


def source_pages(page_spans: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The ``pages`` of a source: its spans with text, in their order."""
    return [
        {
            "page": span["page_number"],
            "text_origin": span["text_origin"],
            "tier": span["tier"],
            "ocr_mean_confidence": span["ocr_mean_confidence"],
            "provenance": page_provenance(span["text_origin"], span["tier"]),
        }
        for span in page_spans
        if span["text_origin"] != TEXT_ORIGIN_NONE
    ]


def source_entry(prompt_number: int, chunk: Mapping[str, Any]) -> Dict[str, Any]:
    """One entry of a record's ``sources``, keys in D9's order."""
    return {
        "prompt_number": prompt_number,
        "chunk_id": chunk["chunk_id"],
        "file_sha256": chunk["file_sha256"],
        "file_path": chunk["file_path"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "clause_label": chunk["clause_label"],
        "pages": source_pages(chunk["page_spans"]),
    }
