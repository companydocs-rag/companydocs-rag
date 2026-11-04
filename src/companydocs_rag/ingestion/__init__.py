"""Ingestion: the parse pass over incoming documents, and text normalization.

Step 1 of the pipeline. For each document it determines, from a single text
extraction, a quality tier (Clean / Decent / Garbage), and from that text plus
OCR text for pages without a text layer, the document type (with a confidence
score). The tier does not select the chunking route: the chunker
(:mod:`companydocs_rag.chunking`) chooses it from its own heading recognition
and copies each page's tier into its chunks as provenance.

:func:`classify_and_assess` is the entry point for the rest of the pipeline.

``quality.py`` is always loaded as part of this package (``.parse_pass``
imports it at module level), so a plain ``import companydocs_rag.ingestion``
does load it. That is safe: ``quality.py`` itself only imports ``preocr``
lazily, inside the functions that call it (see that module's docstring for
why), so this package import never risks pulling in preocr or python-magic.
"""

from .extraction import DocumentText, PageText, extract_pages
from .parse_pass import classify_and_assess
from .quality import (
    TIER_CLEAN,
    TIER_DECENT,
    TIER_GARBAGE,
    QualityAssessment,
    assess_quality,
    run_preocr,
)
from .structure import StructureResult, analyse_structure
from .type_detection import UNKNOWN_TYPE, TypeDetectionResult, detect_type

__all__ = [
    "classify_and_assess",
    "extract_pages",
    "DocumentText",
    "PageText",
    "detect_type",
    "TypeDetectionResult",
    "UNKNOWN_TYPE",
    "analyse_structure",
    "StructureResult",
    "assess_quality",
    "QualityAssessment",
    "run_preocr",
    "TIER_CLEAN",
    "TIER_DECENT",
    "TIER_GARBAGE",
]
