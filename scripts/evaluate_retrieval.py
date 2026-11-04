"""Measure retrieval on the shared question set (evaluation/questions.csv).

Usage (from the repository root, nothing else holding the index):

    uv run python scripts/evaluate_retrieval.py --label baseline
    uv run python scripts/evaluate_retrieval.py --label <name> --index-dir <dir>

Settings come from ``config/settings.yaml`` or, as everywhere in the pipeline,
from the file that ``COMPANYDOCS_RAG_SETTINGS`` names; an experiment points that
variable at its own copy and builds its own index with ``--index-dir``.

For every question it runs the same search as ``companydocs-search`` and finds
the rank of the first candidate that matches an expected source: same source
id (the PDF's file name without ``.pdf``) and, when the question names pages,
a page range that overlaps them. It reports:

- ``doc_hit@k``: share of questions with a matching source id in the top k;
- ``page_hit@k``: the same, also requiring the page overlap;
- ``mrr``: mean reciprocal rank of the first page match (0 when none).

The result goes to ``evaluation/results/<label>.json`` (per question and the
summary, with the settings file and the git commit), so every contribution can
show its numbers against the baseline run on the same corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evaluation" / "questions.csv"
RESULTS = ROOT / "evaluation" / "results"
DEFAULT_INDEX = ROOT / "data" / "processed" / "index"
KS = (1, 3, 5, 10)

DELIMITER = ";"
ENCODING = "utf-8-sig"
COLUMNS = ("question_id", "question", "expected", "author")


def parse_expected(text: str) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
    """``cao-abu-2026:14-15|wtta-stb-2025-385`` -> [(id, (14, 15)), (id, None)]."""
    expected = []
    for part in text.split("|"):
        part = part.strip()
        if not part:
            continue
        source, _, pages = part.partition(":")
        span = None
        if pages:
            first, _, last = pages.partition("-")
            span = (int(first), int(last or first))
        expected.append((source.strip(), span))
    if not expected:
        raise ValueError("no expected source")
    return expected


def read_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding=ENCODING, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=DELIMITER)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        questions = []
        for row in reader:
            if not (row.get("question") or "").strip():
                continue
            try:
                expected = parse_expected(row["expected"] or "")
            except ValueError as error:
                raise ValueError(f"{path}: {row['question_id']}: {error}") from None
            questions.append({**row, "expected_parsed": expected})
    return questions


def matches(
    candidate: Dict[str, Any], expected: List[Tuple[str, Optional[Tuple[int, int]]]]
) -> Tuple[bool, bool]:
    """(source matches, source and pages match) for one candidate."""
    source = Path(candidate["file_path"]).stem
    doc = page = False
    for want, span in expected:
        if source != want:
            continue
        doc = True
        if span is None or (
            candidate["page_start"] <= span[1] and candidate["page_end"] >= span[0]
        ):
            page = True
    return doc, page


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="name of this run")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    args = parser.parse_args(argv)

    from companydocs_rag.retrieval.search import IndexUnusableError, Retriever

    try:
        questions = read_questions(args.questions)
    except (OSError, ValueError) as error:
        print(f"Cannot start: {error}", file=sys.stderr)
        return 2
    if not questions:
        print(f"Cannot start: no questions in {args.questions}", file=sys.stderr)
        return 2

    try:
        retriever = Retriever(args.index_dir)
    except IndexUnusableError as error:
        print(f"Index unusable: {error}", file=sys.stderr)
        return 1

    per_question = []
    with retriever:
        for q in questions:
            result = retriever.search(q["question"])
            doc_rank = page_rank = None
            for candidate in result.candidates:
                doc, page = matches(candidate, q["expected_parsed"])
                if doc and doc_rank is None:
                    doc_rank = candidate["rank"]
                if page and page_rank is None:
                    page_rank = candidate["rank"]
            per_question.append(
                {
                    "question_id": q["question_id"],
                    "author": q["author"],
                    "doc_rank": doc_rank,
                    "page_rank": page_rank,
                    "top3": [
                        f"{Path(c['file_path']).stem}:{c['page_start']}-{c['page_end']}"
                        for c in result.candidates[:3]
                    ],
                }
            )

    n = len(per_question)
    summary: Dict[str, float] = {}
    for k in KS:
        summary[f"doc_hit@{k}"] = (
            sum(1 for r in per_question if r["doc_rank"] and r["doc_rank"] <= k) / n
        )
        summary[f"page_hit@{k}"] = (
            sum(1 for r in per_question if r["page_rank"] and r["page_rank"] <= k) / n
        )
    summary["mrr"] = sum(1 / r["page_rank"] for r in per_question if r["page_rank"]) / n

    record = {
        "label": args.label,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "settings_file": os.environ.get("COMPANYDOCS_RAG_SETTINGS")
        or "config/settings.yaml",
        "index_dir": str(args.index_dir),
        "questions": n,
        "summary": summary,
        "per_question": per_question,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.label}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", "utf-8")

    print(f"{args.label}: {n} questions, commit {record['git_commit']}")
    for key, value in summary.items():
        print(f"  {key:<12} {value:.3f}")
    print(f"written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
