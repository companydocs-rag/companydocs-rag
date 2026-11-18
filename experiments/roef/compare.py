"""Compare retrieval runs of the chunking experiment side by side.

Usage (from the repository root):

    uv run python experiments/roef/compare.py \
        --baseline roef-baseline --variant roef-fixed-1200 [--variant ...] \
        --chunks roef-baseline=data/interim/chunks [--chunks label=dir ...] \
        [--out experiments/roef/results.md]

It reads the result files that ``scripts/evaluate_retrieval.py`` wrote
(``evaluation/results/<label>.json``), the question set and, per run, the
chunk folder the run's index was built from. It writes one Markdown file
with:

- the summary metrics of every run, side by side;
- per variant, the paired difference with the baseline: questions where the
  first page match moved up, down or stayed, with an exact two-sided sign
  test, and a paired bootstrap 95% interval of the difference in MRR and
  page_hit@k (10,000 resamples of the questions, fixed seed, so the file is
  reproducible);
- MRR and page_hit@k per question type, per route of the expected
  document (the route the baseline chunker chose for it) and per kind of
  structure the baseline cut it on (articles, or running page headers that
  formed a numbering run), with the page-header labels per document;
- per chunk folder: documents, routes, chunks, chunk size and pages per
  chunk, the share of chunks with a clause label.

Nothing here changes a result file; it only reads.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evaluation" / "results"
QUESTIONS = ROOT / "evaluation" / "questions.csv"
KS = (1, 3, 5, 10)
RESAMPLES = 10_000
SEED = 20261118


def read_result(label: str) -> Dict[str, Any]:
    return json.loads((RESULTS / f"{label}.json").read_text(encoding="utf-8"))


def read_questions(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["question_id"]: row for row in csv.DictReader(handle, delimiter=";")
        }


def expected_sources(expected: str) -> List[str]:
    return [part.split(":")[0].strip() for part in expected.split("|") if part]


# --- Chunk folders --------------------------------------------------------------------


def chunk_records(folder: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(folder.rglob("*.json"))
        if not path.name.startswith(".")
    ]


def routes_by_source(records: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """The route per source id (the PDF's file name without .pdf)."""
    return {Path(record["file"]["path"]).stem: record["route"] for record in records}


# A clause label that is a publication's running page header rather than a
# heading: a page number, then the name of the official publication. Found by
# reading the baseline chunk files.
PAGE_HEADER_RE = re.compile(
    r"^[0-9]+\s+(Staatscourant|Staatsblad|Tweede Kamer|Kamerstukken)"
)


def page_header_counts(record: Mapping[str, Any]) -> Tuple[int, int]:
    """(chunks with a clause label, of which the label is a page header)."""
    labels = [c["clause_label"] for c in record["chunks"] if c["clause_label"]]
    return len(labels), sum(1 for label in labels if PAGE_HEADER_RE.match(label))


def structure_kinds(records: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Per source id, what the baseline chunker cut on: ``generic route``;
    ``page headers`` when it took the structured route but most of its labels
    are running page headers; else ``articles``."""
    kinds = {}
    for record in records:
        labelled, headers = page_header_counts(record)
        if record["route"] != "structured":
            kind = "generic route"
        elif 2 * headers > labelled:
            kind = "page headers"
        else:
            kind = "articles"
        kinds[Path(record["file"]["path"]).stem] = kind
    return kinds


def header_section(records: Sequence[Mapping[str, Any]]) -> List[str]:
    rows = []
    for record in records:
        labelled, headers = page_header_counts(record)
        if headers:
            rows.append(
                [
                    f"`{Path(record['file']['path']).stem}`",
                    str(record["chunk_count"]),
                    str(labelled),
                    str(headers),
                ]
            )
    return table(["document", "chunks", "with label", "label is a page header"], rows)


def chunk_stats(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    chunks = [chunk for record in records for chunk in record["chunks"]]
    sizes = sorted(len(chunk["text"]) for chunk in chunks)
    pages = [chunk["page_end"] - chunk["page_start"] + 1 for chunk in chunks]
    routes: Dict[str, int] = {}
    for record in records:
        routes[record["route"]] = routes.get(record["route"], 0) + 1
    return {
        "documents": len(records),
        "structured": routes.get("structured", 0),
        "generic": routes.get("generic", 0),
        "chunks": len(chunks),
        "mean_chars": statistics.mean(sizes),
        "median_chars": statistics.median(sizes),
        "p90_chars": sizes[int(0.9 * (len(sizes) - 1))],
        "max_chars": sizes[-1],
        "mean_pages": statistics.mean(pages),
        "multi_page": sum(1 for p in pages if p > 1) / len(pages),
        "labelled": sum(1 for c in chunks if c["clause_label"] is not None)
        / len(chunks),
    }


# --- Metrics --------------------------------------------------------------------------


def reciprocal(rank: Optional[int]) -> float:
    return 1.0 / rank if rank else 0.0


def hit(rank: Optional[int], k: int) -> float:
    return 1.0 if rank and rank <= k else 0.0


def metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    n = len(rows)
    out = {"n": float(n)}
    for k in KS:
        out[f"page_hit@{k}"] = sum(hit(r["page_rank"], k) for r in rows) / n
    out["mrr"] = sum(reciprocal(r["page_rank"]) for r in rows) / n
    return out


def sign_test(up: int, down: int) -> float:
    """Exact two-sided sign test: the chance of a split at least this uneven
    when moving up and moving down are equally likely."""
    n = up + down
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(up, down) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap(
    base: Sequence[float], variant: Sequence[float]
) -> Tuple[float, float, float]:
    """Mean difference (variant - base) and its percentile 95% interval."""
    diffs = [v - b for b, v in zip(base, variant)]
    n = len(diffs)
    rng = random.Random(SEED)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(RESAMPLES)
    )
    return (
        sum(diffs) / n,
        means[int(0.025 * RESAMPLES)],
        means[int(0.975 * RESAMPLES) - 1],
    )


# --- Markdown -------------------------------------------------------------------------


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def fmt(value: float) -> str:
    return f"{value:.3f}"


def signed(value: float) -> str:
    return f"{value:+.3f}"


def summary_section(labels: Sequence[str], results: Mapping[str, Any]) -> List[str]:
    keys = [f"doc_hit@{k}" for k in KS] + [f"page_hit@{k}" for k in KS] + ["mrr"]
    rows = [
        [key] + [fmt(results[label]["summary"][key]) for label in labels]
        for key in keys
    ]
    return table(["metric", *labels], rows)


def paired_section(
    baseline: str, variant: str, results: Mapping[str, Any]
) -> List[str]:
    base = {r["question_id"]: r for r in results[baseline]["per_question"]}
    var = {r["question_id"]: r for r in results[variant]["per_question"]}
    ids = [qid for qid in base if qid in var]
    up = down = same = 0
    for qid in ids:
        b, v = reciprocal(base[qid]["page_rank"]), reciprocal(var[qid]["page_rank"])
        up += v > b
        down += v < b
        same += v == b
    lines = [
        f"First page match of `{variant}` against `{baseline}`, per question: "
        f"higher {up}, lower {down}, the same {same}; two-sided sign test over "
        f"the {up + down} questions that moved: p = {sign_test(up, down):.4f}.",
        "",
    ]
    rows = []
    measures = [("mrr", lambda r: reciprocal(r["page_rank"]))] + [
        (f"page_hit@{k}", lambda r, k=k: hit(r["page_rank"], k)) for k in KS
    ]
    for name, value in measures:
        mean, low, high = paired_bootstrap(
            [value(base[q]) for q in ids], [value(var[q]) for q in ids]
        )
        verdict = "excludes 0" if low > 0 or high < 0 else "includes 0"
        rows.append([name, signed(mean), f"[{signed(low)}, {signed(high)}]", verdict])
    lines.extend(table(["measure", "difference", "95% interval", "interval"], rows))
    return lines


def group_section(
    labels: Sequence[str],
    results: Mapping[str, Any],
    group_of: Mapping[str, str],
    title: str,
) -> List[str]:
    groups = sorted(set(group_of.values()))
    rows = []
    for group in groups:
        for label in labels:
            picked = [
                r
                for r in results[label]["per_question"]
                if group_of.get(r["question_id"]) == group
            ]
            if not picked:
                continue
            m = metrics(picked)
            rows.append(
                [group, str(int(m["n"])), label]
                + [fmt(m[f"page_hit@{k}"]) for k in (1, 3, 10)]
                + [fmt(m["mrr"])]
            )
    return table(
        [title, "n", "run", "page_hit@1", "page_hit@3", "page_hit@10", "mrr"], rows
    )


def chunk_section(stats: Mapping[str, Mapping[str, Any]]) -> List[str]:
    rows = []
    for label, s in stats.items():
        rows.append(
            [
                label,
                f"{s['structured']}/{s['documents']}",
                str(s["chunks"]),
                f"{s['mean_chars']:.0f}",
                f"{s['median_chars']:.0f}",
                str(s["p90_chars"]),
                str(s["max_chars"]),
                f"{s['mean_pages']:.2f}",
                f"{100 * s['multi_page']:.0f}%",
                f"{100 * s['labelled']:.0f}%",
            ]
        )
    return table(
        [
            "run",
            "structured docs",
            "chunks",
            "mean chars",
            "median",
            "p90",
            "max",
            "pages/chunk",
            "multi-page",
            "with label",
        ],
        rows,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument(
        "--chunks",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="the chunk folder a run's index was built from",
    )
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("results.md")
    )
    args = parser.parse_args(argv)

    labels = [args.baseline, *args.variant]
    results = {label: read_result(label) for label in labels}
    questions = read_questions(args.questions)
    folders = dict(item.split("=", 1) for item in args.chunks)
    records = {label: chunk_records(ROOT / folder) for label, folder in folders.items()}

    lines = ["# Chunking experiment: results", ""]
    lines.append(
        "Generated by `experiments/roef/compare.py` from "
        + ", ".join(f"`evaluation/results/{label}.json`" for label in labels)
        + "."
    )
    lines.append("")
    for label in labels:
        r = results[label]
        lines.append(
            f"- `{label}`: {r['questions']} questions, commit {r['git_commit']}, "
            f"settings `{r['settings_file']}`, index `{r['index_dir']}`"
        )
    lines += ["", "## Summary", ""]
    lines += summary_section(labels, results)

    for variant in args.variant:
        lines += ["", f"## Paired: {variant} vs {args.baseline}", ""]
        lines += paired_section(args.baseline, variant, results)

    types = {qid: row["type"] for qid, row in questions.items()}
    lines += ["", "## Per question type", ""]
    lines += group_section(labels, results, types, "type")

    if args.baseline in records:
        route_of = routes_by_source(records[args.baseline])
        routes = {}
        for qid, row in questions.items():
            found = {
                route_of.get(s, "missing") for s in expected_sources(row["expected"])
            }
            routes[qid] = "structured" if "structured" in found else "generic"
        lines += [
            "",
            "## Per route of the expected document (baseline chunker)",
            "",
            "A question counts as `structured` when one of its expected documents "
            "took the structured route in the baseline.",
            "",
        ]
        lines += group_section(labels, results, routes, "route")

        kind_of = structure_kinds(records[args.baseline])
        kinds = {}
        for qid, row in questions.items():
            found = {
                kind_of.get(s, "missing") for s in expected_sources(row["expected"])
            }
            if "page headers" in found:
                kinds[qid] = "page headers"
            elif "articles" in found:
                kinds[qid] = "articles"
            else:
                kinds[qid] = "generic route"
        lines += [
            "",
            "## Per kind of structure the baseline cut on",
            "",
            "Labels that are a running page header (a page number, then "
            "Staatscourant, Staatsblad, Tweede Kamer or Kamerstukken), per "
            "baseline document that has them:",
            "",
        ]
        lines += header_section(records[args.baseline])
        lines += [
            "",
            "A document counts as `page headers` when it took the structured "
            "route and more than half of its labels are page headers; a question "
            "counts as `page headers` when one of its expected documents does, "
            "else as `articles` when one of them took the structured route.",
            "",
        ]
        lines += group_section(labels, results, kinds, "structure")

    if records:
        lines += ["", "## Chunks", ""]
        lines += chunk_section(
            {label: chunk_stats(recs) for label, recs in records.items()}
        )

    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
