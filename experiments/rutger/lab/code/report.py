"""One results table over every run of every benchmark.

    python -m bench.report            -> prints markdown, writes runs/RESULTS.md

Rows are the result files (baseline, TurboQuant, judges ...), columns the
headline metrics per benchmark:
  conditionalqa, legalbench  hit@1, hit@10, char recall@10, MRR@20
  bbsard                     recall@10, recall@100 (deep runs), nDCG@10, MRR@20
"""

from __future__ import annotations

import sys
from typing import List

from .common import RUNS, read_jsonl
from .metrics import evaluate

BENCHES = ("conditionalqa", "legalbench", "bbsard")
COLUMNS = {
    "conditionalqa": ("hit@1", "hit@10", "char_recall@10", "mrr@20"),
    "legalbench": ("hit@1", "hit@10", "char_recall@10", "mrr@20"),
    "bbsard": ("recall@10", "recall@100", "ndcg@10", "mrr@20"),
}


def main(argv: List[str]) -> int:
    lines = ["# bench-lab results", ""]
    for bench in BENCHES:
        folder = RUNS / bench
        files = sorted(p.name for p in folder.glob("*.jsonl")
                       if p.name.startswith(("fast_", "nav_", "baseline_")))
        if not files:
            continue
        cols = COLUMNS[bench]
        lines += [f"## {bench}", "", "| run | n | " + " | ".join(cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
        for name in files:
            summary = evaluate(bench, name)["all"]
            cells = [f"{summary.get(c, float('nan')):.3f}" for c in cols]
            lines.append(f"| {name[:-6]} | {summary['n']} | " + " | ".join(cells) + " |")
        if bench == "legalbench":  # per document type, for the main runs
            lines += ["", "LegalBench-RAG per sub-dataset (hit@10):", ""]
            for name in files:
                if "_deep" in name:
                    continue
                parts = evaluate(bench, name)
                cells = [f"{k} {parts[k]['hit@10']:.3f}" for k in ("contractnli", "cuad", "maud", "privacy_qa") if k in parts]
                lines.append(f"- {name[:-6]}: " + ", ".join(cells))
        nav_files = [f for f in files if f.startswith("nav_")]
        if nav_files:
            lines += ["", f"Judges vs our baseline on the same questions ({bench}):", "",
                      "| judge run | n | " + " | ".join(f"{c} (base -> judge)" for c in cols) + " |",
                      "|---|---|" + "---|" * len(cols)]
            for name in nav_files:
                qids = {r["qid"] for r in read_jsonl(folder / name)}
                base = evaluate(bench, "fast_exact.jsonl", qids)["all"]
                judge = evaluate(bench, name, qids)["all"]
                cells = [f"{base.get(c, float('nan')):.3f} -> {judge.get(c, float('nan')):.3f}" for c in cols]
                lines.append(f"| {name[:-6]} | {judge['n']} | " + " | ".join(cells) + " |")
        lines.append("")
    text = "\n".join(lines)
    (RUNS / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
