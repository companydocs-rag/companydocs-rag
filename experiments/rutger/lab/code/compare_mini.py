"""Compare setups on exactly the same questions, also per question.

    python -m bench.compare_mini   -> runs/analysis/mini.md (and stdout)

For each benchmark, the questions of the mini run (edited judge, old
pipeline) define the set; every other setup is scored on that same set:
our pipeline alone, the untuned d3 where it covered them, the fine-tuned judge
(old pipeline, from the full run), the edited judge on the old pipeline, and
the edited judge on the fixed pipeline. Per question: the rank of the first
gold result in each setup.
"""

from __future__ import annotations

import glob
import os
import sys

from .common import RUNS, read_jsonl, run_dir
from .failures import first_hit
from .metrics import evaluate

def _setups(bench, old, new, base):
    return [("pipeline alone", base),
            ("1 old pipeline + fine-tuned", "nav_d3-lite-tuned-full_" + old.split("_n")[0] + ".jsonl"),
            ("2 old pipeline + edited", "nav_d3-lite-edited2_" + old),
            ("3 new pipeline + fine-tuned", "nav_d3-lite-tuned-full_" + new),
            ("4 new pipeline + edited", "nav_d3-lite-edited2_" + new)]


SETUPS = {
    "conditionalqa": _setups("conditionalqa", "plusbase_sl10_n100_half-heldout.jsonl",
                             "plusbase_sl10_n100_dctx_half-heldout_win1600.jsonl", "baseline_exact.jsonl"),
    "bbsard": _setups("bbsard", "plusbase_sl50_n40_half-heldout.jsonl",
                      "plusbase_sl100_src-fast_exact_deep1000_n16_half-heldout_win1600_bk100.jsonl", "fast_exact_deep.jsonl"),
    "legalbench": _setups("legalbench", "plusbase_sl10_n12_half-heldout.jsonl",
                          "plusbase_sl100_src-fast_exact_deep1000_n12_dctx_flat_flattop2_half-heldout_win1600.jsonl",
                          "fast_exact.jsonl"),
}
KEYS = {"bbsard": ("recall@10", "ndcg@10", "mrr@20"), "conditionalqa": ("hit@1", "hit@10", "mrr@20"),
        "legalbench": ("hit@1", "hit@10", "mrr@20")}


def resolve(bench: str, pattern: str):
    hits = sorted(glob.glob(str(run_dir(bench) / pattern)))
    if not hits:  # a run cut off by the deadline leaves only its partial file
        stem = pattern.replace("nav_", "partial_")
        hits = sorted(glob.glob(str(run_dir(bench) / stem)))
    return os.path.basename(hits[-1]) if hits else None


def main() -> int:
    lines = ["# Mini comparison on the same questions (untouched half)", ""]
    for bench, setups in SETUPS.items():
        files = [(name, resolve(bench, pat)) for name, pat in setups]
        # the reference question set: the smaller of the two edited runs (a step cut off at 30 minutes has fewer)
        runs_ = [f for _, f in files[2:] if f]
        ref = min(runs_, key=lambda f: len(read_jsonl(run_dir(bench) / f))) if runs_ else None
        if not ref:
            lines += [f"## {bench}: no mini run yet", ""]
            continue
        qids = {r["qid"] for r in read_jsonl(run_dir(bench) / ref)}
        lines += [f"## {bench} ({len(qids)} questions)", "", "| setup | " + " | ".join(KEYS[bench]) + " |",
                  "|---" * (len(KEYS[bench]) + 1) + "|"]
        per_q = {}
        qs = {q["qid"]: q for q in read_jsonl(run_dir(bench) / "questions.jsonl")}
        for name, f in files:
            if not f:
                lines.append(f"| {name} | " + " | ".join("not run" for _ in KEYS[bench]) + " |")
                continue
            got = {r["qid"] for r in read_jsonl(run_dir(bench) / f)} & qids
            m = evaluate(bench, f, qids=got)["all"]
            note = "" if len(got) == len(qids) else f" (only {len(got)} of {len(qids)} questions)"
            lines.append(f"| {name}{note} | " + " | ".join(f"{m[k]:.3f}" for k in KEYS[bench]) + " |")
            rows = {r["qid"]: r for r in read_jsonl(run_dir(bench) / f)}
            per_q[name] = {q: first_hit(bench, qs[q], rows[q]["results"]) if q in rows else "-" for q in qids}
        names = [n for n, f in files if f]
        lines += ["", "Rank of the first gold result per question (- = not in the list):", "",
                  "| question | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for q in sorted(qids):
            lines.append(f"| {q} | " + " | ".join(str(per_q[n][q] or "-") for n in names) + " |")
        lines.append("")
    text = "\n".join(lines)
    (RUNS / "analysis").mkdir(exist_ok=True)
    (RUNS / "analysis" / "mini.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
