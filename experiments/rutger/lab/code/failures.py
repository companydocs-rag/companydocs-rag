"""Failure analysis of a judge run against the baseline, per failure type.

    python -m bench.failures   -> runs/analysis/failures.md (and stdout)

Reads only existing result files (no model). For every question:
- search failure: no gold passage among the judged passages (the judge never saw it);
- judge failure: gold was judged but ranked below rank 10.
For the judge failures, the judge's P(yes) on gold passages is compared with
each kind of distractor in the same pool (AUC): another clause of the right
document, a passage of another document of the same type, a passage of another
document type. Also split by passage length, sub-dataset, query length and
category, and the cases where the judge demoted a baseline hit (harm) or
promoted a baseline miss (rescue).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .common import RUNS, read_jsonl, run_dir

RUNS_ = {
    "conditionalqa": ("nav_d3-lite-tuned-full_plusbase_sl10.jsonl", "baseline_exact.jsonl"),
    "legalbench": ("nav_d3-lite-tuned-full_plusbase_sl10.jsonl", "fast_exact.jsonl"),
    "bbsard": ("nav_d3-lite-tuned-full_plusbase_sl50.jsonl", "fast_exact_deep.jsonl"),
}
LEN_BUCKETS = [(0, 800), (800, 1600), (1600, 3200), (3200, 10**9)]


def auc(pos: List[float], neg: List[float]) -> float:
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + 1 + j) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def is_gold(bench, q, item) -> bool:
    if bench == "bbsard":
        return item["doc_id"] in {g["doc_id"] for g in q["gold"]}
    for g in q["gold"]:
        if g["doc_id"] == item["doc_id"] and g["start"] < item["char_end"] and item["char_start"] < g["end"]:
            return True
    return False


def doc_type(doc_id: str) -> str:
    return doc_id.split("/")[0] if "/" in doc_id else "article"


def first_hit(bench, q, results, k=None):
    for rank, item in enumerate(results[:k] if k else results, 1):
        if is_gold(bench, q, item):
            return rank
    return None


def bucket(n: int) -> str:
    for lo, hi in LEN_BUCKETS:
        if lo <= n < hi:
            return f"{lo}-{hi if hi < 10**9 else 'inf'}"
    return "?"


def analyse(bench: str) -> List[str]:
    nav_file, base_file = RUNS_[bench]
    qs = {q["qid"]: q for q in read_jsonl(run_dir(bench) / "questions.jsonl")}
    nav = {r["qid"]: r for r in read_jsonl(run_dir(bench) / nav_file)}
    base = {r["qid"]: r for r in read_jsonl(run_dir(bench) / base_file)}
    out = [f"## {bench} ({len(nav)} questions)", ""]
    group = (lambda q: q["subset"]) if bench == "legalbench" else (
        (lambda q: q.get("category", "all")) if bench == "bbsard" else (lambda q: "all"))
    tally = defaultdict(lambda: defaultdict(int))
    scores = defaultdict(lambda: defaultdict(list))   # (group) -> kind -> P(yes) list
    lens = defaultdict(lambda: defaultdict(list))      # length bucket -> gold/neg
    harm = rescue = 0
    qlen = defaultdict(list)
    for qid, r in nav.items():
        q = qs[qid]; g = group(q); res = r["results"]
        fh = first_hit(bench, q, res)
        bh = first_hit(bench, q, base.get(qid, {}).get("results", []), 20)
        kind = "top10" if fh and fh <= 10 else ("judge: gold ranked >10" if fh else "search: gold never judged")
        tally[g][kind] += 1; tally["ALL"][kind] += 1
        if bh and bh <= 10 and not (fh and fh <= 10):
            harm += 1
        if (fh and fh <= 10) and not (bh and bh <= 10):
            rescue += 1
        qlen[bucket(len(q["query"]) * 4)].append(1 if fh and fh <= 10 else 0)
        gold_docs = {x["doc_id"] for x in q["gold"]}
        gold_types = {doc_type(d) for d in gold_docs}
        for item in res:
            n = item["char_end"] - item["char_start"]
            if is_gold(bench, q, item):
                scores[g]["gold"].append(item["p_yes"]); scores["ALL"]["gold"].append(item["p_yes"])
                lens[bucket(n)]["gold"].append(item["p_yes"])
                continue
            lens[bucket(n)]["neg"].append(item["p_yes"])
            if item["doc_id"] in gold_docs:
                k = "same document, other clause"
            elif doc_type(item["doc_id"]) in gold_types and bench == "legalbench":
                k = "other document, same type"
            elif bench == "legalbench":
                k = "other document type"
            else:
                k = "other document"
            scores[g][k].append(item["p_yes"]); scores["ALL"][k].append(item["p_yes"])
    out += ["Outcome per question (judge run):", ""]
    kinds = ["top10", "judge: gold ranked >10", "search: gold never judged"]
    out.append("| group | n | " + " | ".join(kinds) + " |")
    out.append("|---" * (len(kinds) + 2) + "|")
    for g in sorted(tally, key=lambda x: (x != "ALL", x)):
        n = sum(tally[g].values())
        out.append(f"| {g} | {n} | " + " | ".join(f"{tally[g][k] / n:.2f}" for k in kinds) + " |")
    out += ["", f"Baseline top-10 hits the judge pushed out: {harm}; baseline misses the judge brought into the top 10: {rescue}", ""]
    out += ["Judge AUC, gold passages vs each kind of distractor in the same pools (mean P(yes) in brackets):", ""]
    neg_kinds = sorted({k for g in scores for k in scores[g] if k != "gold"})
    out.append("| group | gold P | " + " | ".join(neg_kinds) + " |")
    out.append("|---" * (len(neg_kinds) + 2) + "|")
    for g in sorted(scores, key=lambda x: (x != "ALL", x)):
        gp = scores[g]["gold"]
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        out.append(f"| {g} | {mean(gp):.2f} | " + " | ".join(
            f"{auc(gp, scores[g][k]):.2f} ({mean(scores[g][k]):.2f}, n {len(scores[g][k])})" for k in neg_kinds) + " |")
    out += ["", "Judge AUC by passage length (characters):", "", "| length | gold n | neg n | AUC | mean P gold | mean P neg |", "|---|---|---|---|---|---|"]
    for b in [bucket(lo) for lo, _ in LEN_BUCKETS]:
        gp, ng = lens[b]["gold"], lens[b]["neg"]
        if gp or ng:
            out.append(f"| {b} | {len(gp)} | {len(ng)} | {auc(gp, ng):.2f} | "
                       f"{(sum(gp) / len(gp)) if gp else float('nan'):.2f} | {(sum(ng) / len(ng)) if ng else float('nan'):.2f} |")
    out += ["", "Top-10 hit rate by query length (characters x4 bucket):", ""]
    for b in sorted(qlen, key=lambda x: int(x.split("-")[0])):
        v = qlen[b]
        out.append(f"- query ~{int(b.split('-')[0]) // 4}+ chars: {sum(v) / len(v):.2f} (n {len(v)})")
    out.append("")
    return out


def main() -> int:
    lines = ["# Failure analysis: d3-lite-tuned-full", ""]
    for bench in RUNS_:
        lines += analyse(bench)
    text = "\n".join(lines)
    out = RUNS / "analysis"
    out.mkdir(exist_ok=True)
    (out / "failures.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
