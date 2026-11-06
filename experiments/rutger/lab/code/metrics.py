"""Retrieval metrics for the three benchmarks.

    python -m bench.metrics <bench> <results.jsonl> [more.jsonl ...]

Span benchmarks (legalbench, conditionalqa): a retrieved chunk is a
(doc_id, char_start, char_end) range.
  char recall@k     gold characters covered by the top k / all gold characters
                    (LegalBench-RAG's character-level recall)
  char precision@k  gold characters covered / retrieved characters
  hit@k             a top-k chunk overlaps a gold span
  doc hit@k         a top-k chunk comes from a gold document
  MRR@k             1 / rank of the first chunk overlapping a gold span
Article benchmark (bbsard): gold is a set of article ids.
  recall@k, MRR@k, nDCG@k over unique articles in rank order.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from .common import read_jsonl, run_dir

KS = (1, 5, 10, 20, 50, 100)


def _merge(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _length(ranges: List[Tuple[int, int]]) -> int:
    return sum(end - start for start, end in ranges)


def _overlap(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> int:
    total, i, j = 0, 0, 0
    while i < len(a) and j < len(b):
        low, high = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        total += max(0, high - low)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def span_metrics(question: Dict[str, Any], results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    gold: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for item in question["gold"]:
        gold[item["doc_id"]].append((item["start"], item["end"]))
    gold = {doc: _merge(spans) for doc, spans in gold.items()}
    gold_docs = set(gold) | ({question["gold_doc"]} if question.get("gold_doc") else set())
    gold_chars = sum(_length(spans) for spans in gold.values())
    out: Dict[str, float] = {}
    first_hit = None
    for rank, item in enumerate(results, 1):
        spans = gold.get(item["doc_id"])
        if spans and _overlap(spans, [(item["char_start"], item["char_end"])]) > 0:
            first_hit = rank
            break
    for k in KS:
        top = results[:k]
        retrieved: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for item in top:
            retrieved[item["doc_id"]].append((item["char_start"], item["char_end"]))
        retrieved = {doc: _merge(spans) for doc, spans in retrieved.items()}
        covered = sum(_overlap(gold[doc], retrieved[doc]) for doc in gold if doc in retrieved)
        retrieved_chars = sum(_length(spans) for spans in retrieved.values())
        if gold_chars:
            out[f"char_recall@{k}"] = covered / gold_chars
            out[f"char_precision@{k}"] = covered / retrieved_chars if retrieved_chars else 0.0
            out[f"hit@{k}"] = float(first_hit is not None and first_hit <= k)
        out[f"doc_hit@{k}"] = float(any(item["doc_id"] in gold_docs for item in top))
    if gold_chars:
        out["mrr@20"] = 1.0 / first_hit if first_hit and first_hit <= 20 else 0.0
    return out


def article_metrics(question: Dict[str, Any], results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    gold = {item["doc_id"] for item in question["gold"]}
    ranked: List[str] = []
    for item in results:
        if item["doc_id"] not in ranked:
            ranked.append(item["doc_id"])
    out: Dict[str, float] = {}
    first = next((i for i, doc in enumerate(ranked, 1) if doc in gold), None)
    for k in KS:
        top = ranked[:k]
        out[f"recall@{k}"] = len(gold & set(top)) / len(gold)
        dcg = sum(1 / math.log2(i + 1) for i, doc in enumerate(top, 1) if doc in gold)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
        out[f"ndcg@{k}"] = dcg / ideal
    out["mrr@20"] = 1.0 / first if first and first <= 20 else 0.0
    return out


def evaluate(bench: str, results_file: str, qids=None) -> Dict[str, Any]:
    """qids: optional set of question ids to restrict to (same-question comparisons)."""
    questions = {q["qid"]: q for q in read_jsonl(run_dir(bench) / "questions.jsonl")}
    rows = read_jsonl(run_dir(bench) / results_file)
    if qids is not None:
        rows = [r for r in rows if r["qid"] in qids]
    per_subset: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        question = questions[row["qid"]]
        if bench == "bbsard":
            values = article_metrics(question, row["results"])
        else:
            if not question["gold"] and not question.get("gold_doc"):
                continue
            values = span_metrics(question, row["results"])
        per_subset["all"].append(values)
        if question.get("subset"):
            per_subset[question["subset"]].append(values)
    summary: Dict[str, Any] = {}
    for subset, items in per_subset.items():
        keys = sorted({key for item in items for key in item})
        summary[subset] = {
            "n": len(items),
            **{key: sum(item.get(key, 0.0) for item in items if key in item)
               / max(1, sum(1 for item in items if key in item)) for key in keys},
        }
    return summary


def main(argv: List[str]) -> int:
    bench, files = argv[0], argv[1:]
    for name in files:
        summary = evaluate(bench, name)
        print(f"== {bench} / {name}")
        for subset, values in summary.items():
            shown = {k: round(v, 3) for k, v in values.items()
                     if k == "n" or any(k.endswith(f"@{k2}") for k2 in (1, 5, 10, 20, 100))}
            print(f"  {subset}: {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
