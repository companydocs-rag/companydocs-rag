"""Training data for the decision judge, built from ConditionalQA *train*.

    python -m bench.traindata    -> runs/conditionalqa/judge_train.jsonl, judge_val.jsonl

Every example is exactly what the navigation asks the judge at test time:
(query, kind, text) -> yes / no, with kind page | section | passage and text
the node description (pages, sections) or the chunk text (passages), as in
bench.navigate. Per train question:

  page      the gold page = yes; pages our own hybrid search ranks high but
            are wrong = no (hard negatives: the lookalike problem), plus two
            random pages
  section   sections of the gold page whose chunks hold the evidence = yes;
            the other sections of that same page = no (hardest distinction)
  passage   chunks overlapping the evidence = yes; other chunks of the gold
            page and retrieved chunks from wrong pages = no

The 285 dev questions (our test set) are never used; 5% of the train
questions are held out as validation. LegalBench-RAG and bBSARD stay pure
test sets.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from .common import read_jsonl, run_dir, write_jsonl

BENCH = "conditionalqa"


def _overlaps(chunk: Dict[str, Any], gold: List[Dict[str, Any]]) -> bool:
    return any(
        g["doc_id"] == chunk["doc_id"] and g["start"] < chunk["char_end"] and chunk["char_start"] < g["end"]
        for g in gold
    )


def main(argv: List[str]) -> int:
    from companydocs_rag.retrieval.fusion import fuse

    from .fastsearch import HybridIndex
    from .pipeline import _embedder

    out = run_dir(BENCH)
    nodes = json.loads((out / "tree.json").read_text("utf-8"))
    live = [n for n in nodes if "merged_into" not in n]
    chunks = {c["chunk_id"]: c for c in read_jsonl(out / "chunks.jsonl")}
    pages = {n["doc_id"]: n for n in live if n.get("kind") == "page"}
    owner = {cid: n for n in live for cid in n["chunk_ids"]}
    questions = [q for q in read_jsonl(out / "questions_train.jsonl") if q["gold"]]

    index = HybridIndex(BENCH)
    embedder = _embedder()
    vectors = embedder._model.encode(
        [q["query"] for q in questions], prompt=embedder._config.query_instruction,
        batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float32)

    rng = random.Random(0)
    page_ids = list(pages)
    rows: List[Dict[str, Any]] = []
    for question, vector in zip(questions, vectors):
        query, gold, gold_doc = question["query"], question["gold"], question["gold_doc"]
        fused = fuse(index.dense_leg(vector, 50), index.sparse_leg(query, 50), rrf_k=60, candidate_count=30)
        retrieved = [chunks[item.chunk_id] for item in fused]
        page = pages.get(gold_doc)
        if page is None:
            continue

        def add(kind: str, text: str, label: int) -> None:
            rows.append({"qid": question["qid"], "query": query, "kind": kind, "text": text, "label": label})

        # page level
        add("page", page["description"], 1)
        # (a page collapsed into its only section is not a page node; skip it)
        wrong_docs = list(dict.fromkeys(
            c["doc_id"] for c in retrieved if c["doc_id"] != gold_doc and c["doc_id"] in pages
        ))[:4]
        wrong_docs += rng.sample([d for d in page_ids if d != gold_doc and d not in wrong_docs], 2)
        for doc in wrong_docs:
            add("page", pages[doc]["description"], 0)

        # section level, within the gold page
        sections = [nodes[c] for c in page["children"]]
        multi = [s for s in sections if len(s["chunk_ids"]) > 1 or s["children"]]
        good = [s for s in multi if any(_overlaps(chunks[c], gold) for c in s["chunk_ids"])]
        bad = [s for s in multi if s not in good]
        for s in good:
            add("section", s["description"], 1)
        for s in rng.sample(bad, min(4, len(bad))):
            add("section", s["description"], 0)

        # passage level
        page_chunks = [cid for s in sections for cid in s["chunk_ids"]] + page["chunk_ids"]
        positives = [cid for cid in page_chunks if _overlaps(chunks[cid], gold)]
        same_page = [cid for cid in page_chunks if cid not in positives]
        elsewhere = [c["chunk_id"] for c in retrieved if c["doc_id"] != gold_doc][:3]
        for cid in positives:
            add("passage", chunks[cid]["text"], 1)
        for cid in rng.sample(same_page, min(3, len(same_page))) + elsewhere:
            add("passage", chunks[cid]["text"], 0)

    qids = sorted({r["qid"] for r in rows})
    val = set(random.Random(1).sample(qids, max(1, len(qids) // 20)))
    train_rows = [r for r in rows if r["qid"] not in val]
    val_rows = [r for r in rows if r["qid"] in val]
    random.Random(2).shuffle(train_rows)
    write_jsonl(out / "judge_train.jsonl", train_rows)
    write_jsonl(out / "judge_val.jsonl", val_rows)
    stats = defaultdict(lambda: [0, 0])
    for r in rows:
        stats[r["kind"]][r["label"]] += 1
    print(f"{len(questions)} train questions -> {len(train_rows)} train / {len(val_rows)} val examples "
          f"({len(val)} val questions)")
    for kind, (neg, pos) in stats.items():
        print(f"  {kind}: {pos} yes / {neg} no")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
