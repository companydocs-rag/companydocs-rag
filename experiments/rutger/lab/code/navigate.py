"""DecisionRAG-style navigation of a benchmark tree with a decision model.

    python -m bench.navigate <bench> <judge> [--limit N] [--with-baseline]
           [--beam B] [--first-beam B1] [--shortlist S] [--baseline-k K]
           [--doc-context] [--judge-window C]

Beam search from the root: at each step the judge scores every child of the
current frontier (inner nodes on their description, chunks on their text),
the top `beam` inner nodes are expanded (`first-beam` at the first level),
and every scored chunk enters the passage pool. Pruned branches are never
revisited, as in the paper. A section that holds exactly one chunk is judged
as that passage directly (with its heading), not as an extra step.

--with-baseline adds our pipeline's top 20 to the pool, judged the same way
(the paper's "+ dense candidates"); --baseline-k changes how many (above 20 the
deep baseline file, which holds 100 results per question, is read).

Optional, all off by default (the default run is unchanged):
  --doc-context     a passage is judged with its document's title in front
                    ("Document: <title>"); the stored text and spans are not changed
  --judge-window C  a judged input longer than C characters is cut into
                    overlapping windows of at most C characters (stride C/2);
                    the passage takes the best window's score
  --flat-docs       a document (legalbench) or page (conditionalqa) in the beam is
                    judged through its chunks, not its sections (see FLAT_KINDS)
  --flat-top K      with --flat-docs: only the K best-scored of them (0 = all)
  --half diag|heldout  only the questions of one half (sha1 of the qid, see main)

Output: runs/<bench>/nav_<judge>[...].jsonl in the baseline's format, so
bench.metrics scores it. Non-default flags are part of the file name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from typing import Any, Dict, List, Tuple

from .common import read_jsonl, run_dir, write_jsonl
from .judges import load_judge

KIND = {"page": "page", "document": "document", "type": "collection",
        "code": "legal code", "part": "part of a legal code", "section": "section"}


def _subset_questions(bench: str, questions: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """The first `limit` questions; for legalbench spread over its four
    sub-datasets, so a screening run covers every document type."""
    if not limit:
        return questions
    if bench != "legalbench":
        return questions[:limit]
    per = max(1, limit // 4)
    picked: List[Dict[str, Any]] = []
    for subset in ("contractnli", "cuad", "maud", "privacy_qa"):
        picked += [q for q in questions if q["subset"] == subset][:per]
    return picked


def subtree_docs(nodes, chunks) -> Dict[int, set]:
    """For every node, the documents that have a chunk somewhere below it."""
    docs: Dict[int, set] = {}

    def walk(node_id: int) -> set:
        node = nodes[node_id]
        found = {chunks[c]["doc_id"] for c in node["chunk_ids"] if c in chunks}
        for child in node["children"]:
            found |= walk(child)
        docs[node_id] = found
        return found

    sys.setrecursionlimit(10000)
    walk(0)
    return docs


DEFAULT_BASELINE_K = 20  # the baseline results --with-baseline adds, unless --baseline-k
DEFAULT_SHORTLIST_SOURCE = "fast_exact_deep"  # the 100-chunk list the shortlist came from so far
# --flat-docs: a node of these kinds that is in the beam is not descended with the
# judge; every chunk below it is judged directly. legalbench "document" and
# conditionalqa "page" have sections below them. bbsard "article" is a leaf document
# whose chunks are already judged directly, and bbsard codes are whole statutes,
# so neither is flattened.
FLAT_KINDS = ("document", "page")


def doc_titles(nodes) -> Dict[str, str]:
    """doc_id -> title of the tree node that holds the document (a conditionalqa
    page, a legalbench document or a bbsard article). Merged-away nodes count
    too: _collapse can hoist a document's only child, and the document's title
    then stays on the old node."""
    titles: Dict[str, str] = {}
    for node in nodes:
        if node.get("doc_id") is not None:
            titles.setdefault(node["doc_id"], node["title"])
    return titles


def document_chunks(nodes, node_id: int, cache: Dict[int, List[str]]) -> List[str]:
    """Every chunk below a node in tree order (its own chunks first), cached."""
    if node_id not in cache:
        found: List[str] = []

        def walk(i: int) -> None:
            found.extend(nodes[i]["chunk_ids"])
            for child in nodes[i]["children"]:
                walk(child)

        walk(node_id)
        cache[node_id] = found
    return cache[node_id]


def judge_inputs(cid, chunks, titles, doc_context: bool, window: int) -> List[str]:
    """The texts the judge scores for one passage. By default the chunk text
    alone. doc_context puts the document's title in front. window (characters,
    0 = off): a judged input longer than that is cut into overlapping windows of
    at most `window` characters, stride window // 2, and the passage takes the
    best window (navigate). The stored chunk text and its spans never change."""
    chunk = chunks[cid]
    head = f"Document: {titles.get(chunk['doc_id'], chunk['doc_id'])}\n\n" if doc_context else ""
    body = chunk["text"]
    if not window or len(head) + len(body) <= window:
        return [head + body]
    if len(head) > window // 2:  # a very long title may not take more than half a window
        head = head[: window // 2]
    width = window - len(head)
    starts = list(range(0, len(body) - width, window // 2)) + [len(body) - width]
    return [head + body[start : start + width] for start in dict.fromkeys(starts)]


def navigate(query, nodes, chunks, judge, beam, first_beam, extra_chunks=(),
             allowed=None, *, doc_context=False, window=0, titles=None,
             flat_docs=False, flat_top=0) -> Tuple[List[Tuple[float, str]], int]:
    """allowed: optional function node_id -> bool; with a shortlist, the judge
    only sees nodes whose subtree holds a shortlisted document.
    doc_context, window: see judge_inputs; titles: doc_titles(nodes), computed
    here when missing. flat_docs: see FLAT_KINDS; flat_top K (0 = all): only the
    K best-scored flat-kind nodes of a frontier are flattened, the rest descended."""
    if doc_context and titles is None:
        titles = doc_titles(nodes)
    pool: Dict[str, float] = {}
    calls = 0
    frontier = [0]
    level = 0
    flat_cache: Dict[int, List[str]] = {}
    while frontier:
        inner: List[int] = []
        passages: List[str] = []
        flattened = 0
        for node_id in frontier:  # frontier is in score order, best first
            node = nodes[node_id]
            if flat_docs and node.get("kind") in FLAT_KINDS and (not flat_top or flattened < flat_top):
                flattened += 1
                passages.extend(document_chunks(nodes, node_id, flat_cache))  # judged directly
                continue
            for child_id in node["children"]:
                child = nodes[child_id]
                if allowed is not None and not allowed(child_id):
                    continue
                if not child["children"] and len(child["chunk_ids"]) == 1:
                    passages.append(child["chunk_ids"][0])  # a one-chunk section
                elif child["children"] or child["chunk_ids"]:
                    inner.append(child_id)
            passages.extend(node["chunk_ids"])
        if level == 0:
            passages.extend(cid for cid in extra_chunks if cid in chunks)
        passages = [cid for cid in dict.fromkeys(passages) if cid not in pool]
        if passages:
            texts, owners = [], []
            for cid in passages:
                inputs = judge_inputs(cid, chunks, titles or {}, doc_context, window)
                texts += inputs
                owners += [cid] * len(inputs)
            scores = judge.relevance(query, texts, "passage")
            calls += len(texts)
            best: Dict[str, float] = {}
            for cid, score in zip(owners, scores):
                best[cid] = max(best.get(cid, score), score)
            pool.update(best)
        if not inner:
            break
        groups: Dict[str, List[int]] = {}
        for node_id in inner:
            groups.setdefault(KIND.get(nodes[node_id].get("kind"), "section"), []).append(node_id)
        scored: List[Tuple[float, int]] = []
        for kind, ids in groups.items():
            scores = judge.relevance(query, [nodes[i]["description"] or nodes[i]["title"] for i in ids], kind)
            calls += len(ids)
            scored += list(zip(scores, ids))
        scored.sort(reverse=True)
        frontier = [node_id for _, node_id in scored[: (first_beam if level == 0 else beam)]]
        level += 1
    ranked = sorted(((score, cid) for cid, score in pool.items()), reverse=True)
    return ranked, calls


def navigate_many(queries, nodes, chunks, judge, beam, first_beam, extras, alloweds,
                  *, doc_context=False, window=0, titles=None, flat_docs=False, flat_top=0):
    """navigate() for many questions at once, level-synchronous: at every
    level the candidates of all still-active questions go to the judge in
    one relevance_batch call (token-budget batches on the GPU), instead of a
    few small calls per question. Same candidates and decisions as navigate().
    doc_context, window, titles: as in navigate()."""
    n = len(queries)
    pools: List[Dict[str, float]] = [dict() for _ in range(n)]
    frontiers = [[0] for _ in range(n)]
    calls = [0] * n
    if doc_context and titles is None:
        titles = doc_titles(nodes)
    flat_cache: Dict[int, List[str]] = {}
    level = 0
    while any(frontiers):
        items, owners = [], []  # owners: (question, "p", chunk_id) or (question, "n", node_id)
        inner_of: List[List[int]] = [[] for _ in range(n)]
        for q in range(n):
            if not frontiers[q]:
                continue
            allowed = alloweds[q]
            passages: List[str] = []
            flattened = 0
            for node_id in frontiers[q]:  # frontier is in score order, best first
                node = nodes[node_id]
                if flat_docs and node.get("kind") in FLAT_KINDS and (not flat_top or flattened < flat_top):
                    flattened += 1
                    passages.extend(document_chunks(nodes, node_id, flat_cache))  # judged directly
                    continue
                for child_id in node["children"]:
                    child = nodes[child_id]
                    if allowed is not None and not allowed(child_id):
                        continue
                    if not child["children"] and len(child["chunk_ids"]) == 1:
                        passages.append(child["chunk_ids"][0])
                    elif child["children"] or child["chunk_ids"]:
                        inner_of[q].append(child_id)
                passages.extend(node["chunk_ids"])
            if level == 0:
                passages.extend(cid for cid in extras[q] if cid in chunks)
            for cid in dict.fromkeys(passages):
                if cid not in pools[q]:
                    for text in judge_inputs(cid, chunks, titles or {}, doc_context, window):
                        items.append((queries[q], "passage", text))
                        owners.append((q, "p", cid))
            for node_id in inner_of[q]:
                kind = KIND.get(nodes[node_id].get("kind"), "section")
                items.append((queries[q], kind, nodes[node_id]["description"] or nodes[node_id]["title"]))
                owners.append((q, "n", node_id))
        scores = judge.relevance_batch(items) if items else []
        scored_nodes: List[List[Tuple[float, int]]] = [[] for _ in range(n)]
        best: List[Dict[str, float]] = [dict() for _ in range(n)]
        for (q, what, ref), score in zip(owners, scores):
            calls[q] += 1
            if what == "p":
                best[q][ref] = max(best[q].get(ref, score), score)
            else:
                scored_nodes[q].append((score, ref))
        for q in range(n):
            pools[q].update(best[q])
        width = first_beam if level == 0 else beam
        for q in range(n):
            scored_nodes[q].sort(reverse=True)
            frontiers[q] = [node_id for _, node_id in scored_nodes[q][:width]]
        level += 1
    ranked = [sorted(((s, cid) for cid, s in pool.items()), reverse=True) for pool in pools]
    return ranked, calls


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="bench.navigate")
    parser.add_argument("bench")
    parser.add_argument("judge")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--with-baseline", action="store_true")
    parser.add_argument("--beam", type=int, default=6)
    parser.add_argument("--first-beam", type=int, default=0)
    parser.add_argument("--shortlist", type=int, default=0,
                        help="only navigate documents in our hybrid top-N (0 = full tree, as the paper)")
    parser.add_argument("--group", type=int, default=20,
                        help="questions navigated together (one judge call per level for all)")
    parser.add_argument("--baseline-k", type=int, default=DEFAULT_BASELINE_K,
                        help="results of the baseline that --with-baseline adds (default 20); above 20 "
                             "the deep baseline file (100 results per question) is read")
    parser.add_argument("--doc-context", action="store_true",
                        help="judge each passage with its document's title in front")
    parser.add_argument("--judge-window", type=int, default=0, metavar="C",
                        help="judge a passage longer than C characters as overlapping windows of at most "
                             "C characters (stride C/2); the passage takes the best window's score")
    parser.add_argument("--flat-docs", action="store_true",
                        help="a document (legalbench) or page (conditionalqa) in the beam is judged through its "
                             "chunks directly, not through its sections (see FLAT_KINDS)")
    parser.add_argument("--flat-top", type=int, default=0, metavar="K",
                        help="with --flat-docs: flatten only the K best-scored document nodes of each beam "
                             "(0 = all); the other beam documents are descended as usual")
    parser.add_argument("--half", choices=["all", "diag", "heldout"], default="all",
                        help="run only one half of the questions: diag = sha1(qid) as an integer, even; "
                             "heldout = the others")
    parser.add_argument("--shortlist-source", default=DEFAULT_SHORTLIST_SOURCE, metavar="STEM",
                        help="runs/<bench>/<STEM>.jsonl whose distinct documents per question form the shortlist "
                             "(default fast_exact_deep, the 100-chunk list; e.g. fast_exact_deep1000)")
    args = parser.parse_args(argv)
    first_beam = args.first_beam or args.beam
    if args.baseline_k < 1:
        parser.error("--baseline-k must be at least 1")
    if args.judge_window and args.judge_window < 100:
        parser.error("--judge-window must be 0 (off) or at least 100 characters")
    if any(sep in args.shortlist_source for sep in "/\\") or not args.shortlist_source:
        parser.error("--shortlist-source must be a file stem inside runs/<bench>/")

    out = run_dir(args.bench)
    nodes = json.loads((out / "tree.json").read_text("utf-8"))
    chunks = {c["chunk_id"]: c for c in read_jsonl(out / "chunks.jsonl")}
    if args.flat_top and not args.flat_docs:
        parser.error("--flat-top needs --flat-docs")
    if args.flat_top < 0:
        parser.error("--flat-top must be 0 or more")
    everything = read_jsonl(out / "questions.jsonl")
    if args.half != "all":
        # the half is chosen before --limit, so --limit samples within it
        wanted = args.half == "diag"
        everything = [q for q in everything
                      if (int(hashlib.sha1(q["qid"].encode()).hexdigest(), 16) % 2 == 0) == wanted]
    questions = _subset_questions(args.bench, everything, args.limit)
    baseline_rows = read_jsonl(out / "fast_exact.jsonl")
    if args.with_baseline and args.baseline_k > min(len(r["results"]) for r in baseline_rows):
        # fast_exact keeps the 20 candidates of the default fusion; the deep file
        # keeps 100 (wider legs, same fusion), so its top 20 can differ slightly
        baseline_rows = read_jsonl(out / "fast_exact_deep.jsonl")
        if args.baseline_k > min(len(r["results"]) for r in baseline_rows):
            parser.error(f"--baseline-k {args.baseline_k}: the baseline files hold fewer results per question")
    baseline = {r["qid"]: [x["chunk_id"] for x in r["results"]] for r in baseline_rows}
    deep: Dict[str, List[str]] = {}
    if args.shortlist:
        source = out / f"{args.shortlist_source}.jsonl"
        if not source.exists():
            parser.error(f"--shortlist-source: {source.name} does not exist")
        deep = {r["qid"]: [x["doc_id"] for x in r["results"]] for r in read_jsonl(source)}
        short = sum(1 for docs in deep.values() if len(dict.fromkeys(docs)) < args.shortlist)
        print(f"shortlist from {source.name}: {short} of {len(deep)} questions have fewer than "
              f"{args.shortlist} distinct documents in it", flush=True)
    docs_under = subtree_docs(nodes, chunks) if args.shortlist else None
    titles = doc_titles(nodes) if args.doc_context else None
    src_suffix = (f"_src-{args.shortlist_source}"
                  if args.shortlist and args.shortlist_source != DEFAULT_SHORTLIST_SOURCE else "")
    suffix = (("_plusbase" if args.with_baseline else "") + (f"_sl{args.shortlist}" if args.shortlist else "")
              + src_suffix + (f"_n{args.limit}" if args.limit else ""))
    new_suffix = (("_dctx" if args.doc_context else "") + ("_flat" if args.flat_docs else "")
                  + (f"_flattop{args.flat_top}" if args.flat_docs and args.flat_top else "")
                  + (f"_half-{args.half}" if args.half != "all" else "")
                  + (f"_win{args.judge_window}" if args.judge_window else "")
                  + (f"_bk{args.baseline_k}" if args.with_baseline and args.baseline_k != DEFAULT_BASELINE_K else ""))
    # partial files of runs with the new flags carry the full name, so no run overwrites another
    partial_name = f"partial_{args.judge}{suffix}{new_suffix}.jsonl" if new_suffix else f"partial_{args.judge}.jsonl"

    started = time.time()
    judge = load_judge(args.judge)
    print(f"{args.judge}: loaded in {time.time() - started:.0f}s")
    rows, total_calls, started = [], 0, time.time()
    for first in range(0, len(questions), args.group):
        group = questions[first : first + args.group]
        extras, alloweds = [], []
        for question in group:
            extras.append(baseline.get(question["qid"], [])[: args.baseline_k] if args.with_baseline else ())
            allowed = None
            if args.shortlist:
                keep = set(list(dict.fromkeys(deep.get(question["qid"], [])))[: args.shortlist])
                allowed = lambda node_id, keep=keep: bool(docs_under[node_id] & keep)
            alloweds.append(allowed)
        ranked_all, calls_all = navigate_many([q["query"] for q in group], nodes, chunks, judge,
                                              args.beam, first_beam, extras, alloweds,
                                              doc_context=args.doc_context, window=args.judge_window,
                                              titles=titles, flat_docs=args.flat_docs, flat_top=args.flat_top)
        for question, ranked, calls in zip(group, ranked_all, calls_all):
            total_calls += calls
            rows.append({
                "qid": question["qid"],
                "judgments": calls,
                "results": [
                    {"rank": rank, "chunk_id": cid, "doc_id": chunks[cid]["doc_id"],
                     "char_start": chunks[cid]["char_start"], "char_end": chunks[cid]["char_end"],
                     "p_yes": round(score, 5)}
                    for rank, (score, cid) in enumerate(ranked[:100], 1)
                ],
            })
        number = len(rows)
        # interim results after every group, so a reporter can score a long run
        write_jsonl(out / partial_name, rows)
        elapsed = time.time() - started
        print(f"  {number}/{len(questions)}  {elapsed / number:.1f} s/query, "
              f"{total_calls / number:.0f} judgments/query", flush=True)
    name = f"nav_{args.judge}{suffix}{new_suffix}.jsonl"
    write_jsonl(out / name, rows)
    print(f"wrote {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
