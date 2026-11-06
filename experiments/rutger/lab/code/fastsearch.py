"""Our hybrid search, in-process and fast, exact like Qdrant local mode.

    python -m bench.fastsearch <bench> [--deep] [--dense exact|turbo4|turbo2]
    python -m bench.fastsearch <bench> --check   (compare with baseline_exact.jsonl)

Why: local mode searches in pure Python (0.5 s/query at 22K points) and the
sandbox blocks a localhost Qdrant server. This reproduces local mode's maths
with matrix products and feeds our own fusion:

  dense   cosine over the normalized chunk vectors (dense.npy), top dense_limit
  sparse  our BM25 document values (bm25_vector, same avgdl/k1/b as the
          index), query values from query_sparse_vector times Qdrant's IDF
          ln((N - df + 0.5) / (df + 0.5) + 1) (local_collection._compute_idf),
          dot product over shared terms, only points that share a term
  fusion  companydocs_rag.retrieval.fusion.fuse (RRF k 60), unchanged

--dense turbo4|turbo2 swaps the dense leg for turbovec's TurboQuant index
(4 or 2 bits per dimension; the algorithm Qdrant >= 1.18 ships built in).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from .common import read_jsonl, run_dir, write_jsonl


class HybridIndex:
    def __init__(self, bench: str) -> None:
        from scipy import sparse

        from companydocs_rag.embeddings.dense import compose_embedding_text
        from companydocs_rag.embeddings.sparse import bm25_vector, sparse_terms
        from companydocs_rag.retrieval.search import retrieval_settings

        out = run_dir(bench)
        self.chunks = read_jsonl(out / "chunks.jsonl")
        self.ids = [c["chunk_id"] for c in self.chunks]
        self.dense = np.load(out / "dense.npy").astype(np.float32)
        norms = np.linalg.norm(self.dense, axis=1, keepdims=True)
        self.dense /= np.where(norms == 0, 1, norms)  # cosine, as Qdrant
        self.config = retrieval_settings()
        terms = [sparse_terms(compose_embedding_text(c)) for c in self.chunks]
        avgdl = sum(len(t) for t in terms) / len(terms)
        rows, cols, vals, self.column = [], [], [], {}
        for row, found in enumerate(terms):
            vector = bm25_vector(found, avgdl=avgdl, k1=self.config.bm25_k1, b=self.config.bm25_b)
            for index, value in zip(vector.indices, vector.values):
                col = self.column.setdefault(index, len(self.column))
                rows.append(row)
                cols.append(col)
                vals.append(value)
        self.matrix = sparse.csc_matrix(
            (np.asarray(vals, dtype=np.float64), (rows, cols)),
            shape=(len(self.chunks), len(self.column)),
        )
        df = np.diff(self.matrix.indptr)  # documents per term
        n = len(self.chunks)
        self.idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)
        self.turbo = None

    def use_turbo(self, bits: int) -> None:
        import turbovec

        self.turbo = turbovec.TurboQuantIndex(dim=self.dense.shape[1], bit_width=bits)
        # TQ+ calibration on a uniform random sample (fixed seed), as the
        # library documents; then add every vector and warm the caches.
        sample = np.random.default_rng(0).choice(len(self.dense), size=min(1024, len(self.dense)), replace=False)
        self.turbo.calibrate(np.ascontiguousarray(self.dense[np.sort(sample)]))
        self.turbo.add(np.ascontiguousarray(self.dense))
        self.turbo.prepare()

    def dense_leg(self, query_vector: np.ndarray, limit: int) -> List[Tuple[str, float]]:
        if self.turbo is not None:
            scores, idx = self.turbo.search(query_vector[None, :].astype(np.float32), k=limit)
            return [(self.ids[int(i)], float(s)) for s, i in zip(scores[0], idx[0]) if int(i) >= 0]
        scores = self.dense @ query_vector
        top = np.argpartition(-scores, min(limit, len(scores) - 1))[:limit]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]

    def sparse_leg(self, query: str, limit: int) -> List[Tuple[str, float]]:
        from companydocs_rag.embeddings.sparse import query_sparse_vector, sparse_terms

        if not sparse_terms(query):
            return []
        vector = query_sparse_vector(query)
        cols, weights = [], []
        for index, value in zip(vector.indices, vector.values):
            col = self.column.get(index)
            if col is not None:
                cols.append(col)
                weights.append(value * self.idf[col])
        if not cols:
            return []
        sub = self.matrix[:, cols]
        scores = np.asarray(sub @ np.asarray(weights)).ravel()
        touched = np.unique(sub.indices)  # points that share a term with the query
        if len(touched) > limit:
            part = np.argpartition(-scores[touched], limit - 1)[:limit]
            touched = touched[part]
        touched = touched[np.argsort(-scores[touched])]
        return [(self.ids[i], float(scores[i])) for i in touched]


def search_all(bench: str, deep: bool, dense_mode: str, depth: int = 100) -> List[Dict[str, Any]]:
    from companydocs_rag.retrieval.fusion import fuse

    from .pipeline import _embedder

    started = time.time()
    index = HybridIndex(bench)
    if dense_mode.startswith("turbo"):
        index.use_turbo(int(dense_mode[-1]))
    print(f"{bench}: index ready in {time.time() - started:.1f}s ({len(index.ids)} chunks)")
    questions = read_jsonl(run_dir(bench) / "questions.jsonl")
    config = index.config
    dense_limit, sparse_limit, count = (
        (depth, depth, depth) if deep else (config.dense_limit, config.sparse_limit, config.candidate_count)
    )
    embedder = _embedder()
    query_vectors = embedder._model.encode(
        [q["query"] for q in questions],
        prompt=embedder._config.query_instruction,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    by_id = {c["chunk_id"]: c for c in index.chunks}
    started = time.time()
    rows = []
    for question, vector in zip(questions, query_vectors):
        fused = fuse(
            index.dense_leg(vector, dense_limit),
            index.sparse_leg(question["query"], sparse_limit),
            rrf_k=config.rrf_k,
            candidate_count=count,
        )
        rows.append({
            "qid": question["qid"],
            "results": [
                {"rank": item.rank, "chunk_id": item.chunk_id,
                 "doc_id": by_id[item.chunk_id]["doc_id"],
                 "char_start": by_id[item.chunk_id]["char_start"],
                 "char_end": by_id[item.chunk_id]["char_end"],
                 "dense_rank": item.dense_rank, "sparse_rank": item.sparse_rank}
                for item in fused
            ],
        })
    elapsed = time.time() - started
    print(f"{bench} [{dense_mode}{' deep' if deep else ''}]: {len(rows)} queries in {elapsed:.2f}s "
          f"({1000 * elapsed / len(rows):.1f} ms/query)")
    return rows


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="bench.fastsearch")
    parser.add_argument("bench")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--depth", type=int, default=100,
                        help="with --deep: candidates per leg and in the list (default 100, as fast_exact_deep); "
                             "any other depth is written to a new file fast_<dense>_deep<depth>.jsonl, never over an old one")
    parser.add_argument("--dense", default="exact", choices=["exact", "turbo4", "turbo2"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.depth != 100 and not args.deep:
        parser.error("--depth needs --deep")
    if args.depth < 1:
        parser.error("--depth must be at least 1")
    out = run_dir(args.bench)
    if args.depth != 100 and (out / f"fast_{args.dense}_deep{args.depth}.jsonl").exists():
        parser.error(f"fast_{args.dense}_deep{args.depth}.jsonl exists; it is not overwritten")
    rows = search_all(args.bench, args.deep, args.dense, args.depth)
    if args.check:
        reference = {r["qid"]: [x["chunk_id"] for x in r["results"]] for r in read_jsonl(out / "baseline_exact.jsonl")}
        same = sum(1 for r in rows if [x["chunk_id"] for x in r["results"]] == reference.get(r["qid"]))
        same_set = sum(1 for r in rows if set(x["chunk_id"] for x in r["results"]) == set(reference.get(r["qid"], [])))
        print(f"check vs Qdrant local mode: identical ranking {same}/{len(rows)}, identical top-20 set {same_set}/{len(rows)}")
        return 0
    if args.deep and args.depth != 100:
        name = f"fast_{args.dense}_deep{args.depth}.jsonl"
    else:
        name = f"fast_{args.dense}{'_deep' if args.deep else ''}.jsonl"
    write_jsonl(out / name, rows)
    print(f"wrote {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
