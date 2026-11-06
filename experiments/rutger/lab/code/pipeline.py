"""Run our pipeline (chunker -> hybrid index -> hybrid search) on a benchmark.

    python -m bench.pipeline chunk  <bench>
    python -m bench.pipeline index  <bench>
    python -m bench.pipeline search <bench> [--deep]

chunk:  our chunker (chunk_document, structure hint unknown) on every
        normalized record, in parallel processes -> runs/<bench>/chunks.jsonl
index:  our embedding text, dense model (GPU fp16 via BENCH_EMBED_*), BM25
        terms and values, and our collection definition (dense cosine +
        sparse with Qdrant IDF), on a local Qdrant server.
search: our two legs and RRF fusion with the pipeline's defaults (dense 50,
        sparse 50, rrf_k 60, 20 candidates) -> runs/<bench>/baseline.jsonl.
        --deep uses 100/100/100 to measure recall@100 as well.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from .common import read_jsonl, run_dir, write_jsonl

os.environ.setdefault("BENCH_EMBED_DEVICE", "cuda")
os.environ.setdefault("BENCH_EMBED_DTYPE", "float16")
os.environ.setdefault("HF_HOME", r"%USERPROFILE%\models\hf")

_CONFIG = None
_MEMORY_CLIENT = None


def _chunk_one(path: str) -> Dict[str, Any]:
    global _CONFIG
    from companydocs_rag.chunking.chunker import chunk_document, chunking_settings

    if _CONFIG is None:
        _CONFIG = chunking_settings()
    record = json.loads(Path(path).read_text("utf-8"))
    return chunk_document(record, config=_CONFIG)


def cmd_chunk(bench: str) -> None:
    out = run_dir(bench)
    paths = sorted(str(p) for p in (out / "normalized").glob("*.json"))
    started = time.time()
    rows: List[Dict[str, Any]] = []
    routes = {"structured": 0, "generic": 0}
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        for doc in pool.map(_chunk_one, paths, chunksize=8):
            routes[doc["route"]] += 1
            for chunk in doc["chunks"]:
                rows.append(
                    {
                        "doc_id": doc["file"]["path"],
                        "route": doc["route"],
                        "language": doc["language"],
                        **{
                            key: chunk[key]
                            for key in (
                                "chunk_id",
                                "chunk_index",
                                "text",
                                "char_start",
                                "char_end",
                                "chunk_method",
                                "clause_label",
                                "clause_number",
                                "heading_family",
                                "numbering_run",
                                "part",
                                "part_count",
                                "exceeds_max_chars",
                                "page_start",
                                "page_end",
                                "text_origins",
                                "page_spans",
                            )
                        },
                    }
                )
    write_jsonl(out / "chunks.jsonl", rows)
    lengths = sorted(len(row["text"]) for row in rows)
    print(
        f"{bench}: {len(paths)} documents -> {len(rows)} chunks in "
        f"{time.time() - started:.0f}s; routes {routes}; chars median "
        f"{lengths[len(lengths) // 2]}, p95 {lengths[int(len(lengths) * 0.95)]}, "
        f"max {lengths[-1]}"
    )


def _collection(bench: str) -> str:
    return f"bench_{bench}"


def _backend() -> str:
    """local = Qdrant local mode, exactly as our pipeline runs (no TurboQuant:
    local mode ignores quantization); server = the localhost Qdrant server
    with TurboQuant. BENCH_QDRANT selects it."""
    return os.environ.get("BENCH_QDRANT", "local")


def _client(bench: str):
    from qdrant_client import QdrantClient

    from companydocs_rag.vectorstore import store

    store.COLLECTION_NAME = _collection(bench)  # functions read it at call time
    if _backend() == "local":
        # Same local-mode engine and exact search as our pipeline, held in RAM
        # (on the D: hard disk its SQLite writes were the bottleneck).
        global _MEMORY_CLIENT
        if _MEMORY_CLIENT is None:
            _MEMORY_CLIENT = QdrantClient(location=":memory:")
        return _MEMORY_CLIENT, store
    from .qdrant_server import URL, ensure_running

    ensure_running()
    return QdrantClient(url=URL, timeout=300), store


def _dense_vectors(bench: str, composed: List[str]):
    """The chunk embeddings, computed once on the GPU and cached as .npy."""
    import numpy as np

    cache = run_dir(bench) / "dense.npy"
    if cache.exists():
        vectors = np.load(cache)
        if len(vectors) == len(composed):
            print(f"{bench}: dense vectors from cache ({cache.name})")
            return vectors
    embedder = _embedder()
    model = embedder._model
    # The pipeline's own limit: companydocs-index refuses an embedding text over
    # min(embeddings.max_tokens, the model's limit); here longer texts are cut
    # at that limit and counted, so one huge article cannot exhaust the GPU.
    limit = min(embedder._config.max_tokens, model.max_seq_length or 10**9)
    model.max_seq_length = limit
    lengths = [min(limit, n) for n in (len(ids) for ids in model.tokenizer(composed, add_special_tokens=True)["input_ids"])]
    over = sum(1 for text_len in lengths if text_len >= limit)
    order = sorted(range(len(composed)), key=lambda i: -lengths[i])
    budget = 24_000  # padded tokens per batch; fits the 8 GB GPU in fp16
    batches, current = [], []
    for i in order:
        if current and lengths[current[0]] * (len(current) + 1) > budget:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    started = time.time()
    vectors = np.zeros((len(composed), embedder.dimension), dtype=np.float32)
    for number, batch in enumerate(batches, 1):
        vectors[batch] = model.encode(
            [composed[i] for i in batch],
            batch_size=len(batch),
            normalize_embeddings=True,  # as encode_documents
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if number % 200 == 0:
            print(f"  {bench}: {number}/{len(batches)} batches, {time.time() - started:.0f}s", flush=True)
    np.save(cache, vectors)
    print(
        f"{bench}: embedded {len(composed)} chunks in {time.time() - started:.0f}s "
        f"({len(batches)} batches; {over} texts cut at {limit} tokens)"
    )
    return vectors


def _embedder():
    from companydocs_rag.embeddings.dense import (
        SentenceTransformerEmbedder,
        embeddings_settings,
    )

    config = dataclasses.replace(embeddings_settings(), batch_size=32)
    return SentenceTransformerEmbedder(config, local_files_only=False)


def cmd_index(bench: str) -> None:
    from companydocs_rag.embeddings.dense import compose_embedding_text
    from companydocs_rag.embeddings.sparse import bm25_vector, sparse_terms
    from companydocs_rag.retrieval.search import retrieval_settings

    out = run_dir(bench)
    chunks = read_jsonl(out / "chunks.jsonl")
    started = time.time()
    composed = [compose_embedding_text(chunk) for chunk in chunks]
    terms = [sparse_terms(text) for text in composed]
    avgdl = sum(len(found) for found in terms) / len(terms)
    config = retrieval_settings()
    sparse = [
        bm25_vector(found, avgdl=avgdl, k1=config.bm25_k1, b=config.bm25_b)
        for found in terms
    ]
    print(f"{bench}: {len(chunks)} chunks, sparse terms in {time.time() - started:.0f}s")

    dense = _dense_vectors(bench, composed)

    client, store = _client(bench)
    if client.collection_exists(store.COLLECTION_NAME):
        client.delete_collection(store.COLLECTION_NAME)
    store.create_collection(client, int(dense.shape[1]))
    if _backend() == "server":
        # TurboQuant (the algorithm turbovec implements), built into Qdrant
        # >= 1.18: 4-bit quantized copies of the dense vectors, originals kept
        # in RAM for rescoring. Our own collection definition stays as it is.
        from qdrant_client import models

        client.update_collection(
            store.COLLECTION_NAME,
            quantization_config=models.TurboQuantization(
                turbo=models.TurboQuantQuantizationConfig(
                    bits=models.TurboQuantBitSize.BITS4, memory=models.Memory.PINNED
                )
            ),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000),
        )
    started = time.time()
    store.upload_points(
        client,
        (
            store.Point(
                chunk_id=chunk["chunk_id"],
                dense=vector.tolist(),
                sparse_indices=values.indices,
                sparse_values=values.values,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                    "text": chunk["text"],
                    "clause_label": chunk["clause_label"],
                },
            )
            for chunk, vector, values in zip(chunks, dense, sparse)
        ),
        256,
    )
    count = store.count_points(client)
    print(f"{bench}: uploaded {count} points in {time.time() - started:.0f}s")
    started = time.time()
    while _backend() == "server":  # wait for the HNSW index and TurboQuant copies
        info = client.get_collection(store.COLLECTION_NAME)
        if str(info.status).lower().endswith("green") and info.indexed_vectors_count:
            break
        time.sleep(2)
    if _backend() == "server":
        print(
            f"{bench}: indexed ({info.indexed_vectors_count} vectors, quantization "
            f"{info.config.quantization_config}) in {time.time() - started:.0f}s"
        )
        client.close()
    (out / "index_info.json").write_text(
        json.dumps({"chunks": len(chunks), "points": count, "avgdl": avgdl}), "utf-8"
    )


def _query_dense(client, store, vector, limit, mode):
    """store.query_dense with the search parameters of the chosen mode:
    exact = brute-force float search (what our local-mode index does);
    turbo = HNSW over the TurboQuant 4-bit copies, rescored with the
    originals (oversampling 2)."""
    from qdrant_client import models

    params = (
        models.SearchParams(exact=True)
        if mode == "exact"
        else models.SearchParams(
            quantization=models.QuantizationSearchParams(
                ignore=False, rescore=True, oversampling=2.0
            )
        )
    )
    response = client.query_points(
        store.COLLECTION_NAME,
        query=list(vector),
        using=store.DENSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
        search_params=params,
    )
    return store._hits(response.points)


def cmd_search(bench: str, deep: bool, mode: str = "turbo") -> None:
    from companydocs_rag.embeddings.sparse import query_sparse_vector, sparse_terms
    from companydocs_rag.retrieval.fusion import fuse
    from companydocs_rag.retrieval.search import retrieval_settings

    out = run_dir(bench)
    questions = read_jsonl(out / "questions.jsonl")
    config = retrieval_settings()
    dense_limit, sparse_limit, count = (
        (100, 100, 100)
        if deep
        else (config.dense_limit, config.sparse_limit, config.candidate_count)
    )
    embedder = _embedder()
    client, store = _client(bench)
    started = time.time()
    query_vectors = embedder._model.encode(  # batched queries, same prompt as encode_query
        [q["query"] for q in questions],
        prompt=embedder._config.query_instruction,
        batch_size=64,
        normalize_embeddings=True,  # as encode_query
        convert_to_numpy=True,
    ).tolist()
    rows = []
    for question, vector in zip(questions, query_vectors):
        dense_hits = _query_dense(client, store, vector, dense_limit, mode)
        sparse_hits = []
        if sparse_terms(question["query"]):
            values = query_sparse_vector(question["query"])
            sparse_hits = store.query_sparse(
                client, values.indices, values.values, sparse_limit
            )
        payloads = {hit.chunk_id: hit.payload for hit in [*dense_hits, *sparse_hits]}
        fused = fuse(
            [(hit.chunk_id, hit.score) for hit in dense_hits],
            [(hit.chunk_id, hit.score) for hit in sparse_hits],
            rrf_k=config.rrf_k,
            candidate_count=count,
        )
        rows.append(
            {
                "qid": question["qid"],
                "results": [
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk_id,
                        "doc_id": payloads[item.chunk_id]["doc_id"],
                        "char_start": payloads[item.chunk_id]["char_start"],
                        "char_end": payloads[item.chunk_id]["char_end"],
                        "dense_rank": item.dense_rank,
                        "sparse_rank": item.sparse_rank,
                    }
                    for item in fused
                ],
            }
        )
    name = f"baseline_{mode}{'_deep' if deep else ''}.jsonl"
    write_jsonl(out / name, rows)
    elapsed = time.time() - started
    print(
        f"{bench} [{mode}{' deep' if deep else ''}]: {len(rows)} queries searched in {elapsed:.1f}s "
        f"({1000 * elapsed / len(rows):.0f} ms/query)"
    )


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="bench.pipeline")
    parser.add_argument("command", choices=["chunk", "embed", "index", "search", "run"])
    parser.add_argument("bench")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--mode", choices=["exact", "turbo"], default="turbo")
    args = parser.parse_args(argv)
    if args.command == "chunk":
        cmd_chunk(args.bench)
    elif args.command == "embed":  # only the GPU embeddings, cached as dense.npy
        from companydocs_rag.embeddings.dense import compose_embedding_text

        chunks = read_jsonl(run_dir(args.bench) / "chunks.jsonl")
        _dense_vectors(args.bench, [compose_embedding_text(c) for c in chunks])
    elif args.command == "index":
        cmd_index(args.bench)
    elif args.command == "run":  # index + search in one process (in-memory local mode)
        cmd_index(args.bench)
        cmd_search(args.bench, False, args.mode)
        cmd_search(args.bench, True, args.mode)
    else:
        cmd_search(args.bench, args.deep, args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
