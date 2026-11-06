"""The decision-model judge: an alternative retrieval route (Rutger's experiment).

With ``retrieval.reranker: judge`` the Retriever keeps the hybrid search and
adds a decision model that judges relevance, in the shape of DecisionRAG
navigation (arXiv 2610.35868):

1. Document step: the judge scores every document of the index on its title
   and opening; the best ``judge_documents`` are kept.
2. Passage step: inside those documents it judges passages directly (at most
   ``judge_doc_chunks`` per document, the ones closest to the query by dense
   score), together with the hybrid candidates (the baseline's top
   ``candidate_count``).
3. Each passage is judged as "Document: <title>" plus its text; a text longer
   than ``judge_window_chars`` is judged in overlapping windows and gets the
   best window's score.
4. The pool is ranked by the judge's P(yes); the first ``candidate_count`` are
   the result.

The judge is vllm-sr's Decision 3.0 lite (``d3-lite``, Qwen3.5-0.8B, Apache
2.0), fine-tuned as a retrieval judge and optionally weight-edited outside
this repository (experiments/rutger/README.md). ``judge_model_dir`` is the
downloaded d3-lite folder, ``judge_adapter`` the judge file (LoRA, readout
rows and weight edits). d3 ships its own model code, so it is loaded with
``trust_remote_code=True``; the code was read before use. torch and
transformers are imported only when a judge is constructed, so importing
this module loads no model library (test_lazy_imports).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# The judge's question, as in its fine-tuning (bench-lab bench/judges.py).
QUESTION = (
    "Does the following {kind} contain information that helps answer the question "
    "in the state? Answer yes only if it is on the right document and topic.\n\n{text}"
)

# Padded tokens per forward pass (8 GB GPU; measured in the benchmark lab).
TOKEN_BUDGET = 12_000

ScoreFn = Callable[[Sequence[Tuple[str, str, str]]], List[float]]


def windows(text: str, size: int) -> List[str]:
    """``text`` cut into windows of at most ``size`` characters with stride
    ``size // 2``; the last window ends at the text's end. A text of at most
    ``size`` characters is one window."""
    if len(text) <= size:
        return [text]
    stride = max(1, size // 2)
    starts = list(range(0, len(text) - size + 1, stride))
    if starts[-1] + size < len(text):
        starts.append(len(text) - size)
    return [text[start : start + size] for start in starts]


def document_title(payload: Mapping[str, Any]) -> str:
    """A readable document title from a chunk payload: the source id (the
    file name without ``.pdf``) with dashes and underscores as spaces."""
    name = str(payload.get("file_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name.replace("-", " ").replace("_", " ").strip()


def passage_inputs(title: str, text: str, window_chars: int) -> List[str]:
    """The texts judged for one passage: "Document: <title>" plus each window
    of the text, every input at most ``window_chars`` characters (a title
    longer than half the window is cut)."""
    head = f"Document: {title[: window_chars // 2]}\n\n"
    room = max(100, window_chars - len(head))
    return [head + part for part in windows(text, room)]


def judge_rank(
    query: str,
    documents: Mapping[str, Dict[str, Any]],
    hybrid: Sequence[Dict[str, Any]],
    doc_chunks: Callable[[str], Sequence[Dict[str, Any]]],
    score: ScoreFn,
    *,
    top_documents: int,
    window_chars: int,
    result_count: int,
) -> List[Dict[str, Any]]:
    """The judge route for one query. ``documents`` maps a document key
    (its file hash) to ``{"title": ..., "opening": ...}``; ``hybrid`` is the
    baseline's candidate list; ``doc_chunks(key)`` gives the passages of one
    document to judge (candidate-shaped dicts); ``score`` judges
    ``(query, kind, text)`` triples. Returns at most ``result_count``
    candidates ranked by P(yes), each with ``judge_score`` and
    ``judge_document_score``, and ``rank`` renumbered from 1."""
    keys = sorted(documents)
    doc_scores = dict(
        zip(
            keys,
            score(
                [
                    (
                        query,
                        "document",
                        f"{documents[k]['title']}: {documents[k]['opening']}",
                    )
                    for k in keys
                ]
            ),
        )
    )
    chosen = sorted(keys, key=lambda k: -doc_scores[k])[:top_documents]

    pool: Dict[str, Dict[str, Any]] = {}
    for candidate in hybrid:
        pool[candidate["chunk_id"]] = dict(candidate)
    for key in chosen:
        for candidate in doc_chunks(key):
            pool.setdefault(candidate["chunk_id"], dict(candidate))

    items: List[Tuple[str, str, str]] = []
    owners: List[str] = []
    for chunk_id, candidate in pool.items():
        title = document_title(candidate)
        for text in passage_inputs(title, str(candidate.get("text", "")), window_chars):
            items.append((query, "passage", text))
            owners.append(chunk_id)
    best: Dict[str, float] = {}
    for chunk_id, value in zip(owners, score(items) if items else []):
        best[chunk_id] = max(value, best.get(chunk_id, float("-inf")))

    ranked = sorted(pool.values(), key=lambda c: (-best[c["chunk_id"]], c["chunk_id"]))
    out: List[Dict[str, Any]] = []
    for position, candidate in enumerate(ranked[:result_count], 1):
        candidate["rank"] = position
        candidate["judge_score"] = best[candidate["chunk_id"]]
        candidate["judge_document_score"] = doc_scores.get(
            candidate.get("file_sha256", "")
        )
        out.append(candidate)
    return out


def _budget_batches(
    lengths: Sequence[int], budget: int, cap: int = 128
) -> List[List[int]]:
    """Indices sorted longest first, cut into batches of at most ``budget``
    padded tokens (and ``cap`` rows)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: List[List[int]] = []
    current: List[int] = []
    for i in order:
        if current and (
            lengths[current[0]] * (len(current) + 1) > budget or len(current) >= cap
        ):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


class DecisionJudge:
    """d3-lite with the judge file applied: P(yes) per (query, kind, text).

    The judge file (``torch.save`` dict) holds ``lora`` (A/B pairs, merged as
    W += alpha / r * B @ A), ``readout_rows`` (the two yes/no readout rows),
    and optionally the weight edits ``deltas`` (per module weight),
    ``param_deltas`` (per parameter) and ``readout_delta``.
    """

    def __init__(self, model_dir: str, adapter: str = "", device: str = "cpu") -> None:
        import os
        import sys

        import torch
        from transformers import AutoModel

        self._sys = sys
        self.model = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, device=device, batch_size=32
        )
        runtime = self.model.runtime
        if device.startswith("cuda") and not torch.cuda.is_bf16_supported(
            including_emulation=False
        ):
            # bf16 is emulated on older GPUs; fp16 was 2.6x faster there
            runtime.backbone.half()
        if adapter:
            saved = torch.load(
                os.fspath(adapter), map_location="cpu", weights_only=True
            )
            scale = saved["alpha"] / saved["r"]
            with torch.no_grad():
                for name, tensor in saved["lora"].items():
                    if ".lora_A." not in name:
                        continue
                    module = name.split(".lora_A.")[0]
                    lora_b = saved["lora"][name.replace(".lora_A.", ".lora_B.")]
                    weight = runtime.backbone.get_submodule(module).weight
                    weight.add_(
                        ((lora_b.float() @ tensor.float()) * scale).to(
                            weight.device, weight.dtype
                        )
                    )
                rows = saved["readout_rows"]
                runtime.readout[: rows.shape[0]] = rows.to(
                    runtime.readout.device, runtime.readout.dtype
                )
                for module, delta in saved.get("deltas", {}).items():
                    weight = runtime.backbone.get_submodule(module).weight
                    weight.add_(delta.to(weight.device, weight.dtype))
                named = dict(runtime.backbone.named_parameters())
                for name, delta in saved.get("param_deltas", {}).items():
                    named[name].add_(delta.to(named[name].device, named[name].dtype))
                if "readout_delta" in saved:
                    delta = saved["readout_delta"]
                    runtime.readout[: delta.shape[0]] += delta.to(
                        runtime.readout.device, runtime.readout.dtype
                    )
        module = sys.modules[type(runtime).__module__]
        self._module = module
        keys = module.normalize_question({"type": "noul", "instructions": "x"}).keys
        self._true_index = list(keys).index("true")
        self._n_codes = len(keys)

    def __call__(self, items: Sequence[Tuple[str, str, str]]) -> List[float]:
        runtime = self.model.runtime
        texts = [
            runtime.text(
                f"Question: {query}",
                self._module.normalize_question(
                    {
                        "type": "noul",
                        "instructions": QUESTION.format(kind=kind, text=text),
                    }
                ).rendered,
            )
            for query, kind, text in items
        ]
        sequences = runtime.tokenizer(texts, add_special_tokens=False)["input_ids"]
        scores: List[float] = [0.0] * len(items)
        for batch in _budget_batches([len(s) for s in sequences], TOKEN_BUDGET):
            rows = runtime.probabilities(
                [sequences[i] for i in batch], [self._n_codes] * len(batch)
            )
            for i, row in zip(batch, rows):
                scores[i] = row[self._true_index]
        import torch

        if torch.cuda.is_available():
            # Long passages leave large cached blocks behind; on an 8 GB GPU they
            # pile up until Windows spills into shared memory (5-10x slower).
            torch.cuda.empty_cache()
        return scores


def load_judge(model_dir: str, adapter: str, device: str) -> Optional[ScoreFn]:
    """The judge as a score function (raises on a load failure)."""
    return DecisionJudge(model_dir, adapter, device)
