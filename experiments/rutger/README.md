# Rutger: a decision model as retrieval judge

## Question

Does a small decision model that judges relevance, used the way DecisionRAG
does ([arXiv 2610.35868](https://arxiv.org/abs/2610.35868)), find the right
page more often than our hybrid baseline (dense + BM25, fused with reciprocal
rank fusion) on the companydocs corpus?

## What I tried locally first (outside this repository)

Before touching this repository I built and tested the judge in a separate
benchmark lab, on public legal retrieval benchmarks, with our own pipeline
(chunker, Qwen3-Embedding-0.6B, BM25, RRF) underneath. Everything ran locally
(RTX 2070 Super, 8 GB); no data went to an external service.

1. **Choosing the model.** Zero-shot comparison of small decision models from
   their official sources: vllm-sr Decision 3.0
   ([d3-lite](https://huggingface.co/vllm-sr/d3-lite), d3-nano), Together
   [Tev1](https://huggingface.co/togethercomputer/Tev1-0.8B-experimental)
   (0.8B, 4B) and Qwen3-Reranker-0.6B. I picked **d3-lite** (Qwen3.5-0.8B,
   Apache 2.0): the best small option with a clear licence.
2. **Fine-tuning it as a judge.** LoRA (r 16) on 30,726 relevance judgments
   built from the ConditionalQA *train* split; validation AUC 0.827 -> 0.918.
3. **Full benchmark runs** (judge on top of our pipeline, beam search over a
   document tree), all questions of each benchmark:

   | benchmark | metric | pipeline alone | + fine-tuned judge |
   |---|---|---|---|
   | [ConditionalQA](https://haitian-sun.github.io/conditionalqa/) dev (285) | hit@1 / MRR | 0.557 / 0.688 | **0.720 / 0.814** |
   | [LegalBench-RAG](https://github.com/ZeroEntropy-AI/legalbenchrag) mini (776) | hit@10 / MRR | 0.338 / 0.154 | **0.527 / 0.319** |
   | bBSARD, Dutch part (222) | nDCG@10 / MRR | 0.219 / 0.262 | **0.304 / 0.363** |

   bBSARD is the bilingual version of
   [BSARD](https://huggingface.co/datasets/maastrichtlawtech/bsard) (CC BY-NC-SA
   4.0, used for evaluation only).

4. **Weight-space editing**, following Experiment 8 of "Sparks of
   Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2026,
   [blog](https://rikiyat.github.io/blog/weight-space-programming/)): a
   Python session with the weights loaded, a budget of forward passes on
   training-side data, no gradients, every edit diagnosed by hand and checked.
   - Round 1 found a real weakness (on texts much longer than the training
     texts, the judge's "no" heads in layer 15 leak attention) and tested four
     edits for it; none beat the unedited judge, so none was kept.
   - Round 2 unlocked information the judge computes but did not use: a
     readout edit along a direction of its own final state, plus one gated
     neuron in layer 23 that only fires on contract questions (5,120 weights,
     0.0006% of the model). Per-question AUC on held-back check questions:
     LegalBench 0.636 -> 0.763, ContractNLI 0.403 -> 0.697; ConditionalQA
     held-out unchanged (0.915 -> 0.916).
5. **Pipeline changes for the judge:** the document title in front of every
   judged passage, long chunks judged in windows of the trained length, a
   deeper document shortlist, and judging the passages of the best two
   documents directly instead of pruning them through article descriptions.
6. **Choice**, from four setups compared on the same questions of untouched
   benchmark halves (ConditionalQA 100, bBSARD 16, LegalBench 56):

   | setup | ConditionalQA hit@1 | bBSARD nDCG@10 | LegalBench MRR (old pipeline) |
   |---|---|---|---|
   | pipeline alone | 0.474 | 0.081 | 0.144 |
   | old pipeline + fine-tuned judge | 0.701 | 0.096 | 0.261 |
   | old pipeline + edited judge | 0.680 | 0.126 | **0.322** |
   | new pipeline + fine-tuned judge | **0.763** | 0.118 | (too slow to finish) |
   | new pipeline + edited judge | 0.753 | 0.117 | (too slow to finish) |

   The new pipeline is the biggest lever; the edited judge is better on legal
   text and slightly worse on ConditionalQA. This corpus is Dutch law, cao's
   and contracts, so this repository uses **the new pipeline with the edited
   judge**, with the fine-tuned judge measured beside it.

## What changed in this repository

All behind one setting, `retrieval.reranker` (default `none`, the baseline):

| file | change |
|---|---|
| `src/companydocs_rag/retrieval/judge.py` | new: the judge (d3-lite plus the judge file: LoRA, readout rows, weight edits) and the judge route |
| `src/companydocs_rag/retrieval/search.py` | seven settings; the `Retriever` runs the judge route when `reranker: judge` |
| `src/companydocs_rag/vectorstore/store.py` | two read functions for the route (all payloads; dense query inside one document) |
| `config/settings.yaml` | the seven keys at the end of `retrieval`, all off by default |
| `pyproject.toml`, `uv.lock` | optional extra `judge` (`accelerate`); the runtime dependencies stay at ten |
| `tests/retrieval/test_judge.py`, `test_search.py` | the route with a fake judge (no model), the settings and their pins |
| `experiments/rutger/settings.yaml` | the judge route with the edited judge (the main setup) |
| `experiments/rutger/settings-judge-tuned.yaml` | the same with the fine-tuned judge |
| `experiments/rutger/models/` | the two judge files (Git LFS, via the `.gitattributes` in that folder) |
| `experiments/rutger/download_d3_lite.py` | fetches the d3-lite base model at the pinned revision and checks it |
| `experiments/rutger/lab/` | the benchmark-lab scripts, reports and weight-editing logs behind the judge |
| `evaluation/questions.csv`, `evaluation/results/rutger-*.json` | the question set (RU01-RU100) and the result files |

The judge route, per question:

1. the judge scores every document on its title and opening and keeps the
   best two (`judge_documents`);
2. it judges the passages of those two documents (at most 300 each, the
   closest by dense score: `judge_doc_chunks`) together with the hybrid top 20;
3. each passage is judged as "Document: <title>" plus its text, in windows of
   1,600 characters (`judge_window_chars`), and gets its best window's score;
4. the 20 passages with the highest P(yes) are the result.

## How to run it

The judge files are in `experiments/rutger/models/` (Git LFS, so run
`git lfs pull` after cloning if they show as small text files):
`judge-edited.pt` (the main setup) and `judge-tuned.pt`. The d3-lite base
model (1.7 GB, [vllm-sr/d3-lite](https://huggingface.co/vllm-sr/d3-lite),
Apache 2.0) is not in git; the download script fetches the revision the
experiment used and checks it against the model's manifest:

```powershell
uv sync --extra judge
uv run python experiments/rutger/download_d3_lite.py
$env:COMPANYDOCS_RAG_SETTINGS = "experiments/rutger/settings.yaml"
uv run python scripts/evaluate_retrieval.py --label rutger-judge-edited
Remove-Item Env:COMPANYDOCS_RAG_SETTINGS
```

d3 ships its own model code, so it loads with `trust_remote_code=True`; that
code was read before use. On a CPU the route works but takes minutes per
question; the measurements below ran with `judge_device: "cuda:0"` (set in the
settings files). How the judge files were made, with the scripts, reports and
session logs: [lab/README.md](lab/README.md).

## Results

Measured on the corpus of `corpus/sources.csv` (26 documents) with the 100
questions of `evaluation/questions.csv` (RU01-RU100: 43 rule, 33 fact, 14
comparison, 10 unanswerable; every document has at least two; each answer was
checked against the text of the cited PDF page). Unanswerable questions count
with their closest source, as in `evaluation/README.md`.

Notes on the setup: the measurement copy used the CUDA build of the same torch
release (2.14.1) and embedded the index on the GPU in float32 (same model,
revision and precision as the CPU default); during the judge runs the query
embedder ran on the CPU, so the GPU held only the judge. Before each long run a
short probe on the heaviest questions checked that GPU memory stays below the
8 GB limit (peak 6.3 GB). Tesseract was not installed, so the one page without
a text layer (1 of 5 pages of `woo-houten-uitzendorganisaties-2025`) has no
text.

### The question set: 100 questions (RU01-RU100), Roef's merged chunker

<!-- FULL:START -->
Questions: 100; result files: `evaluation/results/rutger-baseline.json`, `evaluation/results/rutger-judge-edited.json`, `evaluation/results/rutger-judge-tuned.json`.

| setup | doc_hit@1 | page_hit@1 | page_hit@3 | page_hit@5 | page_hit@10 | mrr |
|---|---|---|---|---|---|---|
| baseline (hybrid) | 0.780 | 0.370 | 0.520 | 0.640 | 0.770 | 0.486 |
| judge route, edited judge | 0.820 | 0.630 | 0.830 | 0.880 | 0.960 | 0.740 |
| judge route, fine-tuned judge | 0.830 | 0.690 | 0.840 | 0.900 | 0.930 | 0.778 |

Judge route (edited) vs baseline: page_hit@1 0.370 -> 0.630, page_hit@3 0.520 -> 0.830, MRR 0.486 -> 0.740.

<details><summary>Page rank of the right source per question (- = not in the top 20)</summary>

| question | type | baseline (hybrid) | judge route, edited judge | judge route, fine-tuned judge |
|---|---|---|---|---|
| RU01 | rule | 8 | 3 | 5 |
| RU02 | fact | 1 | 1 | 1 |
| RU03 | fact | - | 1 | 3 |
| RU04 | fact | - | 11 | 11 |
| RU05 | fact | 1 | 2 | 1 |
| RU06 | rule | 1 | 3 | 1 |
| RU07 | rule | 1 | 6 | 1 |
| RU08 | rule | 1 | 1 | 1 |
| RU09 | rule | 2 | 2 | 2 |
| RU10 | fact | 2 | 1 | 1 |
| RU100 | rule | 4 | 1 | 1 |
| RU11 | comparison | 3 | 1 | 1 |
| RU12 | comparison | 9 | 4 | 3 |
| RU13 | rule | 3 | 1 | 1 |
| RU14 | rule | 11 | 1 | 1 |
| RU15 | fact | 1 | 1 | 1 |
| RU16 | rule | 1 | 1 | 1 |
| RU17 | unanswerable | 1 | 2 | 1 |
| RU18 | unanswerable | - | 2 | 2 |
| RU19 | fact | - | - | - |
| RU20 | rule | 1 | 2 | 2 |
| RU21 | rule | 7 | 2 | 4 |
| RU22 | rule | 1 | 1 | 1 |
| RU23 | fact | 4 | 1 | 1 |
| RU24 | rule | 4 | 1 | 2 |
| RU25 | fact | 10 | 1 | 1 |
| RU26 | rule | 4 | 8 | 10 |
| RU27 | unanswerable | 1 | 4 | 3 |
| RU28 | fact | 1 | 1 | 1 |
| RU29 | comparison | 2 | 1 | 1 |
| RU30 | comparison | - | 1 | 1 |
| RU31 | comparison | 4 | 1 | 1 |
| RU32 | rule | 1 | 1 | 1 |
| RU33 | fact | 2 | 8 | 5 |
| RU34 | fact | 19 | 1 | 1 |
| RU35 | fact | 1 | 1 | 1 |
| RU36 | fact | 3 | 1 | 1 |
| RU37 | rule | 1 | 1 | 1 |
| RU38 | rule | - | 3 | 2 |
| RU39 | fact | 4 | 1 | 1 |
| RU40 | unanswerable | 1 | 1 | 1 |
| RU41 | comparison | 2 | 2 | 1 |
| RU42 | fact | 5 | 5 | 4 |
| RU43 | fact | 1 | 1 | 1 |
| RU44 | rule | - | - | - |
| RU45 | fact | 7 | 1 | 1 |
| RU46 | fact | 13 | 2 | 2 |
| RU47 | comparison | - | 1 | 1 |
| RU48 | unanswerable | 1 | 1 | 1 |
| RU49 | rule | 1 | 1 | 1 |
| RU50 | fact | 6 | 1 | 1 |
| RU51 | rule | 4 | 5 | 1 |
| RU52 | rule | - | 6 | 8 |
| RU53 | rule | 5 | 9 | 8 |
| RU54 | unanswerable | 1 | 1 | 1 |
| RU55 | comparison | 9 | 9 | 13 |
| RU56 | rule | 13 | 3 | 1 |
| RU57 | rule | 13 | 1 | 1 |
| RU58 | fact | 1 | 1 | 1 |
| RU59 | comparison | 2 | 1 | 1 |
| RU60 | fact | 1 | 1 | 1 |
| RU61 | fact | 15 | 6 | 5 |
| RU62 | rule | 7 | 1 | 1 |
| RU63 | rule | 1 | 1 | 1 |
| RU64 | fact | 1 | 1 | 1 |
| RU65 | comparison | 4 | 4 | 18 |
| RU66 | comparison | 5 | 1 | 1 |
| RU67 | unanswerable | 1 | 3 | 2 |
| RU68 | fact | 3 | 1 | 1 |
| RU69 | rule | - | 1 | 1 |
| RU70 | fact | 1 | 1 | 1 |
| RU71 | rule | 1 | 2 | 1 |
| RU72 | fact | 6 | 1 | 1 |
| RU73 | rule | 2 | 3 | 3 |
| RU74 | rule | - | 2 | 4 |
| RU75 | fact | 1 | 1 | 1 |
| RU76 | unanswerable | 8 | 1 | 1 |
| RU77 | rule | 1 | 1 | 1 |
| RU78 | rule | 1 | 1 | 1 |
| RU79 | rule | 1 | 2 | 2 |
| RU80 | rule | 3 | 1 | 1 |
| RU81 | rule | 1 | 1 | 1 |
| RU82 | rule | 1 | 1 | 1 |
| RU83 | comparison | 1 | 1 | 1 |
| RU84 | fact | 3 | 1 | 1 |
| RU85 | fact | 4 | 3 | 2 |
| RU86 | unanswerable | - | - | - |
| RU87 | rule | 15 | 1 | 1 |
| RU88 | rule | 18 | 1 | 1 |
| RU89 | comparison | 12 | 1 | 1 |
| RU90 | rule | - | 8 | - |
| RU91 | rule | 1 | 1 | 1 |
| RU92 | rule | 10 | 1 | 1 |
| RU93 | rule | 1 | 1 | 1 |
| RU94 | fact | 13 | 1 | 1 |
| RU95 | comparison | 8 | 1 | 1 |
| RU96 | fact | 6 | 3 | 2 |
| RU97 | unanswerable | 2 | 2 | 2 |
| RU98 | rule | 3 | 1 | 1 |
| RU99 | fact | 1 | 1 | 1 |

</details>
<!-- FULL:END -->

### Earlier run: the first 27 questions (RU01-RU27)

<!-- PRELIM:START -->
Questions: 27; result files: `evaluation/results/rutger-prelim-baseline.json`, `evaluation/results/rutger-prelim-edited.json`, `evaluation/results/rutger-prelim-tuned.json`.

| setup | doc_hit@1 | page_hit@1 | page_hit@3 | page_hit@5 | page_hit@10 | mrr |
|---|---|---|---|---|---|---|
| baseline (hybrid) | 0.741 | 0.407 | 0.556 | 0.667 | 0.815 | 0.518 |
| judge route, edited judge | 0.704 | 0.481 | 0.778 | 0.852 | 0.926 | 0.650 |
| judge route, fine-tuned judge | 0.704 | 0.556 | 0.815 | 0.889 | 0.926 | 0.690 |

Judge route (edited) vs baseline: page_hit@1 0.407 -> 0.481, page_hit@3 0.556 -> 0.778, MRR 0.518 -> 0.650.

<details><summary>Page rank of the right source per question (- = not in the top 20)</summary>

| question | type | baseline (hybrid) | judge route, edited judge | judge route, fine-tuned judge |
|---|---|---|---|---|
| RU01 |  | 8 | 3 | 5 |
| RU02 |  | 1 | 1 | 1 |
| RU03 |  | - | 1 | 3 |
| RU04 |  | - | 11 | 11 |
| RU05 |  | 1 | 2 | 1 |
| RU06 |  | 1 | 3 | 1 |
| RU07 |  | 1 | 6 | 1 |
| RU08 |  | 1 | 1 | 1 |
| RU09 |  | 2 | 2 | 2 |
| RU10 |  | 2 | 1 | 1 |
| RU11 |  | 3 | 1 | 1 |
| RU12 |  | 9 | 4 | 3 |
| RU13 |  | 3 | 1 | 1 |
| RU14 |  | 11 | 1 | 1 |
| RU15 |  | 1 | 1 | 1 |
| RU16 |  | 1 | 1 | 1 |
| RU17 |  | 1 | 2 | 1 |
| RU18 |  | - | 2 | 2 |
| RU19 |  | - | - | - |
| RU20 |  | 1 | 2 | 2 |
| RU21 |  | 7 | 2 | 4 |
| RU22 |  | 1 | 1 | 1 |
| RU23 |  | 4 | 1 | 1 |
| RU24 |  | 4 | 1 | 2 |
| RU25 |  | 10 | 1 | 1 |
| RU26 |  | 4 | 8 | 10 |
| RU27 |  | 1 | 4 | 3 |

</details>
<!-- PRELIM:END -->

## Conclusion

On 100 questions the judge route finds the right page far more often than the
hybrid baseline: with the edited judge page_hit@1 0.370 -> 0.630, page_hit@3
0.520 -> 0.830, page_hit@10 0.770 -> 0.960 and MRR 0.486 -> 0.740. The
fine-tuned judge is a little ahead at the top (page_hit@1 0.690, MRR 0.778) and
the edited judge further down (page_hit@10 0.960 against 0.930); on this Dutch
corpus the edit's contract-question path rarely fires, so the two are close.
On the English legal benchmarks the edited judge was the better one (see "What
I tried locally first").

The judge route stays behind `retrieval.reranker` and the baseline stays the
default; changing the default needs this result and a team decision
(docs/DEVELOPMENT.md). The cost is time: about 18 seconds per question on an
8 GB GPU (31 minutes for the 100 questions), against about a second for the
hybrid search.
