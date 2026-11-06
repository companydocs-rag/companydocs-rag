# The benchmark lab: how the judge was made

Evidence of the local work behind Rutger's experiment (see
[../README.md](../README.md), "What I tried locally first"). The lab was a
separate copy of our pipeline (the same code this repository started from),
run on public legal retrieval benchmarks on one laptop (RTX 2070 Super, 8 GB).
The files are kept as they ran, only with local paths replaced by
`%USERPROFILE%`, `<lab>` and `<workspace>` and the package name by
`companydocs_rag`; they are not part of the package, are not wired into this
repository and are excluded from ruff (`ruff.toml` here). The benchmark
datasets themselves are not included.

| file | what it is |
|---|---|
| `code/prepare.py` | turns ConditionalQA, LegalBench-RAG (mini) and bBSARD into our pipeline's input (one normalized record per document) |
| `code/pipeline.py`, `code/fastsearch.py` | our chunker, embeddings, BM25 and RRF over each benchmark corpus; `fastsearch.py` is an in-process exact search (and TurboQuant) checked equal to Qdrant local mode |
| `code/hierarchy.py` | the document trees the judge navigates (ConditionalQA pages and sections, LegalBench document types, documents and articles, bBSARD codes and articles) |
| `code/judges.py` | the decision models as judges (d3-lite and d3-nano, Tev1, Qwen3-Reranker), zero-shot and fine-tuned, with the weight edits applied at load |
| `code/navigate.py` | DecisionRAG-style beam search with the judge, batched across questions, plus the pipeline options (document context, judge windows, flat top documents, deeper shortlist, held-out half) |
| `code/traindata.py`, `code/train_d3.py`, `code/decide_extend.py` | judge training data from ConditionalQA *train*, the LoRA fine-tune of d3-lite, and the fixed rule that decided to extend it to all training examples |
| `code/metrics.py`, `code/report.py`, `code/compare_mini.py`, `code/failures.py` | the metrics, the results report, the same-question comparison of four setups, and the failure analysis |
| `code/fastkernels.py` | the kernel choice (plain PyTorch measured faster than flash-linear-attention on this GPU) |
| `code/wsp/kernel.py`, `code/wsp/run.py` | the weight-space programming session: a persistent Python kernel with the weights loaded, a counted forward-pass budget, gradients refused, every cell logged |
| `reports/RESULTS.md` | all benchmark results |
| `reports/mini.md` | the four setups on the same untouched questions (the basis for choosing new pipeline + edited judge) |
| `reports/failures.md`, `reports/SHORTCOMINGS.md` | where the judge fails and whether the ability is inside the model (linear probes) or missing |
| `reports/weight-editing-round1-REPORT.md` | round 1 of the weight editing: the diagnosis and the four edits that were not kept |
| `logs/weight-editing-round1-log.md`, `logs/weight-editing-round2-log.md` | every cell of both editing sessions with its output and the forward passes used |

The judge files that came out of this are in [../models/](../models/) (Git
LFS): `judge-tuned.pt` (the fine-tuned LoRA and readout rows) and
`judge-edited.pt` (the same plus the round-2 weight edits).
