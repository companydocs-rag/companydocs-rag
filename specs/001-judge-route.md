# 001: the judge route (decision model as retrieval judge)

Short change note (docs/DEVELOPMENT.md, Working together). Author: Rutger.

## Goal

Find the right page more often than the hybrid baseline by letting a small
decision model judge relevance, DecisionRAG-style (arXiv 2610.35868), without
changing the baseline: the alternative sits behind a setting.

## What changed

- `retrieval.reranker` (default `none`) with six `judge_*` settings at the end
  of the `retrieval` section; `judge` turns the route on.
- `src/companydocs_rag/retrieval/judge.py`: the judge (d3-lite plus a judge
  file with the fine-tuned LoRA, readout rows and weight edits) and the route:
  score every document on its title and opening, keep the best two, judge
  their passages (at most 300 each, closest by dense score) together with the
  hybrid top 20, each as "Document: <title>" plus its text in windows of 1,600
  characters, and rank by P(yes). The GPU cache is released after every call.
- `retrieval/search.py`: the settings and the `Retriever` hook;
  `vectorstore/store.py`: two read functions for the route.
- `pyproject.toml`, `uv.lock`: optional extra `judge` (`accelerate`); the ten
  runtime dependencies are unchanged.
- `evaluation/questions.csv`: the question set of 100 (RU01-RU100).
- `experiments/rutger/`: settings for both judges, the judge files (Git LFS),
  a download script for the base model, the write-up and the lab evidence.

## How it was tested

- `tests/retrieval/test_judge.py` (the route with a fake judge: windows,
  titles, document step, best window, result size, settings) and the updated
  settings pins in `test_search.py`; the full suite passes (2,085, without the
  OCR tests on a machine without Tesseract); ruff clean.
- Measured with `scripts/evaluate_retrieval.py` on the 100 questions:
  baseline MRR 0.486, edited judge 0.740, fine-tuned judge 0.778;
  page_hit@10 0.770, 0.960 and 0.930 (`evaluation/results/rutger-*.json`,
  `experiments/rutger/README.md`).

## What was left

- The default stays `none`; switching it is a team decision.
- The route needs a GPU in practice (about 18 s per question on 8 GB); on a CPU
  it takes minutes per question.
- The judge files are made outside this repository (the benchmark lab,
  `experiments/rutger/lab/`); the base model is downloaded, not committed.
- The answer layer is not measured with the route yet (only retrieval).
