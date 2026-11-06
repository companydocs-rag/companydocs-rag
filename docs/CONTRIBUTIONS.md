# Individual contributions

The challenge asks for one clearly identifiable experiment, design choice or
improvement per person, each with its code/configuration and its evaluation
outcome. This page is the overview; the detail of each lives in
`experiments/<name>/`. Every number here comes from a file in
`evaluation/results/` and is measured on the same question set
(`evaluation/questions.csv`) and the same corpus (`corpus/sources.csv`) as the
baseline.

The challenge was approved by the lecturer; the challenge form states this.

## Shared foundation (team)

| what | where |
|---|---|
| Pipeline (parse pass with OCR, normalization, structure-aware chunking, hybrid dense + BM25 retrieval, local LLM answers, demo) | `src/companydocs_rag/`, copied from the team's earlier client pipeline and renamed |
| Public corpus definition and download | `corpus/sources.csv`, `scripts/download_corpus.py` |
| Question set (each person writes part of it; the `author` column shows who) | `evaluation/questions.csv` |
| Retrieval evaluation (`doc_hit@k`, `page_hit@k`, MRR) | `scripts/evaluate_retrieval.py` |
| Baseline result | `evaluation/results/baseline.json` |

## Roef

| | |
|---|---|
| Topic | _to fill in, e.g. chunking strategy_ |
| What changed or was tested | _the alternative, behind which setting/flag_ |
| Code / config | _paths, PR link_ |
| Evaluation outcome | _table: baseline vs. alternative, from `evaluation/results/`_ |
| Conclusion | _what the effect was and what the team decided_ |

Detail: [experiments/roef/](../experiments/roef/)

## Rutger

| | |
|---|---|
| Topic | Retrieval strategy: a decision model as relevance judge (DecisionRAG-style route on top of the hybrid search) |
| What changed or was tested | `retrieval.reranker: judge` (default `none`): a fine-tuned and weight-edited d3-lite judge (Qwen3.5-0.8B) picks the two best documents, judges their passages and the hybrid top 20, and ranks by P(yes); compared with the hybrid baseline, with the fine-tuned (unedited) judge beside it. The judge was chosen, fine-tuned and edited in a separate benchmark lab first (ConditionalQA, LegalBench-RAG, bBSARD); see the README |
| Code / config | `src/companydocs_rag/retrieval/judge.py`, `retrieval/search.py`, `vectorstore/store.py`, `tests/retrieval/test_judge.py`; `experiments/rutger/settings.yaml` (edited judge), `settings-judge-tuned.yaml` |
| Evaluation outcome | On 27 questions: page_hit@1 0.407 -> 0.481, page_hit@3 0.556 -> 0.778, page_hit@10 0.815 -> 0.926, MRR 0.518 -> 0.650 (baseline -> judge route, edited judge; `evaluation/results/rutger-full-baseline-20261104-1815.json`, `rutger-full-judge-edited-20261104-1815.json`) |
| Conclusion | The judge route stays behind a setting; the baseline remains the default (team decision). Cost: about half a minute per question on an 8 GB GPU (31 s measured) |

Detail: [experiments/rutger/](../experiments/rutger/)

## Lucan

| | |
|---|---|
| Topic | _to fill in, e.g. the demo interface (frontend) with a small user test_ |
| What changed or was tested | |
| Code / config | |
| Evaluation outcome | |
| Conclusion | |

Detail: [experiments/lucan/](../experiments/lucan/)
