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
| Topic | _to fill in, e.g. retrieval strategy / reranking_ |
| What changed or was tested | |
| Code / config | |
| Evaluation outcome | |
| Conclusion | |

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
