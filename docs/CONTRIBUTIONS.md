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
| Topic | Chunking strategy: structure-aware chunking (architecture decision 2) against fixed-length chunks |
| What changed or was tested | `chunking.strategy: fixed_length` (default `structure_aware`): every document takes the generic route, whole lines packed into pieces of at most `chunk_target_chars` with 150 characters of overlap, without clause labels; the numbering runs are still recorded as evidence. Measured at 600, 1,200 and 2,000 characters against the baseline; with the default settings the chunk files are byte-identical to `main`'s |
| Code / config | `src/companydocs_rag/chunking/chunker.py`, `chunking/cli.py`, `tests/chunking/`; `experiments/roef/settings.yaml` (fixed 1,200), `settings-fixed-600.yaml`, `settings-fixed-2000.yaml`; `experiments/roef/compare.py`; `specs/002-chunking-strategy.md` |
| Evaluation outcome | On 100 questions, baseline -> fixed 1,200: MRR 0.486 -> 0.443, page_hit@1 0.370 -> 0.320, page_hit@10 0.770 -> 0.720; 37 questions worse, 16 better (sign test p = 0.0055). Fixed 600 / 2,000: MRR 0.435 / 0.446, page_hit@1 0.270 / 0.280 (`evaluation/results/roef-baseline.json`, `roef-fixed-*.json`, `experiments/roef/results.md`) |
| Conclusion | Structure-aware chunking puts the right page at rank 1 more often than fixed length of any of the three sizes; chunk size barely matters (MRR 0.435-0.446). The gain comes from documents with real article headings; in the Staatscourant documents the detector accepts running page headers as a numbering run (1,618 of 1,691 chunks of `cao-metaalbewerking-avv-2026`), where the gain disappears. `structure_aware` stays the default; filtering page headers out of the heading detection is the next step |

Detail: [experiments/roef/](../experiments/roef/)

## Rutger

| | |
|---|---|
| Topic | Retrieval strategy: a decision model as relevance judge (DecisionRAG-style route on top of the hybrid search) |
| What changed or was tested | `retrieval.reranker: judge` (default `none`): a fine-tuned and weight-edited d3-lite judge (Qwen3.5-0.8B) picks the two best documents, judges their passages and the hybrid top 20, and ranks by P(yes); compared with the hybrid baseline, with the fine-tuned (unedited) judge beside it. The judge was chosen, fine-tuned and edited in a separate benchmark lab first (ConditionalQA, LegalBench-RAG, bBSARD); see the README |
| Code / config | `src/companydocs_rag/retrieval/judge.py`, `retrieval/search.py`, `vectorstore/store.py`, `tests/retrieval/test_judge.py`; `experiments/rutger/settings.yaml` (edited judge), `settings-judge-tuned.yaml` |
| Evaluation outcome | On 100 questions: page_hit@1 0.370 -> 0.630, page_hit@3 0.520 -> 0.830, page_hit@10 0.770 -> 0.960, MRR 0.486 -> 0.740 (baseline -> judge route, edited judge; `evaluation/results/rutger-baseline.json`, `rutger-judge-edited.json`) |
| Conclusion | The judge route stays behind a setting; the baseline remains the default (team decision). On 100 questions both judges clearly beat the baseline (edited MRR 0.740, fine-tuned 0.778, baseline 0.486). Cost: about 18 s per question on an 8 GB GPU |

Detail: [experiments/rutger/](../experiments/rutger/)

## Lucan

| | |
|---|---|
| Topic | Interface: a workspace that highlights the cited passage on the PDF page, so a user can verify an answer against its source |
| What changed or was tested | The demo becomes a workspace (side bar, chat view, document library, status block, page viewer next to the chat); the lines of each cited passage are highlighted on the page image by aligning the chunk's text with the page's words (pdfplumber for text pages, Tesseract's word boxes for scans), drawn only when at least 60% of the passage is found. Retrieval, prompt and answers are unchanged. Measured against the baseline interface, which shows the page without a highlight |
| Code / config | `src/companydocs_rag/interface/highlight.py`, `library.py`, `server.py`, `page.html` and the other interface modules, `tests/interface/` (incl. `test_highlight.py`, `test_library.py`); `answering.temperature: null` for hosted models; `experiments/lucan/measure_highlight.py`, `settings-fable.yaml` |
| Evaluation outcome | On the 90 answerable questions: a highlight on 465 of 466 answer passages (99.8%; the one without is a 2-word passage) and on at least one answer passage for every question; median 100% of the passage's words found (lowest 95.5%); the user is pointed to a median 7.7% of the page instead of the whole page (baseline 100%); on the retrieved top-5 pages (preview index, 62 questions) a highlight on 100% (`experiments/lucan/highlight_results.md`, `highlight_results.json`) |
| Conclusion | The highlight works on every answer page in the question set and narrows what the user has to read to about 8% of the page. Limits: text-layer pages only (no scanned answer pages in the set), the alignment compares text from the same extraction and the boxes were not checked visually, and whether users check answers faster was not measured (a user test is the next step) |

Detail: [experiments/lucan/](../experiments/lucan/)
