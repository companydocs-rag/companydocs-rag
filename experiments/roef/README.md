# Roef: structure-aware versus fixed-length chunking

## Question

Architecture decision 2 says: chunk on the documents' own article and clause
numbering, not on a fixed length. The chunker does that per document when it
finds an accepted numbering run, and its chunks carry the heading as a clause
label, which is embedded in front of the text (dense and BM25). Does that
actually find the right page more often than structure-blind chunks of a fixed
length, on this corpus and the shared question set? And how much does the size
of fixed-length chunks matter?

## What changed

One setting, `chunking.strategy`, with the baseline as the default:

| value | what the chunker does |
|---|---|
| `structure_aware` (default) | unchanged: per document, the structured route when a numbering run is accepted, else the generic route |
| `fixed_length` | every document takes the generic route: whole lines packed into pieces of at most `chunk_target_chars` with `chunk_overlap_chars` of overlap, without clause labels; the heading candidates and runs are still found and recorded in `heading_evidence`, but decide nothing |

| file | change |
|---|---|
| `src/companydocs_rag/chunking/chunker.py` | the setting (`STRATEGY`, `STRATEGIES`), its validation, and `chunk_document` ignoring the runs under `fixed_length` |
| `src/companydocs_rag/chunking/cli.py` | one INFO line at the start of a run under a non-default strategy |
| `config/settings.yaml` | `chunking.strategy: "structure_aware"`, first key of the section |
| `tests/chunking/test_chunker.py`, `test_cli.py` | the setting and its pins; `fixed_length` cuts a structured document exactly as if its run were rejected, keeps the heading evidence, leaves a document without an accepted run unchanged, takes its sizes from the settings; the command's log and exit code 2 for an invalid strategy |
| `docs/DEVELOPMENT.md`, `specs/002-chunking-strategy.md` | one sentence on the setting; the change note |
| `experiments/roef/settings*.yaml` | the three fixed-length runs (below) |
| `experiments/roef/compare.py`, `results.md` | the comparison script and its output |
| `evaluation/results/roef-*.json` | the result files (`roef-baseline-cpu.json`: the check below) |

The chunk files keep their format (route `generic`, chunk method `generic`), so
the index, search and answer layers read them unchanged. Why line-aligned
pieces and not token windows: the chunker's invariants say every chunk starts
and ends at a line boundary and is an unchanged slice of the text, with
per-page provenance. Lines in these PDFs are short, so the pieces are close to
constant length anyway (for 1,200: p10 1,110, median 1,161, p90 1,193
characters).

**The default is unchanged.** With the default settings, this branch writes
chunk files that are byte-identical to those of `main`'s chunker, for all 26
documents. The full test suite passes (2,515 passed, 1 xfailed, the known
`test_structure.py` limitation); ruff is clean.

## Setup

- Corpus: the 26 public PDFs of `corpus/sources.csv` (1,188 pages, 2 of them
  OCR'd), through the parse pass, normalization and the chunker.
- Questions: the shared set `evaluation/questions.csv`, 100 questions
  (RU01-RU100). It was written before this experiment and not changed;
  nothing was tuned on it. The three sizes were fixed before the first result:
  half of, equal to and the maximum of the baseline's sizes
  (`chunk_target_chars` 1,200, `chunk_max_chars` 2,000).
- Runs, each with its own chunk folder and index, measured with
  `scripts/evaluate_retrieval.py` (hybrid dense + BM25 with RRF, no reranker):

  | label | settings | chunks |
  |---|---|---|
  | `roef-baseline` | `config/settings.yaml` | structure-aware |
  | `roef-fixed-600` | `experiments/roef/settings-fixed-600.yaml` | fixed length, at most 600 characters |
  | `roef-fixed-1200` | `experiments/roef/settings.yaml` (the main alternative) | fixed length, at most 1,200 characters |
  | `roef-fixed-2000` | `experiments/roef/settings-fixed-2000.yaml` | fixed length, at most 2,000 characters |

  Every settings file is a full copy of `config/settings.yaml`; they differ
  from it only in `chunking.strategy` and, for 600 and 2,000,
  `chunking.chunk_target_chars`. The overlap stays 150.
- The baseline reproduces Rutger's baseline (`rutger-baseline.json`, measured
  on another machine): the same summary to three decimals and the same ranks
  for all 100 questions. The only difference is one page range in a top-3
  list, in `woo-houten-uitzendorganisaties-2025`, the one document with OCR
  pages.
- **How the indexes were built.** On this laptop's CPU one index takes about
  two hours. The four indexes above were built on its GPU (RTX 3050 Ti, 4 GB)
  instead, with `companydocs-index` unchanged except for two values set by a
  local wrapper that is not part of the repository: the device (`cuda` instead
  of `cpu`) and the encoding batch size (4 instead of 16, because 16 does not
  fit in 4 GB in float32). The model, its revision and float32 are unchanged.
  A check on 96 chunks gave a cosine of at least 0.99999994 between vectors
  encoded with batch 4 and batch 16. Queries are embedded on the CPU, as
  always. As a check, the baseline index was also built the official way,
  `uv run companydocs-index` on the CPU (7,160 s): `roef-baseline-cpu.json`
  has the same ranks and the same top 3 as `roef-baseline.json` for all 100
  questions.
- Index manifests record the libraries used (`torch 2.14.1+cu130` for the GPU
  builds).

### How to run it

```powershell
uv run companydocs-parse-pass data/raw -r      # from PowerShell (known issue in docs/DEVELOPMENT.md)
uv run companydocs-normalize
uv run companydocs-chunk                       # baseline chunks in data/interim/chunks
uv run companydocs-index --index-dir data/processed/index-roef-baseline
uv run python scripts/evaluate_retrieval.py --label roef-baseline --index-dir data/processed/index-roef-baseline

$env:COMPANYDOCS_RAG_SETTINGS = "experiments/roef/settings.yaml"   # or settings-fixed-600.yaml / -2000.yaml
uv run companydocs-chunk --chunks-dir data/interim/chunks-roef-fixed-1200
uv run companydocs-index --chunks-dir data/interim/chunks-roef-fixed-1200 --index-dir data/processed/index-roef-fixed-1200
uv run python scripts/evaluate_retrieval.py --label roef-fixed-1200 --index-dir data/processed/index-roef-fixed-1200
Remove-Item Env:COMPANYDOCS_RAG_SETTINGS

uv run python experiments/roef/compare.py --baseline roef-baseline `
    --variant roef-fixed-600 --variant roef-fixed-1200 --variant roef-fixed-2000 `
    --chunks roef-baseline=data/interim/chunks `
    --chunks roef-fixed-600=data/interim/chunks-roef-fixed-600 `
    --chunks roef-fixed-1200=data/interim/chunks-roef-fixed-1200 `
    --chunks roef-fixed-2000=data/interim/chunks-roef-fixed-2000
```

## Results

All numbers come from `evaluation/results/roef-*.json`; the full tables,
including per question type, are in [results.md](results.md).

| metric | baseline (structure-aware) | fixed 600 | fixed 1,200 | fixed 2,000 |
|---|---|---|---|---|
| doc_hit@1 | **0.780** | 0.700 | 0.760 | 0.710 |
| doc_hit@10 | **0.970** | **0.970** | 0.960 | 0.930 |
| page_hit@1 | **0.370** | 0.270 | 0.320 | 0.280 |
| page_hit@3 | 0.520 | **0.550** | 0.500 | **0.550** |
| page_hit@5 | 0.640 | **0.660** | 0.600 | 0.640 |
| page_hit@10 | 0.770 | **0.790** | 0.720 | 0.750 |
| MRR | **0.486** | 0.435 | 0.443 | 0.446 |

Paired against the baseline, per question (the rank of the first page match;
sign test over the questions that moved; paired bootstrap of the questions,
10,000 resamples):

| | rank higher / lower / same | sign test p | MRR difference [95% interval] | page_hit@1 difference [95% interval] |
|---|---|---|---|---|
| fixed 600 | 23 / 38 / 39 | 0.072 | -0.051 [-0.111, +0.007] | **-0.100 [-0.180, -0.030]** |
| fixed 1,200 | 16 / 37 / 47 | **0.0055** | -0.044 [-0.089, +0.002] | -0.050 [-0.120, +0.010] |
| fixed 2,000 | 24 / 35 / 41 | 0.19 | -0.040 [-0.091, +0.010] | **-0.090 [-0.160, -0.020]** |

The chunks:

| run | chunks | mean / median characters | pages per chunk | chunks spanning pages | with a clause label |
|---|---|---|---|---|---|
| baseline | 3,986 | 1,047 / 1,147 | 1.29 | 29% | 67% |
| fixed 600 | 8,645 | 555 / 561 | 1.16 | 16% | 0% |
| fixed 1,200 | 3,685 | 1,151 / 1,161 | 1.34 | 34% | 0% |
| fixed 2,000 | 2,092 | 1,942 / 1,962 | 1.58 | 55% | 0% |

A chunk that spans more pages matches a question's page more easily, so
`page_hit` favours larger chunks. At 1,200 and 2,000 that favours the
fixed-length runs, not the baseline, and they still score lower at rank 1.

## What the chunk files show

Splitting the questions by what the baseline chunker actually cut their
document on (from `results.md`):

| expected document | questions | baseline MRR / page_hit@1 | fixed 600 | fixed 1,200 | fixed 2,000 |
|---|---|---|---|---|---|
| cut on article headings | 69 | **0.462 / 0.35** | 0.384 / 0.22 | 0.407 / 0.29 | 0.431 / 0.25 |
| cut on running page headers | 19 | **0.599 / 0.47** | 0.590 / 0.42 | 0.581 / 0.42 | 0.577 / **0.47** |
| generic route in both | 12 | 0.446 / 0.33 | **0.487 / 0.33** | 0.429 / 0.33 | 0.327 / 0.17 |

- **Where the headings are real, structure helps.** In the cao's of ABU and
  NBBU, the SNA regulations and the contracts the chunks follow the articles
  (a long article is split into parts that keep its label), and the label
  tells both retrieval legs what a chunk is about. RU93 (may the same
  inspector keep coming back?) finds the right page at rank 1 with the
  baseline, in the chunk labelled "Artikel 9 Werkwijze", followed by
  "Artikel 6 Inspectiepersoneel" and "Artikel 7 Kwalificatie inspecteurs";
  with fixed 1,200, whose pieces start and end mid-article, it is at rank 5.
- **The detector also accepts running page headers.** In the Staatscourant
  publications every page starts with its page number and the publication
  line ("57 Staatscourant 2026 nr. ..."). Those numbers ascend by one, so they
  pass as a decimal numbering run: in `cao-metaalbewerking-avv-2026` (402
  pages) the chunker accepts a run of 402 members that starts with two real
  items and then follows the page headers, and 1,618 of its 1,691 chunks
  carry a page header as their clause label. The same happens in
  `cao-metalektro-avv-2026` (107 labels), `wet-baadi-stb-2026-41` (56) and
  `wet-wml-indexatie-2026-07` (4). There the structured route is in effect a
  per-page cut with a noisy label, and the advantage over fixed length
  disappears (0.599 against 0.577-0.590).
- **A wrong label can mislead.** In `wet-wml-indexatie-2026-07` the footnote
  line "3 Stb. 1983, 300." became the label of the chunk with the minimum wage
  table; RU03 (minimum wage at 18) misses that page in the baseline top 20,
  where page-header chunks of other Staatscourant pages take the top ranks,
  and finds it at rank 2 with fixed 1,200. One cross-reference at a line start
  ("artikel 11 (lid 1) opgenomen bepalingen.") was taken as the heading of
  article 11 in `cert-sna-reglement-inspectie-2026`.
- **For the generic-route documents** the baseline and fixed 1,200 have the
  same chunks; their scores still move a little because chunks of other
  documents compete for the same ranks.

## Conclusion

- **Structure-aware chunking wins at the top of the list.** Against
  fixed-length chunks of the same size (1,200) the baseline has a higher MRR
  (0.486 against 0.443) and page_hit@1 (0.37 against 0.32); 37 questions get a
  worse first page match and 16 a better one (sign test p = 0.0055). The
  bootstrap interval of the MRR difference just includes 0, so on 100
  questions the size of the effect is uncertain, but its direction is not:
  every fixed size loses MRR and page_hit@1.
- **Chunk size is not the explanation.** The fixed-length MRR is flat over
  600, 1,200 and 2,000 (0.435-0.446). Smaller or larger pieces put the right
  page in the top 10 about as often as the baseline (600: page_hit@10 0.79),
  but less often at rank 1.
- **The advantage comes from real headings.** On documents cut on articles
  the gap is largest (MRR 0.462 against 0.384-0.431); on documents where the
  detector latched onto running page headers it is gone.
- **Decision for the team:** keep `structure_aware` as the default
  (architecture decision 2 holds); `fixed_length` stays available behind the
  setting.
- **Next step:** stop the heading detector from accepting running page headers
  and footnote lines as a numbering run (for example: reject a run whose
  members are the first line of consecutive pages, or remove running headers
  in normalization, recorded like the page labels). About 1,800 of the 2,687
  labelled baseline chunks carry such a label, mostly in the largest
  document, so that is where the structured route can still gain.

### Limitations

- 100 questions, all written by one team member; the comparison and unanswerable
  groups are small (14 and 10).
- The baseline's two ingredients, the cut points and the clause label in the
  embedded text, are measured together; separating them means changing how
  the embedding text is composed (`embeddings/dense.py`), outside this
  experiment.
- Retrieval only; the answer layer was not measured with the fixed-length
  chunks.
