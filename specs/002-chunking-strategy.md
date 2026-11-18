# 002: the chunking strategy (structure-aware or fixed length)

Short change note (docs/DEVELOPMENT.md, Working together). Author: Roef.

## Goal

Measure whether architecture decision 2, structure-aware chunking, finds the
right page more often than structure-blind fixed-length chunks, without
changing the baseline: the alternative sits behind a setting.

## What changed

- `chunking.strategy` (default `structure_aware`), the first key of the
  `chunking` section. `fixed_length` sends every document down the generic
  route: whole lines packed into pieces of at most `chunk_target_chars` with
  `chunk_overlap_chars` of overlap, without clause fields. The heading
  candidates and runs are still found and assessed and recorded in
  `heading_evidence`; they decide nothing. The route is `generic`.
- `src/companydocs_rag/chunking/chunker.py`: the setting, its validation
  (`ChunkingConfigError` for any other value) and `chunk_document` using no
  run under `fixed_length`. `chunking/cli.py`: one INFO line naming a
  non-default strategy at the start of a run.
- The chunk file format, `CHUNKS_SCHEMA_VERSION` and `CHUNKER_VERSION` are
  unchanged: the strategy is a setting, like the sizes, and the default output
  is byte-identical.
- `docs/DEVELOPMENT.md`: one sentence in the chunking bullet.
- `experiments/roef/`: the settings of the three runs (fixed 600, 1,200,
  2,000), the comparison script and its output, and the write-up;
  `evaluation/results/roef-*.json`: the result files.

## How it was tested

- `tests/chunking/test_chunker.py`: the settings pin and defaults; invalid
  values; `fixed_length` cuts a structured two-page document exactly as when
  its run is rejected, keeps the heading evidence, leaves documents without an
  accepted run unchanged and takes its sizes from the settings.
  `tests/chunking/test_cli.py`: the INFO line and summary under
  `fixed_length`, exit 2 for an invalid strategy. Full suite: 2,515 passed,
  1 xfailed; ruff clean.
- With the default settings the chunk files of all 26 documents are
  byte-identical to those of `main`'s chunker.
- Measured with `scripts/evaluate_retrieval.py` on the 100 questions:
  baseline MRR 0.486, fixed 600 / 1,200 / 2,000: 0.435 / 0.443 / 0.446;
  page_hit@1 0.370 against 0.270 / 0.320 / 0.280
  (`experiments/roef/README.md`).

## What was left

- The default stays `structure_aware`.
- The heading detector accepts running page headers ("57 Staatscourant 2026
  nr. ...") as a decimal numbering run, and occasionally a footnote or a
  cross-reference at a line start; about 1,800 of the 2,687 labelled chunks
  carry such a label. Filtering them is the next change, as its own spec.
- The cut points and the clause label in the embedded text are measured
  together; the answer layer was not measured with fixed-length chunks.
