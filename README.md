# companydocs-rag

A local RAG pipeline over public Dutch business documents for staffing,
secondment and technical companies: uitzend-cao's (ABU, NBBU), sector cao's
(Metaal en Techniek, Metalektro), legislation (Wtta, Waadi,
Arbeidstijdenwet, ...), the Belastingdienst model agreements, privacy guidance
for recruitment and selection, and published government contracts. Everything
runs locally: OCR, embeddings and the language model (Ollama). No cloud APIs.

A BUas challenge by Roef, Rutger and Lucan. Each person's own experiment is in
[docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md) and `experiments/<name>/`.

## Quick start

1. Set up the machine (uv, Tesseract, Ollama): [docs/ONBOARDING.md](docs/ONBOARDING.md).
2. Fetch the documents: `uv run python scripts/download_corpus.py`. The script
   reads `corpus/sources.csv`, downloads every PDF to `data/raw/<domain>/`,
   checks its sha256 and writes `data/raw/00_INDEX.csv`. The PDFs themselves
   are not in git.
3. Build the pipeline: `companydocs-parse-pass data/raw -r` →
   `companydocs-normalize` → `companydocs-chunk` → `companydocs-index`.
4. Ask questions: `uv run companydocs-demo` and open `http://127.0.0.1:8765/`.
5. Measure: `uv run python scripts/evaluate_retrieval.py` on
   `evaluation/questions.csv` (see `evaluation/README.md`).

This README describes the parse pass, normalization, chunking,
`companydocs-metadata` and the demo; `companydocs-index`, `companydocs-search`
and `companydocs-answer` are described in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

The documents, the answer prompt and the demo interface are Dutch; the code,
the documentation and the commits are English.

## Parse pass (step 1)

Per document, one text extraction yields a status (`processed` / `failed`)
and, for every page with a text layer, a quality tier (`Clean` / `Decent` /
`Garbage`), summarised in one document label for reporting. The document type
comes from that text, supplemented with OCR text for pages without a text
layer. The chunker (`companydocs-chunk`, see Chunking) chooses its route per
document from the headings it recognises itself, not from the quality tier or
the structure score. Each page's tier travels with the provenance of the
chunks: the headings decide where to cut, the tier how far to trust the
result.

From Python:

```python
from companydocs_rag.ingestion import classify_and_assess

result = classify_and_assess("some_agreement.pdf")
result["status"]  # "processed" | "failed" (reason in result["status_reason"])
result["detected_type"]  # "lease_agreement" ... or "unknown/other"
result["type_confidence"]  # 0.0 - 1.0
result["quality_label"]  # "Clean"/"Decent"/"Garbage", None: failure or no text layer
result["awaiting_ocr"]  # None | "partial" | "all"
result["language"]  # document language for OCR ("nld", ...); None: failure
result["language_source"]  # "override" | "index" | "default"; None: failure
result["quality_signals"]  # evidence behind the label, including source/tier per page
result["ocr"]  # OCR engine and the document's OCR summary
result["page_text"]  # the text of every page (own text layer or OCR), with provenance
```

Over a folder, with a CSV report:

```bash
uv run companydocs-parse-pass <folder> -o report.csv
# also stores the full signals per document as JSON under data/interim/parse_pass_signals/;
# use --signals-dir <folder> to store them elsewhere
```

### Command line

`uv run companydocs-parse-pass <folder>` processes the PDFs in a folder.
Options:

- `-o/--output <file>`: the CSV report (default
  `data/interim/parse_pass_report.csv`).
- `-r/--recursive`: also PDFs in subfolders.
- `--signals-dir <folder>`: the signals per document (default
  `data/interim/parse_pass_signals/`).
- `--page-text-dir <folder>`: the page text store (default
  `data/interim/page_text/`).
- `--ocr-cache-dir <folder>`: the OCR cache (default `data/interim/ocr_cache/`).
- `--language-overrides <file>`: the language override file (default
  `data/manual/language_overrides.csv`).
- `-q/--quiet`: warnings and errors only.

The default locations are under the project folder (the folder with
`pyproject.toml`), in `data/`, which git ignores.

What a run writes:

- **CSV report**: one row per PDF, UTF-8 with BOM, with the columns
  `filename`, `status`, `detected_type`, `type_confidence`, `quality_label`,
  `awaiting_ocr`, `explanation`, `language`, `language_source`, `ocr_pages`,
  `no_text_pages`, `ocr_mean_confidence` and `no_text_reasons`.
- **Signals**: per PDF one JSON file with the full result except
  `page_text`, at the PDF's relative path in the input folder, with `.json`
  instead of `.pdf`.
- **Page text store**: per PDF one JSON file at that same relative path, with
  the text of every page, from the own text layer or from OCR, and the
  provenance per page: `source`, `tier` (only for text from the own text
  layer, otherwise `null`), `text_origin`, `no_text_reason`, `language`,
  `language_source` and `ocr_mean_confidence`. `schema_version` is 2.
- **OCR cache**: one JSON file per page and key (cache format version, file
  hash, page number, dpi, grayscale, language, Tesseract version), with
  temporary files in `tmp/`. A later run reads cached pages from the cache
  instead of running OCR on them again.

**Language.** Every document gets one language, for OCR. First the override
file counts (`;`-separated, UTF-8 with BOM, columns `Bestandsnaam` and `Taal`;
a missing file is not an error), then `00_INDEX.csv` in the PDF's nearest
ancestor folder (columns `Map`, `Bestandsnaam` and `Taal`), and otherwise
`nld`, with a warning. The codes are the two-letter codes from the index: NL,
EN, DE, FR, PL, SK and CS.

**Tesseract.** OCR runs locally with Tesseract, a system program on PATH
(setting `tesseract_cmd`), with the language data `nld`, `eng`, `deu`, `fra`,
`pol`, `slk` and `ces`. Pages without a text layer whose extraction did not
fail go to OCR; with `ocr_garbage_tier_pages` and `ocr_extraction_failed_pages`
also pages with tier `Garbage` and pages whose extraction failed. Without
Tesseract a run logs one ERROR and every page routed to OCR gets an OCR error
(`ocr_error`), even when the page is in the cache: the Tesseract version is
part of the cache key.

The thresholds and the keyword lists per document type are in
`config/settings.yaml`. They are first versions, derived from public sources
(ROZ model lease agreements, BW Boek 2, the layout of the KvK Handelsregister),
meant to be tuned through error analysis on the real document set; see the
documented assumptions at the top of `src/companydocs_rag/ingestion/quality.py`.
The parse pass reads the sections under `ingestion.parse_pass`: `quality`,
`type_detection` and `structure`, and `ocr` (rendering, routing, Tesseract) and
`language` (default language, index file, language codes).

## Normalization

`uv run companydocs-normalize` reads the parse pass's page text store and
writes per document one JSON file with the soft-normalized text of every page,
at the same relative path. Options:

- `--page-text-dir <folder>`: the page text store (default
  `data/interim/page_text/`).
- `--normalized-dir <folder>`: the output (default `data/interim/normalized/`).
- `-q/--quiet`: warnings and errors only.

Soft normalization, per page and in this order:

1. Zero-width characters and the soft hyphen are removed, non-breaking spaces
   (U+00A0, U+202F) become ordinary spaces, then NFC follows.
2. Whitespace at the end of a line is removed, and a run of spaces within a
   line becomes one space.
3. Page labels at the edge of a page are removed. Among the first and last
   three non-empty lines, a line goes when it is a page label as a whole
   (`- 2 -`, `Pagina 2 van 5`, `2/5`, `Seite 2 von 5`, `Strona 2 z 5`,
   `Strana 2 z 5`, `Strana 2`, `Page 2 of 5`), or a bare number that rises in
   step with the page number over at least three pages.
4. A word hyphenated at the end of a line is joined again, with or without its
   hyphen, only when the rules in
   `src/companydocs_rag/ingestion/normalization.py` allow it, and never across a
   page boundary.
5. A run of four or more identical filler characters (`.`, `_`, `-`, `…`)
   becomes three characters.

Nothing else changes: quotes, dashes, diacritics, capitals and digits stay,
and lines are not merged into paragraphs. Every removed line is in the output,
with its original text and its line number in the store's page text (from 0).
So is every join, with the line number, the rule used and the two word parts.
The counts and thresholds are in `config/settings.yaml` under
`ingestion.normalization`; the label patterns and the conjunction lists only
there.

Per document the output holds `schema_version`, `normalization_version`,
`unicode_version`, `file` (path and sha256), `status`, `page_count`,
`page_labels` (the settings used and the number runs found), `header_footer`
(not built yet), `normalized_sha256` and `pages`. Per page: the provenance
fields from the store (including `tier`), `input_sha256`, `text`,
`normalized_sha256`, `removed_lines`, `joins` and `counts`.

A store file with a `schema_version` other than 2 is refused; rerun
`companydocs-parse-pass` then. A file that cannot be read or is not a valid
store file fails. Both are logged, the other files are still written, and the
exit code is then 1. The command stops with exit code 2 before reading a file
when `--page-text-dir` is not an existing folder, or when
`ingestion.normalization` lacks the label patterns or the conjunction lists or
holds a label pattern that does not compile. A second run writes byte-identical
files.

`hard_normalize` in `normalization.py` (NFKC, lower case, straight quotes and
dashes, whitespace to one space; diacritics stay) is meant for keyword indexes
and queries. The command does not use it.

## Chunking

`uv run companydocs-chunk` reads the normalized files of
`companydocs-normalize` and writes per document one JSON file with the chunks,
at the same relative path. Options:

- `--normalized-dir <folder>`: the normalized files (default
  `data/interim/normalized/`).
- `--chunks-dir <folder>`: the output (default `data/interim/chunks/`).
- `-q/--quiet`: warnings and errors only.

The chunker puts the text of the pages one after another in page order, with
one line feed in between. It skips empty pages; their numbers are in
`empty_pages`. The text never changes: every chunk is a slice of that text that
starts and ends on a line boundary.

A heading starts with a keyword and a number (`Artikel 3`, `Art. 3`, `§ 3` and
the other keywords under `chunking` in `config/settings.yaml`), or with a
number such as `8.`, `8)` or `1.2` followed by a title. A line with only a
number is not a heading. Per kind, the chunker looks for runs of headings whose
number rises by one each time; a next run starts again at 1. The kind with the
longest run counts. A run is accepted when it has at least three headings and
its sections, from heading to heading, have a median length of at least 400
characters. A table of contents is ruled out that way.

- Route `structured`: there is an accepted run. Each of its sections becomes
  one chunk (`heading_section`). A section of more than 2000 characters is
  split; the parts that start less than 6000 characters after the heading keep
  the clause fields (`heading_section_part`, with `part` and `part_count`),
  the later ones do not (`heading_section_overflow`).
- Route `generic`: there is no accepted run.

Text outside the sections of accepted runs goes through a splitter that only
cuts between lines (`generic`): whole lines up to 1200 characters per piece, a
longer line as a piece of its own, and every next piece starts with the last
lines of the previous one, at most 150 characters together. A piece of only
whitespace does not become a chunk. A long section is split the same way. The
numbers are the settings under `chunking`.

Per document the output holds `schema_version`, `chunker_version`, `file`,
`status`, `normalized_hash`, `language`, `language_source`, `structure_hint`
(always `unknown`), `route`, `empty_pages`, `heading_evidence` (the chosen
kind, the number of candidates per kind and every run, accepted or with the
reason for rejection), `chunk_count` and `chunks`. Per chunk: `chunk_id`, the
text with its hash and position, the pages (`page_spans`: per page the position
in the page text, `text_origin`, `tier` and `ocr_mean_confidence`),
`chunk_method`, the clause fields (`clause_label`, `clause_number`,
`numbering_run`, `heading_family`) and `exceeds_max_chars`. A chunk has no
quality tier or confidence of its own.

A normalized file with a `schema_version` other than 1 is refused; rerun
`companydocs-normalize` then. A file that cannot be read or is not a valid
normalized file fails. Both are logged, the other files are still written, and
the exit code is then 1. A failed document (`status` `failed`) is written like
any other, without chunks. The command stops with exit code 2 before reading a
file when `--normalized-dir` is not an existing folder, or when the settings
under `chunking` are invalid. A second run writes byte-identical files.

## Metadata

`companydocs-metadata` has two subcommands. `extract` uses a local language
model to take the date and term fields from the chunked documents of the lease
family; `report` computes from them, for a reference date, the expiry date, the
next renewal and the latest notice date. It was built for lease contracts and
still has to be adapted to this corpus (its `lease_family_types` do not occur
in it).

### extract

`uv run companydocs-metadata extract` reads the chunk files of
`companydocs-chunk` and writes per chunk file one extraction file (JSON), at
the same relative path. Options:

- `--chunks-dir <folder>`: the chunk files (default `data/interim/chunks/`).
- `--extracted-dir <folder>`: the output (default
  `data/processed/metadata/extracted/`).
- `--type-overrides <file>`: the type override file (default
  `data/manual/document_type_overrides.csv`).
- `--relations <file>`: the relations file (default
  `data/manual/relations.csv`).

Every document gets a document type: first from the type override file, then
from `00_INDEX.csv` in the PDF's nearest ancestor folder (columns `Map`,
`Bestandsnaam` and `Documenttype`), and otherwise none, with a warning. Only a
document whose type is in `metadata.lease_family_types` (default
`Huurovereenkomst`, `Allonge` and `Lease- / exploitatieovereenkomst`) goes to
the language model. Every other document gets an extraction file with
`extraction_status` `not_in_scope`, without a request.

Per document, `extract` ranks the chunks by the number of keywords from
`metadata.keywords` they contain, plus one when they hold a date (with the
month names from `metadata.month_names`); a chunk without a keyword or a date
is dropped. In that order the chunks fill at most `max_batches_per_document`
requests of at most `selection_budget_chars` characters each; a chunk larger
than that budget is a request of its own. The language model returns the
fields `signing_date`, `commencement_date`, `initial_term`, `initial_term_end`,
`renewal` and `notice_period`, plus `other_termination_provisions`; every quote
it gives is checked by `extract` against its chunk. When every chunk is
dropped, the document is `extracted` without a request, with a warning.

`extract` needs Ollama, with the model from `metadata.model` (default
`companydocs-gemma4-e4b-8k`, created as under Ollama models) at
`metadata.base_url` (default `http://127.0.0.1:11434/v1`). With
`metadata.warmup` one short request that loads the model goes first. `extract`
does not read the index and so does not lock it, but it shares Ollama with
`companydocs-answer`: do not run the two at the same time.

Per run `extract` writes one JSONL log,
`extract-<YYYYMMDDTHHMMSSZ>.jsonl` (the start time in UTC), in
`metadata.log_dir` (default `data/processed/metadata_logs/`). A run without
extraction requests writes no log.

Exit codes:

- 0: done, no document `partial` or `error`. Also for an empty
  `--chunks-dir`, with a warning.
- 1: `--chunks-dir` is not a folder, or a chunk file was refused: a
  `schema_version` other than 1, unreadable or not a valid chunk file. Then
  nothing is written and no request goes out; rerun `companydocs-chunk`.
- 2: the command cannot start: invalid arguments, invalid settings under
  `metadata`, a missing or unreadable prompt file (`metadata.prompt_file`), an
  empty or missing API key variable (only with `metadata.api_key_env`), or an
  unusable type override or relations file. This is checked before
  `--chunks-dir` is read.
- 3: Ollama is unreachable or does not know the model (HTTP 404), or the
  warm-up request failed. The run stops; extraction files already written stay.
- 4: done, with at least one document `partial` or `error`.

### report

`uv run companydocs-metadata report --reference-date YYYY-MM-DD` reads the
extraction files and writes a follow-up overview in Dutch: `followup.md`,
`followup.json` and `followup.csv`. Options:

- `--reference-date <YYYY-MM-DD>`: the reference date (required).
- `--extracted-dir <folder>`: the extraction files (default
  `data/processed/metadata/extracted/`).
- `--report-dir <folder>`: the output (default `data/processed/metadata/`).

Per document, except those with `not_in_scope`, it lists the fields found with
their pages, and for the reference date the expiry date, the next renewal and
the latest notice date, or why a date cannot be computed or does not apply. A
document of a type in `metadata.amendment_types` (default `Allonge`) gets no
derived dates of its own. `followup.csv` is `;`-separated, UTF-8 with BOM.
`report` reads no clock: the reference date is its only date, and two runs over
the same input with the same reference date write byte-identical files.
`report` needs neither Ollama nor the index.

Exit codes: 0 done; 1 when `--extracted-dir` is not a folder, holds no
extraction files or an extraction file was refused (then nothing is written);
2 for invalid arguments or invalid settings under `metadata`.

**Known limitation.** On Windows `report` can fail when one of the three
follow-up files is open in another program: it then stops with a traceback,
and the files it wrote before that file (in the order `followup.md`,
`followup.json`, `followup.csv`) have already been replaced. Run `report` only
when no follow-up file is open elsewhere.

### Files in `data/manual/`

`extract` reads two files that are kept by hand. Both are `;`-separated, UTF-8
with or without BOM, with a header row; values are trimmed. A missing file is
not an error. A file that cannot be read or lacks a column makes `extract` stop
with exit code 2. A row with an empty `Bestandsnaam` is skipped, and for a
repeated `Bestandsnaam` the first row counts; both with a warning.
`Bestandsnaam` is the PDF's file name, without folder; a row applies to every
chunk file with that name.

- `document_type_overrides.csv`: columns `Bestandsnaam` and `Documenttype`.
  The type goes before the one from `00_INDEX.csv`. An empty `Documenttype`
  does not count, with a warning; then the index decides.
- `relations.csv`: columns `Bestandsnaam`, `Relatie`, `Doelbestand` and
  `Detail`. `Relatie` is `amends` or `duplicate_of`; another value makes
  `extract` stop with exit code 2. `Doelbestand` is the file name of the target
  document. When exactly one chunk file of the run has that name and a sha256,
  the relation is recorded with that sha256; otherwise it gets
  `target_not_found`, with a warning. `Detail` is a note that goes with the
  relation into the extraction file.

## Demo

`uv run companydocs-demo` starts a local browser interface over the answer
layer of `companydocs-answer`: ask a question, read the answer with a card per
cited source, and open every cited page of the PDF as an image. Options:

- `--port <port>`: the port on 127.0.0.1 (default `interface.port`, 8765).
- `--index-dir <folder>`: the index of `companydocs-index` (default
  `data/processed/index/`).

The server listens on 127.0.0.1 only, handles one request at a time and opens
no browser; its only stdout line is the URL, by default
`http://127.0.0.1:8765/`. Every question is answered on its own, without the
earlier questions. For an answer with sources, every cited source gets a card
with the document type (from `data/manual/document_type_overrides.csv` and
`00_INDEX.csv`, as for `extract`, otherwise "type onbekend"), the file name,
the clause label as a hint and per page the provenance with a link to that
page. The page is looked up by file hash and page number under `data/raw/`,
checked again on every request and shown as an image without writing anything
to disk. The file name appears only on the card in the browser, never in the
prompt to the language model. Every answer goes, as for `companydocs-answer`,
to the answer log in `data/processed/answer_logs/`.

The demo needs Ollama with the model from `answering.model` (see Ollama
models) and sends one short request at startup that loads the model. It holds
the index while it runs: `companydocs-index`, `companydocs-search` and
`companydocs-answer` cannot reach the index then. Stop it with Ctrl+C.

Exit codes at startup: 1 when the index is unusable or held by another
process; 2 when the command cannot start (invalid arguments, no index folder,
invalid settings under `embeddings`, `retrieval`, `answering` or `interface`,
an unusable prompt file or API key variable, no `data/raw/` folder, an unusable
type override file, or a port in use); 3 when Ollama is unreachable or the
warm-up request fails. After Ctrl+C: 0. A language model error while running
gives a message for that question; the server keeps running.

## Ollama models

`companydocs-answer`, `companydocs-demo` and `companydocs-metadata extract` ask
a local language model in Ollama for their answers. There are two models for
that, each from a Modelfile in `config/ollama/`: it takes a base model
(`gemma4:e4b` or `qwen3.5:4b`) and sets the context window to 8192 tokens, the
window that `answering.context_window` and `metadata.context_window` expect.
A team member pulls the base model (`ollama pull gemma4:e4b`) and creates the
model with `ollama create`, from the project folder. The Qwen model (base model
`qwen3.5:4b`) is only needed as a fallback:

```powershell
ollama create companydocs-gemma4-e4b-8k -f config/ollama/companydocs-gemma4-e4b-8k.Modelfile
ollama create companydocs-qwen3.5-4b-8k -f config/ollama/companydocs-qwen3.5-4b-8k.Modelfile
```

A team member always pulls and creates the models by hand; no script does
that. By default `answering.model` and `metadata.model` use
`companydocs-gemma4-e4b-8k`. Falling back to Qwen means setting
`answering.model` to `companydocs-qwen3.5-4b-8k` in a copy of
`config/settings.yaml` and pointing `COMPANYDOCS_RAG_SETTINGS` at that copy. Do
not change the value in `config/settings.yaml` itself: a test
(`tests/answering/test_settings.py`) requires the default value there.

## Tests

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-tests.ps1 [pytest arguments]
```

In its own process the script removes the PATH entries containing
`Git\usr\bin`, checks that `uv` and `tesseract` are on PATH, runs
`uv run pytest` in the project folder and exits with pytest's exit code. Its
own exit codes: 90 when PATH is empty afterwards, 91 without `uv`, 92 without
`tesseract`, and 5, with its own message, when pytest collected no tests.
