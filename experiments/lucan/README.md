# Lucan: a workspace interface with highlighted sources

## Question

Can the interface point the user to the exact passage an answer is based on,
reliably enough to replace reading the whole page?

A RAG answer is only useful to a company when an employee can verify it
against the source. The baseline interface shows a card per cited source with
a link to the page image; the user then has to find the passage on the page
themselves.

## What changed

All interface changes are in `src/companydocs_rag/interface/` (with tests in
`tests/interface/`); retrieval, prompt and answer layer are unchanged, so both
interfaces show the same answers and sources.

| | baseline (`main` before this PR) | workspace (this PR) |
|---|---|---|
| Layout | one page: question, answer, source cards | side bar, chat view, documents view, page viewer next to the chat |
| Source on the page | page image only | the lines of the cited passage highlighted on the page image (`highlight.py`, `GET /markering`) |
| Documents | not visible | a library of all documents with type and page count (`library.py`, `GET /documenten`) |
| Status | none | model, whether it runs locally, number of documents (`GET /status`) |
| History | each question stands alone | chat history in the browser (local storage only) |

How the highlight works: the words and their boxes come from what the parse
pass used (Tesseract's word boxes from the OCR cache for scanned pages,
pdfplumber's words for text-layer pages). The chunk's text on that page is
aligned with the page's words (`difflib.SequenceMatcher`); a highlight is only
drawn when at least 60% of the passage's words are found (`MIN_COVERAGE`) and
at least half of the words in the highlighted run match (`MIN_DENSITY`), and a
passage of fewer than 3 words gets none: no highlight is better than a wrong
one. The thresholds are named constants at the top of `highlight.py`.

## How it was measured

`measure_highlight.py` (results in `highlight_results.json` and
`highlight_results.md`), on the 90 answerable questions of the shared question
set (`evaluation/questions.csv`; the 10 `unanswerable` ones have no correct
passage):

1. **Answer passages:** every chunk with text on a page the question set names
   as the answer, 466 (chunk, page) pairs over all 26 documents. Per pair: is a
   highlight drawn, what share of the passage's words the alignment found, and
   what share of the page the highlight covers.
2. **Retrieved passages:** the top 5 candidates of the same search the answer
   layer runs, on a preview index of 22 of the 26 documents (the full index
   takes many hours on a laptop without a GPU), for the 62 questions whose
   expected sources are all in it.

`--spotcheck N` also renders N highlighted pages with the highlight drawn on
them, for a visual check; that check was not done for this result.

```powershell
uv run python experiments/lucan/measure_highlight.py --index-dir data/processed/index-preview
```

## Result

| measure | baseline | workspace |
|---|---|---|
| answer passages with a highlight (466 pairs) | 0% | **99.8%** (465; the one without is a 2-word passage, below the minimum) |
| questions with at least one highlighted answer passage (90) | 0% | **100%** |
| median share of the passage's words found | - | **100%** (lowest 95.5%) |
| share of the page the user is pointed to (median, drawn) | 100% (whole page) | **7.7%** (middle half 3.8% to 9.2%, at most 23.6%) |
| retrieved top-5 pages with a highlight (450 pages) | 0% | **100%** |
| retrieved pages on an expected page with a highlight (68) | 0% | **100%** |

Per domain the highlight is drawn on 99-100% of the answer passages
(`highlight_results.md`). That the first retrieved candidate gets a highlight
on an expected page for 37% of the questions is a property of the retrieval,
not of the highlight: it matches the baseline's `page_hit@1` of 0.37.

## Conclusion

Where the baseline leaves the user to search the whole page, the workspace
interface marks the cited passage on every answer page in the question set and
narrows what the user has to read to about 8% of the page (a factor of about
13). The alignment found at least 95% of the passage's words in every case.

Limits:

- **Same source on both sides.** On text-layer pages the passage and the page
  words both come from pdfplumber, so a high coverage is expected; it shows
  that the alignment holds up through normalization and chunking, not that
  the boxes sit on the right lines. A visual check of rendered pages
  (`--spotcheck`) would settle that and was not done; one page looked at while
  building the script showed the right clauses highlighted, with the page footer
  included.
- **No scanned answer pages.** All 466 answer pages have a text layer; the OCR
  path (Tesseract word boxes) is covered by the tests but not by this
  measurement.
- **Whether users check answers faster** with the highlight was not measured:
  a within-subjects user test (time to verdict, verdict correctness, SUS) is
  the natural next step.

## Demo with a hosted model (optional)

For looking at the interface without waiting on the local model, the answer
layer can run on a hosted model through an OpenAI-compatible endpoint:
`settings-fable.yaml` points it at Claude Fable 5.1, with the API key in the
user environment variable `COMPANYDOCS_LLM_API_KEY` (never in a file). To
support it, `answering.temperature: null` now leaves `temperature` out of the
request (newer hosted models reject it); the default stays `0.0`. This departs
from the local-only rule in `docs/DEVELOPMENT.md` and is not used by the
measurement above, which needs no language model.

```powershell
$env:COMPANYDOCS_RAG_SETTINGS = "experiments/lucan/settings-fable.yaml"
uv run companydocs-demo --index-dir data/processed/index-preview
```
