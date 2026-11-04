# Evaluation

## The question set: `questions.csv`

`;`-separated, UTF-8. One row per question:

| column | content |
|---|---|
| `question_id` | unique, with the author's first two letters: `RO01` (Roef), `RU01` (Rutger), `LU01` (Lucan) |
| `question` | the question as an employee of a staffing or technical company would ask it (in Dutch, like the documents) |
| `expected` | where the answer is: `source-id:pages`, several separated by `\|`. For example `cao-abu-2026:14-15` or `wet-wtta-stb-2025-385:3\|wet-wtta-mvt-36446-3:22`. Pages are the PDF's page numbers (1 = first page), not the number printed on the page. Without pages (`cao-abu-2026`) only the document counts. |
| `answer` | the correct answer in a few words, so answers can be scored by hand later |
| `author` | `Roef`, `Rutger` or `Lucan` |
| `type` | `fact` (number, term, date), `rule` (what is allowed or required), `comparison` (two documents), `unanswerable` (not in the corpus; the system should abstain) |

Agreements:

- Everyone writes **at least 15 questions**, spread over the domains. Look up
  the answer in the PDF first and note the page; write the question in your own
  words, not with the literal text of the article (otherwise you only measure
  keyword overlap).
- About 1 in 10 questions is `unanswerable`. Those get the closest source as
  `expected`; the retrieval score counts them, but they are mainly for the
  answer layer.
- Freeze the set **before** anyone tunes an experiment, and do not change it
  quietly afterwards: a new question is a new row in a PR, and all results are
  rerun after it.

## Measuring: `scripts/evaluate_retrieval.py`

```powershell
uv run python scripts/evaluate_retrieval.py --label baseline
```

For every question the script searches like `companydocs-search` and finds the
rank of the first candidate with the right source (and page). It reports
`doc_hit@1/3/5/10`, `page_hit@1/3/5/10` and the MRR, and writes everything to
`results/<label>.json`, with the commit and the settings file.

`results/baseline.json` is the comparison point for all three experiments.
Rerun it when the question set or the corpus changes.
