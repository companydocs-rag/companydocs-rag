# Experiments

One folder per person: `roef/`, `rutger/`, `lucan/`. Each holds:

- `README.md`: the question the experiment answers, what was changed (and
  behind which setting or flag), how it was run (commands), the result table
  (baseline vs. alternative, numbers from `evaluation/results/`) and the
  conclusion;
- `settings.yaml`: the full settings file the alternative ran with (a copy of
  `config/settings.yaml` with the changed values), used through
  `COMPANYDOCS_RAG_SETTINGS`;
- any extra result files (figures, user-test sheets).

Run an alternative without touching the shared index:

```powershell
$env:COMPANYDOCS_RAG_SETTINGS = "experiments/<name>/settings.yaml"
uv run companydocs-chunk --chunks-dir data/interim/chunks-<name>      # only if chunking changes
uv run companydocs-index --chunks-dir data/interim/chunks-<name> --index-dir data/processed/index-<name>
uv run python scripts/evaluate_retrieval.py --label <name>-<variant> --index-dir data/processed/index-<name>
Remove-Item Env:COMPANYDOCS_RAG_SETTINGS
```

Without `--chunks-dir` the index reads the shared `data/interim/chunks/`. The tests pin the committed
`config/settings.yaml`, so an experiment never edits that file to try a value.
