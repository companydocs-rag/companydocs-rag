# Onboarding

From a fresh Windows laptop to a working pipeline and your own branch. Every step says how you know it worked. **[person]** marks a step each team member does by hand: accounts and pulling or creating models.

Commands are for Windows PowerShell, run from the repository root unless a step says otherwise.

## 1. Tools

| tool | version | check |
|---|---|---|
| Git for Windows | 2.5x | `git --version` |
| uv | 0.12 or later | `uv --version` (install: `winget install astral-sh.uv`) |
| Python | 3.12, managed by uv | after step 2: `.venv\Scripts\python.exe --version` |
| Tesseract | 5.5.3 (`v5.5.3.20260724`) with ces deu eng fra nld pol slk | `tesseract --version`; `tesseract --list-langs` |
| Ollama | current | `ollama --version` (install: `winget install Ollama.Ollama`) |

- **Tesseract**: winget only has 5.4. Install 5.5.3 from the official release, <https://github.com/tesseract-ocr/tesseract/releases/tag/5.5.3> (`tesseract-ocr-w64-setup-5.5.3.20260724.exe`, needs admin), then copy the "fast" language files `ces deu eng fra nld pol slk osd` from <https://github.com/tesseract-ocr/tessdata_fast> into `C:\Program Files\Tesseract-OCR\tessdata`, and add `C:\Program Files\Tesseract-OCR` to the system PATH. Open a new terminal. Check: `--list-langs` shows the eight.
- **Anaconda users**: Anaconda sets `VIRTUAL_ENV` globally; uv warns that it ignores it. That warning is harmless.

## 2. Clone and install

1. **[person]** Accept the invitation to the GitHub organisation `companydocs-rag`.
2. Clone, set your identity for this repo, install:

   ```powershell
   git clone https://github.com/companydocs-rag/companydocs-rag.git
   cd companydocs-rag
   git config user.name "<your name>"
   git config user.email "<the email of your GitHub account>"
   uv sync --python 3.12
   ```

   `.venv` takes about 1 GB. **If the clone is inside a OneDrive folder**, keep `.venv` out of OneDrive first, before `uv sync`, so OneDrive does not upload tens of thousands of files:

   ```powershell
   New-Item -ItemType Directory -Force "$env:USERPROFILE\.venvs\companydocs-rag" | Out-Null
   New-Item -ItemType Junction -Path .venv -Target "$env:USERPROFILE\.venvs\companydocs-rag"
   ```

   Check: `uv run companydocs-search --help` prints the usage.

## 3. Tests

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-tests.ps1
.venv\Scripts\ruff check .
```

Check: `2485 passed, 1 xfailed` (or more, as tests are added), exit 0; ruff prints `All checks passed!`. Always run pytest through this script (docs/DEVELOPMENT.md, known issue with `Git\usr\bin`). About 5 minutes.

## 4. The corpus

```powershell
uv run python scripts/download_corpus.py
```

The 26 public PDFs come with the clone (`data/raw/<domain>/<id>.pdf`, plus `data/raw/00_INDEX.csv`). The script checks every one against its sha256 in `corpus/sources.csv` and downloads any that is missing. Check: exit 0 and `26 of 26 sources in place`. For the demo, `--with-demo-only` also fetches the two SNA documents whose copyright forbids redistribution, into `data/raw/demo-only/`; git ignores that folder, so they never get committed. Build the index for the evaluation without them. A `sha256 differs` line means the publisher replaced the file: tell the team; do not edit the hash on your own, since everyone must index the same files.

## 5. Models [person]

```powershell
ollama pull gemma4:e4b
ollama create companydocs-gemma4-e4b-8k -f config/ollama/companydocs-gemma4-e4b-8k.Modelfile
```

Check: `ollama list` shows `companydocs-gemma4-e4b-8k`. The embedding model (Qwen/Qwen3-Embedding-0.6B, about 1.2 GB) is downloaded by the first `companydocs-index` run (step 6); after that, set `$env:HF_HUB_OFFLINE = "1"` in terminals where you run the index.

## 6. Build the pipeline

```powershell
uv run companydocs-parse-pass data/raw -r
uv run companydocs-normalize
uv run companydocs-chunk
uv run companydocs-index
```

The parse pass runs OCR on pages without a text layer (cached in `data/interim/ocr_cache/`, so a rerun is fast). The index build is the slow step on a CPU. Check: each exits 0; `data/processed/index/manifest.json` exists. Run the parse pass from PowerShell, not from Git Bash (`Git\usr\bin`, docs/DEVELOPMENT.md).

## 7. Search, answer, demo

Only one process at a time may hold the index.

```powershell
uv run companydocs-search "Hoe lang duurt fase A in de ABU-cao?"
uv run companydocs-demo
```

The demo prints `http://127.0.0.1:8765/` when ready (the first start loads the LLM and can take a few minutes on a laptop without an NVIDIA GPU; an answer can take one to a few minutes). Ctrl+C stops it.

## 8. Evaluate

```powershell
uv run python scripts/evaluate_retrieval.py --label baseline
```

See `evaluation/README.md` for the question set and the metrics, and `experiments/README.md` for running an alternative against its own index.

## 9. Your work

1. `git switch -c <yourname>/<topic>` (for example `roef/chunking`).
2. Your experiment goes into `experiments/<yourname>/` and, where code changes, into `src/` behind a setting or flag, with tests. Summarise it in `docs/CONTRIBUTIONS.md`.
3. Run the tests and ruff (step 3).
4. Push your branch and open a pull request into `main`; never push to `main`. The PR is the evidence of your contribution, so keep it to your own work.
