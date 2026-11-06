"""Download the d3-lite base model for the judge route (Rutger's experiment).

    uv run python experiments/rutger/download_d3_lite.py

Downloads vllm-sr/d3-lite (Qwen3.5-0.8B decision model, Apache 2.0) at the
revision the experiment used into data/models/d3-lite/ (git-ignored), then
checks every file against the sha256 values in the model's own
MODEL_MANIFEST.json. The judge files (fine-tuned LoRA, readout rows and weight
edits) are in experiments/rutger/models/ (Git LFS); only this base model is
downloaded. Exit 0 when every file is in place and matches, 1 otherwise.
"""

import hashlib
import json
import sys
from pathlib import Path

REPO_ID = "vllm-sr/d3-lite"
REVISION = "6c01fba14e74f63b70e1ef442b0b6d1b934339d5"
ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "data" / "models" / "d3-lite"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    from huggingface_hub import snapshot_download

    TARGET.mkdir(parents=True, exist_ok=True)
    where = TARGET.relative_to(ROOT)
    print(f"downloading {REPO_ID}@{REVISION[:12]} to {where} (about 1.7 GB)")
    snapshot_download(repo_id=REPO_ID, revision=REVISION, local_dir=str(TARGET))
    manifest = json.loads((TARGET / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    expected = manifest["identity"]["files_sha256"]
    bad = [
        name
        for name, value in expected.items()
        if not (TARGET / name).exists() or sha256(TARGET / name) != value
    ]
    for name in bad:
        print(f"sha256 differs or missing: {name}")
    print(
        f"{len(expected) - len(bad)} of {len(expected)} files match the model manifest"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
