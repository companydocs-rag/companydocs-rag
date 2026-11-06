"""Check 1: does the d3 validation curve still rise enough to train on more data?

    python -m bench.decide_extend   -> prints EXTEND or STOP with the numbers, exit 0

Reads runs/logs/train-d3.log (UTF-16 from PowerShell) and compares the
validation AUC at the interim check (step 640) with the AUC after training
(step 1250). Rule, fixed before seeing the result: EXTEND when the second
half of the run still added at least +0.005 AUC (the curve has not
flattened); otherwise STOP. The 400 validation examples come from train
questions never used for training.
"""

from __future__ import annotations

import re
import sys

from .common import RUNS

MIN_GAIN = 0.005


def main() -> int:
    raw = (RUNS / "logs" / "train-d3.log").read_bytes()
    text = raw.decode("utf-16", errors="ignore") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "ignore")
    found = dict(re.findall(r"(before training|step \d+|after training): \{'val_acc': [\d.]+, 'val_auc': ([\d.]+)\}", text))
    before, mid, after = (float(found[k]) if k in found else None
                          for k in ("before training", "step 640", "after training"))
    if mid is None or after is None:
        print(f"STOP (incomplete log: before={before} mid={mid} after={after})")
        return 0
    gain = after - mid
    verdict = "EXTEND" if gain >= MIN_GAIN else "STOP"
    print(f"{verdict}: val AUC before {before}, step 640 {mid}, step 1250 {after}; "
          f"second-half gain {gain:+.4f} (threshold {MIN_GAIN})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
