"""Submit one cell to the running weight-space programming kernel and print
its output.

    python -m bench.wsp.run <cell.py> [--timeout 1800]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time

from .kernel import CELLS, SESSION


def main(argv) -> int:
    parser = argparse.ArgumentParser(prog="bench.wsp.run")
    parser.add_argument("cell", nargs="?")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--ready", action="store_true", help="wait until the kernel is ready")
    args = parser.parse_args(argv)
    if args.ready:
        deadline = time.time() + args.timeout
        since = time.time() - 5  # a READY left by an earlier kernel does not count
        while not ((SESSION / "READY").exists() and (SESSION / "READY").stat().st_mtime > since):
            if time.time() > deadline:
                print("timeout: kernel not ready")
                return 1
            time.sleep(1)
        print((SESSION / "READY").read_text())
        return 0
    CELLS.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.stem) for p in CELLS.glob("*.py") if p.stem.isdigit()]
    target = CELLS / f"{(max(numbers) + 1 if numbers else 1):04d}.py"
    tmp = target.with_suffix(".tmp")
    shutil.copyfile(args.cell, tmp)
    tmp.rename(target)
    out = target.with_suffix(".out")
    deadline = time.time() + args.timeout
    while not out.exists():
        if time.time() > deadline:
            print(f"timeout: {target.name} still running")
            return 1
        time.sleep(0.5)
    time.sleep(0.1)
    sys.stdout.write(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
