"""Download the public corpus listed in corpus/sources.csv into data/raw/.

Usage (from the repository root):

    uv run python scripts/download_corpus.py [--with-demo-only] [--only ID ...]

For every source with ``include`` = ``yes`` (and ``demo-only`` with
``--with-demo-only``) it downloads the URL, checks that the file is a PDF and
that its sha256 equals the ``sha256`` column, and stores it as
``data/raw/<domain>/<id>.pdf``. A file already in place with the right hash is
not downloaded again. A source whose download fails or whose hash differs is
reported and not stored; the others go on (never a silent skip).

Afterwards it writes ``data/raw/00_INDEX.csv`` (``Map;Bestandsnaam;Taal;
Documenttype``) for every PDF in place, which the parse pass reads for the
language and the demo and ``companydocs-metadata`` for the document type.

Exit codes: 0 every selected source in place; 1 one or more failed (the rest
is in place and indexed); 2 cannot start (no sources file, bad columns).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Dict, List

import httpx

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "corpus" / "sources.csv"
RAW = ROOT / "data" / "raw"
INDEX_NAME = "00_INDEX.csv"

DELIMITER = ";"
ENCODING = "utf-8-sig"
COLUMNS = (
    "id",
    "domain",
    "title",
    "publisher",
    "version",
    "url",
    "sha256",
    "pages",
    "text_layer",
    "reuse",
    "include",
)
INCLUDE_VALUES = ("yes", "demo-only", "no")

# The document type the index records per domain; the demo shows it on the
# source cards.
DOMAIN_TYPES: Dict[str, str] = {
    "uitzend-cao": "Uitzend-cao",
    "sector-cao": "Sector-cao",
    "wetgeving": "Wetgeving",
    "modelovereenkomst": "Modelovereenkomst",
    "privacy": "Privacy",
    "certificering": "Certificering",
    "contract-praktijk": "Contract (praktijk)",
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) companydocs-rag corpus download"
TIMEOUT_SECONDS = 120.0


def read_sources(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding=ENCODING, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=DELIMITER)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"{path}: duplicate id {row['id']}")
        seen.add(row["id"])
        if row["domain"] not in DOMAIN_TYPES:
            raise ValueError(f"{path}: {row['id']}: unknown domain {row['domain']}")
        if row["include"] not in INCLUDE_VALUES:
            raise ValueError(f"{path}: {row['id']}: include must be {INCLUDE_VALUES}")
    return rows


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def target_of(row: Dict[str, str]) -> Path:
    return RAW / row["domain"] / f"{row['id']}.pdf"


def fetch(client: httpx.Client, row: Dict[str, str]) -> str:
    """Download one source; return '' when in place, else the failure reason."""
    target = target_of(row)
    expected = row["sha256"].upper()
    if target.exists() and sha256_of(target.read_bytes()) == expected:
        print(f"ok (in place)  {row['id']}")
        return ""
    try:
        response = client.get(row["url"])
        response.raise_for_status()
    except httpx.HTTPError as error:
        return f"download failed: {type(error).__name__}: {error}"
    data = response.content
    if not data.startswith(b"%PDF"):
        return "not a PDF (the URL may now serve an HTML page)"
    actual = sha256_of(data)
    if expected and actual != expected:
        return (
            f"sha256 differs: expected {expected[:12]}..., got {actual[:12]}... "
            "(a new version was published? check it, then update corpus/sources.csv)"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f"ok (downloaded {len(data) / 1e6:.1f} MB)  {row['id']}")
    return ""


def write_index(rows: List[Dict[str, str]]) -> int:
    lines = []
    for row in rows:
        target = target_of(row)
        if target.exists():
            lines.append(
                [row["domain"], target.name, "NL", DOMAIN_TYPES[row["domain"]]]
            )
    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / INDEX_NAME).open("w", encoding=ENCODING, newline="") as handle:
        writer = csv.writer(handle, delimiter=DELIMITER, lineterminator="\n")
        writer.writerow(["Map", "Bestandsnaam", "Taal", "Documenttype"])
        writer.writerows(lines)
    return len(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--with-demo-only",
        action="store_true",
        help="also download sources marked demo-only (not for the submission)",
    )
    parser.add_argument("--only", nargs="+", metavar="ID", help="only these ids")
    args = parser.parse_args(argv)

    try:
        rows = read_sources(SOURCES)
    except (OSError, ValueError) as error:
        print(f"Cannot start: {error}", file=sys.stderr)
        return 2

    wanted = {"yes", "demo-only"} if args.with_demo_only else {"yes"}
    selected = [r for r in rows if r["include"] in wanted]
    if args.only:
        unknown = sorted(set(args.only) - {r["id"] for r in rows})
        if unknown:
            print(f"Cannot start: unknown ids {unknown}", file=sys.stderr)
            return 2
        selected = [r for r in selected if r["id"] in set(args.only)]

    failures = {}
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(
        headers=headers, follow_redirects=True, timeout=TIMEOUT_SECONDS
    ) as client:
        for row in selected:
            reason = fetch(client, row)
            if reason:
                failures[row["id"]] = reason
                print(f"FAILED         {row['id']}: {reason}", file=sys.stderr)

    indexed = write_index(selected)
    print(
        f"\n{len(selected) - len(failures)} of {len(selected)} sources in place; "
        f"{indexed} in {RAW / INDEX_NAME}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
