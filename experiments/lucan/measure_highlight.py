"""Measure the source highlight of the workspace interface (Lucan's experiment).

Usage (from the repository root, nothing else holding the index):

    uv run python experiments/lucan/measure_highlight.py
    uv run python experiments/lucan/measure_highlight.py \
        --index-dir data/processed/index-preview

Three measurements, written to ``experiments/lucan/highlight_results.json`` and
summarised in ``experiments/lucan/highlight_results.md``:

1. **Answer passages.** For every answerable question in
   ``evaluation/questions.csv``, the chunks that hold text on an expected page
   are the passages a user should be shown. Per (chunk, page) pair: is a
   highlight drawn (by text origin and by domain), what share of the passage's
   words the page alignment found (``coverage``), and what share of the page the
   highlight covers (``area``; the baseline interface shows the whole page, 1.0).
2. **Retrieved passages.** With an index, the top 5 candidates of the same
   search the answer layer uses: how often is a highlight drawn, how often on an
   expected page, and does the first candidate get a highlight on an expected
   page.
3. **Spot check** (optional, ``--spotcheck N``). A seeded sample of highlighted
   pages rendered as images with the highlight drawn on them, plus
   ``spotcheck.csv`` for a person's verdict (correct / partly / wrong): the
   alignment cannot judge itself.

The highlight code is used unchanged (``companydocs_rag.interface.highlight``);
this script only calls it and counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "lucan"
QUESTIONS = ROOT / "evaluation" / "questions.csv"
RAW = ROOT / "data" / "raw"
CHUNKS = ROOT / "data" / "interim" / "chunks"
OCR_CACHE = ROOT / "data" / "interim" / "ocr_cache"
TOP_K = 5
SPOTCHECK_DPI = 100
SEED = 7


def parse_expected(text: str) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
    """``a:3-4|b`` -> [("a", (3, 4)), ("b", None)], as the evaluation script."""
    found = []
    for part in (text or "").split("|"):
        part = part.strip()
        if not part:
            continue
        source, _, pages = part.partition(":")
        span = None
        if pages:
            first, _, last = pages.partition("-")
            span = (int(first), int(last or first))
        found.append((source.strip(), span))
    return found


def read_questions() -> List[Dict[str, Any]]:
    with QUESTIONS.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    return [
        {**row, "expected_parsed": parse_expected(row["expected"])}
        for row in rows
        if (row.get("question") or "").strip()
    ]


DOMAINS: Dict[str, str] = {}


def read_chunk_files() -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    """Per source id its chunks, and per source id its file hash; the domain
    (the chunk file's folder, as in data/raw/) goes into DOMAINS."""
    chunks: Dict[str, List[Dict[str, Any]]] = {}
    hashes: Dict[str, str] = {}
    for path in sorted(CHUNKS.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        source = Path(data["file"]["path"]).stem
        chunks[source] = data.get("chunks") or []
        hashes[source] = data["file"]["sha256"]
        DOMAINS[source] = path.parent.name
    return chunks, hashes


def measure(highlighter, highlight, digest: str, page: int, chunk_id: str):
    """One (chunk, page) pair: drawn, coverage, area, origin and the boxes."""
    _, passages = highlighter.chunks[chunk_id]
    passage = passages[page]
    words = highlighter._words(digest, page, passage.text_origin)
    page_keys = [word.key for word in words]
    passage_keys = highlight._keys(passage.text)
    coverage = 0.0
    if page_keys and passage_keys:
        smallest = (
            highlight.MIN_BLOCK if len(passage_keys) >= highlight.SHORT_PASSAGE else 1
        )
        blocks = SequenceMatcher(
            None, page_keys, passage_keys, autojunk=False
        ).get_matching_blocks()
        matched = sum(b.size for b in blocks if b.size >= smallest)
        coverage = matched / len(passage_keys)
    boxes = highlighter.boxes(digest, page, chunk_id)
    return {
        "origin": passage.text_origin,
        "passage_words": len(passage_keys),
        "page_words": len(page_keys),
        "coverage": round(coverage, 4),
        "drawn": bool(boxes),
        "area": round(sum(box[2] * box[3] for box in boxes), 5),
        "boxes": boxes,
    }


def share(values: List[bool]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def median(values: List[float]) -> Optional[float]:
    return round(statistics.median(values), 3) if values else None


def summary(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    drawn = [p for p in pairs if p["drawn"]]
    return {
        "pairs": len(pairs),
        "drawn": share([p["drawn"] for p in pairs]),
        "median_coverage_all": median([p["coverage"] for p in pairs]),
        "median_coverage_drawn": median([p["coverage"] for p in drawn]),
        "median_area_drawn": median([p["area"] for p in drawn]),
    }


def answer_passages(questions, chunks, hashes, highlighter, highlight):
    pairs: List[Dict[str, Any]] = []
    per_question: List[Dict[str, Any]] = []
    for q in questions:
        if q.get("type") == "unanswerable":
            continue
        seen = set()
        found: List[Dict[str, Any]] = []
        for source, span in q["expected_parsed"]:
            if span is None or source not in chunks:
                continue
            digest = hashes[source]
            for chunk in chunks[source]:
                chunk_id = chunk.get("chunk_id")
                if chunk_id not in highlighter.chunks:
                    continue
                for page in highlighter.chunks[chunk_id][1]:
                    if span[0] <= page <= span[1] and (chunk_id, page) not in seen:
                        seen.add((chunk_id, page))
                        result = measure(highlighter, highlight, digest, page, chunk_id)
                        result.update(
                            question_id=q["question_id"],
                            source=source,
                            domain=DOMAINS[source],
                            page=page,
                            chunk_id=chunk_id,
                        )
                        found.append(result)
        pairs.extend(found)
        per_question.append(
            {
                "question_id": q["question_id"],
                "pairs": len(found),
                "any_drawn": any(p["drawn"] for p in found),
            }
        )
    return pairs, per_question


def retrieved_passages(questions, index_dir: Path, hashes, highlighter, highlight):
    from companydocs_rag.retrieval.search import Retriever

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    indexed = {Path(doc["chunk_file"]).stem for doc in manifest["inputs"]["documents"]}
    rows: List[Dict[str, Any]] = []
    asked = 0
    with Retriever(index_dir) as retriever:
        for q in questions:
            if q.get("type") == "unanswerable":
                continue
            expected = [(s, sp) for s, sp in q["expected_parsed"] if sp is not None]
            if not expected or not all(s in indexed for s, _ in expected):
                continue
            asked += 1
            result = retriever.search(q["question"])
            for candidate in result.candidates[:TOP_K]:
                source = Path(candidate["file_path"]).stem
                chunk_id = candidate["chunk_id"]
                if chunk_id not in highlighter.chunks:
                    continue
                for page in sorted(highlighter.chunks[chunk_id][1]):
                    on_expected = any(
                        s == source and sp[0] <= page <= sp[1] for s, sp in expected
                    )
                    m = measure(highlighter, highlight, hashes[source], page, chunk_id)
                    m.pop("boxes")
                    m.update(
                        question_id=q["question_id"],
                        rank=candidate["rank"],
                        source=source,
                        page=page,
                        on_expected_page=on_expected,
                    )
                    rows.append(m)
    top1 = {}
    for row in rows:
        if row["rank"] == 1:
            top1.setdefault(row["question_id"], []).append(row)
    return {
        "questions": asked,
        "candidate_pages": summary(rows),
        "on_expected_page": summary([r for r in rows if r["on_expected_page"]]),
        "top1_highlight_on_expected_page": share(
            [
                any(r["drawn"] and r["on_expected_page"] for r in rs)
                for rs in top1.values()
            ]
        ),
        "rows": rows,
    }


def spot_check(pairs, files, hashes, count: int) -> List[Dict[str, Any]]:
    import pypdfium2 as pdfium
    from PIL import Image, ImageDraw

    drawn = [p for p in pairs if p["drawn"]]
    rng = random.Random(SEED)
    ocr = [p for p in drawn if p["origin"] == "ocr"]
    native = [p for p in drawn if p["origin"] != "ocr"]
    sample = rng.sample(ocr, min(len(ocr), max(1, count // 4))) if ocr else []
    sample += rng.sample(native, min(len(native), count - len(sample)))
    folder = OUT / "spotcheck"
    folder.mkdir(exist_ok=True)
    rows = []
    for number, pair in enumerate(sample, start=1):
        pdf = pdfium.PdfDocument(str(files[hashes[pair["source"]]]))
        image = pdf[pair["page"] - 1].render(scale=SPOTCHECK_DPI / 72).to_pil()
        image = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image.size
        for left, top, w, h in pair["boxes"]:
            draw.rectangle(
                [left * width, top * height, (left + w) * width, (top + h) * height],
                fill=(255, 210, 0, 90),
                outline=(220, 120, 0, 255),
            )
        name = f"{number:02d}_{pair['source']}_p{pair['page']}.jpg"
        Image.alpha_composite(image, overlay).convert("RGB").save(
            folder / name, quality=80
        )
        pdf.close()
        rows.append(
            {
                "nr": number,
                "question_id": pair["question_id"],
                "source": pair["source"],
                "page": pair["page"],
                "origin": pair["origin"],
                "coverage": pair["coverage"],
                "image": f"spotcheck/{name}",
                "verdict": "",
                "notes": "",
            }
        )
    with (OUT / "spotcheck.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def by(pairs, key) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for pair in pairs:
        groups.setdefault(str(pair[key]), []).append(pair)
    return {name: summary(group) for name, group in sorted(groups.items())}


def pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def write_markdown(result: Dict[str, Any]) -> None:
    a = result["answer_passages"]
    lines = [
        "# Highlight measurement",
        "",
        f"Generated by `measure_highlight.py` on {result['questions_answerable']} "
        "answerable questions of `evaluation/questions.csv`.",
        "",
        "## 1. Answer passages",
        "",
        "| group | (chunk, page) pairs | highlight drawn | median coverage "
        "(drawn) | median page area highlighted (drawn) |",
        "|---|---|---|---|---|",
    ]
    rows = [("all", a["all"])] + [(f"origin {k}", v) for k, v in a["by_origin"].items()]
    rows += [(f"domain {k}", v) for k, v in a["by_domain"].items()]
    for name, s in rows:
        lines.append(
            f"| {name} | {s['pairs']} | {pct(s['drawn'])} | "
            f"{pct(s['median_coverage_drawn'])} | {pct(s['median_area_drawn'])} |"
        )
    lines += [
        "",
        f"Questions with at least one highlighted answer passage: "
        f"{pct(a['questions_with_highlight'])}. Baseline interface: no highlight, "
        "the user reads the whole page (area 100%).",
    ]
    r = result.get("retrieved_passages")
    if r:
        lines += [
            "",
            "## 2. Retrieved passages (top 5)",
            "",
            f"Index `{result['index_dir']}`, {r['questions']} questions whose expected "
            "sources are all in that index.",
            "",
            "| | pages | highlight drawn |",
            "|---|---|---|",
            f"| all candidate pages | {r['candidate_pages']['pairs']} | "
            f"{pct(r['candidate_pages']['drawn'])} |",
            "| candidate pages on an expected page | "
            f"{r['on_expected_page']['pairs']} | "
            f"{pct(r['on_expected_page']['drawn'])} |",
            "",
            "First candidate highlighted on an expected page: "
            f"{pct(r['top1_highlight_on_expected_page'])} of the questions.",
        ]
    if result.get("spot_check"):
        lines += [
            "",
            "## 3. Spot check",
            "",
            f"{len(result['spot_check'])} highlighted pages in `spotcheck/`; the "
            "verdict per image goes into `spotcheck.csv` (correct / partly / wrong).",
        ]
    lines.append("")
    (OUT / "highlight_results.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument(
        "--spotcheck",
        type=int,
        default=0,
        help="render this many highlighted pages for a visual check (default: none)",
    )
    args = parser.parse_args(argv)

    from companydocs_rag.interface import highlight
    from companydocs_rag.interface.pageview import build_file_map

    questions = read_questions()
    files = build_file_map(RAW)
    chunks, hashes = read_chunk_files()
    highlighter = highlight.Highlighter(files, CHUNKS, OCR_CACHE)

    pairs, per_question = answer_passages(
        questions, chunks, hashes, highlighter, highlight
    )
    result: Dict[str, Any] = {
        "questions_answerable": len(per_question),
        "thresholds": {
            "MIN_COVERAGE": highlight.MIN_COVERAGE,
            "MIN_DENSITY": highlight.MIN_DENSITY,
            "MIN_BLOCK": highlight.MIN_BLOCK,
            "MIN_PASSAGE_WORDS": highlight.MIN_PASSAGE_WORDS,
        },
        "answer_passages": {
            "all": summary(pairs),
            "by_origin": by(pairs, "origin"),
            "by_domain": by(pairs, "domain"),
            "questions_with_highlight": share([q["any_drawn"] for q in per_question]),
        },
    }
    if args.index_dir is not None:
        result["index_dir"] = str(args.index_dir)
        result["retrieved_passages"] = retrieved_passages(
            questions, ROOT / args.index_dir, hashes, highlighter, highlight
        )
    if args.spotcheck > 0:
        result["spot_check"] = spot_check(pairs, files, hashes, args.spotcheck)
    result["answer_passages"]["pairs"] = [
        {k: v for k, v in p.items() if k != "boxes"} for p in pairs
    ]
    (OUT / "highlight_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(result)
    print((OUT / "highlight_results.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
