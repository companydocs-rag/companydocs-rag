"""Turn each benchmark into our pipeline's input plus a gold file.

    python -m bench.prepare legalbench|conditionalqa|bbsard

Writes, under runs/<bench>/:
  normalized/<n>.json  one normalized record per document (chunker input)
  docs.jsonl           doc_id, title, language, text (+ paragraph spans)
  questions.jsonl      qid, query, gold (doc ids and, where known, char spans)

Evaluation sets:
  legalbench     LegalBench-RAG-mini style: 194 queries per sub-dataset
                 (contractnli, cuad, maud, privacy_qa), seed 0; full corpus.
  conditionalqa  the 285 dev questions (271 carry evidence); query is the
                 scenario followed by the question.
  bbsard         the Dutch test questions over the Dutch statute corpus.
"""

from __future__ import annotations

import ast
import html
import json
import random
import re
import sys
from typing import Any, Dict, List, Tuple

from .common import BENCHMARKS, normalized_record, run_dir, write_jsonl

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"[ \t]+")


def _strip_html(fragment: str) -> str:
    return _SPACE.sub(" ", html.unescape(_TAG.sub(" ", fragment))).strip()


def _write_normalized(out, docs: List[Dict[str, Any]]) -> None:
    folder = out / "normalized"
    folder.mkdir(exist_ok=True)
    for old in folder.glob("*.json"):
        old.unlink()
    for number, doc in enumerate(docs):
        record = normalized_record(doc["doc_id"], doc["text"], doc["language"])
        (folder / f"{number:06d}.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )


def prepare_legalbench() -> Tuple[int, int]:
    root = BENCHMARKS / "legalbenchrag"
    docs = []
    for path in sorted((root / "corpus").rglob("*.txt")):
        doc_id = path.relative_to(root / "corpus").as_posix()
        text = path.read_text(encoding="utf-8")
        docs.append(
            {"doc_id": doc_id, "title": path.stem, "language": "eng", "text": text}
        )
    questions = []
    for subset in ("contractnli", "cuad", "maud", "privacy_qa"):
        tests = json.loads((root / "benchmarks" / f"{subset}.json").read_text("utf-8"))
        tests = tests["tests"]
        picked = random.Random(0).sample(tests, min(194, len(tests)))
        for number, test in enumerate(picked):
            questions.append(
                {
                    "qid": f"{subset}-{number}",
                    "subset": subset,
                    "query": test["query"],
                    "gold": [
                        {
                            "doc_id": snippet["file_path"],
                            "start": snippet["span"][0],
                            "end": snippet["span"][1],
                        }
                        for snippet in test["snippets"]
                    ],
                }
            )
    out = run_dir("legalbench")
    write_jsonl(out / "docs.jsonl", docs)
    write_jsonl(out / "questions.jsonl", questions)
    _write_normalized(out, docs)
    return len(docs), len(questions)


def prepare_conditionalqa() -> Tuple[int, int]:
    root = BENCHMARKS / "conditionalqa" / "v1_0"
    raw = json.loads((root / "documents.json").read_text("utf-8"))
    docs = []
    by_url: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        lines, spans, position = [], [], 0
        for fragment in item["contents"]:
            line = _strip_html(fragment)
            if not line:
                spans.append(None)
                continue
            spans.append([position, position + len(line)])
            lines.append(line)
            position += len(line) + 1  # the line feed that joins the lines
        doc = {
            "doc_id": item["url"],
            "title": item["title"],
            "language": "eng",
            "text": "\n".join(lines),
            "paragraphs": [
                {"html": fragment, "span": span}
                for fragment, span in zip(item["contents"], spans)
            ],
        }
        docs.append(doc)
        by_url[item["url"]] = doc

    out = run_dir("conditionalqa")
    for split, name in (("train", "questions_train.jsonl"), ("dev", "questions.jsonl")):
        questions, missing = _conditionalqa_questions(root / f"{split}.json", by_url)
        write_jsonl(out / name, questions)
        if missing:
            print(f"conditionalqa {split}: {missing} evidence fragments not found in documents")
    write_jsonl(out / "docs.jsonl", docs)
    _write_normalized(out, docs)
    return len(docs), len(questions)


def _conditionalqa_questions(path, by_url) -> Tuple[List[Dict[str, Any]], int]:
    questions = []
    missing = 0
    for item in json.loads(path.read_text("utf-8")):
        evidences = item["evidences"]
        if isinstance(evidences, str):
            evidences = ast.literal_eval(evidences)
        doc = by_url.get(item["url"])
        gold = []
        if doc is not None:
            for evidence in evidences:
                span = next(
                    (
                        para["span"]
                        for para in doc["paragraphs"]
                        if para["html"] == evidence and para["span"] is not None
                    ),
                    None,
                )
                if span is None:
                    missing += 1
                    continue
                gold.append({"doc_id": doc["doc_id"], "start": span[0], "end": span[1]})
        questions.append(
            {
                "qid": item["id"],
                "query": f"{item['scenario']} {item['question']}".strip(),
                "gold_doc": item["url"],
                "gold": gold,
                "has_evidence": bool(evidences),
            }
        )
    return questions, missing


_REFERENCE = re.compile(r"^Art\.\s*(?P<number>\S+)\s+(?P<code>.*?)(?:\s*\((?P<path>[^()]*)\))?\s*$")


def parse_reference(reference: str) -> Dict[str, Any]:
    """'Art. 1.1.1 <code name> (boek 1, titel 1)' -> number, code, path list.
    The path is the statute's own structure (book, title, chapter, ...)."""
    match = _REFERENCE.match(reference.strip())
    if match is None:
        return {"number": None, "code": reference.strip(), "path": []}
    path = [part.strip() for part in (match["path"] or "").split(",") if part.strip()]
    return {"number": match["number"], "code": match["code"].strip(), "path": path}


def prepare_bbsard() -> Tuple[int, int]:
    import csv

    csv.field_size_limit(10**9)
    root = BENCHMARKS / "bbsard" / "nl"
    with (root / "corpus.csv").open(encoding="utf-8") as handle:
        corpus = list(csv.DictReader(handle))
    docs = []
    for row in corpus:
        ref = parse_reference(row["reference"])
        docs.append(
            {
                "doc_id": row["id"],
                "title": row["reference"],
                "language": "nld",
                "text": row["article"],
                "code": ref["code"],
                "path": ref["path"],
                "number": ref["number"],
            }
        )
    with (root / "test.csv").open(encoding="utf-8") as handle:
        tests = list(csv.DictReader(handle))
    questions = [
        {
            "qid": f"bbsard-{row['id']}",
            "query": row["question"],
            "category": row["category"],
            "gold": [{"doc_id": item.strip()} for item in row["article_ids"].split(",")],
        }
        for row in tests
    ]
    out = run_dir("bbsard")
    write_jsonl(out / "docs.jsonl", docs)
    write_jsonl(out / "questions.jsonl", questions)
    _write_normalized(out, docs)
    return len(docs), len(questions)


def main(argv: List[str]) -> int:
    preparers = {
        "legalbench": prepare_legalbench,
        "conditionalqa": prepare_conditionalqa,
        "bbsard": prepare_bbsard,
    }
    for name in argv or list(preparers):
        docs, questions = preparers[name]()
        print(f"{name}: {docs} documents, {questions} questions")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
