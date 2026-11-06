"""Build a navigation tree per benchmark from the documents' own structure.

    python -m bench.hierarchy <bench>   -> runs/<bench>/tree.json + checks

Leaves are our own chunks (runs/<bench>/chunks.jsonl). Inner levels follow the
documents' real structure, never arbitrary splits:

  conditionalqa  collection -> page (title + section headings)
                 -> section (h1/h2/h3 heading + opening text) -> chunk
  legalbench     collection -> document type (NDA, CUAD contract, MAUD merger
                 agreement, privacy policy) -> document (title + opening)
                 -> article (our chunker's numbering run; generic documents
                 get consecutive sections) -> chunk
  bbsard         collection -> code -> book/title/chapter/... (the statute's
                 own path) -> article

Every inner node gets a description a judge can decide on. Statute path
levels carry only numbers ("titel 3"), so their description is the article
number range plus the opening of the articles under them.

Checks printed: nodes and branching per level, the largest fan-out, and that
every gold passage is reachable (it is a leaf under some path by
construction; the check guards against chunks that match no node).
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .common import read_jsonl, run_dir

DESCRIPTION_CHARS = 400
GROUP_SIZE = 12  # chunks per section for documents without accepted numbering
SUBSET_TITLES = {
    "contractnli": "Non-disclosure agreements (NDAs)",
    "cuad": "Commercial contracts (CUAD: licence, supply, distribution, services ...)",
    "maud": "Merger and acquisition agreements (MAUD)",
    "privacy_qa": "Privacy policies of consumer apps and websites",
}
CODE_ALIASES = {"Wetboak van Strafvordering": "Wetboek van Strafvordering"}


def _clip(text: str, limit: int = DESCRIPTION_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


class Tree:
    def __init__(self, title: str) -> None:
        self.nodes: List[Dict[str, Any]] = []
        self.root = self.add(None, title, "", kind="root")

    def add(self, parent: Optional[int], title: str, description: str, **extra) -> int:
        node = {
            "id": len(self.nodes),
            "parent": parent,
            "title": title,
            "description": description,
            "children": [],
            "chunk_ids": [],
            **extra,
        }
        self.nodes.append(node)
        if parent is not None:
            self.nodes[parent]["children"].append(node["id"])
        return node["id"]

    def attach(self, node: int, chunk_id: str) -> None:
        self.nodes[node]["chunk_ids"].append(chunk_id)


def _chunks_by_doc(bench: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in read_jsonl(run_dir(bench) / "chunks.jsonl"):
        grouped[chunk["doc_id"]].append(chunk)
    for chunks in grouped.values():
        chunks.sort(key=lambda chunk: chunk["chunk_index"])
    return grouped


def build_conditionalqa() -> Tree:
    tree = Tree("UK government guidance pages (gov.uk)")
    chunks_by_doc = _chunks_by_doc("conditionalqa")
    for doc in read_jsonl(run_dir("conditionalqa") / "docs.jsonl"):
        sections: List[Dict[str, Any]] = []
        for para in doc["paragraphs"]:
            if para["span"] is None:
                continue
            text = doc["text"][para["span"][0] : para["span"][1]]
            if re.match(r"^<h[1-3]", para["html"]) or not sections:
                sections.append({"title": text, "start": para["span"][0], "body": []})
            else:
                sections[-1]["body"].append(text)
        headings = "; ".join(section["title"] for section in sections)
        page = tree.add(
            tree.root,
            doc["title"],
            _clip(f"{doc['title']}. Sections: {headings}"),
            kind="page",
            doc_id=doc["doc_id"],
        )
        # Our chunker cuts across section boundaries, so a section node owns
        # the chunks that start in it and names every section they cover;
        # sections in which no chunk starts get no node (no dead branches).
        starts = [section["start"] for section in sections]
        owned: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for chunk in chunks_by_doc.get(doc["doc_id"], []):
            owner = max(
                (i for i, start in enumerate(starts) if start <= chunk["char_start"]),
                default=0,
            )
            owned[owner].append(chunk)
        for owner in sorted(owned):
            chunks = owned[owner]
            low, high = chunks[0]["char_start"], chunks[-1]["char_end"]
            covered = [
                sections[i]["title"]
                for i, start in enumerate(starts)
                if i == owner or low <= start < high
            ]
            node = tree.add(
                page,
                sections[owner]["title"],
                _clip(
                    f"{doc['title']} - {'; '.join(covered)}: "
                    f"{' '.join(sections[owner]['body'])}"
                ),
                kind="section",
            )
            for chunk in chunks:
                tree.attach(node, chunk["chunk_id"])
    _collapse(tree)
    return tree


def _major(clause_number: Optional[str]) -> Optional[str]:
    if not clause_number:
        return None
    return re.split(r"[.\-()]", clause_number.strip())[0] or None


_BOILERPLATE = re.compile(
    r"(certain (identified )?(confidential )?information (has been|marked)[^.]*\.|"
    r"\[\*+\][^.]*\.|confidential treatment (has been )?requested[^.]*\.)",
    re.IGNORECASE,
)
_PARTIES = re.compile(r"\b(by and )?between\b", re.IGNORECASE)


def _document_opening(text: str) -> str:
    """What identifies a contract: its opening without redaction boilerplate,
    starting at the 'by and between <parties>' sentence when there is one."""
    head = _BOILERPLATE.sub(" ", text[:4000])
    match = _PARTIES.search(head)
    if match:
        start = max(0, head.rfind("\n", 0, match.start()) + 1)
        start = max(start, match.start() - 200)
        return head[:200].strip() + " ... " + head[start : start + 600]
    return head[:800]


MAX_GROUP = 40  # largest article a judge sees in one step


def _minor(clause_number: Optional[str]) -> Optional[str]:
    if not clause_number:
        return None
    parts = [p for p in re.split(r"[.\-()]", clause_number.strip()) if p]
    return ".".join(parts[:2]) if len(parts) >= 2 else None


def _split_large(group: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """An article with more than MAX_GROUP chunks is split by its own
    sub-numbering (5.01, 5.02, ...) when that exists, else into consecutive
    runs of GROUP_SIZE chunks."""
    if len(group) <= MAX_GROUP:
        return [group]
    parts: List[List[Dict[str, Any]]] = []
    current: object = object()
    for chunk in group:
        key = _minor(chunk["clause_number"]) or current
        if not parts or key != current:
            parts.append([])
            current = key
        parts[-1].append(chunk)
    if len(parts) == 1 or any(len(part) > MAX_GROUP for part in parts):
        flat = [chunk for part in parts for chunk in part]
        return [flat[i : i + GROUP_SIZE] for i in range(0, len(flat), GROUP_SIZE)]
    return parts


def build_legalbench() -> Tree:
    tree = Tree("Legal documents: contracts, agreements and policies")
    chunks_by_doc = _chunks_by_doc("legalbench")
    subsets: Dict[str, int] = {}
    for doc in read_jsonl(run_dir("legalbench") / "docs.jsonl"):
        subset = doc["doc_id"].split("/")[0]
        if subset not in subsets:
            subsets[subset] = tree.add(
                tree.root, SUBSET_TITLES[subset], SUBSET_TITLES[subset], kind="type"
            )
        title = doc["title"].replace("_", " ")
        node = tree.add(
            subsets[subset],
            title,
            _clip(f"{title}: {_document_opening(doc['text'])}"),
            kind="document",
            doc_id=doc["doc_id"],
        )
        chunks = chunks_by_doc.get(doc["doc_id"], [])
        structured = chunks and chunks[0]["route"] == "structured"
        groups: List[List[Dict[str, Any]]] = []
        if structured:
            current_key = object()
            for chunk in chunks:
                key = _major(chunk["clause_number"])
                if key is None and groups:
                    key = current_key  # unnumbered text stays with its article
                if not groups or key != current_key:
                    groups.append([])
                    current_key = key
                groups[-1].append(chunk)
        else:
            groups = [chunks[i : i + GROUP_SIZE] for i in range(0, len(chunks), GROUP_SIZE)]
        groups = [part for group in groups for part in _split_large(group)]
        if len(groups) <= 1:  # short document: chunks hang directly under it
            for chunk in chunks:
                tree.attach(node, chunk["chunk_id"])
            continue
        for group in groups:
            label = group[0]["clause_label"] or _clip(group[0]["text"], 80)
            section = tree.add(
                node,
                label,
                _clip(f"{title} - {label}: {' '.join(c['text'] for c in group)[:1200]}"),
                kind="section",
            )
            for chunk in group:
                tree.attach(section, chunk["chunk_id"])
    _collapse(tree)
    return tree


def build_bbsard() -> Tree:
    tree = Tree("Belgische wetgeving (wetboeken, Nederlandstalig)")
    chunks_by_doc = _chunks_by_doc("bbsard")
    docs = read_jsonl(run_dir("bbsard") / "docs.jsonl")
    for doc in docs:
        doc["code"] = CODE_ALIASES.get(doc["code"], doc["code"])
    index: Dict[tuple, int] = {}
    members: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        key: tuple = (doc["code"],)
        if key not in index:
            index[key] = tree.add(tree.root, doc["code"], doc["code"], kind="code")
        parent = index[key]
        members[parent].append(doc)
        for part in doc["path"]:
            key = key + (part,)
            if key not in index:
                index[key] = tree.add(parent, f"{doc['code']}, {', '.join(key[1:])}", "", kind="part")
            parent = index[key]
            members[parent].append(doc)
        leaf = tree.add(parent, doc["title"], _clip(f"{doc['title']}: {doc['text']}"), kind="article", doc_id=doc["doc_id"])
        for chunk in chunks_by_doc.get(doc["doc_id"], []):
            tree.attach(leaf, chunk["chunk_id"])
    for node_id, docs_under in members.items():
        node = tree.nodes[node_id]
        numbers = [d["number"] for d in docs_under if d["number"]]
        opening = " | ".join(_clip(d["text"], 110) for d in docs_under[:3])
        span = f"art. {numbers[0]} - {numbers[-1]}" if numbers else ""
        node["description"] = _clip(f"{node['title']} ({span}, {len(docs_under)} artikelen): {opening}")
    _collapse(tree)
    return tree


def _collapse(tree: Tree) -> None:
    """Remove decisions without a choice. An inner node with exactly one child
    and no chunks of its own is replaced by that child (the child takes its
    place under the parent); an inner node with neither children nor chunks
    is dropped. Repeats until nothing changes."""
    nodes = tree.nodes
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node["id"] == tree.root or "merged_into" in node:
                continue
            parent = nodes[node["parent"]]
            if not node["children"] and not node["chunk_ids"] and node.get("kind") != "article":
                parent["children"] = [c for c in parent["children"] if c != node["id"]]
                node["merged_into"] = parent["id"]
                changed = True
            elif len(node["children"]) == 1 and not node["chunk_ids"]:
                child = nodes[node["children"][0]]
                parent["children"] = [
                    child["id"] if c == node["id"] else c for c in parent["children"]
                ]
                child["parent"] = parent["id"]
                node["children"] = []
                node["merged_into"] = child["id"]
                changed = True


def check(bench: str, tree: Tree) -> None:
    nodes = tree.nodes
    live = [n for n in nodes if "merged_into" not in n]
    depth: Dict[int, int] = {}
    stack = [(tree.root, 0)]
    while stack:  # walk from the root, so hoisted nodes get their new depth
        node_id, level = stack.pop()
        depth[node_id] = level
        stack.extend((child, level + 1) for child in nodes[node_id]["children"])
    live = [n for n in live if n["id"] in depth]
    by_level: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for node in live:
        by_level[depth[node["id"]]].append(node)
    print(f"{bench}: {len(live)} nodes")
    for level in sorted(by_level):
        items = by_level[level]
        fan = [len(n["children"]) + len(n["chunk_ids"]) for n in items]
        desc = sorted(len(n["description"]) for n in items)
        print(
            f"  level {level}: {len(items):6d} nodes, fan-out median {sorted(fan)[len(fan)//2]}, "
            f"max {max(fan)}; description chars median {desc[len(desc)//2]}, empty {desc.count(0)}"
        )
    chunk_owner = {cid: n["id"] for n in live for cid in n["chunk_ids"]}
    all_chunks = {c["chunk_id"] for c in read_jsonl(run_dir(bench) / "chunks.jsonl")}
    print(f"  chunks attached {len(chunk_owner)} / {len(all_chunks)}")


BUILDERS = {
    "conditionalqa": build_conditionalqa,
    "legalbench": build_legalbench,
    "bbsard": build_bbsard,
}


def main(argv: List[str]) -> int:
    for bench in argv or list(BUILDERS):
        tree = BUILDERS[bench]()
        (run_dir(bench) / "tree.json").write_text(
            json.dumps(tree.nodes, ensure_ascii=False), encoding="utf-8"
        )
        check(bench, tree)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
