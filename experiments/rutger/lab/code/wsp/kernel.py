"""Weight-space programming session for the tuned d3 judge.

Follows Experiment 8 of Takehi, Teoh, Zhang and Khattab, "Sparks of
Weight-Space Programming" (MIT CSAIL, November 2026,
https://rikiyat.github.io/blog/weight-space-programming/): a Python session
with the model's weights loaded, in which the editor reads the weights,
spends a fixed budget of forward passes on prompts from the training side of
the data, and edits the weights by hand. No training: no gradients, no
optimizer, no training loop.

    python -m bench.wsp.kernel [--device cpu|cuda:0] [--budget 50]
    python -m bench.wsp.run <cell.py>          (submit one cell, print its output)

The session is a notebook without a socket (localhost is blocked here): a cell
is a file runs/wsp/cells/NNNN.py; the kernel runs it in one persistent
namespace and writes NNNN.out beside it. Every cell, its output and the
passes used so far are appended to runs/wsp/log.md, the full log of the run.

Data (all ConditionalQA, the dataset the judge was fine-tuned on):
- TRAIN: judge_train.jsonl, the 30,726 examples the fine-tune saw.
- PRACTICE: the examples of half of the 112 judge_val questions (seed 7).
  No gradient ever touched them.
- held-out: the other half of judge_val, not in the namespace; scored once by
  finalize(), unedited and edited. The benchmark test sets (ConditionalQA dev,
  LegalBench-RAG, bBSARD) are never touched here.

Budget: one forward pass is one call of at most MAX_ROWS (512) prompts, however
it is cut into GPU batches; a call with more prompts costs ceil(n / 512)
passes. Every route to the model's forward goes through the counter (the
judge's relevance_batch, so navigation on train questions counts too). Reading
or analysing the weights costs nothing. In the CPU reading phase no forward
pass is possible; to_gpu() moves the session to the GPU, carrying the edits.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from ..common import RUNS, read_jsonl, run_dir
from ..judges import MODELS, QUESTION, D3TunedJudge, _budget_batches, _local

SESSION = Path(os.environ.get("WSP_SESSION", str(RUNS / "wsp")))
CELLS = SESSION / "cells"
STATE = SESSION / "state.json"
LOG = SESSION / "log.md"
LATEST = SESSION / "latest_edit.pt"
TUNED = os.environ.get("WSP_TUNED", os.path.join(MODELS, "finetuned", "d3-lite-judge-full"))
OUT = os.environ.get("WSP_OUT", os.path.join(MODELS, "finetuned", "d3-lite-edited"))
MAX_ROWS = 512


# ---------------------------------------------------------------- no training

def forbid_gradients() -> None:
    """No gradients in the session: autograd off, and backward / grad raise.
    Installed after the model has loaded (torch._dynamo registers the
    originals while transformers imports)."""
    def refuse(what):
        def _refuse(*_a, **_k):
            raise RuntimeError(f"weight-space programming: {what} is not allowed (no gradients)")
        return _refuse
    torch.set_grad_enabled(False)
    torch.Tensor.backward = refuse("Tensor.backward")
    torch.autograd.backward = refuse("autograd.backward")
    torch.autograd.grad = refuse("autograd.grad")


# ---------------------------------------------------------------- helpers

def auc(scores, labels) -> float:
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return math.nan
    rank_sum = sum(rank for rank, (_, label) in enumerate(pairs, 1) if label == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def metrics(scores, labels) -> dict:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    eps = 1e-6
    return {
        "n": int(len(y)),
        "auc": round(auc(s.tolist(), y.astype(int).tolist()), 4),
        "acc": round(float(((s > 0.5) == (y > 0.5)).mean()), 4),
        "logloss": round(float(-(y * np.log(s + eps) + (1 - y) * np.log(1 - s + eps)).mean()), 4),
        "brier": round(float(((s - y) ** 2).mean()), 4),
        "pos_rate_pred": round(float((s > 0.5).mean()), 4),
        "pos_rate_true": round(float(y.mean()), 4),
    }


class Session:
    def __init__(self, device: str, budget: int) -> None:
        self.budget = budget
        self.device = device
        self.state = json.loads(STATE.read_text()) if STATE.exists() else {
            "passes_used": 0, "budget": budget, "turn": 0, "finalized": False, "history": []}
        val = read_jsonl(run_dir("conditionalqa") / "judge_val.jsonl")
        qids = sorted({r["qid"] for r in val})
        practice_q = set(random.Random(7).sample(qids, len(qids) // 2))
        self.TRAIN = read_jsonl(run_dir("conditionalqa") / "judge_train.jsonl")
        self.PRACTICE = [r for r in val if r["qid"] in practice_q]
        self._heldout = [r for r in val if r["qid"] not in practice_q]
        self.inside = False
        self.ctx = {}  # set per GPU batch: ctx["batch"] = row indices of the batch
        self._load(device)
        if LATEST.exists():
            self._apply(torch.load(LATEST, map_location="cpu", weights_only=True))
            print(f"resumed edits from {LATEST}", flush=True)

    # ------------------------------------------------------------ model

    def _load(self, device: str) -> None:
        self.judge = D3TunedJudge(_local("vllm-sr/d3-lite"), TUNED, device=device)
        self.rt = self.judge.model.runtime
        self.layers = [layer for layer, _ in self.rt.encoder.stack]
        self.layer_types = list(self.rt.backbone.language_model.config.layer_types[: len(self.layers)])
        self.params = dict(self.rt.backbone.named_parameters())
        # the unedited tuned judge, as the benchmark loader builds it
        self._orig = {n: p.detach().to("cpu", copy=True) for n, p in self.params.items()}
        self._orig_readout = self.rt.readout[:2].detach().to("cpu", copy=True)
        module = sys.modules[type(self.rt).__module__]
        self._module = module
        keys = module.normalize_question({"type": "noul", "instructions": "x"}).keys
        self.true_index, self.n_codes = list(keys).index("true"), len(keys)
        rt, sess = self.rt, self
        for attr in ("logits", "probabilities"):
            original = getattr(rt, attr)

            def guarded(*a, _original=original, _attr=attr, **k):
                if not sess.inside:
                    raise RuntimeError(f"rt.{_attr} outside forward(): use forward() so the pass is counted")
                return _original(*a, **k)
            setattr(rt, attr, guarded)
        self.judge.relevance_batch = self._judge_batch
        self.judge.relevance = lambda q, texts, kind="passage": self._judge_batch([(q, kind, t) for t in texts])

    def to_gpu(self) -> str:
        """End the CPU reading phase: reload on the GPU exactly as the benchmark
        loader does (fp16 cast, LoRA merge) and carry the edits made so far."""
        if self.device.startswith("cuda"):
            return "already on the GPU"
        edits = self.edit_state()
        del self.judge, self.rt, self.layers, self.params
        import gc
        gc.collect()
        self.device = "cuda:0"
        self._load("cuda:0")
        self._apply(edits)
        return f"on {self.device}; carried {len(edits['param_deltas'])} edited tensors"

    def W(self, name: str) -> torch.Tensor:
        """The live parameter (edit it in place)."""
        return self.params[name]

    def orig(self, name: str) -> torch.Tensor:
        """The unedited parameter (CPU copy)."""
        return self._orig[name]

    def names(self, pattern: str = "") -> list:
        return [n for n in self.params if pattern in n]

    @property
    def readout(self) -> torch.Tensor:
        """The 2 noul readout rows (live view; row order = noul codes)."""
        return self.rt.readout[:2]

    def prompt(self, row) -> str:
        q, kind, text = (row["query"], row["kind"], row["text"]) if isinstance(row, dict) else row
        rendered = self._module.normalize_question(
            {"type": "noul", "instructions": QUESTION.format(kind=kind, text=text)}).rendered
        return self.rt.text(f"Question: {q}", rendered)

    def tokens(self, row) -> list:
        return self.rt.tokenizer(self.prompt(row), add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------ forward passes

    def _spend(self, n_rows: int) -> int:
        cost = max(1, math.ceil(n_rows / MAX_ROWS))
        if not self.device.startswith("cuda"):
            raise RuntimeError("CPU reading phase: no forward passes (call to_gpu() after the benchmark)")
        if self.state["passes_used"] + cost > self.budget:
            raise RuntimeError(f"forward-pass budget: {self.state['passes_used']}/{self.budget} used, "
                               f"this call needs {cost}")
        self.state["passes_used"] += cost
        self._save_state()
        return cost

    def _score(self, items, on_batch=None):
        rt = self.rt
        texts = [self.prompt(it) for it in items]
        sequences = rt.tokenizer(texts, add_special_tokens=False)["input_ids"]
        scores = [0.0] * len(items)
        self.inside = True
        try:
            for batch in _budget_batches([len(s) for s in sequences], self.judge.TOKEN_BUDGET):
                self.ctx = {"batch": batch, "lengths": [len(sequences[i]) for i in batch]}
                if on_batch:
                    on_batch(self.ctx)
                rows = rt.probabilities([sequences[i] for i in batch], [self.n_codes] * len(batch))
                for i, row in zip(batch, rows):
                    scores[i] = row[self.true_index]
        finally:
            self.inside = False
            self.ctx = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # long batches otherwise fragment the cache into a shared-memory spill
        return scores

    def _judge_batch(self, items):
        self._spend(len(items))
        return self._score(items)

    def forward(self, rows, capture=(), layers=None):
        """P(yes) for each row (dict with query/kind/text or a triple); one
        pass per 512 rows. capture: any of "resid" (layer output), "attn_out"
        (token-mixer output projection), "mlp_in" (input to down_proj) and
        "mlp_out" (down_proj output), all at the answer (last) position, as
        float32 arrays [n, layers, dim] in row order."""
        items = [(r["query"], r["kind"], r["text"]) if isinstance(r, dict) else tuple(r) for r in rows]
        self._spend(len(items))
        layers = list(range(len(self.layers))) if layers is None else list(layers)
        store = {c: {L: [None] * len(items) for L in layers} for c in capture}
        hooks = []

        def put(kind, L, tensor):
            last = tensor[:, -1].float().cpu()
            for j, i in enumerate(self.ctx["batch"]):
                store[kind][L][i] = last[j]

        for L in layers:
            layer = self.layers[L]
            mixer = layer.linear_attn if self.layer_types[L] == "linear_attention" else layer.self_attn
            proj = mixer.out_proj if hasattr(mixer, "out_proj") else mixer.o_proj
            if "resid" in capture:
                hooks.append(layer.register_forward_hook(
                    lambda m, i, o, L=L: put("resid", L, o[0] if isinstance(o, tuple) else o)))
            if "attn_out" in capture:
                hooks.append(proj.register_forward_hook(lambda m, i, o, L=L: put("attn_out", L, o)))
            if "mlp_in" in capture:
                hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
                    lambda m, i, L=L: put("mlp_in", L, i[0])))
            if "mlp_out" in capture:
                hooks.append(layer.mlp.down_proj.register_forward_hook(lambda m, i, o, L=L: put("mlp_out", L, o)))
        try:
            scores = self._score(items)
        finally:
            for h in hooks:
                h.remove()
        result = {"p": np.asarray(scores)}
        for c in capture:
            result[c] = np.stack([torch.stack(store[c][L]).numpy() for L in layers], axis=1)
        return result

    def evaluate(self, rows) -> dict:
        """forward() plus metrics overall and per kind (costs passes)."""
        p = self.forward(rows)["p"]
        out = {"all": metrics(p, [r["label"] for r in rows])}
        for kind in sorted({r["kind"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["kind"] == kind]
            out[kind] = metrics(p[idx], [rows[i]["label"] for i in idx])
        return out | {"_p": p}

    # ------------------------------------------------------------ edits

    def edit_state(self) -> dict:
        deltas = {}
        for n, p in self.params.items():
            cur = p.detach().cpu()
            if not torch.equal(cur, self._orig[n].to(cur.dtype)):
                deltas[n] = cur.float() - self._orig[n].float()
        readout = self.rt.readout[:2].detach().cpu().float() - self._orig_readout.float()
        return {"param_deltas": deltas, "readout_delta": readout if readout.abs().sum() > 0 else None}

    def _apply(self, edits: dict) -> None:
        for n, d in edits["param_deltas"].items():
            p = self.params[n]
            p.copy_((self._orig[n].float() + d).to(p.device, p.dtype))
        if edits.get("readout_delta") is not None:
            self.rt.readout[:2] = (self._orig_readout.float() + edits["readout_delta"]).to(
                self.rt.readout.device, self.rt.readout.dtype)

    def changed(self) -> dict:
        """Changed weights per edited tensor, and in total (vs the unedited judge)."""
        out = {}
        for n, p in self.params.items():
            cur = p.detach().cpu()
            diff = int((cur != self._orig[n].to(cur.dtype)).sum())
            if diff:
                out[n] = diff
        r = int((self.rt.readout[:2].detach().cpu() != self._orig_readout).sum())
        if r:
            out["readout[:2]"] = r
        total = sum(p.numel() for p in self.params.values()) + self.rt.readout[:2].numel()
        out["_total"] = sum(v for k, v in out.items() if not k.startswith("_"))
        out["_share"] = f"{100 * out['_total'] / total:.4f}% of {total:,}"
        return out

    def revert(self, name: str | None = None) -> None:
        names = [name] if name else list(self.params)
        for n in names:
            p = self.params[n]
            p.copy_(self._orig[n].to(p.device, p.dtype))
        if name is None:
            self.rt.readout[:2] = self._orig_readout.to(self.rt.readout.device, self.rt.readout.dtype)

    def save_latest(self, note: str = "") -> str:
        edits = self.edit_state()
        torch.save(edits, LATEST)
        self.state["history"].append({"turn": self.state["turn"], "passes_used": self.state["passes_used"],
                                      "note": note, "changed": self.changed()["_total"]})
        self._save_state()
        return f"saved {len(edits['param_deltas'])} edited tensors ({note})"

    def snapshot(self, tag: str) -> str:
        path = SESSION / "snapshots"
        path.mkdir(exist_ok=True)
        torch.save(self.edit_state() | {"passes_used": self.state["passes_used"]}, path / f"{tag}.pt")
        return f"snapshot {tag} at {self.state['passes_used']} passes"

    # ------------------------------------------------------------ the end

    def finalize(self, note: str, snapshots=()) -> dict:
        """Once: the held-out half of judge_val, unedited vs edited (and any
        snapshots), then judge.pt for d3-lite-edited. Not a practice pass."""
        if self.state["finalized"]:
            raise RuntimeError("already finalized")
        hl = [r["label"] for r in self._heldout]
        items = [(r["query"], r["kind"], r["text"]) for r in self._heldout]
        edits = self.edit_state()
        report = {}
        self.revert()
        report["unedited"] = metrics(self._score(items), hl)
        for tag in snapshots:
            snap = torch.load(SESSION / "snapshots" / f"{tag}.pt", map_location="cpu", weights_only=True)
            self.revert()
            self._apply(snap)
            report[f"{tag} ({snap['passes_used']} passes)"] = metrics(self._score(items), hl)
        self.revert()
        self._apply(edits)
        report["edited"] = metrics(self._score(items), hl)
        changed = self.changed()
        report["changed"] = {"total": changed["_total"], "share": changed["_share"]}
        report["passes_used"] = self.state["passes_used"]
        not_worse = report["edited"]["auc"] >= report["unedited"]["auc"]
        saved = torch.load(os.path.join(TUNED, "judge.pt"), map_location="cpu", weights_only=True)
        deltas, others = {}, {}
        for n, d in edits["param_deltas"].items():
            (deltas.__setitem__(n[: -len(".weight")], d) if n.endswith(".weight") else others.__setitem__(n, d))
        saved["deltas"] = deltas
        saved["param_deltas"] = others
        if edits.get("readout_delta") is not None:
            saved["readout_delta"] = edits["readout_delta"]
        saved["edit"] = {"method": "manual weight-space programming (bench.wsp)", "note": note,
                         "passes_used": self.state["passes_used"], "budget": self.budget,
                         "heldout": report, "heldout_not_worse": not_worse,
                         "changed_weights": changed["_total"]}
        os.makedirs(OUT, exist_ok=True)
        torch.save(saved, os.path.join(OUT, "judge.pt"))
        self.state["finalized"] = True
        self._save_state()
        (SESSION / "heldout.json").write_text(json.dumps(report, indent=2))
        return report

    def _save_state(self) -> None:
        STATE.write_text(json.dumps(self.state, indent=2))


def main(argv) -> int:
    parser = argparse.ArgumentParser(prog="bench.wsp.kernel")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    CELLS.mkdir(parents=True, exist_ok=True)
    # markers are never deleted: READY and STOP count only when newer than this start
    started = time.time()
    torch.set_grad_enabled(False)
    s = Session(args.device, args.budget)
    forbid_gradients()
    ns = {"S": s, "torch": torch, "np": np, "metrics": metrics, "auc": auc,
          "TRAIN": s.TRAIN, "PRACTICE": s.PRACTICE, "forward": s.forward, "evaluate": s.evaluate,
          "W": s.W, "orig": s.orig, "names": s.names, "changed": s.changed, "revert": s.revert,
          "save_latest": s.save_latest, "snapshot": s.snapshot, "to_gpu": s.to_gpu}
    ready = (f"kernel ready on {s.device} in {time.time() - started:.0f}s; passes "
             f"{s.state['passes_used']}/{s.budget}; TRAIN {len(s.TRAIN)}, PRACTICE {len(s.PRACTICE)}, "
             f"held-out {len(s._heldout)} (hidden)")
    print(ready, flush=True)
    (SESSION / "READY").write_text(ready)
    if not LOG.exists():
        LOG.write_text("# Weight-space programming log: d3-lite judge\n\n" + __doc__.split("\n\n")[1] + "\n\n",
                       encoding="utf-8")
    stop = SESSION / "STOP"
    while not (stop.exists() and stop.stat().st_mtime > started):
        pending = sorted(p for p in CELLS.glob("*.py") if not p.with_suffix(".out").exists())
        if not pending:
            time.sleep(0.3)
            continue
        cell = pending[0]
        code = cell.read_text(encoding="utf-8")
        buffer = io.StringIO()
        t0 = time.time()
        before = s.state["passes_used"]
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            try:
                exec(compile(code, cell.name, "exec"), ns)
            except Exception:
                traceback.print_exc()
        s.state["turn"] += 1
        s._save_state()
        output = buffer.getvalue()
        footer = (f"[turn {s.state['turn']} | {time.time() - t0:.1f}s | passes this cell "
                  f"{s.state['passes_used'] - before}, used {s.state['passes_used']}/{s.budget} | {s.device}]")
        cell.with_suffix(".out").write_text(output + "\n" + footer + "\n", encoding="utf-8")
        shown = output if len(output) < 20000 else output[:20000] + "\n... (truncated in the log; full in the .out)"
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"## Turn {s.state['turn']} ({time.strftime('%H:%M:%S')}, {cell.name})\n\n"
                      f"```python\n{code.rstrip()}\n```\n\n```\n{shown.rstrip()}\n{footer}\n```\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
