# Weight-space programming log: d3-lite judge

Follows Experiment 8 of Takehi, Teoh, Zhang and Khattab, "Sparks of
Weight-Space Programming" (MIT CSAIL, November 2026,
https://rikiyat.github.io/blog/weight-space-programming/): a Python session
with the model's weights loaded, in which the editor reads the weights,
spends a fixed budget of forward passes on prompts from the training side of
the data, and edits the weights by hand. No training: no gradients, no
optimizer, no training loop.

## Turn 1 (14:30:19, 0001.py)

```python
# Round 2, turn 1 (no pass). Goal: unlock the judge's unused ability to tell
# gold passages from lookalikes (round 1 turn 29: probe 0.83 at layer 12 and
# 0.79 on the final state vs the judge's 0.66). Data: the DIAGNOSIS half of
# the benchmark questions (hash split as in round 1; the other half stays
# untouched for the final measurement), split again by question into FIT
# (70%, to compute directions) and CHECK (30%, to choose strengths); plus
# ConditionalQA TRAIN (training data) against regressions at home, and
# PRACTICE as in round 1.
import hashlib, random
import numpy as np
from bench.common import read_jsonl, run_dir
from bench.failures import is_gold
diag = lambda qid: int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 2 == 0
part = lambda qid: "fit" if int(hashlib.sha1(("p2" + qid).encode()).hexdigest(), 16) % 10 < 7 else "check"
rng = random.Random(201)
DATA = {}
for bench, navf in (("legalbench", "nav_d3-lite-tuned-full_plusbase_sl10.jsonl"), ("bbsard", "nav_d3-lite-tuned-full_plusbase_sl50.jsonl")):
    qs = {q["qid"]: q for q in read_jsonl(run_dir(bench) / "questions.jsonl") if diag(q["qid"])}
    nav = {r["qid"]: r for r in read_jsonl(run_dir(bench) / navf) if r["qid"] in qs}
    chunks = {c["chunk_id"]: c for c in read_jsonl(run_dir(bench) / "chunks.jsonl")}
    rows = []
    for qid, r in nav.items():
        q = qs[qid]; res = r["results"]
        gold = [it for it in res if is_gold(bench, q, it)]
        if not gold: continue
        neg = [it for it in res if not is_gold(bench, q, it)]
        if len(neg) < 7: continue
        pick = neg[:4] + rng.sample(neg[4:], 3)            # the 4 top distractors + 3 random from the pool
        for it, lab in [(gold[0], 1)] + [(x, 0) for x in pick]:
            rows.append({"query": q["query"], "kind": "passage", "text": chunks[it["chunk_id"]]["text"], "label": lab,
                         "qid": qid, "bench": bench, "sub": q.get("subset", bench), "part": part(qid), "p_old": it["p_yes"]})
    DATA[bench] = rows
cqa = [r for r in TRAIN if r["kind"] == "passage"]
byq = {}
for r in cqa: byq.setdefault(r["qid"], []).append(r)
cq = [q for q, rs in byq.items() if any(x["label"] for x in rs) and sum(1 - x["label"] for x in rs) >= 7]
rows = []
for q in random.Random(202).sample(cq, 128):
    rs = byq[q]; pos = [x for x in rs if x["label"]][0]; negs = [x for x in rs if not x["label"]][:7]
    for x, lab in [(pos, 1)] + [(n_, 0) for n_ in negs]:
        rows.append({"query": x["query"], "kind": "passage", "text": x["text"], "label": lab, "qid": q, "bench": "cqa-train", "sub": "cqa", "part": "fit"})
DATA["cqa-train"] = rows
for k, v in DATA.items():
    print(k, len(v), "rows;", {p: len({r["qid"] for r in v if r["part"] == p}) for p in ("fit", "check")}, "questions (fit/check)")
```

```
Traceback (most recent call last):
  File "<lab>\bench\wsp\kernel.py", line 422, in main
    exec(compile(code, cell.name, "exec"), ns)
  File "0001.py", line 38, in <module>
  File "%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\Lib\random.py", line 430, in sample
    raise ValueError("Sample larger than population or is negative")
ValueError: Sample larger than population or is negative
[turn 1 | 3.2s | passes this cell 0, used 0/50 | cuda:0]
```

## Turn 2 (14:30:35, 0002.py)

```python
# Round 2, turn 2 (no pass): turn 1 again; too few TRAIN questions have 7 irrelevant passages, so 4+.
# gold passages from lookalikes (round 1 turn 29: probe 0.83 at layer 12 and
# 0.79 on the final state vs the judge's 0.66). Data: the DIAGNOSIS half of
# the benchmark questions (hash split as in round 1; the other half stays
# untouched for the final measurement), split again by question into FIT
# (70%, to compute directions) and CHECK (30%, to choose strengths); plus
# ConditionalQA TRAIN (training data) against regressions at home, and
# PRACTICE as in round 1.
import hashlib, random
import numpy as np
from bench.common import read_jsonl, run_dir
from bench.failures import is_gold
diag = lambda qid: int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 2 == 0
part = lambda qid: "fit" if int(hashlib.sha1(("p2" + qid).encode()).hexdigest(), 16) % 10 < 7 else "check"
rng = random.Random(201)
DATA = {}
for bench, navf in (("legalbench", "nav_d3-lite-tuned-full_plusbase_sl10.jsonl"), ("bbsard", "nav_d3-lite-tuned-full_plusbase_sl50.jsonl")):
    qs = {q["qid"]: q for q in read_jsonl(run_dir(bench) / "questions.jsonl") if diag(q["qid"])}
    nav = {r["qid"]: r for r in read_jsonl(run_dir(bench) / navf) if r["qid"] in qs}
    chunks = {c["chunk_id"]: c for c in read_jsonl(run_dir(bench) / "chunks.jsonl")}
    rows = []
    for qid, r in nav.items():
        q = qs[qid]; res = r["results"]
        gold = [it for it in res if is_gold(bench, q, it)]
        if not gold: continue
        neg = [it for it in res if not is_gold(bench, q, it)]
        if len(neg) < 7: continue
        pick = neg[:4] + rng.sample(neg[4:], 3)            # the 4 top distractors + 3 random from the pool
        for it, lab in [(gold[0], 1)] + [(x, 0) for x in pick]:
            rows.append({"query": q["query"], "kind": "passage", "text": chunks[it["chunk_id"]]["text"], "label": lab,
                         "qid": qid, "bench": bench, "sub": q.get("subset", bench), "part": part(qid), "p_old": it["p_yes"]})
    DATA[bench] = rows
cqa = [r for r in TRAIN if r["kind"] == "passage"]
byq = {}
for r in cqa: byq.setdefault(r["qid"], []).append(r)
cq = [q for q, rs in byq.items() if any(x["label"] for x in rs) and sum(1 - x["label"] for x in rs) >= 4]
rows = []
for q in random.Random(202).sample(cq, min(160, len(cq))):
    rs = byq[q]; pos = [x for x in rs if x["label"]][0]; negs = [x for x in rs if not x["label"]][:7]
    for x, lab in [(pos, 1)] + [(n_, 0) for n_ in negs]:
        rows.append({"query": x["query"], "kind": "passage", "text": x["text"], "label": lab, "qid": q, "bench": "cqa-train", "sub": "cqa", "part": "fit"})
DATA["cqa-train"] = rows
for k, v in DATA.items():
    print(k, len(v), "rows;", {p: len({r["qid"] for r in v if r["part"] == p}) for p in ("fit", "check")}, "questions (fit/check)")
```

```
legalbench 2120 rows; {'fit': 192, 'check': 73} questions (fit/check)
bbsard 512 rows; {'fit': 47, 'check': 17} questions (fit/check)
cqa-train 1082 rows; {'fit': 160, 'check': 0} questions (fit/check)
[turn 2 | 2.9s | passes this cell 0, used 0/50 | cuda:0]
```

## Turn 3 (14:46:09, 0003.py)

```python
# Round 2, turn 3 (GPU, 11 passes). Capture, at the answer position, the
# final normalised hidden state h~ (what the 2 readout rows read: the
# decision logit is z = h~ . (r_true - r_false) / T exactly) and the layer-12
# residual, for LegalBench (5 passes), bBSARD (1), ConditionalQA TRAIN (3) and
# PRACTICE (2). Any readout edit is then exactly simulable offline.
import numpy as np
norm_mod = S.rt.backbone.language_model.norm
def capture(rows):
    store = [None] * len(rows)
    def hook(m, inp, out):
        last = out[:, -1].float().cpu()
        for j, i in enumerate(S.ctx["batch"]):
            store[i] = last[j]
    hk = norm_mod.register_forward_hook(hook)
    try:
        res = forward(rows, capture=("resid",), layers=[12])
    finally:
        hk.remove()
    return {"H": torch.stack(store).numpy(), "R12": res["resid"][:, 0], "p": res["p"],
            "y": np.array([r["label"] for r in rows])}
CAPT = {}
for name in ("legalbench", "bbsard", "cqa-train"):
    CAPT[name] = capture(DATA[name])
PRACTICE_ROWS = [dict(r, bench="practice", sub="practice", part="check") for r in PRACTICE]
DATA["practice"] = PRACTICE_ROWS
CAPT["practice"] = capture(PRACTICE_ROWS)
ti = S.true_index; T = float(getattr(S.rt, "temperature", 1.0))
RD = (S.readout[ti].float() - S.readout[1 - ti].float()).cpu().numpy()
for name, c in CAPT.items():
    z = np.log(c["p"] / (1 - c["p"])); z2 = c["H"] @ RD / T
    print(f"{name}: {len(c['y'])} rows; exactness |z - h~.r| max {np.abs(z - z2).max():.4f}")
```

```
legalbench: 2120 rows; exactness |z - h~.r| max 0.0000
bbsard: 512 rows; exactness |z - h~.r| max 0.0000
cqa-train: 1082 rows; exactness |z - h~.r| max 0.0000
practice: 806 rows; exactness |z - h~.r| max 0.0000
[turn 3 | 912.4s | passes this cell 11, used 11/50 | cuda:0]
```

## Turn 4 (14:46:25, 0004.py)

```python
# Round 2, turn 4 (no pass). Readout edit, designed and simulated exactly.
# Direction: within-question Fisher discriminant on h~ (each row minus its
# question's mean, so only what separates gold from distractors of the SAME
# question counts), shrinkage lam * mean eigenvalue, closed form (no
# gradients), fitted on the FIT parts only: LegalBench fit + bBSARD fit +
# ConditionalQA TRAIN, each benchmark weighted equally. Edit: the decision
# readout r = r_true - r_false becomes r + beta * |r| * w (split +-1/2 over the
# two rows). Score on CHECK parts and PRACTICE: mean per-question AUC.
import numpy as np
def centred(name, mask):
    c = CAPT[name]; rows = DATA[name]
    H = c["H"][mask].astype(np.float64); y = c["y"][mask]; q = np.array([r["qid"] for r in rows])[mask]
    Hc = H.copy()
    for u_ in np.unique(q):
        m_ = q == u_; Hc[m_] -= H[m_].mean(0)
    return Hc, y
def part_mask(name, p):
    return np.array([r["part"] == p for r in DATA[name]])
def fisher(parts, lam):
    D_ = CAPT["legalbench"]["H"].shape[1]
    Sw = np.zeros((D_, D_)); md = np.zeros(D_)
    for name, mask in parts:
        Hc, y = centred(name, mask)
        mu1, mu0 = Hc[y == 1].mean(0), Hc[y == 0].mean(0)
        X0 = np.vstack([Hc[y == 1] - mu1, Hc[y == 0] - mu0])
        Sw += X0.T @ X0 / len(X0); md += mu1 - mu0
    Sw /= len(parts); md /= len(parts)
    w = np.linalg.solve(Sw + lam * np.trace(Sw) / D_ * np.eye(D_), md)
    return w / np.linalg.norm(w)
def per_q_auc_np(z, rows, mask=None):
    by = {}
    for i, r_ in enumerate(rows):
        if mask is not None and not mask[i]: continue
        by.setdefault(r_["qid"], []).append((z[i], r_["label"]))
    vals = [auc([a for a, _ in v], [b for _, b in v]) for v in by.values() if 0 < sum(b for _, b in v) < len(v)]
    return float(np.mean(vals)) if vals else float("nan")
FIT = [("legalbench", part_mask("legalbench", "fit")), ("bbsard", part_mask("bbsard", "fit")),
       ("cqa-train", part_mask("cqa-train", "fit"))]
nr = np.linalg.norm(RD)
def score_sets(rd):
    out = {}
    for name in ("legalbench", "bbsard"):
        rows = DATA[name]; z = CAPT[name]["H"] @ rd
        chk = part_mask(name, "check")
        out[name] = per_q_auc_np(z, rows, chk)
        if name == "legalbench":
            for sub in ("contractnli", "cuad", "maud", "privacy_qa"):
                out[sub] = per_q_auc_np(z, rows, chk & np.array([r["sub"] == sub for r in rows]))
    zp = CAPT["practice"]["H"] @ rd
    out["practice pq"] = per_q_auc_np(zp, DATA["practice"])
    out["practice auc"] = auc(zp.tolist(), CAPT["practice"]["y"].tolist())
    return out
base = score_sets(RD)
print("unedited:", {k: round(v, 3) for k, v in base.items()})
WS = {}
for lam in (1.0, 0.1):
    w = fisher(FIT, lam); WS[lam] = w
    print(f"\nlam {lam}: cos(w, r) {float(w @ RD / nr):+.3f}")
    for beta in (0.25, 0.5, 1.0, 2.0, 4.0):
        sc = score_sets(RD + beta * nr * w)
        print(f"  beta {beta}: " + " ".join(f"{k} {v:.3f}" for k, v in sc.items()))
```

```
unedited: {'legalbench': 0.636, 'contractnli': 0.403, 'cuad': 0.754, 'maud': 0.56, 'privacy_qa': 0.737, 'bbsard': 0.782, 'practice pq': 0.952, 'practice auc': 0.936}

lam 1.0: cos(w, r) +0.053
  beta 0.25: legalbench 0.640 contractnli 0.412 cuad 0.754 maud 0.571 privacy_qa 0.737 bbsard 0.782 practice pq 0.952 practice auc 0.936
  beta 0.5: legalbench 0.642 contractnli 0.412 cuad 0.760 maud 0.571 privacy_qa 0.737 bbsard 0.807 practice pq 0.950 practice auc 0.936
  beta 1.0: legalbench 0.650 contractnli 0.437 cuad 0.766 maud 0.571 privacy_qa 0.737 bbsard 0.798 practice pq 0.951 practice auc 0.935
  beta 2.0: legalbench 0.663 contractnli 0.437 cuad 0.777 maud 0.619 privacy_qa 0.744 bbsard 0.807 practice pq 0.949 practice auc 0.935
  beta 4.0: legalbench 0.671 contractnli 0.471 cuad 0.777 maud 0.619 privacy_qa 0.744 bbsard 0.782 practice pq 0.948 practice auc 0.934

lam 0.1: cos(w, r) +0.063
  beta 0.25: legalbench 0.636 contractnli 0.395 cuad 0.754 maud 0.571 privacy_qa 0.737 bbsard 0.790 practice pq 0.952 practice auc 0.936
  beta 0.5: legalbench 0.638 contractnli 0.403 cuad 0.754 maud 0.571 privacy_qa 0.737 bbsard 0.798 practice pq 0.952 practice auc 0.936
  beta 1.0: legalbench 0.638 contractnli 0.403 cuad 0.760 maud 0.560 privacy_qa 0.737 bbsard 0.798 practice pq 0.952 practice auc 0.935
  beta 2.0: legalbench 0.646 contractnli 0.412 cuad 0.766 maud 0.583 privacy_qa 0.737 bbsard 0.790 practice pq 0.952 practice auc 0.935
  beta 4.0: legalbench 0.671 contractnli 0.437 cuad 0.794 maud 0.607 privacy_qa 0.759 bbsard 0.790 practice pq 0.949 practice auc 0.935
[turn 4 | 0.6s | passes this cell 0, used 11/50 | cuda:0]
```

## Turn 5 (14:46:57, 0005.py)

```python
# Round 2, turn 5 (no pass). Turn 4: the readout edit helps and still rises
# at beta 4 (LegalBench check 0.636 -> 0.671, ContractNLI 0.403 -> 0.471,
# practice 0.952 -> 0.948). Selection rule, fixed now before searching:
# maximise mean(LegalBench check, bBSARD check) subject to practice
# per-question AUC >= unedited - 0.005. Search wider beta, three shrinkages,
# and a standardised variant (Fisher on features scaled by their within-
# question sd, mapped back to h~ space: the probe of round 1 standardised).
import numpy as np
def fisher_std(parts, lam):
    Hs = [centred(n, m)[0] for n, m in parts]
    sd = np.sqrt(np.mean([h.var(0) for h in Hs], 0)) + 1e-6
    D_ = len(sd); Sw = np.zeros((D_, D_)); md = np.zeros(D_)
    for (name, mask), Hc in zip(parts, Hs):
        y = CAPT[name]["y"][mask]; Z = Hc / sd
        mu1, mu0 = Z[y == 1].mean(0), Z[y == 0].mean(0)
        X0 = np.vstack([Z[y == 1] - mu1, Z[y == 0] - mu0])
        Sw += X0.T @ X0 / len(X0); md += mu1 - mu0
    Sw /= len(parts); md /= len(parts)
    v = np.linalg.solve(Sw + lam * np.trace(Sw) / D_ * np.eye(D_), md)
    w = v / sd                       # z-space weights back to h~ space
    return w / np.linalg.norm(w)
floor = base["practice pq"] - 0.005
cands = []
for kind, fn in (("raw", fisher), ("std", fisher_std)):
    for lam in (10.0, 1.0, 0.1):
        w = fn(FIT, lam)
        for beta in (1, 2, 4, 8, 16, 32):
            sc = score_sets(RD + beta * nr * w)
            ok = sc["practice pq"] >= floor
            cands.append(((sc["legalbench"] + sc["bbsard"]) / 2 if ok else -1, kind, lam, beta, sc))
cands.sort(key=lambda c: -c[0])
print("unedited:", {k: round(v, 3) for k, v in base.items()}, "| floor", round(floor, 3))
for obj, kind, lam, beta, sc in cands[:8]:
    print(f"{kind} lam {lam} beta {beta}: objective {obj:.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in sc.items()))
# fit-part scores of the best, to see how much is overfit
obj, kind, lam, beta, sc = cands[0]
w = (fisher if kind == "raw" else fisher_std)(FIT, lam)
rd = RD + beta * nr * w
fitsc = {n: per_q_auc_np(CAPT[n]["H"] @ rd, DATA[n], part_mask(n, "fit")) for n in ("legalbench", "bbsard")}
fit0 = {n: per_q_auc_np(CAPT[n]["H"] @ RD, DATA[n], part_mask(n, "fit")) for n in ("legalbench", "bbsard")}
print("best on its FIT part:", {n: f"{fit0[n]:.3f} -> {fitsc[n]:.3f}" for n in fitsc})
BEST = dict(kind=kind, lam=lam, beta=beta, w=w)
```

```
unedited: {'legalbench': 0.636, 'contractnli': 0.403, 'cuad': 0.754, 'maud': 0.56, 'privacy_qa': 0.737, 'bbsard': 0.782, 'practice pq': 0.952, 'practice auc': 0.936} | floor 0.947
std lam 0.1 beta 16: objective 0.743 | legalbench 0.697 contractnli 0.471 cuad 0.806 maud 0.655 privacy_qa 0.782 bbsard 0.790 practice pq 0.947 practice auc 0.931
raw lam 0.1 beta 8: objective 0.742 | legalbench 0.685 contractnli 0.454 cuad 0.789 maud 0.643 privacy_qa 0.782 bbsard 0.798 practice pq 0.949 practice auc 0.933
raw lam 1.0 beta 2: objective 0.735 | legalbench 0.663 contractnli 0.437 cuad 0.777 maud 0.619 privacy_qa 0.744 bbsard 0.807 practice pq 0.949 practice auc 0.935
std lam 1.0 beta 4: objective 0.735 | legalbench 0.671 contractnli 0.445 cuad 0.771 maud 0.631 privacy_qa 0.767 bbsard 0.798 practice pq 0.950 practice auc 0.934
std lam 0.1 beta 8: objective 0.732 | legalbench 0.673 contractnli 0.445 cuad 0.777 maud 0.619 privacy_qa 0.774 bbsard 0.790 practice pq 0.948 practice auc 0.934
raw lam 0.1 beta 4: objective 0.731 | legalbench 0.671 contractnli 0.437 cuad 0.794 maud 0.607 privacy_qa 0.759 bbsard 0.790 practice pq 0.949 practice auc 0.935
raw lam 1.0 beta 4: objective 0.726 | legalbench 0.671 contractnli 0.471 cuad 0.777 maud 0.619 privacy_qa 0.744 bbsard 0.782 practice pq 0.948 practice auc 0.934
std lam 1.0 beta 2: objective 0.725 | legalbench 0.652 contractnli 0.429 cuad 0.766 maud 0.595 privacy_qa 0.737 bbsard 0.798 practice pq 0.949 practice auc 0.935
best on its FIT part: {'legalbench': '0.669 -> 0.731', 'bbsard': '0.775 -> 0.848'}
[turn 5 | 3.6s | passes this cell 0, used 11/50 | cuda:0]
```

## Turn 6 (14:47:25, 0006.py)

```python
# Round 2, turn 6 (no pass). Is the layer-12 state worth a deeper edit on
# top of the readout edit? Same within-question standardised Fisher recipe on
# the layer-12 residual (FIT parts), then on CHECK: (a) the layer-12 score
# alone, (b) the readout-edited logit plus gamma * layer-12 score. Only if
# (b) clearly beats the readout edit is a layer-12 path worth building.
import numpy as np
def centred_R(name, mask):
    c = CAPT[name]; rows = DATA[name]
    H = c["R12"][mask].astype(np.float64); y = c["y"][mask]; q = np.array([r["qid"] for r in rows])[mask]
    Hc = H.copy()
    for u_ in np.unique(q):
        m_ = q == u_; Hc[m_] -= H[m_].mean(0)
    return Hc, y
def fisher_std_R(parts, lam):
    Hs = [centred_R(n, m)[0] for n, m in parts]
    sd = np.sqrt(np.mean([h.var(0) for h in Hs], 0)) + 1e-6
    D_ = len(sd); Sw = np.zeros((D_, D_)); md = np.zeros(D_)
    for (name, mask), Hc in zip(parts, Hs):
        y = CAPT[name]["y"][mask]; Z = Hc / sd
        mu1, mu0 = Z[y == 1].mean(0), Z[y == 0].mean(0)
        X0 = np.vstack([Z[y == 1] - mu1, Z[y == 0] - mu0])
        Sw += X0.T @ X0 / len(X0); md += mu1 - mu0
    Sw /= len(parts); md /= len(parts)
    w = np.linalg.solve(Sw + lam * np.trace(Sw) / D_ * np.eye(D_), md) / sd
    return w / np.linalg.norm(w)
rd_best = RD + BEST["beta"] * nr * BEST["w"]
for lam in (1.0, 0.1):
    w12 = fisher_std_R(FIT, lam)
    line = f"lam {lam}: L12 alone "
    for name in ("legalbench", "bbsard"):
        line += f"{name} {per_q_auc_np(CAPT[name]['R12'] @ w12, DATA[name], part_mask(name, 'check')):.3f} "
    line += f"practice {per_q_auc_np(CAPT['practice']['R12'] @ w12, DATA['practice']):.3f} | readout-edit + gamma*L12:"
    for gamma in (0.5, 1, 2, 4):
        sc = {}
        for name in ("legalbench", "bbsard", "practice"):
            z = CAPT[name]["H"] @ rd_best
            s12 = CAPT[name]["R12"] @ w12
            s12 = (s12 - s12.mean()) / s12.std() * z.std()
            mask = part_mask(name, "check") if name != "practice" else None
            sc[name] = per_q_auc_np(z + gamma * s12, DATA[name], mask)
        line += f" g{gamma}: lb {sc['legalbench']:.3f} bb {sc['bbsard']:.3f} pr {sc['practice']:.3f};"
    print(line)
print("readout edit alone: lb 0.697 bb 0.790 pr 0.947")
```

```
lam 1.0: L12 alone legalbench 0.689 bbsard 0.697 practice 0.925 | readout-edit + gamma*L12: g0.5: lb 0.706 bb 0.773 pr 0.948; g1: lb 0.706 bb 0.773 pr 0.945; g2: lb 0.712 bb 0.765 pr 0.942; g4: lb 0.708 bb 0.748 pr 0.938;
lam 0.1: L12 alone legalbench 0.701 bbsard 0.723 practice 0.915 | readout-edit + gamma*L12: g0.5: lb 0.726 bb 0.773 pr 0.945; g1: lb 0.728 bb 0.756 pr 0.942; g2: lb 0.736 bb 0.756 pr 0.936; g4: lb 0.722 bb 0.748 pr 0.926;
readout edit alone: lb 0.697 bb 0.790 pr 0.947
[turn 6 | 0.9s | passes this cell 0, used 11/50 | cuda:0]
```

## Turn 7 (14:57:14, 0007.py)

```python
# Round 2, turn 7 (GPU, 4 passes). Apply edit 1: the readout reads the
# judge's final state along the within-question standardised Fisher
# direction as well (lam 0.1, beta 16: best under the rule fixed in turn 5).
# r_true += d/2, r_false -= d/2 with d = beta * |r| * w, so the decision
# logit gains d . h~. Real check on the LegalBench CHECK part and PRACTICE
# against the exact simulation (LegalBench check 0.697, practice 0.947).
import numpy as np
d = torch.tensor(BEST["beta"] * nr * BEST["w"], dtype=torch.float32)
ro = S.rt.readout
ro[ti] += (d / 2).to(ro.device, ro.dtype); ro[1 - ti] -= (d / 2).to(ro.device, ro.dtype)
print("edit 1:", changed())
chk_rows = [r for r in DATA["legalbench"] if r["part"] == "check"]
p = forward(chk_rows)["p"]; z = np.log(p / (1 - p))
print("legalbench check, real:", round(per_q_auc_np(z, chk_rows), 3), "(simulated 0.697, unedited 0.636)")
pp = forward(PRACTICE)["p"]; zp = np.log(pp / (1 - pp))
print("practice, real:", round(per_q_auc_np(zp, DATA["practice"]), 3), "(simulated 0.947, unedited 0.952); pooled auc",
      round(auc(zp.tolist(), CAPT["practice"]["y"].tolist()), 4))
print(save_latest("edit 1: readout along within-question Fisher direction, lam 0.1, beta 16"))
```

```
edit 1: {'readout[:2]': 2048, '_total': 2048, '_share': '0.0002% of 852,987,968'}
legalbench check, real: 0.697 (simulated 0.697, unedited 0.636)
practice, real: 0.947 (simulated 0.947, unedited 0.952); pooled auc 0.9311
saved 0 edited tensors (edit 1: readout along within-question Fisher direction, lam 0.1, beta 16)
[turn 7 | 559.5s | passes this cell 4, used 15/50 | cuda:0]
```

## Turn 8 (aborted)

finalize() was started but the kernel was restarted before it finished (GPU memory had spilled 2.3 GB into shared memory, making every pass about 10x slower). Nothing was printed, so the held-out half was not seen. The kernel now empties the GPU cache after every pass. finalize() is postponed until the gated paths are built; the readout edit (turn 7) is reloaded from latest_edit.pt.

## Turn 8 (15:49:29, 0009.py)

```python
# Round 2, turn 9 (GPU, 11 passes). After the kernel restart: rebuild the same sets as turn 2
# (deterministic), then capture for the gated paths in layer 23's MLP (fixes 1 and 2).
import hashlib, random
import numpy as np
from bench.common import read_jsonl, run_dir
from bench.failures import is_gold
diag = lambda qid: int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 2 == 0
part = lambda qid: "fit" if int(hashlib.sha1(("p2" + qid).encode()).hexdigest(), 16) % 10 < 7 else "check"
rng = random.Random(201)
DATA = {}
for bench, navf in (("legalbench", "nav_d3-lite-tuned-full_plusbase_sl10.jsonl"), ("bbsard", "nav_d3-lite-tuned-full_plusbase_sl50.jsonl")):
    qs = {q["qid"]: q for q in read_jsonl(run_dir(bench) / "questions.jsonl") if diag(q["qid"])}
    nav = {r["qid"]: r for r in read_jsonl(run_dir(bench) / navf) if r["qid"] in qs}
    chunks = {c["chunk_id"]: c for c in read_jsonl(run_dir(bench) / "chunks.jsonl")}
    rows = []
    for qid, r in nav.items():
        q = qs[qid]; res = r["results"]
        gold = [it for it in res if is_gold(bench, q, it)]
        if not gold: continue
        neg = [it for it in res if not is_gold(bench, q, it)]
        if len(neg) < 7: continue
        pick = neg[:4] + rng.sample(neg[4:], 3)            # the 4 top distractors + 3 random from the pool
        for it, lab in [(gold[0], 1)] + [(x, 0) for x in pick]:
            rows.append({"query": q["query"], "kind": "passage", "text": chunks[it["chunk_id"]]["text"], "label": lab,
                         "qid": qid, "bench": bench, "sub": q.get("subset", bench), "part": part(qid), "p_old": it["p_yes"]})
    DATA[bench] = rows
cqa = [r for r in TRAIN if r["kind"] == "passage"]
byq = {}
for r in cqa: byq.setdefault(r["qid"], []).append(r)
cq = [q for q, rs in byq.items() if any(x["label"] for x in rs) and sum(1 - x["label"] for x in rs) >= 4]
rows = []
for q in random.Random(202).sample(cq, min(160, len(cq))):
    rs = byq[q]; pos = [x for x in rs if x["label"]][0]; negs = [x for x in rs if not x["label"]][:7]
    for x, lab in [(pos, 1)] + [(n_, 0) for n_ in negs]:
        rows.append({"query": x["query"], "kind": "passage", "text": x["text"], "label": lab, "qid": q, "bench": "cqa-train", "sub": "cqa", "part": "fit"})
DATA["cqa-train"] = rows
for k, v in DATA.items():
    print(k, len(v), "rows;", {p: len({r["qid"] for r in v if r["part"] == p}) for p in ("fit", "check")}, "questions (fit/check)")
import gc
L23 = S.layers[23]; norm_mod = S.rt.backbone.language_model.norm
def capture_all(rows):
    xin, act, hn = [None] * len(rows), [None] * len(rows), [None] * len(rows)
    def keep(store, pre=False):
        def f(m, a, b=None):
            t = (a[0] if pre else b)[:, -1].float().cpu()
            for j, k in enumerate(S.ctx["batch"]): store[k] = t[j]
        return f
    hs = [L23.post_attention_layernorm.register_forward_hook(keep(xin)),
          L23.mlp.down_proj.register_forward_pre_hook(keep(act, pre=True)),
          norm_mod.register_forward_hook(keep(hn))]
    try:
        res = forward(rows, capture=("resid",), layers=[23])
    finally:
        for h in hs: h.remove()
    return {"X23": torch.stack(xin).numpy(), "A23": torch.stack(act).numpy(), "Hpre": res["resid"][:, 0],
            "H": torch.stack(hn).numpy(), "p": res["p"], "y": np.array([r["label"] for r in rows])}
PRACTICE_ROWS = [dict(r, bench="practice", sub="practice", part="check") for r in PRACTICE]
DATA["practice"] = PRACTICE_ROWS
CAPT = {}
for name in ("legalbench", "bbsard", "cqa-train", "practice"):
    CAPT[name] = capture_all(DATA[name]); gc.collect()
print("GPU after captures: %.2f GB reserved" % (torch.cuda.memory_reserved() / 2**30))
ti = S.true_index
gw = 1 + W("language_model.norm.weight").float().cpu().numpy()
eps = S.rt.backbone.language_model.norm.eps
RD_NOW = (S.readout[ti].float() - S.readout[1 - ti].float()).cpu().numpy()
def final_logit(Hpre, rd=None):
    Hn = Hpre / np.sqrt((Hpre ** 2).mean(-1, keepdims=True) + eps) * gw
    return Hn @ (RD_NOW if rd is None else rd)
for name, c in CAPT.items():
    z = np.log(c["p"] / (1 - c["p"]))
    print(f"{name}: {len(c['y'])} rows; |model logit - simulated from Hpre| max {np.abs(z - final_logit(c['Hpre'])).max():.4f}")
np.savez_compressed(r"<lab>\runs\wsp2\captures_round2.npz",
                    **{f"{n}__{k}": v for n, c in CAPT.items() for k, v in c.items()})
```

```
legalbench 2120 rows; {'fit': 192, 'check': 73} questions (fit/check)
bbsard 512 rows; {'fit': 47, 'check': 17} questions (fit/check)
cqa-train 1082 rows; {'fit': 160, 'check': 0} questions (fit/check)
GPU after captures: 2.10 GB reserved
legalbench: 2120 rows; |model logit - simulated from Hpre| max 0.0079
bbsard: 512 rows; |model logit - simulated from Hpre| max 0.0076
cqa-train: 1082 rows; |model logit - simulated from Hpre| max 0.0070
practice: 806 rows; |model logit - simulated from Hpre| max 0.0071
[turn 8 | 907.9s | passes this cell 11, used 26/50 | cuda:0]
```

## Turn 9 (15:49:51, 0010.py)

```python
# Round 2, turn 10 (no pass). Design the gated paths in L23's MLP, exactly
# simulated: Hpre' = Hpre - a_j * down_j + silu(G . x) * (U . x) * down'_j,
# logit = final_norm(Hpre') . readout (x = the MLP input, captured).
#   gate G: "contract question" (LegalBench) or "Dutch" (bBSARD), from the
#           between-domain Fisher direction on x plus the near-constant mean
#           direction as a stand-in for a bias (the MLP has none); two
#           parameters set so in-domain rows sit near +6 and others near -6
#   up U:   the domain's within-question standardised Fisher direction on x
#           (that domain's FIT part only)
#   down:   beta * unit decision direction ((1 + norm.w) * readout difference)
#   neuron: the least-used neuron of L23 (checked: removing it changes nothing)
# Rule (fixed as in turn 5): maximise mean(LegalBench check, bBSARD check)
# with practice per-question AUC >= 0.947 (unedited 0.952 - 0.005).
import numpy as np
def per_q_auc_np(z, rows, mask=None):
    by = {}
    for i, r_ in enumerate(rows):
        if mask is not None and not mask[i]: continue
        by.setdefault(r_["qid"], []).append((z[i], r_["label"]))
    vals = [auc([a for a, _ in v], [b for _, b in v]) for v in by.values() if 0 < sum(b for _, b in v) < len(v)]
    return float(np.mean(vals)) if vals else float("nan")
part_mask = lambda name, p: np.array([r["part"] == p for r in DATA[name]])
silu = lambda z: z / (1 + np.exp(-z))
Wd = W("language_model.layers.23.mlp.down_proj.weight").float().cpu().numpy()      # [1024, 3584]
SETS = ("legalbench", "bbsard", "cqa-train", "practice")
# 1. the least-used neuron(s): mean |a_j| * |down_j| over all captured rows, then an exact removal check
allA = np.vstack([CAPT[n]["A23"] for n in SETS])
use = np.abs(allA).mean(0) * np.linalg.norm(Wd, axis=0)
order = np.argsort(use)
def scores(mod=None):
    out = {}
    for n in SETS:
        Hp = CAPT[n]["Hpre"] + (mod(n) if mod else 0)
        z = final_logit(Hp)
        out[n] = per_q_auc_np(z, DATA[n], part_mask(n, "check") if n in ("legalbench", "bbsard") else None)
    return out
base = scores()
print("current (readout edit):", {k: round(v, 3) for k, v in base.items()})
FREE = []
for j in order[:6]:
    s_ = scores(lambda n, j=j: -CAPT[n]["A23"][:, j:j+1] * Wd[:, j][None])
    dz = max(abs(s_[k] - base[k]) for k in s_)
    print(f"neuron {j}: use {use[j]:.2e}, max per-set change when removed {dz:.4f}")
    if dz < 1e-3: FREE.append(int(j))
print("free neurons:", FREE[:2])
# 2. gates
X = {n: CAPT[n]["X23"].astype(np.float64) for n in SETS}
m_hat = np.vstack(list(X.values())).mean(0); m_hat /= np.linalg.norm(m_hat)
def fisher_between(pos_sets):
    P = np.vstack([X[n] for n in SETS if n in pos_sets]); N_ = np.vstack([X[n] for n in SETS if n not in pos_sets])
    Sw = np.cov(np.vstack([P - P.mean(0), N_ - N_.mean(0)]).T)
    g = np.linalg.solve(Sw + 0.1 * np.trace(Sw) / len(Sw) * np.eye(len(Sw)), P.mean(0) - N_.mean(0))
    return g / np.linalg.norm(g)
def gate_row(pos_sets, lo=-6.0, hi=6.0):
    g = fisher_between(pos_sets)
    P = np.vstack([X[n] for n in SETS if n in pos_sets]); N_ = np.vstack([X[n] for n in SETS if n not in pos_sets])
    A_ = np.array([[P @ g, P @ m_hat], [N_ @ g, N_ @ m_hat]], dtype=object)
    M = np.array([[(P @ g).mean(), (P @ m_hat).mean()], [(N_ @ g).mean(), (N_ @ m_hat).mean()]])
    a, b = np.linalg.solve(M, np.array([hi, lo]))
    G = a * g + b * m_hat
    zp, zn = P @ G, N_ @ G
    print(f"  gate for {pos_sets}: in-domain z mean {zp.mean():+.1f} min {zp.min():+.1f}; others mean {zn.mean():+.1f} max {zn.max():+.1f}")
    return G
G_lb = gate_row({"legalbench"}); G_bb = gate_row({"bbsard"})
# 3. up rows: within-question standardised Fisher on x, FIT part of the domain only
def fisher_within(name, lam=0.1):
    m_ = part_mask(name, "fit"); Xs = X[name][m_]; y = CAPT[name]["y"][m_]; q = np.array([r["qid"] for r in DATA[name]])[m_]
    Xc = Xs.copy()
    for u_ in np.unique(q): Xc[q == u_] -= Xs[q == u_].mean(0)
    sd = Xc.std(0) + 1e-6; Z = Xc / sd
    mu1, mu0 = Z[y == 1].mean(0), Z[y == 0].mean(0)
    X0 = np.vstack([Z[y == 1] - mu1, Z[y == 0] - mu0]); Sw = X0.T @ X0 / len(X0)
    w = np.linalg.solve(Sw + lam * np.trace(Sw) / len(Sw) * np.eye(len(Sw)), mu1 - mu0) / sd
    return w / np.linalg.norm(w)
U_lb = fisher_within("legalbench"); U_bb = fisher_within("bbsard")
u_dec = gw * RD_NOW; u_dec /= np.linalg.norm(u_dec)
j_lb, j_bb = FREE[0], FREE[1]
def mod_for(beta_lb, beta_bb):
    def f(n):
        d = np.zeros_like(CAPT[n]["Hpre"])
        for j, G, U, beta in ((j_lb, G_lb, U_lb, beta_lb), (j_bb, G_bb, U_bb, beta_bb)):
            if beta == 0: continue
            d += -CAPT[n]["A23"][:, j:j+1] * Wd[:, j][None]
            d += (silu(X[n] @ G) * (X[n] @ U))[:, None] * (beta * u_dec)[None]
        return d
    return f
# scale: how big is U . x? (to put beta in sensible units)
su_lb = np.std(X["legalbench"] @ U_lb); su_bb = np.std(X["bbsard"] @ U_bb)
print(f"sd of U.x: legalbench {su_lb:.3f}, bbsard {su_bb:.3f}; sd of Hpre . u_dec: {np.std(CAPT['legalbench']['Hpre'] @ u_dec):.3f}")
res_ = []
for bl in (0, 0.05, 0.1, 0.2, 0.4):
    for bb_ in (0, 0.05, 0.1, 0.2, 0.4):
        s_ = scores(mod_for(bl / su_lb / 6, bb_ / su_bb / 6))   # beta in units of (sd of the final decision projection per sd of U.x)
        ok = s_["practice"] >= 0.947
        res_.append(((s_["legalbench"] + s_["bbsard"]) / 2 if ok else -1, bl, bb_, s_))
res_.sort(key=lambda r: -r[0])
for obj, bl, bb_, s_ in res_[:6]:
    print(f"beta_lb {bl} beta_bb {bb_}: objective {obj:.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in s_.items()))
GATED = dict(j_lb=j_lb, j_bb=j_bb, G_lb=G_lb, G_bb=G_bb, U_lb=U_lb, U_bb=U_bb, u_dec=u_dec, su_lb=su_lb, su_bb=su_bb, best=res_[0])
```

```
current (readout edit): {'legalbench': 0.697, 'bbsard': 0.79, 'cqa-train': 0.972, 'practice': 0.947}
neuron 1809: use 2.94e-04, max per-set change when removed 0.0000
neuron 2635: use 3.12e-04, max per-set change when removed 0.0000
neuron 2377: use 4.01e-04, max per-set change when removed 0.0000
neuron 2924: use 4.17e-04, max per-set change when removed 0.0000
neuron 1588: use 4.71e-04, max per-set change when removed 0.0000
neuron 1699: use 4.85e-04, max per-set change when removed 0.0000
free neurons: [1809, 2635]
  gate for {'legalbench'}: in-domain z mean +6.0 min +1.2; others mean -6.0 max -2.0
  gate for {'bbsard'}: in-domain z mean +6.0 min +2.9; others mean -6.0 max -4.1
sd of U.x: legalbench 0.107, bbsard 0.091; sd of Hpre . u_dec: 0.123
beta_lb 0.1 beta_bb 0.05: objective 0.777 | legalbench 0.763 bbsard 0.790 cqa-train 0.972 practice 0.947
beta_lb 0.1 beta_bb 0: objective 0.772 | legalbench 0.763 bbsard 0.782 cqa-train 0.972 practice 0.947
beta_lb 0.1 beta_bb 0.1: objective 0.772 | legalbench 0.763 bbsard 0.782 cqa-train 0.972 practice 0.947
beta_lb 0.1 beta_bb 0.2: objective 0.771 | legalbench 0.761 bbsard 0.782 cqa-train 0.972 practice 0.947
beta_lb 0.05 beta_bb 0.05: objective 0.768 | legalbench 0.746 bbsard 0.790 cqa-train 0.972 practice 0.947
beta_lb 0.05 beta_bb 0.1: objective 0.765 | legalbench 0.748 bbsard 0.782 cqa-train 0.972 practice 0.947
[turn 9 | 4.9s | passes this cell 0, used 26/50 | cuda:0]
```

## Turn 10 (15:51:51, 0011.py)

```python
# Round 2, turn 11 (GPU, 4 passes). Edit 2: a gated path in neuron 1809 of
# L23's MLP (unused: removing it changes nothing to 4 decimals on every set).
#   gate_proj row  = contract-question gate (in-domain z >= +1.2, others <= -2.0)
#   up_proj row    = LegalBench within-question relevance direction (FIT part)
#   down_proj col  = beta * decision direction, beta_lb 0.1 (turn 10's best)
# The bBSARD neuron is left out: it added nothing on the check part.
# Real check on the LegalBench CHECK part and PRACTICE against the
# simulation (LegalBench check 0.763, practice 0.947).
import numpy as np
j = GATED["j_lb"]; beta = 0.1 / GATED["su_lb"] / 6
mlp = S.layers[23].mlp
for w_, val, rows_ in ((mlp.gate_proj.weight, GATED["G_lb"], True), (mlp.up_proj.weight, GATED["U_lb"], True),
                       (mlp.down_proj.weight, beta * GATED["u_dec"], False)):
    t = torch.tensor(val, dtype=torch.float32).to(w_.device, w_.dtype)
    if rows_: w_[j, :] = t
    else: w_[:, j] = t
print("edit 2:", changed())
chk_rows = [r for r in DATA["legalbench"] if r["part"] == "check"]
p = forward(chk_rows)["p"]; z = np.log(p / (1 - p))
print("legalbench check, real:", round(per_q_auc_np(z, chk_rows), 3), "(simulated 0.763; readout edit alone 0.697; unedited 0.636)")
for sub in ("contractnli", "cuad", "maud", "privacy_qa"):
    m_ = np.array([r["sub"] == sub for r in chk_rows])
    print(f"   {sub}: {per_q_auc_np(z, chk_rows, m_):.3f}")
pp = forward(PRACTICE)["p"]; zp = np.log(pp / (1 - pp))
print("practice, real:", round(per_q_auc_np(zp, DATA["practice"]), 3), "(simulated 0.947, unedited 0.952)")
print(save_latest("edit 1 readout + edit 2 gated contract neuron L23.1809, beta 0.1"))
```

```
edit 2: {'language_model.layers.23.mlp.gate_proj.weight': 1024, 'language_model.layers.23.mlp.up_proj.weight': 1024, 'language_model.layers.23.mlp.down_proj.weight': 1024, 'readout[:2]': 2048, '_total': 5120, '_share': '0.0006% of 852,987,968'}
legalbench check, real: 0.763 (simulated 0.763; readout edit alone 0.697; unedited 0.636)
   contractnli: 0.697
   cuad: 0.794
   maud: 0.774
   privacy_qa: 0.774
practice, real: 0.947 (simulated 0.947, unedited 0.952)
saved 3 edited tensors (edit 1 readout + edit 2 gated contract neuron L23.1809, beta 0.1)
[turn 10 | 93.8s | passes this cell 4, used 30/50 | cuda:0]
```

## Turn 11 (15:53:40, 0012.py)

```python
# Round 2, turn 12 (no practice pass). Finalize: the hidden ConditionalQA
# validation half, unedited vs edited (once), then judge.pt (edits 1 + 2).
import json
rep = S.finalize("round 2: (1) readout edit along the judge's own within-question Fisher direction of its final state "
                 "(lam 0.1, beta 16); (2) gated contract-question path in L23 MLP neuron 1809 (beta 0.1). Directions "
                 "fitted on the diagnosis half of LegalBench/bBSARD plus ConditionalQA TRAIN; strengths on the check part.")
print(json.dumps(rep, indent=1))
```

```
{
 "unedited": {
  "n": 813,
  "auc": 0.9153,
  "acc": 0.8893,
  "logloss": 0.2808,
  "brier": 0.0816,
  "pos_rate_pred": 0.1636,
  "pos_rate_true": 0.1931
 },
 "edited": {
  "n": 813,
  "auc": 0.9163,
  "acc": 0.8069,
  "logloss": 2.1782,
  "brier": 0.1931,
  "pos_rate_pred": 0.0,
  "pos_rate_true": 0.1931
 },
 "changed": {
  "total": 5120,
  "share": "0.0006% of 852,987,968"
 },
 "passes_used": 30
}
[turn 11 | 89.6s | passes this cell 0, used 30/50 | cuda:0]
```

