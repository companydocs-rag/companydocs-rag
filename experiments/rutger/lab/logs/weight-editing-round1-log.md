# Weight-space programming log: d3-lite judge

Follows Experiment 8 of Takehi, Teoh, Zhang and Khattab, "Sparks of
Weight-Space Programming" (MIT CSAIL, November 2026,
https://rikiyat.github.io/blog/weight-space-programming/): a Python session
with the model's weights loaded, in which the editor reads the weights,
spends a fixed budget of forward passes on prompts from the training side of
the data, and edits the weights by hand. No training: no gradients, no
optimizer, no training loop.

## Turn 1 (22:41:38, 0001.py)

```python
# Turn 1 (reading, no forward pass). Hypothesis to start from: the fine-tune
# taught the judge mainly through (a) the two readout rows and (b) LoRA updates
# that write into the yes-vs-no direction. Map where that direction is written:
# the decision direction u = (1 + final_norm.weight) * (r_true - r_false),
# because logit_true - logit_false = rmsnorm(h) . (r_true - r_false).
import torch, re
from bench.wsp.kernel import TUNED
saved = torch.load(TUNED + "/judge.pt", map_location="cpu", weights_only=True)
print("judge.pt keys:", list(saved), "r", saved["r"], "alpha", saved["alpha"])
ti = S.true_index; fi = 1 - ti
R = S.readout.float().cpu()
from safetensors.torch import load_file
base_readout = load_file(str(S.rt.root / "readout.safetensors"))["weight"][:2].float()
print("readout rows: tuned norms", R.norm(dim=1).tolist(), "base norms", base_readout.norm(dim=1).tolist())
print("cos(tuned, base) per row", torch.nn.functional.cosine_similarity(R, base_readout, dim=1).tolist())
r = R[ti] - R[fi]
g = W("language_model.norm.weight").float().cpu()
u = (1 + g) * r
print(f"decision direction: |r| {r.norm():.3f}, |u| {u.norm():.3f}, cos(r, u) {torch.nn.functional.cosine_similarity(r, u, 0):.3f}")
print("final norm (1+w): mean", float((1 + g).mean()), "max", float((1 + g).abs().max()), "top dims", (1 + g).abs().topk(5).indices.tolist())
un = u / u.norm()

# LoRA size per layer and module: |scale * B A| / |W0| (W0 = base weight = merged - delta)
scale = saved["alpha"] / saved["r"]
rows = []
for name, A in saved["lora"].items():
    if ".lora_A." not in name:
        continue
    mod = name.split(".lora_A.")[0]
    B = saved["lora"][name.replace(".lora_A.", ".lora_B.")]
    delta = (B.float() @ A.float()) * scale
    Wm = W(mod + ".weight").float().cpu()
    W0 = Wm - delta
    L = int(re.search(r"layers\.(\d+)\.", mod).group(1))
    short = mod.split(".")[-1]
    entry = {"L": L, "m": short, "rel": float(delta.norm() / W0.norm())}
    if short in ("out_proj", "o_proj", "down_proj"):
        # how much this writer's contribution along u changed: |u^T dW| vs |u^T W0|
        entry["u_write_base"] = float((un @ W0).norm())
        entry["u_write_delta"] = float((un @ delta).norm())
        entry["u_write_tuned"] = float((un @ Wm).norm())
    rows.append(entry)
by_layer = {}
for e in rows:
    by_layer.setdefault(e["L"], {})[e["m"]] = e
print("\nlayer type          rel|dW|/|W0| per module (x1000)                              writes along u: base -> tuned (delta)")
for L in sorted(by_layer):
    mods = by_layer[L]
    rel = " ".join(f"{m[:6]}={1000*e['rel']:.1f}" for m, e in sorted(mods.items()))
    w = [e for e in mods.values() if "u_write_base" in e]
    wr = "  ".join(f"{e['m'][:4]} {e['u_write_base']:.2f}->{e['u_write_tuned']:.2f} ({e['u_write_delta']:.2f})" for e in w)
    print(f"{L:2d} {S.layer_types[L][:6]:7s} {rel}\n      {wr}")
```

```
judge.pt keys: ['lora', 'readout_rows', 'targets', 'r', 'alpha', 'base'] r 16 alpha 32
readout rows: tuned norms [0.7306782603263855, 0.6874667406082153] base norms [0.733228325843811, 0.6964792013168335]
cos(tuned, base) per row [0.9988741874694824, 0.9988183379173279]
decision direction: |r| 0.629, |u| 2.732, cos(r, u) 0.992
final norm (1+w): mean 4.309151649475098 max 6.3125 top dims [713, 687, 111, 196, 160]

layer type          rel|dW|/|W0| per module (x1000)                              writes along u: base -> tuned (delta)
 0 linear  down_p=23.2 gate_p=29.7 in_pro=21.7 in_pro=21.1 out_pr=19.7 up_pro=41.2
      out_ 0.58->0.58 (0.01)  down 0.51->0.51 (0.01)
 1 linear  down_p=23.4 gate_p=26.8 in_pro=20.8 in_pro=18.3 out_pr=19.3 up_pro=40.9
      out_ 0.59->0.59 (0.01)  down 0.50->0.50 (0.01)
 2 linear  down_p=24.1 gate_p=22.3 in_pro=20.4 in_pro=18.4 out_pr=19.0 up_pro=44.1
      out_ 0.55->0.55 (0.02)  down 0.48->0.48 (0.01)
 3 full_a  down_p=24.6 gate_p=22.1 k_proj=27.0 o_proj=20.0 q_proj=17.9 up_pro=43.0 v_proj=26.3
      o_pr 0.55->0.55 (0.01)  down 0.48->0.48 (0.01)
 4 linear  down_p=24.2 gate_p=23.1 in_pro=18.4 in_pro=20.0 out_pr=21.2 up_pro=41.8
      out_ 0.54->0.54 (0.01)  down 0.46->0.46 (0.01)
 5 linear  down_p=24.2 gate_p=25.0 in_pro=18.7 in_pro=21.2 out_pr=20.6 up_pro=42.2
      out_ 0.53->0.53 (0.01)  down 0.48->0.48 (0.01)
 6 linear  down_p=23.7 gate_p=26.0 in_pro=20.4 in_pro=22.2 out_pr=20.1 up_pro=42.3
      out_ 0.53->0.53 (0.01)  down 0.47->0.47 (0.01)
 7 full_a  down_p=22.9 gate_p=27.6 k_proj=31.9 o_proj=20.2 q_proj=22.3 up_pro=39.3 v_proj=27.7
      o_pr 0.50->0.50 (0.01)  down 0.48->0.48 (0.01)
 8 linear  down_p=21.1 gate_p=29.0 in_pro=19.6 in_pro=23.5 out_pr=20.1 up_pro=39.3
      out_ 0.53->0.53 (0.02)  down 0.47->0.47 (0.01)
 9 linear  down_p=19.7 gate_p=30.2 in_pro=19.8 in_pro=23.0 out_pr=19.2 up_pro=37.2
      out_ 0.53->0.53 (0.01)  down 0.48->0.49 (0.01)
10 linear  down_p=21.5 gate_p=29.8 in_pro=18.9 in_pro=24.3 out_pr=18.1 up_pro=37.4
      out_ 0.53->0.53 (0.01)  down 0.48->0.48 (0.01)
11 full_a  down_p=17.9 gate_p=29.0 k_proj=27.8 o_proj=16.1 q_proj=22.6 up_pro=36.3 v_proj=18.7
      o_pr 0.54->0.54 (0.01)  down 0.49->0.49 (0.01)
12 linear  down_p=17.3 gate_p=24.8 in_pro=16.6 in_pro=21.0 out_pr=17.1 up_pro=34.4
      out_ 0.53->0.53 (0.01)  down 0.47->0.47 (0.01)
13 linear  down_p=16.1 gate_p=23.5 in_pro=17.2 in_pro=20.9 out_pr=17.3 up_pro=34.8
      out_ 0.54->0.54 (0.01)  down 0.48->0.48 (0.01)
14 linear  down_p=14.6 gate_p=19.6 in_pro=17.5 in_pro=19.4 out_pr=16.2 up_pro=32.3
      out_ 0.53->0.53 (0.01)  down 0.46->0.46 (0.01)
15 full_a  down_p=11.9 gate_p=18.8 k_proj=21.7 o_proj=10.5 q_proj=17.1 up_pro=27.3 v_proj=14.9
      o_pr 0.72->0.72 (0.02)  down 0.49->0.49 (0.02)
16 linear  down_p=11.9 gate_p=17.5 in_pro=14.0 in_pro=17.1 out_pr=13.5 up_pro=27.1
      out_ 0.47->0.47 (0.02)  down 0.48->0.48 (0.02)
17 linear  down_p=11.0 gate_p=16.1 in_pro=12.8 in_pro=16.1 out_pr=13.8 up_pro=26.5
      out_ 0.42->0.42 (0.02)  down 0.48->0.48 (0.02)
18 linear  down_p=10.5 gate_p=15.8 in_pro=11.8 in_pro=16.1 out_pr=14.2 up_pro=24.1
      out_ 0.46->0.46 (0.02)  down 0.49->0.49 (0.01)
19 full_a  down_p=9.0 gate_p=15.2 k_proj=20.2 o_proj=7.6 q_proj=18.3 up_pro=20.5 v_proj=10.3
      o_pr 0.94->0.94 (0.03)  down 0.57->0.57 (0.03)
20 linear  down_p=8.1 gate_p=14.5 in_pro=11.6 in_pro=13.6 out_pr=11.3 up_pro=19.6
      out_ 0.47->0.47 (0.02)  down 0.61->0.61 (0.01)
21 linear  down_p=7.7 gate_p=14.5 in_pro=12.8 in_pro=14.5 out_pr=10.8 up_pro=17.6
      out_ 0.57->0.57 (0.03)  down 0.65->0.65 (0.02)
22 linear  down_p=6.8 gate_p=12.5 in_pro=13.2 in_pro=14.5 out_pr=8.6 up_pro=18.0
      out_ 0.59->0.59 (0.02)  down 0.61->0.61 (0.01)
23 full_a  down_p=3.4 gate_p=9.6 k_proj=16.1 o_proj=4.7 q_proj=10.1 up_pro=15.4 v_proj=8.7
      o_pr 0.83->0.83 (0.02)  down 0.52->0.52 (0.01)
[turn 1 | 1.5s | passes this cell 0, used 0/50 | cpu]
```

## Turn 2 (22:42:09, 0002.py)

```python
# Turn 2 (reading). Turn 1: readout rows barely moved; the late full-attention
# o_proj (layers 15, 19, 23) write most strongly along u. Hypothesis: a few
# heads in those layers carry the relevance decision, and the fine-tune
# strengthened them. Read it per head: o_proj columns of head h are
# [h*256:(h+1)*256]; their write along u is |u^T o_proj[:, cols]|.
# Also the MLP neurons whose output column aligns with u (signed: +yes / -no).
import re
from bench.wsp.kernel import TUNED
saved = torch.load(TUNED + "/judge.pt", map_location="cpu", weights_only=True)
scale = saved["alpha"] / saved["r"]
def lora_delta(mod):
    A = saved["lora"].get(mod + ".lora_A.default.weight")
    if A is None:
        return None
    B = saved["lora"][mod + ".lora_B.default.weight"]
    return (B.float() @ A.float()) * scale
ti = S.true_index
R = S.readout.float().cpu(); r = R[ti] - R[1 - ti]
u = (1 + W("language_model.norm.weight").float().cpu()) * r; un = u / u.norm()
cfg = S.rt.backbone.language_model.config
hd, nh = cfg.head_dim, cfg.num_attention_heads
print("full-attention heads: |u^T o_proj[:, head]| base -> tuned (head dim", hd, ")")
for L in [i for i, t in enumerate(S.layer_types) if t == "full_attention"]:
    mod = f"language_model.layers.{L}.self_attn.o_proj"
    Wt = W(mod + ".weight").float().cpu(); d = lora_delta(mod); W0 = Wt - d
    cells = []
    for h in range(nh):
        cols = slice(h * hd, (h + 1) * hd)
        cells.append(f"h{h} {float((un @ W0[:, cols]).norm()):.2f}->{float((un @ Wt[:, cols]).norm()):.2f}")
    print(f"  L{L:2d}: " + "  ".join(cells))
print("\nMLP neurons most aligned with u (cos of down_proj column with u; + writes yes, - writes no), tuned; base cos in ()")
summary = []
for L in range(len(S.layers)):
    mod = f"language_model.layers.{L}.mlp.down_proj"
    Wt = W(mod + ".weight").float().cpu(); W0 = Wt - lora_delta(mod)
    ct = (un @ Wt) / Wt.norm(dim=0); c0 = (un @ W0) / W0.norm(dim=0)
    top = ct.abs().topk(4).indices.tolist()
    summary.append((L, float(ct.abs().max()), float((ct.abs() > 0.2).sum()), float((c0.abs() > 0.2).sum())))
    print(f"  L{L:2d}: " + "  ".join(f"n{j} {ct[j]:+.2f} ({c0[j]:+.2f}) |col| {Wt[:, j].norm():.2f}" for j in top))
print("\nneurons with |cos|>0.2 per layer: tuned vs base:", [(L, int(a), int(b)) for L, _, a, b in summary])
```

```
full-attention heads: |u^T o_proj[:, head]| base -> tuned (head dim 256 )
  L 3: h0 0.17->0.17  h1 0.20->0.20  h2 0.24->0.24  h3 0.20->0.20  h4 0.18->0.18  h5 0.19->0.19  h6 0.19->0.19  h7 0.18->0.18
  L 7: h0 0.16->0.16  h1 0.16->0.16  h2 0.17->0.17  h3 0.20->0.20  h4 0.18->0.18  h5 0.20->0.20  h6 0.18->0.18  h7 0.17->0.17
  L11: h0 0.23->0.23  h1 0.17->0.17  h2 0.17->0.17  h3 0.18->0.18  h4 0.24->0.24  h5 0.17->0.17  h6 0.20->0.20  h7 0.16->0.16
  L15: h0 0.25->0.25  h1 0.23->0.23  h2 0.24->0.24  h3 0.29->0.29  h4 0.26->0.26  h5 0.30->0.30  h6 0.22->0.22  h7 0.23->0.23
  L19: h0 0.19->0.19  h1 0.48->0.47  h2 0.19->0.19  h3 0.20->0.20  h4 0.42->0.42  h5 0.48->0.48  h6 0.28->0.28  h7 0.23->0.23
  L23: h0 0.26->0.26  h1 0.38->0.38  h2 0.24->0.23  h3 0.35->0.35  h4 0.21->0.21  h5 0.44->0.44  h6 0.15->0.16  h7 0.20->0.20

MLP neurons most aligned with u (cos of down_proj column with u; + writes yes, - writes no), tuned; base cos in ()
  L 0: n3381 -0.12 (-0.12) |col| 0.27  n1412 -0.10 (-0.10) |col| 0.28  n3399 -0.10 (-0.10) |col| 0.28  n612 +0.10 (+0.10) |col| 0.27
  L 1: n456 -0.11 (-0.11) |col| 0.25  n3063 -0.11 (-0.11) |col| 0.28  n32 +0.11 (+0.11) |col| 0.19  n1940 -0.10 (-0.10) |col| 0.26
  L 2: n1397 +0.10 (+0.10) |col| 0.27  n584 +0.10 (+0.10) |col| 0.25  n2578 -0.10 (-0.10) |col| 0.27  n3146 -0.10 (-0.10) |col| 0.27
  L 3: n3095 -0.12 (-0.12) |col| 0.26  n28 +0.11 (+0.11) |col| 0.26  n2415 -0.11 (-0.11) |col| 0.26  n2623 -0.10 (-0.10) |col| 0.24
  L 4: n2642 -0.11 (-0.10) |col| 0.25  n965 +0.10 (+0.10) |col| 0.24  n1687 +0.10 (+0.10) |col| 0.28  n1392 -0.10 (-0.10) |col| 0.26
  L 5: n484 +0.11 (+0.11) |col| 0.29  n7 +0.11 (+0.11) |col| 0.16  n3478 +0.10 (+0.10) |col| 0.25  n2469 +0.10 (+0.10) |col| 0.27
  L 6: n2450 -0.11 (-0.11) |col| 0.26  n935 -0.11 (-0.11) |col| 0.26  n2952 +0.10 (+0.10) |col| 0.27  n305 +0.09 (+0.09) |col| 0.19
  L 7: n2190 -0.13 (-0.13) |col| 0.26  n3324 +0.12 (+0.12) |col| 0.26  n165 +0.12 (+0.12) |col| 0.25  n285 -0.11 (-0.11) |col| 0.27
  L 8: n2120 -0.11 (-0.11) |col| 0.27  n1520 -0.11 (-0.11) |col| 0.24  n445 +0.10 (+0.10) |col| 0.33  n2025 +0.10 (+0.10) |col| 0.25
  L 9: n2111 +0.12 (+0.12) |col| 0.27  n2186 -0.12 (-0.12) |col| 0.22  n875 -0.11 (-0.11) |col| 0.27  n2362 -0.11 (-0.11) |col| 0.25
  L10: n3033 -0.11 (-0.11) |col| 0.24  n127 +0.11 (+0.11) |col| 0.24  n2391 +0.10 (+0.10) |col| 0.23  n3092 -0.10 (-0.10) |col| 0.27
  L11: n1476 +0.12 (+0.13) |col| 0.19  n1798 -0.12 (-0.12) |col| 0.25  n1353 +0.12 (+0.12) |col| 0.23  n835 -0.12 (-0.12) |col| 0.32
  L12: n1778 -0.11 (-0.11) |col| 0.22  n183 +0.11 (+0.11) |col| 0.25  n2195 -0.11 (-0.11) |col| 0.25  n1330 +0.10 (+0.10) |col| 0.30
  L13: n2182 +0.12 (+0.12) |col| 0.24  n3307 +0.12 (+0.12) |col| 0.27  n2751 +0.11 (+0.11) |col| 0.25  n543 -0.11 (-0.11) |col| 0.24
  L14: n253 -0.17 (-0.17) |col| 0.26  n55 +0.17 (+0.17) |col| 0.26  n45 -0.14 (-0.14) |col| 0.23  n81 -0.13 (-0.13) |col| 0.30
  L15: n433 -0.29 (-0.28) |col| 0.16  n1326 +0.20 (+0.20) |col| 0.23  n584 -0.18 (-0.18) |col| 0.20  n344 -0.16 (-0.15) |col| 0.17
  L16: n637 -0.19 (-0.19) |col| 0.24  n2868 -0.11 (-0.11) |col| 0.30  n690 +0.10 (+0.10) |col| 0.27  n500 -0.10 (-0.10) |col| 0.28
  L17: n1726 +0.13 (+0.13) |col| 0.29  n2306 +0.12 (+0.12) |col| 0.24  n1422 +0.11 (+0.11) |col| 0.29  n3468 -0.10 (-0.10) |col| 0.30
  L18: n3190 +0.32 (+0.32) |col| 0.27  n360 +0.30 (+0.30) |col| 0.21  n2170 -0.25 (-0.25) |col| 0.31  n17 -0.22 (-0.22) |col| 0.22
  L19: n173 +0.37 (+0.37) |col| 0.17  n123 -0.29 (-0.28) |col| 0.17  n3458 +0.28 (+0.28) |col| 0.29  n2717 +0.21 (+0.21) |col| 0.33
  L20: n1082 -0.42 (-0.42) |col| 0.28  n980 +0.26 (+0.26) |col| 0.28  n3545 +0.24 (+0.24) |col| 0.30  n2852 +0.23 (+0.23) |col| 0.30
  L21: n803 +0.41 (+0.41) |col| 0.33  n3443 +0.32 (+0.32) |col| 0.29  n2094 -0.31 (-0.30) |col| 0.32  n3110 -0.30 (-0.30) |col| 0.31
  L22: n519 +0.52 (+0.52) |col| 0.29  n2202 -0.40 (-0.40) |col| 0.24  n415 +0.37 (+0.37) |col| 0.25  n929 -0.36 (-0.35) |col| 0.23
  L23: n748 +0.60 (+0.60) |col| 0.19  n1068 -0.55 (-0.55) |col| 0.24  n2614 -0.52 (-0.52) |col| 0.25  n133 +0.42 (+0.42) |col| 0.16

neurons with |cos|>0.2 per layer: tuned vs base: [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0), (7, 0, 0), (8, 0, 0), (9, 0, 0), (10, 0, 0), (11, 0, 0), (12, 0, 0), (13, 0, 0), (14, 0, 0), (15, 1, 1), (16, 0, 0), (17, 0, 0), (18, 4, 4), (19, 4, 4), (20, 4, 4), (21, 5, 5), (22, 10, 10), (23, 16, 16)]
[turn 2 | 0.6s | passes this cell 0, used 0/50 | cpu]
```

## Turn 3 (22:55:13, 0003.py)

```python
# Turn 3 (reading). Length hypothesis, weight side: Gated DeltaNet layers keep
# a recurrent state with per-head decay g = exp(-exp(A_log) * softplus(a + dt_bias)).
# With the input-dependent a set to 0, the half-life in tokens is
# ln 2 / (exp(A_log) * softplus(dt_bias)). Short half-lives mean these causal
# layers carry little from early in a long passage to the answer position;
# only the 6 bidirectional full-attention layers can.
import math
F = torch.nn.functional
print("linear-attention layers: per-head half-life in tokens (a=0), sorted")
for L, t in enumerate(S.layer_types):
    if t != "linear_attention":
        continue
    A = W(f"language_model.layers.{L}.linear_attn.A_log").float().cpu()
    dt = W(f"language_model.layers.{L}.linear_attn.dt_bias").float().cpu()
    rate = A.exp() * F.softplus(dt)
    hl = (math.log(2) / rate).sort().values
    print(f"  L{L:2d}: heads {len(hl)}  min {hl[0]:.1f}  median {hl[len(hl)//2]:.1f}  max {hl[-1]:.1f}   "
          f"share >1000 tok {float((hl > 1000).float().mean()):.2f}")
```

```
linear-attention layers: per-head half-life in tokens (a=0), sorted
  L 0: heads 16  min 0.1  median 504.2  max 238466208.0   share >1000 tok 0.25
  L 1: heads 16  min 0.3  median 1.2  max 2272.1   share >1000 tok 0.06
  L 2: heads 16  min 0.4  median 5.8  max 519.5   share >1000 tok 0.00
  L 4: heads 16  min 0.3  median 74.3  max 2346.0   share >1000 tok 0.06
  L 5: heads 16  min 1.1  median 183.3  max 510.8   share >1000 tok 0.00
  L 6: heads 16  min 0.6  median 6.1  max 648.5   share >1000 tok 0.00
  L 8: heads 16  min 1.4  median 18.9  max 623.8   share >1000 tok 0.00
  L 9: heads 16  min 1.3  median 47.0  max 965.2   share >1000 tok 0.00
  L10: heads 16  min 0.7  median 17.5  max 398.9   share >1000 tok 0.00
  L12: heads 16  min 1.6  median 283.0  max 877.7   share >1000 tok 0.00
  L13: heads 16  min 1.3  median 12.3  max 434.1   share >1000 tok 0.00
  L14: heads 16  min 1.6  median 15.8  max 1770.0   share >1000 tok 0.06
  L16: heads 16  min 0.8  median 193.9  max 636.8   share >1000 tok 0.00
  L17: heads 16  min 1.6  median 7.9  max 597.7   share >1000 tok 0.00
  L18: heads 16  min 0.5  median 107.5  max 1769.1   share >1000 tok 0.06
  L20: heads 16  min 0.3  median 34.1  max 448.1   share >1000 tok 0.00
  L21: heads 16  min 0.5  median 3.2  max 396.6   share >1000 tok 0.00
  L22: heads 16  min 0.3  median 7.7  max 791.6   share >1000 tok 0.00
[turn 3 | 0.0s | passes this cell 0, used 0/50 | cpu]
```

## Turn 4 (02:33:11, 0004.py)

```python
# Turn 4: end of the reading phase. Move the session to the GPU, reloaded
# exactly as the benchmark loader builds the judge (no edits were made while
# reading, so nothing is carried).
print(to_gpu())
print(changed())
```

```

Loading weights:   0%|          | 0/473 [00:00<?, ?it/s]
Loading weights:   4%|3         | 18/473 [00:00<00:03, 130.90it/s]
Loading weights:  12%|#2        | 58/473 [00:00<00:01, 263.65it/s]
Loading weights:  18%|#8        | 87/473 [00:00<00:01, 260.95it/s]
Loading weights:  25%|##4       | 118/473 [00:00<00:01, 277.06it/s]
Loading weights:  31%|###1      | 147/473 [00:00<00:01, 256.01it/s]
Loading weights:  38%|###8      | 181/473 [00:00<00:01, 280.92it/s]
Loading weights:  46%|####5     | 216/473 [00:00<00:00, 299.82it/s]
Loading weights:  52%|#####2    | 247/473 [00:00<00:00, 269.52it/s]
Loading weights:  61%|######    | 287/473 [00:01<00:00, 297.72it/s]
Loading weights:  68%|######7   | 321/473 [00:01<00:00, 309.07it/s]
Loading weights:  84%|########3 | 395/473 [00:01<00:00, 425.86it/s]
Loading weights:  97%|#########7| 461/473 [00:01<00:00, 486.07it/s]
Loading weights: 100%|##########| 473/473 [00:01<00:00, 346.17it/s]
on cuda:0; carried 0 edited tensors
{'_total': 0, '_share': '0.0000% of 852,987,968'}
[turn 4 | 8.5s | passes this cell 0, used 0/50 | cuda:0]
```

## Turn 5 (02:34:09, 0005.py)

```python
# Turn 5 (GPU, first passes). Baseline of the tuned judge on PRACTICE (806
# examples of val questions no gradient touched = 2 passes), with the residual
# stream at the answer position after every layer. Logit lens on the decision
# direction: where in depth do yes and no separate, and where do the errors go
# wrong (false positives = no-examples judged yes, false negatives = the reverse)?
import numpy as np
res = forward(PRACTICE, capture=("resid",))
base_p = res["p"]; base_resid = res["resid"]
y = np.array([r["label"] for r in PRACTICE])
for kind in sorted({r["kind"] for r in PRACTICE}):
    idx = [i for i, r in enumerate(PRACTICE) if r["kind"] == kind]
    print(kind, metrics(base_p[idx], y[idx]))
print("all", metrics(base_p, y))
ti = S.true_index
R = S.readout.float().cpu(); r = (R[ti] - R[1 - ti]).numpy()
g = 1 + W("language_model.norm.weight").float().cpu().numpy()
eps = S.rt.backbone.language_model.norm.variance_epsilon if hasattr(S.rt.backbone.language_model.norm, "variance_epsilon") else 1e-6
def lens(h):  # decision logit if the network stopped here: rmsnorm(h) * g . r
    return ((h / np.sqrt((h ** 2).mean(-1, keepdims=True) + eps)) * g) @ r
D = lens(base_resid)  # [n, layers]
groups = {"TP": (y == 1) & (base_p > .5), "FN": (y == 1) & (base_p <= .5),
          "TN": (y == 0) & (base_p <= .5), "FP": (y == 0) & (base_p > .5)}
print({k: int(v.sum()) for k, v in groups.items()})
print("logit lens (mean decision logit per layer):")
for k, m in groups.items():
    print(f"  {k}: " + " ".join(f"{v:+.1f}" for v in D[m].mean(0)))
print("final logit check vs p: corr", np.corrcoef(D[:, -1], np.log(base_p / (1 - base_p)))[0, 1])
# the worst errors, to read
order = np.argsort(-np.abs(base_p - y))
for i in order[:8]:
    r_ = PRACTICE[i]
    print(f"\n[{r_['kind']} label {r_['label']} p {base_p[i]:.3f} len {len(r_['text'])}] Q: {r_['query'][:200]}\n  T: {r_['text'][:300]}")
```

```
page {'n': 392, 'auc': 0.977, 'acc': 0.9541, 'logloss': 0.1307, 'brier': 0.0358, 'pos_rate_pred': 0.1276, 'pos_rate_true': 0.1429}
passage {'n': 411, 'auc': 0.8812, 'acc': 0.8516, 'logloss': 0.3602, 'brier': 0.1092, 'pos_rate_pred': 0.1776, 'pos_rate_true': 0.2238}
section {'n': 3, 'auc': nan, 'acc': 1.0, 'logloss': 0.0894, 'brier': 0.0102, 'pos_rate_pred': 0.0, 'pos_rate_true': 0.0}
all {'n': 806, 'auc': 0.9357, 'acc': 0.902, 'logloss': 0.2476, 'brier': 0.0731, 'pos_rate_pred': 0.1526, 'pos_rate_true': 0.1836}
{'TP': 96, 'FN': 52, 'TN': 631, 'FP': 27}
logit lens (mean decision logit per layer):
  TP: -2.1 -2.7 -1.4 +2.0 +1.4 +0.7 -1.8 +0.2 +0.5 +1.8 +1.0 +0.6 +0.5 +1.8 +3.6 +11.8 +10.2 +9.0 +7.7 +14.5 +12.4 +11.3 +14.3 +2.1
  FN: -2.1 -2.7 -1.4 +1.9 +1.3 +0.7 -1.8 +0.3 +0.6 +1.8 +0.9 +1.2 +0.9 +3.0 +3.9 -3.8 -3.9 -3.8 -4.3 -5.7 -4.9 -5.1 -9.5 -2.0
  TN: -2.1 -2.8 -1.4 +2.1 +1.4 +0.7 -1.8 +0.2 +0.5 +1.8 +1.0 +1.4 +1.1 +2.8 +3.8 -9.7 -9.3 -8.5 -8.7 -12.5 -11.1 -11.1 -19.6 -5.1
  FP: -2.1 -2.7 -1.4 +1.9 +1.3 +0.7 -1.8 +0.3 +0.6 +1.8 +1.0 +0.9 +0.6 +2.3 +4.0 +9.0 +7.6 +6.5 +5.4 +11.6 +9.7 +8.9 +10.7 +1.1
final logit check vs p: corr 0.9999999989050506

[page label 1 p 0.003 len 400] Q: My sister and her husband are in the process of selling their house and have just learned that they have to go abroad for work for the next six months. They have asked me to handle the sale of their h
  T: Make decisions on behalf of someone. Sections: When you can make decisions for someone; When someone can choose you; Short-term help; Long-term help; When you apply to a court; One-off decisions; Long-term help; How to make decisions; Helping someone make decisions; Making decisions in someone’s bes

[page label 1 p 0.005 len 400] Q: My partner and i have decided to buy a property. Fortunately we have found a perfect house going at £520000.It has a green deal and the seller has informed has that it still has an outstanding amount 
  T: Green Deal: energy saving for your home. Sections: Overview; If you need help paying for home improvements; Find out if your home will benefit; Green Deal mark; Improvements and benefits to your home; Eligible improvements; Get an assessment; What to expect from an assessment; When you book; When th

[passage label 1 p 0.008 len 1197] Q: Due to the current pandemic my local vehicle service garage is closed, therefore it is very difficult for me to service my vehicles at regular times. Can the required service times on my farm vehicles
  T: Most new machines or pieces of equipment should also have:
a ‘declaration of conformity’ to show that it meets EU standards
a manual
information on noise levels
Second-hand equipment and vehicles
Before using second hand machinery, you should:
make sure that it complies with the Provision and Use of

[passage label 1 p 0.010 len 1200] Q: Due to the current pandemic my local vehicle service garage is closed, therefore it is very difficult for me to service my vehicles at regular times. Can the required service times on my farm vehicles
  T: Overview
If you own or manage a farm, you’re responsible for ensuring the health and safety of your workers and anyone who’s affected by what they do.
You must:
carry out a assessment of any risks related to your farm
have a plan to manage these risks and protect people from harm
plan and set standa

[passage label 1 p 0.011 len 1176] Q: I'm Bob and I'm working at a corner shop in my town. My boss is really mean and I am wondering if I'm being diddled. He pays my bus fare and he says I should be grateful. I'm 18 years old. Am I entitl
  T: Voluntary work
You’re classed as doing voluntary work if you can only get certain limited benefits (for example reasonable travel or lunch expenses) and you’re working for a:
charity
voluntary organisation or associated fundraising body
statutory body
Contact the Acas helpline to find out if you sho

[page label 0 p 0.982 len 334] Q: My friend Martin and i are sport fanatics and we have decided to set up a community amateur sports club. We need to come up with a binding document to set up the purpose and structure of the club. Whe
  T: Register as a community amateur sports club (CASC). Sections: Overview; CASCs and charities; Eligibility; Governing document; Open to the whole community; Membership fees; Organised on an amateur basis; Register; What you need to complete the form; How long the application takes; If your application

[passage label 1 p 0.024 len 1059] Q: I am a licensed haulier and have established my yard opposite a posh house. The owner is threatening to take me to a tribunal as he does't like being near a truck yard. Can I be taken to a tribual or 
  T: the police
some trade associations and trade unions
When a public body can object
A public body can object to a licence about the:
professional competence of the operator or its transport manager
operator’s finances
operator’s reputation or fitness to hold a licence
operator’s arrangements for vehic

[passage label 1 p 0.032 len 812] Q: My partner and i have decided to buy a property. Fortunately we have found a perfect house going at £520000.It has a green deal and the seller has informed has that it still has an outstanding amount 
  T: You can pay off your Green Deal early, but you might be charged a fee - check with your provider.
Moving into a property with a Green Deal
If you move into a property with a Green Deal, the landlord or seller must show you a copy of the Energy Performance Certificate (EPC). This will explain what im
[turn 5 | 41.6s | passes this cell 2, used 2/50 | cuda:0]
```

## Turn 6 (02:34:29, 0006.py)

```python
# Turn 6 (GPU). Hypothesis from reading the input distribution: the fine-tune
# only saw texts up to ~1,200 characters (ConditionalQA), but 22.5% of
# LegalBench and 15.4% of bBSARD chunks are longer than 1,600. If the judge's
# yes-evidence is diluted in long texts, a relevant passage buried in filler
# gets a lower P(yes). Probe with TRAIN data only: 128 relevant and 128
# irrelevant passages, each alone and padded with irrelevant passages of the
# same question (same topic) to ~3,000 chars (passage first / last) and
# ~6,000 chars (passage in the middle). 1,024 prompts = 2 passes.
import random
rng = random.Random(21)
byq = {}
for r_ in TRAIN:
    if r_["kind"] == "passage":
        byq.setdefault(r_["qid"], []).append(r_)
cands = [q for q, rs in byq.items() if any(x["label"] for x in rs) and sum(1 - x["label"] for x in rs) >= 8]
qs = rng.sample(cands, 128)
probe = []
def pad(core, fillers, target, where):
    parts, size = [], len(core["text"])
    for f in fillers:
        if size >= target:
            break
        parts.append(f["text"]); size += len(f["text"]) + 1
    if where == "first":
        body = [core["text"]] + parts
    elif where == "last":
        body = parts + [core["text"]]
    else:
        h = len(parts) // 2
        body = parts[:h] + [core["text"]] + parts[h:]
    return "\n".join(body)
for q in qs:
    rs = byq[q]
    pos = rng.choice([x for x in rs if x["label"]])
    negs = [x for x in rs if not x["label"]]
    rng.shuffle(negs)
    neg, fill = negs[0], negs[1:]
    for core in (pos, neg):
        for variant, target, where in (("alone", 0, "first"), ("3k-first", 3000, "first"),
                                       ("3k-last", 3000, "last"), ("6k-middle", 6000, "middle")):
            text = core["text"] if variant == "alone" else pad(core, fill, target, where)
            probe.append({"query": core["query"], "kind": "passage", "text": text, "label": core["label"],
                          "variant": variant, "qid": q})
p = forward(probe)["p"]
import numpy as np
for variant in ("alone", "3k-first", "3k-last", "6k-middle"):
    idx = [i for i, x in enumerate(probe) if x["variant"] == variant]
    lens_ = [len(probe[i]["text"]) for i in idx]
    print(f"{variant:10s} median len {int(np.median(lens_)):5d}  ", metrics(p[idx], [probe[i]["label"] for i in idx]),
          f" mean P(yes) pos {np.mean([p[i] for i in idx if probe[i]['label']]):.3f} neg {np.mean([p[i] for i in idx if not probe[i]['label']]):.3f}")
probe_p = p
```

```
Traceback (most recent call last):
  File "<lab>\bench\wsp\kernel.py", line 421, in main
    exec(compile(code, cell.name, "exec"), ns)
  File "0006.py", line 16, in <module>
  File "%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\Lib\random.py", line 430, in sample
    raise ValueError("Sample larger than population or is negative")
ValueError: Sample larger than population or is negative
[turn 6 | 0.0s | passes this cell 0, used 2/50 | cuda:0]
```

## Turn 7 (02:37:16, 0007.py)

```python
# Turn 7 (GPU), retry of turn 6 with a pool that exists: questions with >= 4 irrelevant passages.
# only saw texts up to ~1,200 characters (ConditionalQA), but 22.5% of
# LegalBench and 15.4% of bBSARD chunks are longer than 1,600. If the judge's
# yes-evidence is diluted in long texts, a relevant passage buried in filler
# gets a lower P(yes). Probe with TRAIN data only: 128 relevant and 128
# irrelevant passages, each alone and padded with irrelevant passages of the
# same question (same topic) to ~3,000 chars (passage first / last) and
# ~6,000 chars (passage in the middle). 1,024 prompts = 2 passes.
import random
rng = random.Random(21)
byq = {}
for r_ in TRAIN:
    if r_["kind"] == "passage":
        byq.setdefault(r_["qid"], []).append(r_)
cands = [q for q, rs in byq.items() if any(x["label"] for x in rs) and sum(1 - x["label"] for x in rs) >= 4]
print("candidate questions", len(cands))
qs = rng.sample(cands, min(128, len(cands)))
probe = []
def pad(core, fillers, target, where):
    parts, size = [], len(core["text"])
    for f in fillers:
        if size >= target:
            break
        parts.append(f["text"]); size += len(f["text"]) + 1
    if where == "first":
        body = [core["text"]] + parts
    elif where == "last":
        body = parts + [core["text"]]
    else:
        h = len(parts) // 2
        body = parts[:h] + [core["text"]] + parts[h:]
    return "\n".join(body)
for q in qs:
    rs = byq[q]
    pos = rng.choice([x for x in rs if x["label"]])
    negs = [x for x in rs if not x["label"]]
    rng.shuffle(negs)
    neg, fill = negs[0], negs[1:]
    for core in (pos, neg):
        for variant, target, where in (("alone", 0, "first"), ("3k-first", 3000, "first"),
                                       ("3k-last", 3000, "last"), ("6k-middle", 6000, "middle")):
            text = core["text"] if variant == "alone" else pad(core, fill, target, where)
            probe.append({"query": core["query"], "kind": "passage", "text": text, "label": core["label"],
                          "variant": variant, "qid": q})
p = forward(probe)["p"]
import numpy as np
for variant in ("alone", "3k-first", "3k-last", "6k-middle"):
    idx = [i for i, x in enumerate(probe) if x["variant"] == variant]
    lens_ = [len(probe[i]["text"]) for i in idx]
    print(f"{variant:10s} median len {int(np.median(lens_)):5d}  ", metrics(p[idx], [probe[i]["label"] for i in idx]),
          f" mean P(yes) pos {np.mean([p[i] for i in idx if probe[i]['label']]):.3f} neg {np.mean([p[i] for i in idx if not probe[i]['label']]):.3f}")
probe_p = p
```

```
candidate questions 2103
alone      median len  1150   {'n': 256, 'auc': 0.9683, 'acc': 0.8555, 'logloss': 0.3293, 'brier': 0.1051, 'pos_rate_pred': 0.3945, 'pos_rate_true': 0.5}  mean P(yes) pos 0.673 neg 0.077
3k-first   median len  3436   {'n': 256, 'auc': 0.8952, 'acc': 0.8164, 'logloss': 0.4425, 'brier': 0.1416, 'pos_rate_pred': 0.3945, 'pos_rate_true': 0.5}  mean P(yes) pos 0.630 neg 0.172
3k-last    median len  3436   {'n': 256, 'auc': 0.8654, 'acc': 0.7695, 'logloss': 0.5192, 'brier': 0.1688, 'pos_rate_pred': 0.3555, 'pos_rate_true': 0.5}  mean P(yes) pos 0.573 neg 0.175
6k-middle  median len  6503   {'n': 256, 'auc': 0.8252, 'acc': 0.7617, 'logloss': 0.5449, 'brier': 0.18, 'pos_rate_pred': 0.3945, 'pos_rate_true': 0.5}  mean P(yes) pos 0.574 neg 0.247
[turn 7 | 155.7s | passes this cell 2, used 4/50 | cuda:0]
```

## Turn 8 (02:39:03, 0008.py)

```python
# Turn 8 (GPU, 1 pass). Turn 7: padding with on-topic non-answers raises the
# irrelevant texts' P(yes) 0.08 -> 0.25 and lowers AUC 0.97 -> 0.83. Find
# the component that adds the false yes: direct contribution of every token
# mixer (attention / linear attention output) and MLP output, at the answer
# position, to the decision direction, divided by the final RMS of the row
# (the final norm's scaling): dla = (c . ((1 + g) * r)) / rms(h_final).
# Compare irrelevant alone vs irrelevant 6k (and relevant alone vs 6k).
import numpy as np
sel = [i for i, x in enumerate(probe) if x["variant"] in ("alone", "6k-middle")]
rows = [probe[i] for i in sel]
res = forward(rows, capture=("resid", "attn_out", "mlp_out"))
ti = S.true_index
R = S.readout.float().cpu(); r = (R[ti] - R[1 - ti]).numpy()
g = 1 + W("language_model.norm.weight").float().cpu().numpy()
u = g * r
rms = np.sqrt((res["resid"][:, -1] ** 2).mean(-1) + 1e-6)       # [n]
dla_attn = (res["attn_out"] @ u) / rms[:, None]                  # [n, L]
dla_mlp = (res["mlp_out"] @ u) / rms[:, None]
lab = np.array([x["label"] for x in rows]); var = np.array([x["variant"] for x in rows])
def m(a, mask): return a[mask].mean(0)
print("final decision logit: neg alone %.2f  neg 6k %.2f  pos alone %.2f  pos 6k %.2f" % tuple(
    np.log(res["p"][(lab == l) & (var == v)] / (1 - res["p"][(lab == l) & (var == v)])).mean()
    for l, v in ((0, "alone"), (0, "6k-middle"), (1, "alone"), (1, "6k-middle"))))
print("final rms: neg alone %.1f  neg 6k %.1f" % (rms[(lab == 0) & (var == "alone")].mean(), rms[(lab == 0) & (var == "6k-middle")].mean()))
print("\nlayer type    attn: neg alone -> 6k (shift) | pos alone -> 6k       mlp: neg alone -> 6k (shift) | pos alone -> 6k")
for L in range(24):
    na, n6 = m(dla_attn, (lab == 0) & (var == "alone"))[L], m(dla_attn, (lab == 0) & (var == "6k-middle"))[L]
    pa, p6 = m(dla_attn, (lab == 1) & (var == "alone"))[L], m(dla_attn, (lab == 1) & (var == "6k-middle"))[L]
    nma, nm6 = m(dla_mlp, (lab == 0) & (var == "alone"))[L], m(dla_mlp, (lab == 0) & (var == "6k-middle"))[L]
    pma, pm6 = m(dla_mlp, (lab == 1) & (var == "alone"))[L], m(dla_mlp, (lab == 1) & (var == "6k-middle"))[L]
    print(f"L{L:2d} {S.layer_types[L][:6]}  {na:+6.2f} -> {n6:+6.2f} ({n6-na:+5.2f}) | {pa:+6.2f} -> {p6:+6.2f}     "
          f"{nma:+6.2f} -> {nm6:+6.2f} ({nm6-nma:+5.2f}) | {pma:+6.2f} -> {pm6:+6.2f}")
P8 = dict(rows=rows, res=res, lab=lab, var=var, dla_attn=dla_attn, dla_mlp=dla_mlp, u=u)
```

```
final decision logit: neg alone -3.52  neg 6k -1.55  pos alone 1.00  pos 6k 0.37
final rms: neg alone 2.1  neg 6k 2.1

layer type    attn: neg alone -> 6k (shift) | pos alone -> 6k       mlp: neg alone -> 6k (shift) | pos alone -> 6k
L 0 linear   -0.01 ->  -0.01 (+0.00) |  -0.01 ->  -0.01      +0.00 ->  +0.00 (-0.00) |  +0.00 ->  +0.00
L 1 linear   +0.01 ->  +0.01 (+0.01) |  +0.01 ->  +0.01      -0.03 ->  -0.03 (-0.00) |  -0.02 ->  -0.03
L 2 linear   -0.01 ->  -0.01 (-0.00) |  -0.01 ->  -0.01      +0.02 ->  +0.02 (-0.00) |  +0.02 ->  +0.02
L 3 full_a   +0.07 ->  +0.08 (+0.01) |  +0.07 ->  +0.08      -0.00 ->  -0.01 (-0.01) |  -0.00 ->  -0.01
L 4 linear   -0.03 ->  -0.03 (+0.00) |  -0.03 ->  -0.03      +0.02 ->  +0.02 (+0.00) |  +0.02 ->  +0.02
L 5 linear   -0.02 ->  -0.03 (-0.01) |  -0.02 ->  -0.03      +0.01 ->  +0.02 (+0.01) |  +0.01 ->  +0.02
L 6 linear   -0.03 ->  -0.03 (+0.00) |  -0.03 ->  -0.03      -0.03 ->  -0.03 (+0.00) |  -0.03 ->  -0.03
L 7 full_a   +0.07 ->  +0.06 (-0.00) |  +0.06 ->  +0.06      -0.01 ->  -0.00 (+0.01) |  -0.01 ->  -0.00
L 8 linear   +0.00 ->  +0.00 (-0.00) |  +0.00 ->  +0.00      +0.00 ->  +0.00 (-0.00) |  +0.00 ->  +0.00
L 9 linear   +0.00 ->  -0.01 (-0.01) |  +0.00 ->  -0.01      +0.03 ->  +0.03 (-0.01) |  +0.03 ->  +0.03
L10 linear   -0.00 ->  -0.01 (-0.01) |  -0.01 ->  -0.01      -0.03 ->  -0.03 (-0.00) |  -0.02 ->  -0.03
L11 full_a   +0.05 ->  +0.06 (+0.01) |  +0.03 ->  +0.05      -0.03 ->  -0.02 (+0.00) |  -0.03 ->  -0.03
L12 linear   +0.01 ->  +0.01 (-0.00) |  -0.00 ->  -0.00      -0.02 ->  -0.01 (+0.00) |  +0.00 ->  -0.00
L13 linear   +0.15 ->  +0.15 (-0.00) |  +0.06 ->  +0.10      -0.01 ->  -0.00 (+0.01) |  +0.04 ->  +0.03
L14 linear   +0.04 ->  +0.03 (-0.01) |  -0.00 ->  +0.01      +0.06 ->  +0.06 (-0.00) |  +0.13 ->  +0.10
L15 full_a   -1.38 ->  -0.56 (+0.82) |  +0.50 ->  +0.15      +0.10 ->  +0.02 (-0.08) |  -0.02 ->  -0.03
L16 linear   -0.09 ->  -0.09 (+0.00) |  +0.03 ->  +0.00      -0.04 ->  -0.02 (+0.02) |  +0.01 ->  +0.00
L17 linear   -0.02 ->  -0.02 (+0.00) |  -0.01 ->  -0.02      +0.02 ->  -0.02 (-0.03) |  -0.04 ->  -0.04
L18 linear   -0.15 ->  -0.10 (+0.04) |  -0.06 ->  -0.08      -0.08 ->  -0.04 (+0.04) |  +0.00 ->  +0.00
L19 full_a   -0.84 ->  -0.06 (+0.78) |  +1.02 ->  +0.97      -0.08 ->  -0.06 (+0.02) |  +0.14 ->  +0.07
L20 linear   -0.07 ->  -0.06 (+0.01) |  -0.02 ->  -0.04      +0.21 ->  +0.10 (-0.12) |  -0.15 ->  -0.08
L21 linear   -0.05 ->  -0.04 (+0.02) |  -0.02 ->  -0.02      -0.17 ->  -0.09 (+0.08) |  +0.02 ->  +0.00
L22 linear   +0.06 ->  +0.13 (+0.07) |  +0.02 ->  -0.01      -1.90 ->  -1.13 (+0.76) |  +0.26 ->  +0.03
L23 full_a   +0.01 ->  -0.12 (-0.13) |  -0.39 ->  -0.43      +0.68 ->  +0.37 (-0.32) |  -0.55 ->  -0.41
[turn 8 | 75.8s | passes this cell 1, used 5/50 | cuda:0]
```

## Turn 9 (02:40:51, 0009.py)

```python
# Turn 9 (GPU, 1 pass). Turn 8: the length shift sits in the full-attention
# outputs of L15 and L19 and in the L22 MLP. Split it further: per head of
# L15 / L19 / L23 (input of o_proj, 8 heads x 256, at the answer position)
# and per neuron of the L22 MLP (input of down_proj). Per-head contribution
# to the decision = (o_proj[:, head] @ x_head) . u / rms_final.
import numpy as np
rows, u = P8["rows"], P8["u"]
lab, var = P8["lab"], P8["var"]
heads_in = {L: [None] * len(rows) for L in (15, 19, 23)}
mlp22 = [None] * len(rows)
hooks = []
def keep(store, key=None):
    def hook(mod, inp):
        x = inp[0][:, -1].float().cpu()
        for j, i in enumerate(S.ctx["batch"]):
            (store if key is None else store[key])[i] = x[j]
    return hook
for L in heads_in:
    hooks.append(S.layers[L].self_attn.o_proj.register_forward_pre_hook(keep(heads_in, L)))
hooks.append(S.layers[22].mlp.down_proj.register_forward_pre_hook(keep(mlp22)))
try:
    res = forward(rows, capture=("resid",), layers=[23])
finally:
    for h in hooks: h.remove()
rms = np.sqrt((res["resid"][:, 0] ** 2).mean(-1) + 1e-6)
ut = torch.tensor(u)
groups = {"neg alone": (lab == 0) & (var == "alone"), "neg 6k": (lab == 0) & (var == "6k-middle"),
          "pos alone": (lab == 1) & (var == "alone"), "pos 6k": (lab == 1) & (var == "6k-middle")}
HEAD = {}
for L, store in heads_in.items():
    X = torch.stack(store)                       # [n, 2048]
    Wo = W(f"language_model.layers.{L}.self_attn.o_proj.weight").float().cpu()   # [1024, 2048]
    hd = Wo.shape[1] // 8
    per = torch.stack([(X[:, h*hd:(h+1)*hd] @ Wo[:, h*hd:(h+1)*hd].T) @ ut for h in range(8)], 1).numpy() / rms[:, None]
    HEAD[L] = per
    print(f"L{L} per-head contribution (mean): " + " | ".join(f"{k}: " + " ".join(f"{v:+.2f}" for v in per[mk].mean(0)) for k, mk in groups.items()))
Wd = W("language_model.layers.22.mlp.down_proj.weight").float().cpu()
A = torch.stack(mlp22)                            # [n, 3584]
contrib = (A * (ut @ Wd)).numpy() / rms[:, None]   # per-neuron contribution
shift = contrib[groups["neg 6k"]].mean(0) - contrib[groups["neg alone"]].mean(0)
top = np.argsort(-np.abs(contrib[groups["neg alone"]].mean(0)))[:8]
print("\nL22 MLP: neurons with the largest mean contribution on irrelevant texts (alone -> 6k; pos alone -> 6k)")
for j in top:
    print(f"  n{j}: {contrib[groups['neg alone']][:, j].mean():+.3f} -> {contrib[groups['neg 6k']][:, j].mean():+.3f}   | "
          f"{contrib[groups['pos alone']][:, j].mean():+.3f} -> {contrib[groups['pos 6k']][:, j].mean():+.3f}")
print("L22 total shift explained by top 8:", float(shift[top].sum()), "of", float(shift.sum()))
P9 = dict(HEAD=HEAD, contrib22=contrib, groups=groups)
```

```
L15 per-head contribution (mean): neg alone: -0.42 -0.07 -0.39 +0.05 -0.20 -0.29 -0.05 -0.01 | neg 6k: -0.22 +0.00 -0.25 +0.07 +0.01 -0.13 -0.03 -0.01 | pos alone: +0.01 +0.02 +0.01 +0.03 +0.35 +0.04 +0.03 +0.01 | pos 6k: -0.04 +0.02 -0.07 +0.06 +0.17 +0.00 +0.01 -0.00
L19 per-head contribution (mean): neg alone: +0.02 +0.03 -0.03 -0.02 -0.09 -0.86 +0.09 +0.00 | neg 6k: +0.01 +0.13 -0.04 -0.03 +0.01 -0.22 +0.09 -0.01 | pos alone: +0.01 +0.09 -0.04 -0.02 +0.01 +0.92 +0.07 -0.02 | pos 6k: +0.01 +0.18 -0.05 -0.03 +0.02 +0.78 +0.08 -0.02
L23 per-head contribution (mean): neg alone: +0.02 -0.01 +0.34 -0.08 +0.02 -0.16 -0.04 -0.08 | neg 6k: +0.02 -0.07 +0.21 -0.06 +0.03 -0.20 +0.01 -0.06 | pos alone: +0.02 -0.31 -0.07 +0.01 +0.01 -0.12 +0.10 -0.03 | pos 6k: +0.02 -0.26 -0.03 -0.02 +0.03 -0.20 +0.08 -0.04

L22 MLP: neurons with the largest mean contribution on irrelevant texts (alone -> 6k; pos alone -> 6k)
  n2202: -1.505 -> -1.061   | -0.256 -> -0.428
  n577: -0.433 -> -0.250   | -0.056 -> -0.072
  n929: +0.344 -> +0.194   | +0.055 -> +0.058
  n1174: -0.212 -> -0.140   | -0.034 -> -0.063
  n519: -0.101 -> -0.090   | -0.051 -> -0.070
  n77: -0.059 -> -0.041   | -0.009 -> -0.017
  n5: +0.037 -> +0.036   | +0.035 -> +0.035
  n7: +0.021 -> +0.021   | +0.021 -> +0.021
L22 total shift explained by top 8: 0.5760494470596313 of 0.7608702182769775
[turn 9 | 68.9s | passes this cell 1, used 6/50 | cuda:0]
```

## Turn 10 (02:43:12, 0010.py)

```python
# Turn 10 (GPU, 1 pass). Turn 9: L19 head 5 is the main yes/no head; its "no"
# fades on long texts (-0.86 -> -0.22) while "yes" holds (+0.92 -> +0.78).
# Hypothesis: "no" is written by attending to a default (sink) position whose
# value means no; long on-topic text pulls attention mass off it. Test: from
# the answer position, where does head 5 put its mass (template / question /
# instruction / passage / suffix), and what does each segment's value say
# (value-vector contribution to the decision)? Recomputed inside the layer
# from its own weights and inputs (q/k norms, rotary, gate) - exact.
import numpy as np
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
rows, u, lab, var = P8["rows"], P8["u"], P8["lab"], P8["var"]
rt = S.rt; tok = rt.tokenizer
LAYER, HEAD = 19, 5
att = S.layers[LAYER].self_attn
hd = att.head_dim; kvh = HEAD // att.num_key_value_groups
Wo = W(f"language_model.layers.{LAYER}.self_attn.o_proj.weight").float()
ut = torch.tensor(u, device=Wo.device)
w_dec = (ut @ Wo[:, HEAD*hd:(HEAD+1)*hd])        # [256]: decision read of the head's output
# segments of each prompt (character spans -> token spans)
def segments(row):
    text = S.prompt(row)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    q0 = text.find(row["query"]); q1 = q0 + len(row["query"])
    i0 = text.find("Does the following"); p0 = text.find(row["text"]); p1 = p0 + len(row["text"])
    seg = []
    for a, b in enc["offset_mapping"]:
        c = (a + b) // 2
        seg.append("question" if q0 <= c < q1 else "passage" if p0 <= c < p1 else
                   "instruction" if i0 <= c < p0 else "suffix" if c >= p1 else "template")
    return seg
SEG = [segments(r_) for r_ in rows]
out = [None] * len(rows)
def hook(mod, args, kwargs, output):
    h = kwargs.get("hidden_states", args[0] if args else None)
    cos, sin = kwargs["position_embeddings"]
    B, Lw, _ = h.shape
    qg = mod.q_proj(h).view(B, Lw, -1, hd * 2)
    q, gate = torch.chunk(qg, 2, dim=-1)
    q = mod.q_norm(q).transpose(1, 2); k = mod.k_norm(mod.k_proj(h).view(B, Lw, -1, hd)).transpose(1, 2)
    v = mod.v_proj(h).view(B, Lw, -1, hd).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    logits = (q[:, HEAD, -1:].float() @ k[:, kvh].float().transpose(-1, -2)).squeeze(1) * mod.scaling   # [B, Lw]
    for j, i in enumerate(S.ctx["batch"]):
        n = S.ctx["lengths"][j]
        lg = logits[j, Lw - n:]
        pr = torch.softmax(lg, -1)
        g = torch.sigmoid(gate[j, -1, HEAD].float())
        vdec = (v[j, kvh, Lw - n:].float() * g) @ w_dec          # each position's value, read as decision
        out[i] = (lg.cpu().numpy(), pr.cpu().numpy(), vdec.cpu().numpy())
hk = att.register_forward_hook(hook, with_kwargs=True)
try:
    res = forward(rows, capture=("resid",), layers=[23])
finally:
    hk.remove()
rms = np.sqrt((res["resid"][:, 0] ** 2).mean(-1) + 1e-6)
names = ("template", "question", "instruction", "passage", "suffix")
print("check: head-5 contribution from recomputation vs turn 9:",
      np.corrcoef([float((o[1] * o[2]).sum()) / rms[i] for i, o in enumerate(out)], P9["HEAD"][19][:, 5])[0, 1])
for gname, mk in P9["groups"].items():
    idx = np.where(mk)[0]
    mass = {s: np.mean([out[i][1][np.array(SEG[i]) == s].sum() for i in idx]) for s in names}
    vals = {s: np.mean([(out[i][1][np.array(SEG[i]) == s] * out[i][2][np.array(SEG[i]) == s]).sum() / rms[i] for i in idx]) for s in names}
    vmean = {s: np.mean([out[i][2][np.array(SEG[i]) == s].mean() / rms[i] if (np.array(SEG[i]) == s).any() else np.nan for i in idx]) for s in names}
    top = np.mean([np.sort(out[i][1])[-1] for i in idx])
    print(f"\n{gname:9s} (n={len(idx)}) max single-token mass {top:.3f}")
    print("   mass:          " + "  ".join(f"{s} {mass[s]:.3f}" for s in names))
    print("   contribution:  " + "  ".join(f"{s} {vals[s]:+.3f}" for s in names))
    print("   value (mean):  " + "  ".join(f"{s} {vmean[s]:+.3f}" for s in names))
# the 5 most-attended tokens in a few irrelevant-alone and irrelevant-6k rows
for gname in ("neg alone", "neg 6k"):
    for i in np.where(P9["groups"][gname])[0][:2]:
        ids = S.tokens(rows[i]); pr = out[i][1]; vd = out[i][2] / rms[i]
        best = np.argsort(-pr)[:6]
        print(f"{gname} row {i}: " + "; ".join(f"{tok.decode([ids[t]])!r}@{t}/{len(ids)} [{SEG[i][t]}] m={pr[t]:.3f} v={vd[t]:+.2f}" for t in best))
P10 = dict(out=out, SEG=SEG, rms=rms)
```

```
check: head-5 contribution from recomputation vs turn 9: 0.9999999266402781

neg alone (n=128) max single-token mass 0.741
   mass:          template 0.018  question 0.006  instruction 0.002  passage 0.002  suffix 0.971
   contribution:  template -0.001  question -0.000  instruction -0.000  passage -0.000  suffix -0.855
   value (mean):  template -0.082  question -0.102  instruction -0.067  passage -0.099  suffix -0.037

neg 6k    (n=128) max single-token mass 0.655
   mass:          template 0.012  question 0.003  instruction 0.001  passage 0.005  suffix 0.978
   contribution:  template -0.001  question -0.000  instruction -0.000  passage -0.000  suffix -0.223
   value (mean):  template -0.100  question -0.115  instruction -0.080  passage -0.115  suffix -0.052

pos alone (n=128) max single-token mass 0.797
   mass:          template 0.011  question 0.002  instruction 0.001  passage 0.004  suffix 0.981
   contribution:  template -0.001  question -0.000  instruction -0.000  passage -0.000  suffix +0.922
   value (mean):  template -0.106  question -0.101  instruction -0.078  passage -0.105  suffix -0.076

pos 6k    (n=128) max single-token mass 0.764
   mass:          template 0.008  question 0.002  instruction 0.001  passage 0.007  suffix 0.983
   contribution:  template -0.001  question -0.000  instruction -0.000  passage -0.000  suffix +0.784
   value (mean):  template -0.115  question -0.122  instruction -0.089  passage -0.126  suffix -0.070
neg alone row 2: '\n'@384/410 [suffix] m=0.727 v=-1.05; 'A'@379/410 [suffix] m=0.146 v=-1.38; '\n\n'@390/410 [suffix] m=0.047 v=+1.20; 'B'@385/410 [suffix] m=0.014 v=+1.04; '<|im_start|>'@0/410 [template] m=0.007 v=-0.07; ' false'@383/410 [suffix] m=0.006 v=-0.04
neg alone row 6: '\n'@449/475 [suffix] m=0.722 v=-1.03; 'A'@444/475 [suffix] m=0.161 v=-1.44; '\n\n'@455/475 [suffix] m=0.039 v=+1.30; '\n\n'@474/475 [suffix] m=0.007 v=+0.16; '</think>'@473/475 [suffix] m=0.006 v=+0.09; '<|im_start|>'@0/475 [template] m=0.006 v=-0.06
neg 6k row 3: '\n'@1523/1549 [suffix] m=0.604 v=-0.93; '\n\n'@1529/1549 [suffix] m=0.184 v=+1.29; 'A'@1518/1549 [suffix] m=0.137 v=-1.27; 'B'@1524/1549 [suffix] m=0.026 v=+1.35; '<|im_start|>'@0/1549 [template] m=0.009 v=-0.09; '\n\n'@1548/1549 [suffix] m=0.006 v=+0.00
neg 6k row 7: '\n\n'@1557/1577 [suffix] m=0.610 v=+1.43; '\n'@1551/1577 [suffix] m=0.252 v=-0.93; 'A'@1546/1577 [suffix] m=0.052 v=-1.37; 'B'@1552/1577 [suffix] m=0.037 v=+1.65; '\n\n'@1576/1577 [suffix] m=0.010 v=-0.09; '<|im_start|>'@0/1577 [template] m=0.008 v=-0.09
[turn 10 | 73.8s | passes this cell 1, used 7/50 | cuda:0]
```

## Turn 11 (02:45:09, 0011.py)

```python
# Turn 11 (GPU, 1 pass). Turn 10: L19 h5 is an option pointer (97% of its
# mass on the answer options; A-tokens mean no, B-tokens yes). The length
# effect must be upstream, in what builds its query. Same exact recomputation
# for every head of L15 and L11: mass on each prompt segment from the answer
# position, its entropy over the passage, and (L15) the direct contribution.
import numpy as np
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
rows, u, lab, var = P8["rows"], P8["u"], P8["lab"], P8["var"]
SEG = [np.array(s) for s in P10["SEG"]]; rms = P10["rms"]
names = ("template", "question", "instruction", "passage", "suffix")
ut = torch.tensor(u, device="cuda:0")
store = {L: [None] * len(rows) for L in (11, 15)}
def make_hook(L):
    att = S.layers[L].self_attn; hd = att.head_dim; G = att.num_key_value_groups
    Wo = W(f"language_model.layers.{L}.self_attn.o_proj.weight").float()
    def hook(mod, args, kwargs, output):
        h = kwargs.get("hidden_states", args[0] if args else None)
        cos, sin = kwargs["position_embeddings"]
        B, Lw, _ = h.shape
        q, gate = torch.chunk(mod.q_proj(h).view(B, Lw, -1, hd * 2), 2, dim=-1)
        q = mod.q_norm(q).transpose(1, 2); k = mod.k_norm(mod.k_proj(h).view(B, Lw, -1, hd)).transpose(1, 2)
        v = mod.v_proj(h).view(B, Lw, -1, hd).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        for j, i in enumerate(S.ctx["batch"]):
            n = S.ctx["lengths"][j]
            per = []
            for hh in range(q.shape[1]):
                lg = (q[j, hh, -1].float() @ k[j, hh // G, Lw - n:].float().T) * mod.scaling
                pr = torch.softmax(lg, -1)
                g = torch.sigmoid(gate[j, -1, hh].float())
                vdec = (v[j, hh // G, Lw - n:].float() * g) @ (ut @ Wo[:, hh*hd:(hh+1)*hd])
                per.append((pr.cpu().numpy(), vdec.cpu().numpy()))
            store[L][i] = per
    return hook
hooks = [S.layers[L].self_attn.register_forward_hook(make_hook(L), with_kwargs=True) for L in store]
try:
    forward(rows)
finally:
    for hk in hooks: hk.remove()
for L in store:
    print(f"\n=== L{L} ===  per head: mass on passage / suffix / question, passage entropy (nats), contribution")
    for hh in range(8):
        line = []
        for gname, mk in P9["groups"].items():
            idx = np.where(mk)[0]
            mp = np.mean([store[L][i][hh][0][SEG[i] == "passage"].sum() for i in idx])
            ms = np.mean([store[L][i][hh][0][SEG[i] == "suffix"].sum() for i in idx])
            mq = np.mean([store[L][i][hh][0][SEG[i] == "question"].sum() for i in idx])
            def ent(i):
                p = store[L][i][hh][0][SEG[i] == "passage"]; p = p / max(p.sum(), 1e-9)
                return float(-(p * np.log(p + 1e-12)).sum())
            e = np.mean([ent(i) for i in idx])
            c = np.mean([(store[L][i][hh][0] * store[L][i][hh][1]).sum() / rms[i] for i in idx])
            line.append(f"{gname}: P{mp:.2f} S{ms:.2f} Q{mq:.2f} H{e:.1f} c{c:+.2f}")
        print(f" h{hh}: " + " | ".join(line))
P11 = store
```

```

=== L11 ===  per head: mass on passage / suffix / question, passage entropy (nats), contribution
 h0: neg alone: P0.30 S0.36 Q0.05 H3.5 c-0.04 | neg 6k: P0.59 S0.22 Q0.02 H4.4 c-0.02 | pos alone: P0.43 S0.33 Q0.03 H3.4 c-0.02 | pos 6k: P0.66 S0.19 Q0.02 H4.3 c-0.01
 h1: neg alone: P0.03 S0.87 Q0.00 H2.3 c+0.04 | neg 6k: P0.06 S0.85 Q0.00 H3.5 c+0.04 | pos alone: P0.03 S0.88 Q0.00 H2.6 c+0.04 | pos 6k: P0.06 S0.85 Q0.00 H3.6 c+0.04
 h2: neg alone: P0.02 S0.71 Q0.00 H2.7 c-0.01 | neg 6k: P0.05 S0.73 Q0.00 H3.8 c-0.00 | pos alone: P0.02 S0.73 Q0.00 H2.9 c-0.01 | pos 6k: P0.05 S0.73 Q0.00 H3.9 c-0.00
 h3: neg alone: P0.14 S0.67 Q0.01 H2.6 c+0.02 | neg 6k: P0.33 S0.52 Q0.01 H3.6 c+0.02 | pos alone: P0.21 S0.64 Q0.01 H2.6 c+0.01 | pos 6k: P0.40 S0.47 Q0.01 H3.6 c+0.02
 h4: neg alone: P0.22 S0.34 Q0.04 H2.7 c+0.04 | neg 6k: P0.55 S0.20 Q0.02 H3.2 c+0.03 | pos alone: P0.50 S0.26 Q0.02 H2.3 c+0.01 | pos 6k: P0.71 S0.13 Q0.01 H2.9 c+0.01
 h5: neg alone: P0.20 S0.49 Q0.01 H2.6 c-0.01 | neg 6k: P0.40 S0.36 Q0.01 H3.8 c-0.01 | pos alone: P0.17 S0.47 Q0.01 H2.9 c-0.01 | pos 6k: P0.38 S0.37 Q0.01 H3.8 c-0.01
 h6: neg alone: P0.05 S0.70 Q0.02 H3.3 c-0.01 | neg 6k: P0.10 S0.67 Q0.02 H4.4 c-0.01 | pos alone: P0.11 S0.67 Q0.02 H3.1 c-0.01 | pos 6k: P0.13 S0.65 Q0.02 H4.2 c-0.01
 h7: neg alone: P0.01 S0.42 Q0.00 H2.8 c+0.01 | neg 6k: P0.02 S0.47 Q0.00 H3.6 c+0.01 | pos alone: P0.01 S0.41 Q0.00 H2.7 c+0.01 | pos 6k: P0.03 S0.46 Q0.00 H3.4 c+0.01

=== L15 ===  per head: mass on passage / suffix / question, passage entropy (nats), contribution
 h0: neg alone: P0.06 S0.91 Q0.01 H2.2 c-0.42 | neg 6k: P0.33 S0.63 Q0.01 H2.7 c-0.22 | pos alone: P0.45 S0.51 Q0.01 H1.8 c+0.01 | pos 6k: P0.62 S0.34 Q0.01 H2.3 c-0.04
 h1: neg alone: P0.19 S0.53 Q0.12 H2.2 c-0.07 | neg 6k: P0.50 S0.25 Q0.11 H2.8 c+0.00 | pos alone: P0.65 S0.19 Q0.06 H1.8 c+0.02 | pos 6k: P0.75 S0.11 Q0.05 H2.4 c+0.02
 h2: neg alone: P0.03 S0.95 Q0.00 H2.3 c-0.39 | neg 6k: P0.22 S0.75 Q0.01 H3.0 c-0.25 | pos alone: P0.23 S0.72 Q0.01 H1.9 c+0.01 | pos 6k: P0.42 S0.54 Q0.01 H2.5 c-0.07
 h3: neg alone: P0.21 S0.64 Q0.02 H2.2 c+0.05 | neg 6k: P0.37 S0.45 Q0.02 H3.7 c+0.07 | pos alone: P0.17 S0.57 Q0.02 H2.7 c+0.03 | pos 6k: P0.32 S0.46 Q0.02 H3.8 c+0.06
 h4: neg alone: P0.13 S0.85 Q0.00 H2.4 c-0.20 | neg 6k: P0.43 S0.54 Q0.00 H3.6 c+0.01 | pos alone: P0.11 S0.86 Q0.00 H2.7 c+0.35 | pos 6k: P0.38 S0.59 Q0.00 H3.7 c+0.17
 h5: neg alone: P0.11 S0.86 Q0.02 H2.8 c-0.29 | neg 6k: P0.36 S0.59 Q0.02 H3.7 c-0.13 | pos alone: P0.36 S0.59 Q0.02 H2.4 c+0.04 | pos 6k: P0.54 S0.41 Q0.02 H3.1 c+0.00
 h6: neg alone: P0.03 S0.96 Q0.00 H2.2 c-0.05 | neg 6k: P0.09 S0.89 Q0.00 H3.5 c-0.03 | pos alone: P0.04 S0.92 Q0.00 H2.4 c+0.03 | pos 6k: P0.11 S0.86 Q0.00 H3.4 c+0.01
 h7: neg alone: P0.08 S0.90 Q0.00 H2.3 c-0.01 | neg 6k: P0.21 S0.76 Q0.00 H3.7 c-0.01 | pos alone: P0.06 S0.92 Q0.00 H2.5 c+0.01 | pos 6k: P0.18 S0.79 Q0.00 H3.7 c-0.00
[turn 11 | 77.2s | passes this cell 1, used 8/50 | cuda:0]
```

## Turn 12 (02:47:42, 0012.py)

```python
# Turn 12 (GPU, 3 passes). Edit 1: log-n attention sharpening of L15, the
# layer whose default-no heads (h0, h2, h5) leak mass to long passages.
# q_norm is shared by the 8 heads and removes any q_proj scale, so the
# sharpening goes into (1 + q_norm.weight): x tau, tau = log(1500)/log(350)
# ~ 1.25 (long legal chunk vs training prompt length). Score on the probe
# (alone + 6k, 1 pass) and PRACTICE (2 passes) against the unedited numbers
# of turns 5 and 7.
import numpy as np
def sharpen(L, tau):
    w = W(f"language_model.layers.{L}.self_attn.q_norm.weight")
    w.copy_((((1 + w.float()) * tau) - 1).to(w.dtype))
def probe_eval(sel_variants=("alone", "6k-middle")):
    rows = [x for x in probe if x["variant"] in sel_variants]
    p = forward(rows)["p"]
    out = {}
    for v in sel_variants:
        idx = [i for i, x in enumerate(rows) if x["variant"] == v]
        lab_ = [rows[i]["label"] for i in idx]
        out[v] = metrics(p[idx], lab_) | {"pos": round(float(np.mean([p[i] for i in idx if rows[i]["label"]])), 3),
                                         "neg": round(float(np.mean([p[i] for i in idx if not rows[i]["label"]])), 3)}
    # mixed-length ranking: all rows together (long and short compete, as in navigation)
    out["mixed"] = metrics(p, [x["label"] for x in rows])
    return out
def practice_eval():
    e = evaluate(PRACTICE); e.pop("_p")
    return {k: e[k] for k in ("all", "page", "passage")}
sharpen(15, 1.25)
print("edit: L15 q_norm x1.25;", changed())
print("probe:", probe_eval())
print("practice:", practice_eval())
print("unedited reference: probe alone auc 0.9683 / 6k 0.8252 (turn 7); practice all auc 0.9357 acc 0.902, passage auc 0.8812 (turn 5)")
```

```
edit: L15 q_norm x1.25; {'language_model.layers.15.self_attn.q_norm.weight': 256, '_total': 256, '_share': '0.0000% of 852,987,968'}
probe: {'alone': {'n': 256, 'auc': 0.9686, 'acc': 0.8594, 'logloss': 0.3274, 'brier': 0.1047, 'pos_rate_pred': 0.3984, 'pos_rate_true': 0.5, 'pos': 0.68, 'neg': 0.074}, '6k-middle': {'n': 256, 'auc': 0.8254, 'acc': 0.7578, 'logloss': 0.5424, 'brier': 0.1788, 'pos_rate_pred': 0.4141, 'pos_rate_true': 0.5, 'pos': 0.587, 'neg': 0.247}, 'mixed': {'n': 512, 'auc': 0.904, 'acc': 0.8086, 'logloss': 0.4349, 'brier': 0.1418, 'pos_rate_pred': 0.4062, 'pos_rate_true': 0.5}}
practice: {'all': {'n': 806, 'auc': 0.9356, 'acc': 0.9007, 'logloss': 0.2505, 'brier': 0.0737, 'pos_rate_pred': 0.1514, 'pos_rate_true': 0.1836}, 'page': {'n': 392, 'auc': 0.977, 'acc': 0.9541, 'logloss': 0.1322, 'brier': 0.036, 'pos_rate_pred': 0.1276, 'pos_rate_true': 0.1429}, 'passage': {'n': 411, 'auc': 0.8809, 'acc': 0.8491, 'logloss': 0.3646, 'brier': 0.1102, 'pos_rate_pred': 0.1752, 'pos_rate_true': 0.2238}}
unedited reference: probe alone auc 0.9683 / 6k 0.8252 (turn 7); practice all auc 0.9357 acc 0.902, passage auc 0.8812 (turn 5)
[turn 12 | 106.1s | passes this cell 3, used 11/50 | cuda:0]
```

## Turn 13 (02:49:49, 0013.py)

```python
# Turn 13 (GPU, 1 pass), edit 1 still applied. Why did x1.25 sharpening of
# L15 change nothing? Measure L15 heads 0/2/5 directly: passage mass now vs
# unedited (turn 11), and decompose the passage-vs-suffix competition into
# count and per-token terms: LSE(passage) - LSE(suffix) = log n_passage +
# [log mean exp(l_passage) - LSE(suffix)]. If the leak were count-driven, the
# per-token term would be stable across lengths; if it is query-driven, the
# per-token term itself rises in long texts.
import numpy as np
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
rows = P8["rows"]; SEG = [np.array(s) for s in P10["SEG"]]
L = 15; att = S.layers[L].self_attn; hd = att.head_dim; G = att.num_key_value_groups
HEADS = (0, 2, 5, 4)
got = [None] * len(rows)
def hook(mod, args, kwargs, output):
    h = kwargs.get("hidden_states", args[0] if args else None)
    cos, sin = kwargs["position_embeddings"]
    B, Lw, _ = h.shape
    q, gate = torch.chunk(mod.q_proj(h).view(B, Lw, -1, hd * 2), 2, dim=-1)
    q = mod.q_norm(q).transpose(1, 2); k = mod.k_norm(mod.k_proj(h).view(B, Lw, -1, hd)).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    for j, i in enumerate(S.ctx["batch"]):
        n = S.ctx["lengths"][j]
        got[i] = {hh: ((q[j, hh, -1].float() @ k[j, hh // G, Lw - n:].float().T) * mod.scaling).cpu().numpy() for hh in HEADS}
hk = att.register_forward_hook(hook, with_kwargs=True)
try:
    p = forward(rows)["p"]
finally:
    hk.remove()
from scipy.special import logsumexp, softmax
for hh in HEADS:
    print(f"L15 h{hh}:")
    for gname, mk in P9["groups"].items():
        idx = np.where(mk)[0]
        mass_now = np.mean([softmax(got[i][hh])[SEG[i] == "passage"].sum() for i in idx])
        mass_before = np.mean([P11[15][i][hh][0][SEG[i] == "passage"].sum() for i in idx])
        lp = [got[i][hh][SEG[i] == "passage"] for i in idx]; ls = [got[i][hh][SEG[i] == "suffix"] for i in idx]
        logn = np.mean([np.log(len(a)) for a in lp])
        per_tok = np.mean([logsumexp(a) - np.log(len(a)) - logsumexp(b) for a, b in zip(lp, ls)])
        top_gap = np.mean([a.max() - b.max() for a, b in zip(lp, ls)])
        print(f"   {gname:9s} passage mass {mass_before:.2f} -> {mass_now:.2f} (x1.25)   log n {logn:.2f}   "
              f"per-token term {per_tok:+.2f}   max(passage) - max(suffix) {top_gap:+.2f}")
```

```
L15 h0:
   neg alone passage mass 0.06 -> 0.04 (x1.25)   log n 5.44   per-token term -11.07   max(passage) - max(suffix) -6.45
   neg 6k    passage mass 0.33 -> 0.27 (x1.25)   log n 7.23   per-token term -9.08   max(passage) - max(suffix) -2.77
   pos alone passage mass 0.45 -> 0.45 (x1.25)   log n 5.56   per-token term -5.94   max(passage) - max(suffix) -0.64
   pos 6k    passage mass 0.62 -> 0.61 (x1.25)   log n 7.24   per-token term -6.66   max(passage) - max(suffix) +0.04
L15 h2:
   neg alone passage mass 0.03 -> 0.02 (x1.25)   log n 5.44   per-token term -11.63   max(passage) - max(suffix) -7.10
   neg 6k    passage mass 0.22 -> 0.15 (x1.25)   log n 7.23   per-token term -9.96   max(passage) - max(suffix) -3.87
   pos alone passage mass 0.23 -> 0.18 (x1.25)   log n 5.56   per-token term -7.44   max(passage) - max(suffix) -2.39
   pos 6k    passage mass 0.42 -> 0.36 (x1.25)   log n 7.24   per-token term -7.99   max(passage) - max(suffix) -1.52
L15 h5:
   neg alone passage mass 0.11 -> 0.06 (x1.25)   log n 5.44   per-token term -9.18   max(passage) - max(suffix) -4.83
   neg 6k    passage mass 0.36 -> 0.27 (x1.25)   log n 7.23   per-token term -8.61   max(passage) - max(suffix) -2.64
   pos alone passage mass 0.36 -> 0.32 (x1.25)   log n 5.56   per-token term -6.45   max(passage) - max(suffix) -1.41
   pos 6k    passage mass 0.54 -> 0.50 (x1.25)   log n 7.24   per-token term -7.19   max(passage) - max(suffix) -0.70
L15 h4:
   neg alone passage mass 0.13 -> 0.08 (x1.25)   log n 5.44   per-token term -8.36   max(passage) - max(suffix) -3.82
   neg 6k    passage mass 0.43 -> 0.32 (x1.25)   log n 7.23   per-token term -8.05   max(passage) - max(suffix) -2.12
   pos alone passage mass 0.11 -> 0.06 (x1.25)   log n 5.56   per-token term -8.84   max(passage) - max(suffix) -4.36
   pos 6k    passage mass 0.38 -> 0.25 (x1.25)   log n 7.24   per-token term -8.46   max(passage) - max(suffix) -2.75
[turn 13 | 71.1s | passes this cell 1, used 12/50 | cuda:0]
```

## Turn 14 (02:50:32, 0014.py)

```python
# Turn 14 (no pass). Revert edit 1 (no gain). Then, from the unedited
# captures of turn 8 (answer position, every layer): is "more on-topic
# filler" a consistent direction in the residual stream, and how does it
# relate to the relevance direction?
#   d_len(L) = mean(6k) - mean(alone), pooled over labels
#   d_lab(L) = mean(pos) - mean(neg), pooled over lengths
# Consistency: AUC of the projection on d_len separating alone vs 6k within
# each label; overlap: cos(d_len, d_lab); harm: the projection of d_len on
# the decision lens direction at the last layers.
import numpy as np
revert(); print("reverted:", changed())
res, lab, var = P8["res"], P8["lab"], P8["var"]
H = res["resid"]                                   # [n, 24, 1024]
is6 = var == "6k-middle"
print("layer | |d_len| |d_lab| cos(len,lab) | len-AUC neg / pos | lab-AUC alone / 6k (proj on d_lab)")
DL = {}
for L in range(24):
    X = H[:, L]
    d_len = X[is6].mean(0) - X[~is6].mean(0); d_lab = X[lab == 1].mean(0) - X[lab == 0].mean(0)
    DL[L] = d_len
    c = float(d_len @ d_lab / np.linalg.norm(d_len) / np.linalg.norm(d_lab))
    pl = X @ (d_len / np.linalg.norm(d_len)); pb = X @ (d_lab / np.linalg.norm(d_lab))
    a_neg = auc(pl[lab == 0].tolist(), is6[lab == 0].astype(int).tolist())
    a_pos = auc(pl[lab == 1].tolist(), is6[lab == 1].astype(int).tolist())
    b_al = auc(pb[~is6].tolist(), lab[~is6].tolist()); b_6 = auc(pb[is6].tolist(), lab[is6].tolist())
    print(f"L{L:2d} | {np.linalg.norm(d_len):6.2f} {np.linalg.norm(d_lab):6.2f} {c:+.2f} | {a_neg:.3f} / {a_pos:.3f} | {b_al:.3f} / {b_6:.3f}")
# which writers produce the length shift at the input of L15 (= output of L14)?
target = DL[14] / np.linalg.norm(DL[14])
print("\nwriters' shift along d_len(L14) (6k minus alone, mean over rows):")
contrib = []
for L in range(15):
    for kind in ("attn_out", "mlp_out"):
        Y = res[kind][:, L]
        contrib.append((float((Y[is6].mean(0) - Y[~is6].mean(0)) @ target), L, kind))
tot = sum(c for c, _, _ in contrib)
for c, L, kind in sorted(contrib, key=lambda x: -abs(x[0]))[:10]:
    print(f"  L{L:2d} {kind:8s} {c:+.3f}  ({100 * c / tot:.0f}% of {tot:.2f})")
P14 = dict(DL=DL)
```

```
reverted: {'_total': 0, '_share': '0.0000% of 852,987,968'}
layer | |d_len| |d_lab| cos(len,lab) | len-AUC neg / pos | lab-AUC alone / 6k (proj on d_lab)
L 0 |   0.09   0.00 +0.64 | 1.000 / 1.000 | 0.647 / 0.538
L 1 |   0.11   0.01 +0.58 | 1.000 / 1.000 | 0.656 / 0.546
L 2 |   0.12   0.01 +0.54 | 1.000 / 1.000 | 0.716 / 0.549
L 3 |   0.19   0.01 +0.54 | 1.000 / 1.000 | 0.714 / 0.556
L 4 |   0.20   0.01 +0.64 | 1.000 / 1.000 | 0.663 / 0.536
L 5 |   0.19   0.01 +0.67 | 1.000 / 1.000 | 0.668 / 0.538
L 6 |   0.19   0.01 +0.64 | 1.000 / 1.000 | 0.687 / 0.543
L 7 |   0.18   0.01 +0.64 | 1.000 / 1.000 | 0.733 / 0.549
L 8 |   0.17   0.01 +0.50 | 1.000 / 1.000 | 0.778 / 0.560
L 9 |   0.19   0.03 +0.24 | 1.000 / 1.000 | 0.812 / 0.599
L10 |   0.18   0.05 +0.24 | 1.000 / 1.000 | 0.881 / 0.629
L11 |   0.37   0.29 +0.29 | 0.999 / 1.000 | 0.934 / 0.799
L12 |   0.44   0.47 +0.26 | 0.998 / 1.000 | 0.949 / 0.795
L13 |   0.66   1.35 +0.43 | 0.935 / 0.921 | 0.964 / 0.804
L14 |   0.75   1.60 +0.35 | 0.939 / 0.953 | 0.968 / 0.807
L15 |   1.32   3.82 +0.53 | 0.908 / 0.736 | 0.967 / 0.820
L16 |   1.32   4.00 +0.51 | 0.901 / 0.746 | 0.967 / 0.818
L17 |   1.32   3.98 +0.50 | 0.906 / 0.753 | 0.967 / 0.818
L18 |   1.36   4.08 +0.49 | 0.909 / 0.759 | 0.967 / 0.817
L19 |   1.59   5.31 +0.57 | 0.899 / 0.683 | 0.968 / 0.819
L20 |   1.67   5.30 +0.55 | 0.899 / 0.728 | 0.968 / 0.819
L21 |   1.75   5.49 +0.54 | 0.900 / 0.747 | 0.968 / 0.819
L22 |   2.00   6.38 +0.59 | 0.894 / 0.705 | 0.968 / 0.823
L23 |   2.75   6.80 +0.02 | 0.651 / 0.816 | 0.956 / 0.740

writers' shift along d_len(L14) (6k minus alone, mean over rows):
  L14 attn_out +0.169  (23% of 0.75)
  L13 attn_out +0.158  (21% of 0.75)
  L13 mlp_out  +0.127  (17% of 0.75)
  L11 attn_out +0.088  (12% of 0.75)
  L12 attn_out +0.084  (11% of 0.75)
  L14 mlp_out  +0.055  (7% of 0.75)
  L11 mlp_out  +0.017  (2% of 0.75)
  L 6 attn_out +0.015  (2% of 0.75)
  L 9 attn_out +0.015  (2% of 0.75)
  L 5 attn_out +0.013  (2% of 0.75)
[turn 14 | 2.5s | passes this cell 0, used 12/50 | cuda:0]
```

## Turn 15 (02:52:46, 0015.py)

```python
# Turn 15 (GPU, 3 passes). Edit 2: remove the length ("more on-topic
# filler") direction d = d_len(L14)/|d_len(L14)| from what the layers that
# write it can write: W <- W - s * d d^T W for the token-mixer output
# projection (o_proj of L11, out_proj of L12-14) and the MLP down_proj of
# L11-14 (93% of the shift into L15). s = 1.0 first.
import numpy as np
d = torch.tensor(P14["DL"][14] / np.linalg.norm(P14["DL"][14]), dtype=torch.float32, device="cuda:0")
WRITERS = []
for L in (11, 12, 13, 14):
    mix = "self_attn.o_proj" if S.layer_types[L] == "full_attention" else "linear_attn.out_proj"
    WRITERS += [f"language_model.layers.{L}.{mix}.weight", f"language_model.layers.{L}.mlp.down_proj.weight"]
def remove_direction(names_, d, s):
    for n in names_:
        w = W(n); wf = w.float()
        w.copy_((wf - s * torch.outer(d, d @ wf)).to(w.dtype))
remove_direction(WRITERS, d, 1.0)
print("edit 2 at s=1.0:", {k: v for k, v in changed().items() if k.startswith("_")})
print("probe:", probe_eval())
print("practice:", practice_eval())
print("unedited: probe alone 0.9683 / 6k 0.8252 / mixed 0.904 (turn 12, ~unchanged); practice all 0.9357, passage 0.8812")
```

```
edit 2 at s=1.0: {'_total': 22549726, '_share': '2.6436% of 852,987,968'}
probe: {'alone': {'n': 256, 'auc': 0.9669, 'acc': 0.8516, 'logloss': 0.3428, 'brier': 0.1097, 'pos_rate_pred': 0.3906, 'pos_rate_true': 0.5, 'pos': 0.658, 'neg': 0.079}, '6k-middle': {'n': 256, 'auc': 0.8195, 'acc': 0.7148, 'logloss': 0.5789, 'brier': 0.1926, 'pos_rate_pred': 0.3398, 'pos_rate_true': 0.5, 'pos': 0.528, 'neg': 0.215}, 'mixed': {'n': 512, 'auc': 0.9029, 'acc': 0.7832, 'logloss': 0.4609, 'brier': 0.1512, 'pos_rate_pred': 0.3652, 'pos_rate_true': 0.5}}
practice: {'all': {'n': 806, 'auc': 0.9366, 'acc': 0.9057, 'logloss': 0.2435, 'brier': 0.073, 'pos_rate_pred': 0.1439, 'pos_rate_true': 0.1836}, 'page': {'n': 392, 'auc': 0.9775, 'acc': 0.9541, 'logloss': 0.1278, 'brier': 0.0358, 'pos_rate_pred': 0.1276, 'pos_rate_true': 0.1429}, 'passage': {'n': 411, 'auc': 0.8828, 'acc': 0.8589, 'logloss': 0.3548, 'brier': 0.1089, 'pos_rate_pred': 0.1606, 'pos_rate_true': 0.2238}}
unedited: probe alone 0.9683 / 6k 0.8252 / mixed 0.904 (turn 12, ~unchanged); practice all 0.9357, passage 0.8812
[turn 15 | 105.7s | passes this cell 3, used 15/50 | cuda:0]
```

## Turn 16 (02:57:25, 0016.py)

```python
# Turn 16 (GPU, 6 passes). Revert edit 2 (no gain). New hypothesis from the
# same-60-question comparison (zero-shot vs tuned): the fine-tune is a large
# in-domain gain but slightly negative out of domain (LegalBench CUAD MRR
# 0.55 -> 0.38). Robust fine-tuning (WiSE-FT, Wortsman et al. 2022):
# interpolate in weight space between the zero-shot and the fine-tuned
# judge, W = W0 + alpha * dLoRA, readout rows the same way.
# Selection rule, fixed before any result: maximise mean(practice AUC,
# probe mixed-length AUC) over alpha in {0, 0.5, 0.75, 1.0, 1.25}.
# This turn: alpha = 0 (the zero-shot judge, reference) and alpha = 0.5.
import numpy as np
from safetensors.torch import load_file
from bench.wsp.kernel import TUNED
revert(); print("reverted:", changed()["_total"])
saved = torch.load(TUNED + "/judge.pt", map_location="cpu", weights_only=True)
scale = saved["alpha"] / saved["r"]
DLORA = {}
for name, A in saved["lora"].items():
    if ".lora_A." in name:
        mod = name.split(".lora_A.")[0]
        B = saved["lora"][name.replace(".lora_A.", ".lora_B.")]
        DLORA[mod + ".weight"] = ((B.float() @ A.float()) * scale)
R_TUNED = S._orig_readout.float()
R_BASE = load_file(str(S.rt.root / "readout.safetensors"))["weight"][:2].float()
print(f"{len(DLORA)} LoRA-merged matrices; readout rows tuned vs base |d| {float((R_TUNED - R_BASE).norm()):.4f}")
def set_alpha(alpha, layers=None):
    """W = tuned + (alpha - 1) * dLoRA (from the unedited originals); layers: only these layer indices."""
    revert()
    for n, d in DLORA.items():
        L = int(n.split("layers.")[1].split(".")[0])
        if layers is not None and L not in layers:
            continue
        w = W(n)
        w.copy_((orig(n).float() + (alpha - 1) * d).to(w.device, w.dtype))
    if layers is None:
        S.rt.readout[:2] = (R_TUNED + (alpha - 1) * (R_TUNED - R_BASE)).to(S.rt.readout.device, S.rt.readout.dtype)
RESULTS = {}
for a in (0.0, 0.5):
    set_alpha(a)
    pr = probe_eval(); pa = practice_eval()
    RESULTS[a] = (pa["all"]["auc"], pr["mixed"]["auc"], pa, pr)
    print(f"\nalpha {a}: practice auc {pa['all']['auc']} (page {pa['page']['auc']}, passage {pa['passage']['auc']}, acc {pa['all']['acc']}) | "
          f"probe alone {pr['alone']['auc']} 6k {pr['6k-middle']['auc']} mixed {pr['mixed']['auc']} | score {(pa['all']['auc'] + pr['mixed']['auc']) / 2:.4f}")
RESULTS[1.0] = (0.9357, 0.904, None, None)
print("\nalpha 1.0 (unedited, turns 5/12): practice auc 0.9357 | probe mixed 0.904 | score", round((0.9357 + 0.904) / 2, 4))
```

```
reverted: 0
150 LoRA-merged matrices; readout rows tuned vs base |d| 0.0493

alpha 0.0: practice auc 0.8462 (page 0.9124, passage 0.7817, acc 0.7878) | probe alone 0.8543 6k 0.6893 mixed 0.7703 | score 0.8082

alpha 0.5: practice auc 0.923 (page 0.9733, passage 0.8621, acc 0.8809) | probe alone 0.9427 6k 0.795 mixed 0.8685 | score 0.8958

alpha 1.0 (unedited, turns 5/12): practice auc 0.9357 | probe mixed 0.904 | score 0.9199
[turn 16 | 213.8s | passes this cell 6, used 21/50 | cuda:0]
```

## Turn 17 (03:01:18, 0017.py)

```python
# Turn 17 (GPU, 6 passes). Turn 16: interpolating toward zero-shot lowers
# both signals (alpha 0: 0.808, 0.5: 0.896, 1.0: 0.920); the fine-tune made
# the judge more length-robust. Finish the pre-stated grid: 0.75 and 1.25
# (1.25 = extrapolating along the fine-tune direction).
for a in (0.75, 1.25):
    set_alpha(a)
    pr = probe_eval(); pa = practice_eval()
    RESULTS[a] = (pa["all"]["auc"], pr["mixed"]["auc"], pa, pr)
    print(f"alpha {a}: practice auc {pa['all']['auc']} (page {pa['page']['auc']}, passage {pa['passage']['auc']}, acc {pa['all']['acc']}, "
          f"logloss {pa['all']['logloss']}) | probe alone {pr['alone']['auc']} 6k {pr['6k-middle']['auc']} mixed {pr['mixed']['auc']} | "
          f"score {(pa['all']['auc'] + pr['mixed']['auc']) / 2:.4f}")
set_alpha(1.0); print("back to alpha 1.0:", changed()["_total"])
```

```
alpha 0.75: practice auc 0.9325 (page 0.9764, passage 0.8753, acc 0.8983, logloss 0.2503) | probe alone 0.9619 6k 0.8204 mixed 0.8952 | score 0.9139
alpha 1.25: practice auc 0.9347 (page 0.9757, passage 0.8793, acc 0.902, logloss 0.2673) | probe alone 0.9658 6k 0.8111 mixed 0.8999 | score 0.9173
back to alpha 1.0: 0
[turn 17 | 215.6s | passes this cell 6, used 27/50 | cuda:0]
```

## Turn 18 (03:02:34, 0018.py)

```python
# Turn 18 (GPU, 2 passes). Turn 17: alpha = 1.0 is the optimum of the grid
# (0.808 / 0.896 / 0.914 / 0.920 / 0.917); no interpolation edit. Last
# hypothesis: a component whose direct write to the decision tracks the
# errors (lookalike false positives, missed positives) more than the labels.
# Screen every head (full attention: o_proj columns per head; linear
# attention: out_proj columns per head) and every MLP: approximate practice
# AUC if its direct write were removed, z' = z - dla_c / T. Direct effect
# only; the best candidates are then checked with real passes.
import numpy as np
T = float(getattr(S.rt, "temperature", 1.0)); print("readout temperature", T)
ut = torch.tensor(P8["u"], device="cuda:0")
heads = {}   # (L, kind) -> list of per-row [n_heads] contributions (before /rms)
def make(L, kind, mod, nh):
    Wo = mod.weight.float()
    hd = Wo.shape[1] // nh
    proj = torch.stack([ut @ Wo[:, h*hd:(h+1)*hd] for h in range(nh)])   # [nh, hd]
    store = heads.setdefault((L, kind), [None] * len(PRACTICE))
    def hook(m, inp):
        x = inp[0][:, -1].float().view(inp[0].shape[0], nh, hd)
        c = (x * proj[None]).sum(-1).cpu().numpy()
        for j, i in enumerate(S.ctx["batch"]):
            store[i] = c[j]
    return hook
hooks = []
for L, layer in enumerate(S.layers):
    if S.layer_types[L] == "full_attention":
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make(L, "attn", layer.self_attn.o_proj, 8)))
    else:
        nv = S.rt.backbone.language_model.config.linear_num_value_heads
        hooks.append(layer.linear_attn.out_proj.register_forward_pre_hook(make(L, "lin", layer.linear_attn.out_proj, nv)))
try:
    res = forward(PRACTICE, capture=("mlp_out", "resid"), layers=list(range(24)))
finally:
    for h in hooks: h.remove()
p = res["p"]; z = np.log(p / (1 - p)); y = np.array([r["label"] for r in PRACTICE])
rms = np.sqrt((res["resid"][:, 23] ** 2).mean(-1) + 1e-6)
base_auc = auc(z.tolist(), y.tolist()); print("practice auc", round(base_auc, 4))
cands = []
for (L, kind), st in heads.items():
    C = np.stack(st) / rms[:, None]
    for h in range(C.shape[1]):
        cands.append((auc((z - C[:, h] / T).tolist(), y.tolist()) - base_auc, f"L{L} {kind} h{h}", float(np.abs(C[:, h]).mean())))
mo = (res["mlp_out"] @ P8["u"]) / rms[:, None]
for L in range(24):
    cands.append((auc((z - mo[:, L] / T).tolist(), y.tolist()) - base_auc, f"L{L} mlp", float(np.abs(mo[:, L]).mean())))
cands.sort(reverse=True)
print("removal would most help (approx AUC gain, mean |contribution|):")
for g, name, m in cands[:10]: print(f"  {name:16s} {g:+.4f}  |c| {m:.3f}")
print("removal would most hurt:")
for g, name, m in cands[-6:]: print(f"  {name:16s} {g:+.4f}  |c| {m:.3f}")
P18 = dict(heads=heads, mo=mo, rms=rms, z=z, y=y, T=T)
```

```
readout temperature 1.0
practice auc 0.9357
removal would most help (approx AUC gain, mean |contribution|):
  L19 attn h5      +0.0016  |c| 1.110
  L22 mlp          +0.0009  |c| 2.152
  L15 attn h1      +0.0002  |c| 0.142
  L19 mlp          +0.0001  |c| 0.143
  L7 attn h3       +0.0001  |c| 0.012
  L16 lin h7       +0.0001  |c| 0.062
  L23 mlp          +0.0001  |c| 0.726
  L15 attn h5      +0.0001  |c| 0.304
  L14 lin h12      +0.0001  |c| 0.014
  L23 attn h2      +0.0000  |c| 0.341
removal would most hurt:
  L14 lin h3       -0.0001  |c| 0.014
  L13 lin h11      -0.0001  |c| 0.035
  L15 attn h6      -0.0002  |c| 0.049
  L23 attn h5      -0.0002  |c| 0.117
  L19 attn h4      -0.0004  |c| 0.168
  L22 lin h12      -0.0023  |c| 0.261
[turn 18 | 35.6s | passes this cell 2, used 29/50 | cuda:0]
```

## Turn 19 (03:02:55, 0019.py)

```python
# Turn 19 (no pass). Turn 18: no component's removal moves practice AUC by
# more than +0.0016 (direct effect); the errors are spread, as for Gemma in
# the blog. Mirror check, free on the turn-18 captures: amplify one
# component's direct write, z' = z + beta * c, beta in {0.5, 1.0}.
import numpy as np
h, mo, rms, z, y = P18["heads"], P18["mo"], P18["rms"], P18["z"], P18["y"]
base = auc(z.tolist(), y.tolist()); out = []
for (L, kind), st in h.items():
    C = np.stack(st) / rms[:, None]
    for hh in range(C.shape[1]):
        for b in (0.5, 1.0):
            out.append((auc((z + b * C[:, hh]).tolist(), y.tolist()) - base, f"L{L} {kind} h{hh} x{1 + b}"))
for L in range(24):
    for b in (0.5, 1.0):
        out.append((auc((z + b * mo[:, L]).tolist(), y.tolist()) - base, f"L{L} mlp x{1 + b}"))
out.sort(reverse=True)
print("best amplifications (approx AUC gain):", [(n, round(g, 4)) for g, n in out[:8]])
```

```
best amplifications (approx AUC gain): [('L22 lin h12 x2.0', 0.0018), ('L22 lin h12 x1.5', 0.0007), ('L19 attn h4 x2.0', 0.0003), ('L23 mlp x2.0', 0.0002), ('L19 attn h4 x1.5', 0.0002), ('L23 attn h5 x2.0', 0.0001), ('L19 attn h1 x1.5', 0.0001), ('L19 attn h1 x2.0', 0.0001)]
[turn 19 | 0.3s | passes this cell 0, used 29/50 | cuda:0]
```

## Turn 20 (03:05:05, 0020.py)

```python
# Turn 20 (GPU, 3 passes). Verify the best screened candidate for real
# (downstream effects included): L22 linear-attention head 12 x2 (out_proj
# columns of that head). Keep only on a clear gain (>= +0.003 AUC on both
# practice and the probe mixed set); otherwise revert and end without edit.
nv = S.rt.backbone.language_model.config.linear_num_value_heads
n = "language_model.layers.22.linear_attn.out_proj.weight"
w = W(n); hd = w.shape[1] // nv
w[:, 12*hd:13*hd] *= 2.0
print("edit:", {k: v for k, v in changed().items()})
pr = probe_eval(); pa = practice_eval()
print(f"practice auc {pa['all']['auc']} (passage {pa['passage']['auc']}, acc {pa['all']['acc']}) | probe alone {pr['alone']['auc']} "
      f"6k {pr['6k-middle']['auc']} mixed {pr['mixed']['auc']}")
print("unedited: practice 0.9357 (passage 0.8812, acc 0.902) | probe alone 0.9683 6k 0.8252 mixed 0.904")
gain = (pa["all"]["auc"] - 0.9357, pr["mixed"]["auc"] - 0.904)
print("gain", gain, "->", "keep" if min(gain) >= 0.003 else "revert")
if min(gain) < 0.003:
    revert(); print("reverted:", changed()["_total"])
```

```
edit: {'language_model.layers.22.linear_attn.out_proj.weight': 131072, '_total': 131072, '_share': '0.0154% of 852,987,968'}
practice auc 0.9366 (passage 0.883, acc 0.902) | probe alone 0.9677 6k 0.8256 mixed 0.9041
unedited: practice 0.9357 (passage 0.8812, acc 0.902) | probe alone 0.9683 6k 0.8252 mixed 0.904
gain (0.0009000000000000119, 9.999999999998899e-05) -> revert
reverted: 0
[turn 20 | 108.9s | passes this cell 3, used 32/50 | cuda:0]
```

## Turn 21 (03:05:26, 0021.py)

```python
# Turn 21 (no pass). Conclusion of the session; no edit is kept.
print(changed())
print("""Conclusion after 20 turns and 32 of 50 forward passes (18 left unused on purpose):
1. Found by reading + passes: the judge's decision forms in one step at L15 (full attention), is
   sharpened in L19 (head 5, an option pointer over the answer template) and L22; the fine-tune left
   these writers unchanged and changed what feeds them (early layers, up_proj).
2. Found failure: long texts (beyond the 1,600-character training range; 22.5% of LegalBench and 15.4%
   of bBSARD chunks) raise the irrelevant passages' P(yes) 0.08 -> 0.25 and lower AUC 0.97 -> 0.83.
   Cause: L15's default-no heads (h0, h2, h5) leak attention to the passage, because a long on-topic
   text contains a better-matching token (max(passage) - max(suffix) +3.7 logits) and more tokens
   (log n +1.8); and the filler writes along the relevance direction before L15 (cos 0.35-0.43).
3. Edits tried, none kept: log-n sharpening of L15 (x1.25; no change), removing the filler direction
   from the L11-14 writers (bias shift only, 6k AUC 0.825 -> 0.820), WiSE-FT interpolation and
   extrapolation of the fine-tune (alpha 1.0 is the optimum of 0 / 0.5 / 0.75 / 1.0 / 1.25), and the
   best screened component amplification (L22 lin h12 x2; +0.0009, noise).
4. As in the blog's Gemma experiment: no single component carries the remaining errors, so a weight
   edit cannot add what the model lacks. The failures that remain need data or pipeline changes:
   long-text and legal training examples, and document context for passages (ContractNLI doc_hit@1
   0.51 -> 0.07: a chunk does not say which NDA it is from).""")
```

```
{'_total': 0, '_share': '0.0000% of 852,987,968'}
Conclusion after 20 turns and 32 of 50 forward passes (18 left unused on purpose):
1. Found by reading + passes: the judge's decision forms in one step at L15 (full attention), is
   sharpened in L19 (head 5, an option pointer over the answer template) and L22; the fine-tune left
   these writers unchanged and changed what feeds them (early layers, up_proj).
2. Found failure: long texts (beyond the 1,600-character training range; 22.5% of LegalBench and 15.4%
   of bBSARD chunks) raise the irrelevant passages' P(yes) 0.08 -> 0.25 and lower AUC 0.97 -> 0.83.
   Cause: L15's default-no heads (h0, h2, h5) leak attention to the passage, because a long on-topic
   text contains a better-matching token (max(passage) - max(suffix) +3.7 logits) and more tokens
   (log n +1.8); and the filler writes along the relevance direction before L15 (cos 0.35-0.43).
3. Edits tried, none kept: log-n sharpening of L15 (x1.25; no change), removing the filler direction
   from the L11-14 writers (bias shift only, 6k AUC 0.825 -> 0.820), WiSE-FT interpolation and
   extrapolation of the fine-tune (alpha 1.0 is the optimum of 0 / 0.5 / 0.75 / 1.0 / 1.25), and the
   best screened component amplification (L22 lin h12 x2; +0.0009, noise).
4. As in the blog's Gemma experiment: no single component carries the remaining errors, so a weight
   edit cannot add what the model lacks. The failures that remain need data or pipeline changes:
   long-text and legal training examples, and document context for passages (ContractNLI doc_hit@1
   0.51 -> 0.07: a chunk does not say which NDA it is from).
[turn 21 | 2.1s | passes this cell 0, used 32/50 | cuda:0]
```

## Turn 22 (13:25:22, 0022.py)

```python
# Turn 22 (no pass). Session restarted (user: fix the found mechanism directly). Rebuild the
# turn-7 length probe exactly (same seed, TRAIN only, no forward) and the evaluation helpers.
import random
rng = random.Random(21)
byq = {}
for r_ in TRAIN:
    if r_["kind"] == "passage":
        byq.setdefault(r_["qid"], []).append(r_)
cands = [q for q, rs in byq.items() if any(x["label"] for x in rs) and sum(1 - x["label"] for x in rs) >= 4]

qs = rng.sample(cands, min(128, len(cands)))
probe = []
def pad(core, fillers, target, where):
    parts, size = [], len(core["text"])
    for f in fillers:
        if size >= target:
            break
        parts.append(f["text"]); size += len(f["text"]) + 1
    if where == "first":
        body = [core["text"]] + parts
    elif where == "last":
        body = parts + [core["text"]]
    else:
        h = len(parts) // 2
        body = parts[:h] + [core["text"]] + parts[h:]
    return "\n".join(body)
for q in qs:
    rs = byq[q]
    pos = rng.choice([x for x in rs if x["label"]])
    negs = [x for x in rs if not x["label"]]
    rng.shuffle(negs)
    neg, fill = negs[0], negs[1:]
    for core in (pos, neg):
        for variant, target, where in (("alone", 0, "first"), ("3k-first", 3000, "first"),
                                       ("3k-last", 3000, "last"), ("6k-middle", 6000, "middle")):
            text = core["text"] if variant == "alone" else pad(core, fill, target, where)
            probe.append({"query": core["query"], "kind": "passage", "text": text, "label": core["label"],
                          "variant": variant, "qid": q})
def probe_eval(sel_variants=("alone", "6k-middle")):
    rows = [x for x in probe if x["variant"] in sel_variants]
    p = forward(rows)["p"]
    out = {}
    for v in sel_variants:
        idx = [i for i, x in enumerate(rows) if x["variant"] == v]
        lab_ = [rows[i]["label"] for i in idx]
        out[v] = metrics(p[idx], lab_) | {"pos": round(float(np.mean([p[i] for i in idx if rows[i]["label"]])), 3),
                                         "neg": round(float(np.mean([p[i] for i in idx if not rows[i]["label"]])), 3)}
    # mixed-length ranking: all rows together (long and short compete, as in navigation)
    out["mixed"] = metrics(p, [x["label"] for x in rows])
    return out
def practice_eval():
    e = evaluate(PRACTICE); e.pop("_p")
    return {k: e[k] for k in ("all", "page", "passage")}
import numpy as np
print("probe rows", len(probe), {v: sum(x["variant"] == v for x in probe) for v in ("alone", "3k-first", "3k-last", "6k-middle")})
```

```
probe rows 1024 {'alone': 256, '3k-first': 256, '3k-last': 256, '6k-middle': 256}
[turn 22 | 0.0s | passes this cell 0, used 32/50 | cuda:0]
```

## Turn 23 (13:27:24, 0023.py)

```python
# Turn 23 (GPU, 1 pass). Targeted fix, step 1: capture what is needed to
# simulate a query edit of L15's default-no heads (0, 2, 5) exactly, offline.
# Per row (probe alone + 6k): x~ = the layer input at the answer position
# (after input_layernorm), the pre-norm query of each head there, every key
# rotated into the answer position's frame (k_rel = R(-pos_last) k_abs, so
# logit_j = scale * q' . k_rel_j with q' the normed, unrotated query), and
# each position's value read as decision (value * gate . o_proj-head . u).
import numpy as np
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
rows = [x for x in probe if x["variant"] in ("alone", "6k-middle")]
lab = np.array([x["label"] for x in rows]); var = np.array([x["variant"] for x in rows])
LAYER, HEADS = 15, (0, 2, 5)
att = S.layers[LAYER].self_attn; hd = att.head_dim; G = att.num_key_value_groups
lm = S.rt.backbone.language_model
ti = S.true_index; R_ = S.readout.float(); r = R_[ti] - R_[1 - ti]
u = (1 + W("language_model.norm.weight").float()) * r
Wo = W(f"language_model.layers.{LAYER}.self_attn.o_proj.weight").float()
wdec = {h: u @ Wo[:, h*hd:(h+1)*hd] for h in HEADS}
CAP = [None] * len(rows)
def hook(mod, args, kwargs, output):
    h = kwargs.get("hidden_states", args[0] if args else None)
    cos, sin = kwargs["position_embeddings"]
    B, Lw, _ = h.shape
    qg = mod.q_proj(h).view(B, Lw, -1, hd * 2)
    qraw, gate = torch.chunk(qg, 2, dim=-1)
    qn = mod.q_norm(qraw)                                   # [B, Lw, 8, hd] normed, unrotated
    k = mod.k_norm(mod.k_proj(h).view(B, Lw, -1, hd)).transpose(1, 2)
    v = mod.v_proj(h).view(B, Lw, -1, hd).transpose(1, 2)
    q_rot, k_abs = apply_rotary_pos_emb(qn.transpose(1, 2), k, cos, sin)
    back = torch.full((3, B, 1), -(Lw - 1), device=h.device, dtype=torch.long)
    cb, sb = lm.rotary_emb(h, back)
    _, k_rel = apply_rotary_pos_emb(k_abs, k_abs, cb, sb)  # R(-(Lw-1)) k_abs
    for j, i in enumerate(S.ctx["batch"]):
        n = S.ctx["lengths"][j]
        item = {"x": h[j, -1].float().cpu(), "qraw": {}, "vdec": {}, "chk": {}}
        item["krel"] = k_rel[j, :, Lw - n:].half().cpu()  # [kvh, n, hd]
        for hh in HEADS:
            item["qraw"][hh] = qraw[j, -1, hh].float().cpu()
            g = torch.sigmoid(gate[j, -1, hh].float())
            item["vdec"][hh] = ((v[j, hh // G, Lw - n:].float() * g) @ wdec[hh]).cpu()
            true_logit = (q_rot[j, hh, -1].float() @ k_abs[j, hh // G, Lw - n:].float().T) * mod.scaling
            mine = (qn[j, -1, hh].float() @ k_rel[j, hh // G, Lw - n:].float().T) * mod.scaling
            item["chk"][hh] = float((true_logit - mine).abs().max())
        CAP[i] = item
hk = att.register_forward_hook(hook, with_kwargs=True)
try:
    res = forward(rows, capture=("resid",), layers=[23])
finally:
    hk.remove()
RMS = np.sqrt((res["resid"][:, 0] ** 2).mean(-1) + 1e-6)
print("relative-frame logits vs true logits, max |diff|:", max(max(c["chk"].values()) for c in CAP))
SEGS = []
tok = S.rt.tokenizer
for x in rows:
    text = S.prompt(x); enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    p0 = text.find(x["text"]); p1 = p0 + len(x["text"])
    SEGS.append(np.array(["passage" if p0 <= (a + b) // 2 < p1 else ("suffix" if (a + b) // 2 >= p1 else "pre")
                          for a, b in enc["offset_mapping"]]))
suf_len = {int((s == "suffix").sum()) for s in SEGS}
print("suffix lengths in tokens (fixed template?):", sorted(suf_len)[:5], "...")
qn_mod = att.q_norm
def head_out(i, hh, qraw=None):
    c = CAP[i]; q = c["qraw"][hh] if qraw is None else qraw
    qn = qn_mod(q.to("cuda:0", qn_mod.weight.dtype)).float().cpu()
    lg = (qn @ c["krel"][hh // G].float().T) * att.scaling
    pr = torch.softmax(lg, -1)
    return float((pr * c["vdec"][hh]).sum()) / RMS[i], float(pr[torch.tensor(SEGS[i] == "passage")].sum())
groups = {"neg alone": (lab == 0) & (var == "alone"), "neg 6k": (lab == 0) & (var == "6k-middle"),
          "pos alone": (lab == 1) & (var == "alone"), "pos 6k": (lab == 1) & (var == "6k-middle")}
for hh in HEADS:
    print(f"h{hh} contribution / passage mass (recomputed, unedited): " + " | ".join(
        f"{g}: {np.mean([head_out(i, hh)[0] for i in np.where(m)[0]]):+.3f} / {np.mean([head_out(i, hh)[1] for i in np.where(m)[0]]):.2f}"
        for g, m in groups.items()))
T23 = dict(rows=rows, lab=lab, var=var, groups=groups, p=res["p"])
```

```
relative-frame logits vs true logits, max |diff|: 0.002100825309753418
suffix lengths in tokens (fixed template?): [35] ...
h0 contribution / passage mass (recomputed, unedited): neg alone: -0.416 / 0.06 | neg 6k: -0.224 / 0.33 | pos alone: +0.010 / 0.45 | pos 6k: -0.045 / 0.62
h2 contribution / passage mass (recomputed, unedited): neg alone: -0.388 / 0.03 | neg 6k: -0.252 / 0.22 | pos alone: +0.013 / 0.23 | pos 6k: -0.070 / 0.42
h5 contribution / passage mass (recomputed, unedited): neg alone: -0.288 / 0.11 | neg 6k: -0.133 / 0.36 | pos alone: +0.041 / 0.36 | pos 6k: +0.003 / 0.54
[turn 23 | 85.6s | passes this cell 1, used 33/50 | cuda:0]
```

## Turn 24 (13:28:10, 0024.py)

```python
# Turn 24 (no pass). Design the targeted fix and simulate it exactly on the
# turn-23 captures. For head h of L15 (0, 2, 5):
#   s(x~) = d' . x~ : the filler score, d' = unit (mean6k - meanAlone) of x~,
#           orthogonalised to mean(x~ alone) so a typical short text scores ~0
#   t_h   = the head's own "no" default: mean over irrelevant-alone rows of the
#           attention-weighted relative keys of the 35 template tokens
#   edit  : q_raw += beta_h * (s / s6k) * e_h,  e_h = (t_h / (1 + q_norm.w)) * |q_raw|
#           (pre-norm direction that raises q' . t_h), s6k = mean s of 6k rows
# i.e. W_q[rows of head h] += beta_h / s6k * outer(e_h, d'). Choose beta per
# head so that irrelevant-6k rows get back their alone verdict while the
# alone rows stay where they are.
import numpy as np
rows, lab, var, groups = T23["rows"], T23["lab"], T23["var"], T23["groups"]
X = torch.stack([c["x"] for c in CAP])                        # [n, 1024]
alone = torch.tensor(var == "alone"); six = ~alone
mu_a, mu_6 = X[alone].mean(0), X[six].mean(0)
d = mu_6 - mu_a; ma = mu_a / mu_a.norm()
dp = d - (d @ ma) * ma; dp = dp / dp.norm()
s = (X @ dp).numpy()
print("filler score s: alone mean %.3f (sd %.3f), 6k mean %.3f (sd %.3f); AUC alone vs 6k %.3f" % (
    s[var == "alone"].mean(), s[var == "alone"].std(), s[six.numpy()].mean(), s[six.numpy()].std(),
    auc(s.tolist(), six.numpy().astype(int).tolist())))
s6k = float(s[six.numpy()].mean())
w1 = (1 + att.q_norm.weight.float().cpu())
E = {}
for hh in HEADS:
    acc_t = torch.zeros(hd); cnt = 0
    for i in np.where(groups["neg alone"])[0]:
        c = CAP[i]; suf = torch.tensor(SEGS[i] == "suffix")
        qn = qn_mod(c["qraw"][hh].to("cuda:0", qn_mod.weight.dtype)).float().cpu()
        lg = (qn @ c["krel"][hh // G].float().T) * att.scaling
        pr = torch.softmax(lg, -1)
        kk = c["krel"][hh // G].float()[suf]; pw = pr[suf]
        acc_t += (pw[:, None] * kk).sum(0) / pw.sum(); cnt += 1
    t = acc_t / cnt
    qnorm_typ = float(np.mean([CAP[i]["qraw"][hh].norm() for i in range(len(rows))]))
    e = (t / w1); e = e / e.norm() * qnorm_typ
    E[hh] = e
def simulate(hh, beta):
    out = {}
    for g, m in groups.items():
        vals = [head_out(i, hh, CAP[i]["qraw"][hh] + beta * (s[i] / s6k) * E[hh]) for i in np.where(m)[0]]
        out[g] = (np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals]))
    return out
for hh in HEADS:
    print(f"\nh{hh}: beta -> contribution / passage mass")
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        o = simulate(hh, beta)
        print(f"  beta {beta:4.2f}: " + " | ".join(f"{g} {o[g][0]:+.3f}/{o[g][1]:.2f}" for g in o))
T24 = dict(dp=dp, s=s, s6k=s6k, E=E)
```

```
filler score s: alone mean 0.000 (sd 4.460), 6k mean 5.927 (sd 2.771); AUC alone vs 6k 0.899

h0: beta -> contribution / passage mass
  beta 0.00: neg alone -0.416/0.06 | neg 6k -0.224/0.33 | pos alone +0.010/0.45 | pos 6k -0.045/0.62
  beta 0.25: neg alone -0.275/0.16 | neg 6k -0.295/0.15 | pos alone +0.022/0.33 | pos 6k -0.101/0.33
  beta 0.50: neg alone -0.144/0.47 | neg 6k -0.321/0.09 | pos alone +0.020/0.22 | pos 6k -0.158/0.15
  beta 0.75: neg alone -0.109/0.59 | neg 6k -0.334/0.08 | pos alone +0.007/0.15 | pos 6k -0.199/0.08
  beta 1.00: neg alone -0.090/0.65 | neg 6k -0.340/0.08 | pos alone -0.011/0.11 | pos 6k -0.227/0.05
  beta 1.50: neg alone -0.074/0.70 | neg 6k -0.344/0.08 | pos alone -0.050/0.07 | pos 6k -0.259/0.03

h2: beta -> contribution / passage mass
  beta 0.00: neg alone -0.388/0.03 | neg 6k -0.252/0.22 | pos alone +0.013/0.23 | pos 6k -0.070/0.42
  beta 0.25: neg alone -0.298/0.09 | neg 6k -0.314/0.08 | pos alone +0.006/0.14 | pos 6k -0.149/0.17
  beta 0.50: neg alone -0.182/0.43 | neg 6k -0.332/0.06 | pos alone -0.017/0.09 | pos 6k -0.207/0.08
  beta 0.75: neg alone -0.153/0.56 | neg 6k -0.340/0.06 | pos alone -0.044/0.06 | pos 6k -0.243/0.04
  beta 1.00: neg alone -0.139/0.62 | neg 6k -0.344/0.06 | pos alone -0.069/0.05 | pos 6k -0.266/0.03
  beta 1.50: neg alone -0.127/0.69 | neg 6k -0.347/0.07 | pos alone -0.110/0.04 | pos 6k -0.292/0.02

h5: beta -> contribution / passage mass
  beta 0.00: neg alone -0.288/0.11 | neg 6k -0.133/0.36 | pos alone +0.041/0.36 | pos 6k +0.003/0.54
  beta 0.25: neg alone -0.274/0.12 | neg 6k -0.145/0.30 | pos alone +0.045/0.32 | pos 6k -0.001/0.44
  beta 0.50: neg alone -0.215/0.19 | neg 6k -0.147/0.27 | pos alone +0.045/0.28 | pos 6k -0.008/0.37
  beta 0.75: neg alone -0.154/0.31 | neg 6k -0.143/0.26 | pos alone +0.043/0.26 | pos 6k -0.015/0.34
  beta 1.00: neg alone -0.118/0.40 | neg 6k -0.135/0.26 | pos alone +0.038/0.24 | pos 6k -0.021/0.32
  beta 1.50: neg alone -0.086/0.52 | neg 6k -0.122/0.28 | pos alone +0.028/0.22 | pos 6k -0.027/0.30
[turn 24 | 7.6s | passes this cell 0, used 33/50 | cuda:0]
```

## Turn 25 (13:28:39, 0025.py)

```python
# Turn 25 (no pass). Turn 24: the mean-difference filler score is too noisy
# on short texts (sd 4.5 vs a 6k mean of 5.9), so short texts with s < 0 get
# a reversed boost and lose their default-no. Cleaner length signal: a
# shrinkage Fisher discriminant in x~ space, w = (S_w + lam I)^-1 (mu6 - mua),
# fitted on one half of the probe questions and checked on the other half
# (split by question, so no passage is in both).
import numpy as np
qid = np.array([x["qid"] for x in rows]); uq = sorted(set(qid))
rng_ = np.random.RandomState(5); rng_.shuffle(uq); fit_q = set(uq[: len(uq) // 2])
fit = np.array([q in fit_q for q in qid]); chk = ~fit
Xn = X.numpy().astype(np.float64)
is6 = (var == "6k-middle")
def lda(mask, lam):
    A, B = Xn[mask & ~is6], Xn[mask & is6]
    Sw = np.cov(np.vstack([A - A.mean(0), B - B.mean(0)]).T)
    w = np.linalg.solve(Sw + lam * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0]), B.mean(0) - A.mean(0))
    return w / np.linalg.norm(w)
print("lam   | check half: AUC alone vs 6k, separation (gap / sd alone), sd alone / sd 6k")
best = None
for lam in (100.0, 10.0, 3.0, 1.0, 0.3, 0.1):
    w = lda(fit, lam)
    sc = Xn @ w
    a_ = auc(sc[chk].tolist(), is6[chk].astype(int).tolist())
    gap = sc[chk & is6].mean() - sc[chk & ~is6].mean(); sda = sc[chk & ~is6].std(); sd6 = sc[chk & is6].std()
    print(f"{lam:5.1f} | {a_:.3f}  {gap / sda:5.2f}  {sda:.3f} / {sd6:.3f}")
    if best is None or gap / sda > best[0]:
        best = (gap / sda, lam)
print("mean-difference direction for comparison: gap/sd", round(float((s[is6].mean() - s[~is6].mean()) / s[~is6].std()), 2))
print("best lam", best[1])
LAM = best[1]
```

```
lam   | check half: AUC alone vs 6k, separation (gap / sd alone), sd alone / sd 6k
100.0 | 0.992   3.61  1.513 / 1.298
 10.0 | 1.000   5.84  0.845 / 0.908
  3.0 | 1.000   6.18  0.720 / 0.763
  1.0 | 1.000   6.53  0.578 / 0.615
  0.3 | 1.000   6.89  0.434 / 0.481
  0.1 | 1.000   7.14  0.342 / 0.396
mean-difference direction for comparison: gap/sd 1.33
best lam 0.1
[turn 25 | 1.4s | passes this cell 0, used 33/50 | cuda:0]
```

## Turn 26 (13:29:11, 0026.py)

```python
# Turn 26 (no pass). Length score from the Fisher direction (lam 1.0, a
# middle choice: 0.1 scores best but may key on artefacts of the synthetic
# concatenation). q_proj has no bias, so the score must be ~0 for a typical
# short text by construction: orthogonalise w to the mean short-text input.
# Check on the held-out half, then re-simulate the three heads.
import numpy as np
w = lda(fit, 1.0)
ma_ = Xn[fit & ~is6].mean(0); ma_ /= np.linalg.norm(ma_)
w2 = w - (w @ ma_) * ma_; w2 /= np.linalg.norm(w2)
sc = Xn @ w2
print("check half: short mean %.3f sd %.3f | 6k mean %.3f sd %.3f | AUC %.3f | gap/sd %.2f" % (
    sc[chk & ~is6].mean(), sc[chk & ~is6].std(), sc[chk & is6].mean(), sc[chk & is6].std(),
    auc(sc[chk].tolist(), is6[chk].astype(int).tolist()), (sc[chk & is6].mean() - sc[chk & ~is6].mean()) / sc[chk & ~is6].std()))
# final direction: fit on all rows, same recipe
wf = lda(np.ones(len(rows), bool), 1.0)
maf = Xn[~is6].mean(0); maf /= np.linalg.norm(maf)
wf = wf - (wf @ maf) * maf; wf /= np.linalg.norm(wf)
s = Xn @ wf; s6k = float(s[is6].mean())
print("all rows: short mean %.3f sd %.3f | 6k mean %.3f sd %.3f" % (s[~is6].mean(), s[~is6].std(), s[is6].mean(), s[is6].std()))
for hh in HEADS:
    print(f"\nh{hh}: beta -> contribution / passage mass")
    for beta in (0.0, 0.15, 0.25, 0.35, 0.5, 0.75):
        o = simulate(hh, beta)
        print(f"  beta {beta:4.2f}: " + " | ".join(f"{g} {o[g][0]:+.3f}/{o[g][1]:.2f}" for g in o))
T26 = dict(wf=wf, s=s, s6k=s6k)
```

```
check half: short mean 0.096 sd 0.583 | 6k mean 3.892 sd 0.628 | AUC 1.000 | gap/sd 6.51
all rows: short mean 0.000 sd 0.394 | 6k mean 3.965 sd 0.410

h0: beta -> contribution / passage mass
  beta 0.00: neg alone -0.416/0.06 | neg 6k -0.224/0.33 | pos alone +0.010/0.45 | pos 6k -0.045/0.62
  beta 0.15: neg alone -0.416/0.06 | neg 6k -0.280/0.20 | pos alone +0.010/0.45 | pos 6k -0.074/0.46
  beta 0.25: neg alone -0.415/0.06 | neg 6k -0.305/0.15 | pos alone +0.009/0.45 | pos 6k -0.094/0.37
  beta 0.35: neg alone -0.415/0.06 | neg 6k -0.323/0.11 | pos alone +0.009/0.45 | pos 6k -0.113/0.29
  beta 0.50: neg alone -0.413/0.06 | neg 6k -0.342/0.07 | pos alone +0.008/0.45 | pos 6k -0.140/0.20
  beta 0.75: neg alone -0.407/0.06 | neg 6k -0.360/0.04 | pos alone +0.007/0.45 | pos 6k -0.177/0.12

h2: beta -> contribution / passage mass
  beta 0.00: neg alone -0.388/0.03 | neg 6k -0.252/0.22 | pos alone +0.013/0.23 | pos 6k -0.070/0.42
  beta 0.15: neg alone -0.388/0.03 | neg 6k -0.300/0.12 | pos alone +0.013/0.23 | pos 6k -0.111/0.27
  beta 0.25: neg alone -0.387/0.03 | neg 6k -0.319/0.08 | pos alone +0.013/0.23 | pos 6k -0.136/0.20
  beta 0.35: neg alone -0.387/0.03 | neg 6k -0.333/0.06 | pos alone +0.012/0.23 | pos 6k -0.158/0.15
  beta 0.50: neg alone -0.386/0.03 | neg 6k -0.347/0.04 | pos alone +0.012/0.23 | pos 6k -0.187/0.10
  beta 0.75: neg alone -0.383/0.03 | neg 6k -0.360/0.02 | pos alone +0.011/0.23 | pos 6k -0.222/0.06

h5: beta -> contribution / passage mass
  beta 0.00: neg alone -0.288/0.11 | neg 6k -0.133/0.36 | pos alone +0.041/0.36 | pos 6k +0.003/0.54
  beta 0.15: neg alone -0.288/0.11 | neg 6k -0.143/0.32 | pos alone +0.041/0.36 | pos 6k +0.001/0.49
  beta 0.25: neg alone -0.288/0.11 | neg 6k -0.146/0.30 | pos alone +0.041/0.36 | pos 6k -0.001/0.45
  beta 0.35: neg alone -0.288/0.11 | neg 6k -0.147/0.28 | pos alone +0.041/0.36 | pos 6k -0.003/0.43
  beta 0.50: neg alone -0.288/0.11 | neg 6k -0.147/0.26 | pos alone +0.041/0.36 | pos 6k -0.005/0.39
  beta 0.75: neg alone -0.287/0.11 | neg 6k -0.141/0.25 | pos alone +0.040/0.36 | pos 6k -0.009/0.35
[turn 26 | 7.0s | passes this cell 0, used 33/50 | cuda:0]
```

## Turn 27 (13:37:33, 0027.py)

```python
# Turn 27 (GPU, 4 passes). Edit 3, the targeted fix: for L15 heads 0 and 2,
# W_q[query rows of head h] += beta_h / s6k * outer(e_h, w), with w the
# centred Fisher length direction of the layer input and e_h the pre-norm
# direction towards the head's own default-no keys (turns 24-26). beta: h0
# 0.35, h2 0.25 (h5 barely responds; left out). Short texts are left alone by
# construction (score ~0). Test on all four probe variants (the 3k ones were
# not used to design the score) and PRACTICE.
import numpy as np
BETA = {0: 0.35, 2: 0.25}
qp = W(f"language_model.layers.15.self_attn.q_proj.weight")
wt = torch.tensor(T26["wf"], dtype=torch.float32)
for hh, b in BETA.items():
    rows_q = slice(hh * 2 * hd, hh * 2 * hd + hd)       # the query half of head hh's q_proj block
    dW = (b / T26["s6k"]) * torch.outer(E[hh], wt)
    qp[rows_q] = (qp[rows_q].float() + dW.to(qp.device)).to(qp.dtype)
print("edit 3:", changed())
pr = probe_eval(("alone", "3k-first", "3k-last", "6k-middle")); pa = practice_eval()
for v in ("alone", "3k-first", "3k-last", "6k-middle", "mixed"):
    m = pr[v]; print(f"probe {v:9s} auc {m['auc']}  " + (f"P(yes) pos {m['pos']} neg {m['neg']}" if "pos" in m else ""))
print(f"practice auc {pa['all']['auc']} (page {pa['page']['auc']}, passage {pa['passage']['auc']}, acc {pa['all']['acc']})")
print("unedited (turn 7 / 5): alone 0.9683 (pos 0.673 neg 0.077) | 3k-first 0.8952 (0.630/0.172) | 3k-last 0.8654 (0.573/0.175) | "
      "6k 0.8252 (0.574/0.247) | practice 0.9357 (page 0.977, passage 0.8812, acc 0.902)")
```

```
edit 3: {'language_model.layers.15.self_attn.q_proj.weight': 518202, '_total': 518202, '_share': '0.0608% of 852,987,968'}
probe alone     auc 0.9689  P(yes) pos 0.673 neg 0.079
probe 3k-first  auc 0.8958  P(yes) pos 0.585 neg 0.153
probe 3k-last   auc 0.865  P(yes) pos 0.525 neg 0.155
probe 6k-middle auc 0.8203  P(yes) pos 0.496 neg 0.206
probe mixed     auc 0.8937  
practice auc 0.9355 (page 0.9768, passage 0.8809, acc 0.9007)
unedited (turn 7 / 5): alone 0.9683 (pos 0.673 neg 0.077) | 3k-first 0.8952 (0.630/0.172) | 3k-last 0.8654 (0.573/0.175) | 6k 0.8252 (0.574/0.247) | practice 0.9357 (page 0.977, passage 0.8812, acc 0.902)
[turn 27 | 474.4s | passes this cell 4, used 37/50 | cuda:0]
```

## Turn 28 (13:38:23, 0028.py)

```python
# Turn 28 (no pass). Edit 3 did what it was built for at head level (short
# texts untouched: alone 0.673/0.079 as before; long irrelevant P(yes)
# 0.247 -> 0.206) but long relevant texts fell more (0.574 -> 0.496) and
# AUC within every length is unchanged (6k 0.8252 -> 0.8203; practice
# 0.9357 -> 0.9355). Restoring the default-no moves the threshold for long
# texts, it does not restore the evidence: the best-match gap between
# relevant and irrelevant long texts is 2.8 logits against 5.8 for short
# ones (turn 13). Revert; no edit kept. 13 passes left unused.
revert(); print(changed())
```

```
{'_total': 0, '_share': '0.0000% of 852,987,968'}
[turn 28 | 3.2s | passes this cell 0, used 37/50 | cuda:0]
```

## Turn 29 (14:08:15, 0029.py)

```python
# Turn 29 (GPU, 6 passes). Diagnosis, not an edit: for each benchmark failure
# type, is the ability inside the judge but unused (a linear probe on its
# residual stream ranks gold above distractors better than its own P(yes)),
# or missing? Benchmark questions are split by a hash of their id: only the
# DIAGNOSIS half is used here; the other half stays untouched for measuring
# any later edit. Metric for judge and probe alike: mean per-question AUC
# (gold vs distractors of the same question). Probe: standardised logistic
# regression, 5-fold cross-validation grouped by question.
import json, hashlib, random
import numpy as np
from bench.common import read_jsonl, run_dir
from bench.navigate import KIND
from bench.failures import is_gold
diag = lambda qid: int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 2 == 0
rng = random.Random(29)
LAYERS = [4, 8, 12, 15, 17, 19, 21, 23]
def load(bench, nav_file):
    qs = [q for q in read_jsonl(run_dir(bench) / "questions.jsonl") if diag(q["qid"])]
    nav = {r["qid"]: r for r in read_jsonl(run_dir(bench) / nav_file)}
    chunks = {c["chunk_id"]: c for c in read_jsonl(run_dir(bench) / "chunks.jsonl")}
    return qs, nav, chunks
SETS = {}
# --- A and D: LegalBench tree levels; B: LegalBench passages
qs, nav, chunks = load("legalbench", "nav_d3-lite-tuned-full_plusbase_sl10.jsonl")
nodes = json.load(open(run_dir("legalbench") / "tree.json", encoding="utf-8"))
doc_node = {n["doc_id"]: n for n in nodes if n.get("kind") == "document"}
owner = {}
for n in nodes:
    for c in n["chunk_ids"]:
        owner[c] = n["id"]
ntext = lambda n: n["description"] or n["title"]
A, B, D = [], [], []
by_sub = {}
for q in qs:
    by_sub.setdefault(q["subset"], []).append(q)
for sub, qq in by_sub.items():
    rng.shuffle(qq)
    for q in qq[:16]:                                    # A: 16 questions per subset, gold doc + 15 siblings
        g = doc_node.get(q["gold"][0]["doc_id"])
        if g is None: continue
        sib = [nodes[i] for i in nodes[g["parent"]]["children"] if i != g["id"]]
        rng.shuffle(sib)
        for n, lab in [(g, 1)] + [(s, 0) for s in sib[:15]]:
            A.append({"query": q["query"], "kind": KIND.get(n["kind"], "section"), "text": ntext(n), "label": lab, "qid": q["qid"]})
pool_q = [q for q in qs if q["qid"] in nav]
rng.shuffle(pool_q)
for q in pool_q:
    res = nav[q["qid"]]["results"]
    gold = [it for it in res if is_gold("legalbench", q, it)]
    if gold and len(B) < 1024:                           # B: 1 gold + 4 top-P + 3 random distractors
        neg = [it for it in res if not is_gold("legalbench", q, it)]
        pick = neg[:4] + rng.sample(neg[4:], min(3, len(neg[4:])))
        for it, lab in [(gold[0], 1)] + [(x, 0) for x in pick]:
            B.append({"query": q["query"], "kind": "passage", "text": chunks[it["chunk_id"]]["text"], "label": lab, "qid": q["qid"],
                      "subset": q["subset"]})
    if len(D) < 512:                                     # D: gold passage's article vs the document's other articles
        gc = [c for c, ch in chunks.items() if ch["doc_id"] == q["gold"][0]["doc_id"]
              and ch.get("char_start", None) is None] if False else None
        g = q["gold"][0]; dn = doc_node.get(g["doc_id"])
        if dn is None: continue
        gold_chunks = [c for c in dn and [] or []]
        # the chunk that overlaps the gold span, found through the pool or the doc's chunks
        cand = [it for it in res if is_gold("legalbench", q, it)]
        if not cand: continue
        nid = owner.get(cand[0]["chunk_id"])
        path = []
        while nid is not None and str(nid) != str(dn["id"]):
            path.append(nid); nid = nodes[int(nid)]["parent"] if nodes[int(nid)]["parent"] is not None else None
            if nid is not None: nid = int(nid)
        if not path: continue
        art = nodes[int(path[-1])]                       # the child of the document on the gold path
        sib = [nodes[i] for i in dn["children"] if i != art["id"]]
        if len(sib) < 3: continue
        rng.shuffle(sib)
        for n, lab in [(art, 1)] + [(s, 0) for s in sib[:7]]:
            D.append({"query": q["query"], "kind": KIND.get(n["kind"], "section"), "text": ntext(n), "label": lab, "qid": q["qid"]})
SETS["A legalbench document level"] = A[:1024]
SETS["B legalbench passages"] = B[:1024]
SETS["D legalbench article inside gold doc"] = D[:512]
# --- C: bBSARD articles
qs, nav, chunks = load("bbsard", "nav_d3-lite-tuned-full_plusbase_sl50.jsonl")
art = {}
for c in chunks.values():
    art.setdefault(c["doc_id"], c["text"])
C = []
for q in rng.sample(qs, len(qs)):
    if q["qid"] not in nav or len(C) >= 512: continue
    res = nav[q["qid"]]["results"]; gd = {g["doc_id"] for g in q["gold"]}
    neg = [it for it in res if it["doc_id"] not in gd]
    gid = [g for g in gd if g in art]
    if not gid or len(neg) < 7: continue
    pick = neg[:4] + rng.sample(neg[4:], 3)
    for t_, lab in [(art[gid[0]], 1)] + [(chunks[x["chunk_id"]]["text"], 0) for x in pick]:
        C.append({"query": q["query"], "kind": "passage", "text": t_, "label": lab, "qid": q["qid"]})
SETS["C bbsard articles"] = C
print({k: (len(v), len({r['qid'] for r in v})) for k, v in SETS.items()})
# --- capture and probe
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
def per_q_auc(scores, rows_):
    by = {}
    for sc_, r_ in zip(scores, rows_):
        by.setdefault(r_["qid"], []).append((sc_, r_["label"]))
    vals = [auc([a for a, _ in v], [b for _, b in v]) for v in by.values() if 0 < sum(b for _, b in v) < len(v)]
    return float(np.mean(vals)), len(vals)
DIAG = {}
for name, rows_ in SETS.items():
    res_ = forward(rows_, capture=("resid",), layers=LAYERS)
    y_ = np.array([r_["label"] for r_ in rows_]); grp = np.array([r_["qid"] for r_ in rows_])
    j_auc, nq = per_q_auc(res_["p"].tolist(), rows_)
    line = f"{name}: {nq} questions | judge per-question AUC {j_auc:.3f} | probe by layer:"
    best = (0, None)
    for li, L in enumerate(LAYERS):
        Xl = res_["resid"][:, li]
        pred = np.zeros(len(rows_))
        for tr, te in GroupKFold(5).split(Xl, y_, grp):
            m_ = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))
            m_.fit(Xl[tr], y_[tr]); pred[te] = m_.decision_function(Xl[te])
        a_, _ = per_q_auc(pred.tolist(), rows_)
        line += f" L{L} {a_:.3f}"
        best = max(best, (a_, L))
    print(line, f"| best L{best[1]} {best[0]:.3f} (gap {best[0] - j_auc:+.3f})")
    DIAG[name] = dict(rows=rows_, res=res_, judge=j_auc, best=best)
```

```
{'A legalbench document level': (850, 61), 'B legalbench passages': (1024, 128), 'D legalbench article inside gold doc': (512, 68), 'C bbsard articles': (512, 64)}
A legalbench document level: 61 questions | judge per-question AUC 0.987 | probe by layer: L4 0.728 L8 0.963 L12 0.999 L15 0.998 L17 0.998 L19 0.998 L21 0.998 L23 0.998 | best L12 0.999 (gap +0.011)
B legalbench passages: 128 questions | judge per-question AUC 0.663 | probe by layer: L4 0.713 L8 0.762 L12 0.830 L15 0.792 L17 0.791 L19 0.779 L21 0.789 L23 0.790 | best L12 0.830 (gap +0.167)
D legalbench article inside gold doc: 68 questions | judge per-question AUC 0.825 | probe by layer: L4 0.554 L8 0.701 L12 0.786 L15 0.791 L17 0.784 L19 0.797 L21 0.786 L23 0.792 | best L19 0.797 (gap -0.028)
C bbsard articles: 64 questions | judge per-question AUC 0.529 | probe by layer: L4 0.614 L8 0.554 L12 0.661 L15 0.663 L17 0.627 L19 0.638 L21 0.645 L23 0.656 | best L15 0.663 (gap +0.134)
[turn 29 | 367.8s | passes this cell 6, used 43/50 | cuda:0]
```

