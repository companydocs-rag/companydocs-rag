"""Fine-tune a d3 decision model (code readout) as a retrieval judge.

    %USERPROFILE%\\.venvs\\bench-train\\Scripts\\python.exe -m bench.train_d3 \\
        --model %USERPROFILE%\\models\\vllm-sr__d3-lite \\
        --out %USERPROFILE%\\models\\finetuned\\d3-lite-judge

d3 runs with non-causal full attention through its own encoder, so it is not
a stock Qwen3.5 model and Unsloth cannot load it as one. Training therefore
goes through d3's own runtime, on exactly the path system_one uses:

  prompt   runtime.text(state, rendered noul question), tokenized without
           special tokens, left-padded (as runtime.logits)
  hidden   runtime.encoder(input embeddings, mask)[:, -1]
  logits   hidden @ readout[:2].T / temperature over the noul codes
  loss     cross-entropy against the label (the code index of "true"/"false")

LoRA (rank 16, PEFT) is injected into the backbone's projections, which the
encoder shares; the two noul readout rows are trained too. The backbone runs
in fp16 (Turing has no native bf16; measured equal within 0.003). State and
question text are those of bench.judges.D3Judge. Saved: the LoRA weights and
the readout rows (judge.pt) in --out.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "runs" / "conditionalqa"
QUESTION = (
    "Does the following {kind} contain information that helps answer the question "
    "in the state? Answer yes only if it is on the right document and topic.\n\n{text}"
)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
           "gate_proj", "up_proj", "down_proj"]


def read_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def auc(scores, labels) -> float:
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return math.nan
    rank_sum = sum(rank for rank, (_, label) in enumerate(pairs, 1) if label == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="bench.train_d3")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=640)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--text-chars", type=int, default=1600,
                        help="candidate text cut for training (ConditionalQA chunks are ~1,150 chars)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0,
                        help="skip the first N training examples (continue on unseen data)")
    parser.add_argument("--init", default="",
                        help="judge.pt to start from (LoRA + readout rows of an earlier run)")
    args = parser.parse_args(argv)

    import torch
    from peft import LoraConfig, inject_adapter_in_model
    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, device="cuda:0")
    rt = model.runtime
    rt.backbone.half()
    module = sys.modules[type(rt).__module__]  # the d3 runtime module (normalize_question)
    rendered_keys = module.normalize_question({"type": "noul", "instructions": "x"}).keys
    true_index = list(rendered_keys).index("true")

    for p in rt.backbone.parameters():
        p.requires_grad_(False)
    inject_adapter_in_model(
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS), rt.backbone
    )
    lora = [p for n, p in rt.backbone.named_parameters() if "lora_" in n]
    for p in lora:
        p.data = p.data.float()
        p.requires_grad_(True)
    readout = torch.nn.Parameter(rt.readout[: len(rendered_keys)].detach().clone().float())
    if args.init:
        saved = torch.load(args.init, map_location="cpu", weights_only=True)
        named = dict(rt.backbone.named_parameters())
        for name, tensor in saved["lora"].items():
            named[name].data.copy_(tensor.to(named[name].device, named[name].dtype))
        readout.data.copy_(saved["readout_rows"].to(readout.device))
        print(f"continued from {args.init} ({len(saved['lora'])} LoRA tensors)", flush=True)
    # d3's PromptEncoder loops over the decoder layers itself, so the backbone's
    # gradient_checkpointing does not apply; without it the activations spilled
    # 5.5 GB into shared memory. Checkpoint every layer call here instead.
    from torch.utils.checkpoint import checkpoint

    class Checkpointed:
        def __init__(self, layer):
            self.layer = layer

        def __call__(self, hidden, **kwargs):
            return checkpoint(self.layer, hidden, use_reentrant=False, **kwargs)

    rt.encoder.stack = [(Checkpointed(layer), softmax) for layer, softmax in rt.encoder.stack]
    print(f"LoRA params {sum(p.numel() for p in lora):,}; noul codes {list(rendered_keys)}; "
          f"temperature {rt.temperature}", flush=True)

    def encode(rows: List[Dict]):
        seqs = []
        for r in rows:
            question = module.normalize_question({
                "type": "noul",
                "instructions": QUESTION.format(kind=r["kind"], text=r["text"][: args.text_chars])})
            text = rt.text(f"Question: {r['query']}", question.rendered)
            seqs.append(rt.tokenizer(text, add_special_tokens=False)["input_ids"][-args.max_len:])
        # pad to a multiple of 128: the Triton kernels of flash-linear-attention
        # compile once per shape, so fixed buckets keep recompiles to a handful
        width = -(-max(len(s) for s in seqs) // 128) * 128
        ids = torch.full((len(seqs), width), rt.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, width - len(s):] = torch.as_tensor(s)
            mask[i, width - len(s):] = 1
        return ids.to(rt.device), mask.to(rt.device)

    def logits(rows: List[Dict]):
        ids, mask = encode(rows)
        with torch.autocast("cuda", dtype=torch.float16), module.sdpa_backends(rt.device):
            embeds = rt.backbone.get_input_embeddings()(ids)
            hidden = rt.encoder(embeds, mask)[:, -1]
        return (hidden.float() @ readout.T) / rt.temperature

    train = read_jsonl(DATA / "judge_train.jsonl")
    val = read_jsonl(DATA / "judge_val.jsonl")
    # a fixed random subset for the interim checks (the full measure is the
    # navigation on the benchmarks afterwards)
    val = random.Random(3).sample(val, min(args.val_size, len(val)))
    train = train[args.offset:]
    if args.limit:
        train = train[: args.limit]

    def evaluate():
        scores, labels = [], []
        with torch.no_grad():
            for i in range(0, len(val), 16):
                rows = val[i: i + 16]
                scores += logits(rows).softmax(-1)[:, true_index].tolist()
                labels += [r["label"] for r in rows]
        acc = sum((s > 0.5) == bool(l) for s, l in zip(scores, labels)) / len(labels)
        return {"val_acc": round(acc, 4), "val_auc": round(auc(scores, labels), 4)}

    params = lora + [readout]
    steps_total = math.ceil(len(train) * args.epochs / (args.batch * args.accum))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / 50) * max(0.1, 1 - s / max(1, steps_total)))
    scaler = torch.amp.GradScaler("cuda")
    print(f"train {len(train)}, val {len(val)}, {steps_total} steps", flush=True)
    print("before training:", evaluate(), flush=True)

    rng = random.Random(0)
    order = list(range(len(train)))
    step, micro, started, running = 0, 0, time.time(), 0.0
    while step < steps_total:
        rng.shuffle(order)
        for start in range(0, int(len(train) * args.epochs), args.batch):
            rows = [train[i] for i in order[start: start + args.batch]]
            target = torch.tensor([true_index if r["label"] == 1 else 1 - true_index for r in rows],
                                  device=rt.device)
            loss = torch.nn.functional.cross_entropy(logits(rows), target) / args.accum
            scaler.scale(loss).backward()
            running += loss.item()
            micro += 1
            if micro % args.accum:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if step in (5, 10, 20) or step % 50 == 0:
                print(f"step {step}/{steps_total} loss {running / max(1, step - getattr(main, '_last', 0)):.4f} "
                      f"{(time.time() - started) / step:.2f} s/step", flush=True)
                running = 0.0
                main._last = step
            if step % args.eval_every == 0:
                print(f"step {step}:", evaluate(), flush=True)
            if step >= steps_total:
                break
    print("after training:", evaluate(), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "lora": {n: p.detach().cpu() for n, p in rt.backbone.named_parameters() if "lora_" in n},
        "readout_rows": readout.detach().cpu(),
        "targets": TARGETS, "r": 16, "alpha": 32, "base": args.model,
    }, out / "judge.pt")
    print(f"saved to {out / 'judge.pt'} ({(time.time() - started) / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
