# Weight-space programming of the d3 judge: report

Protocol: Experiment 8 of Takehi, Teoh, Zhang and Khattab, "Sparks of Weight-Space Programming"
(MIT CSAIL, November 2026, https://rikiyat.github.io/blog/weight-space-programming/). A Python
session with the fine-tuned judge (`d3-lite-judge-full`) loaded; reading and analysing weights is
free; a budget of 50 forward passes (at most 512 prompts each) on training-side data only; no
gradients, no optimizer, no training loop (the session refuses `backward` and every uncounted forward
call). Every edit was chosen by hand after a diagnosis. Practice data: ConditionalQA train examples and
half of the judge_val questions; the other half of judge_val stayed hidden for a final check. The
benchmark test sets were not used for any edit decision.

Full log of every turn (code, output, passes used): `runs/wsp/log.md`. Session code: `bench/wsp/`.

## Result: no edit kept

28 turns, 37 of 50 passes (turns 22-28 added the targeted head fix); 13 passes left unused on purpose (spending them on noise-level changes
would only fit the practice set). No edit beat the unedited judge, so `d3-lite-edited` was not written
and the benchmark numbers of `d3-lite-tuned-full` stand.

## What the weights and passes showed

- The decision forms in one step at layer 15 (full attention); before it, yes and no examples are
  indistinguishable on the decision direction. Layer 19 head 5 sharpens it: an option pointer that puts
  97% of its attention on the answer template (option A tokens mean no, option B tokens yes). The
  layer 22 MLP (neuron 2202) adds the largest "no" write. Layer 23 compresses the logit about 6x.
- The fine-tune left these decision writers unchanged (cosine to base 0.999) and changed what feeds
  them: the early layers (2-4% relative change in layers 0-10) and `up_proj` most.

## The failure found: long texts

The fine-tune saw texts up to ~1,200 characters; 22.5% of LegalBench and 15.4% of bBSARD chunks are
longer than 1,600. Probe from training data (relevant and irrelevant passages, padded with irrelevant
passages of the same question):

| variant | median chars | AUC | P(yes) relevant | P(yes) irrelevant |
|---|---|---|---|---|
| alone | 1,150 | 0.968 | 0.673 | 0.077 |
| 3k, passage first | 3,436 | 0.895 | 0.630 | 0.172 |
| 3k, passage last | 3,436 | 0.865 | 0.573 | 0.175 |
| 6k, passage in the middle | 6,503 | 0.825 | 0.574 | 0.247 |

Cause: layer 15's default-no heads (0, 2, 5) say "no" by attending to the answer options when nothing
in the passage matches. In a long on-topic text the best-matching passage token comes 3.7 logits closer
to the default and there are more tokens (log n +1.8), so mass leaks to the passage (head 0: 0.06 ->
0.33) and the "no" fades. The filler also writes along the relevance direction just before layer 15
(cosine 0.35-0.43): more on-topic text looks like more relevance.

## Edits tried (none kept)

| edit | weights changed | result |
|---|---|---|
| log-n sharpening of layer 15 (`(1 + q_norm.weight)` x1.25) | 256 | no change (6k AUC 0.8252 -> 0.8254): sharpening amplifies the best match as much as the default |
| remove the filler direction from the layer 11-14 writers (W - d d^T W) | 2.6% | bias shift only (6k AUC 0.825 -> 0.820) |
| WiSE-FT interpolation / extrapolation of the fine-tune, alpha 0 / 0.5 / 0.75 / 1.0 / 1.25 | all LoRA matrices | alpha 1.0 is the optimum (score 0.808 / 0.896 / 0.914 / 0.920 / 0.917) |
| best screened component amplification (layer 22 linear-attention head 12 x2) | 0.015% | +0.0009 practice AUC, +0.0001 probe: noise |
| targeted, length-aware fix of the default-no heads 15.0 and 15.2 (turns 22-28, see below) | 0.061% | long irrelevant P(yes) 0.247 -> 0.206 and short texts untouched, but long relevant 0.574 -> 0.496; AUC per length unchanged (6k 0.8252 -> 0.8203, practice 0.9357 -> 0.9355) |

### The targeted fix (turns 22-28, after the first report)

Rank-1 edit of the query rows of heads 15.0 and 15.2: q += beta * s(x) * e_h, where s is the judge's
own length signal in layer 15's input (a centred, regularised Fisher direction: separates short from
6k texts with AUC 1.000 on a held-out half, ~0 for short texts by construction) and e_h points the
query at the head's own default-no keys (the 35-token answer template sits at a fixed distance from
the end, so its keys are the same in every prompt). Simulated exactly offline on captured q, k and v,
then applied for real (beta 0.35 and 0.25). It restored the heads' "no" on long irrelevant texts and
left short texts unchanged (also on the 3k variants, which were not used to build it). But it moved
long relevant texts towards "no" even more, through layer 19's pointer and layer 22, so ranking within
each length did not improve. The leak was a symptom: in long texts the evidence itself is weaker (the
best-match gap between relevant and irrelevant texts is 2.8 logits at 6k against 5.8 for short
texts), and restoring the default only moves the threshold. Reverted. 37 of 50 passes used in total.

The selection rule for the interpolation (mean of practice AUC and mixed-length probe AUC) was fixed
before its first result. A screen of every head and MLP showed no component whose removal or
amplification moves practice AUC by more than 0.002.

## Conclusion

As in the blog's Gemma 3 270M experiment: the remaining errors are spread over the network, with no
single component behind them to switch off, so a weight edit cannot add what the model lacks. What
would help instead:

1. Training data with long texts and legal text (the judge only saw short ConditionalQA texts).
2. Document context for passages: a chunk does not say which contract it comes from, so on
   ContractNLI (near-identical NDAs) the judge's top passage is from the wrong document far more often
   than the baseline's (doc hit@1 0.51 -> 0.07). Passing the document's description with each passage,
   or combining the passage score with its document node's score, gives the judge that information.
3. Keeping passages within the trained length range (split long chunks before judging).
