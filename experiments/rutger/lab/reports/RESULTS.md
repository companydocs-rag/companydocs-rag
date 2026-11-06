# bench-lab results

## conditionalqa

| run | n | hit@1 | hit@10 | char_recall@10 | mrr@20 |
|---|---|---|---|---|---|
| baseline_exact | 285 | 0.557 | 0.919 | 0.877 | 0.688 |
| baseline_exact_deep | 285 | 0.557 | 0.915 | 0.870 | 0.687 |
| fast_exact | 285 | 0.557 | 0.919 | 0.877 | 0.688 |
| fast_exact_deep | 285 | 0.557 | 0.915 | 0.870 | 0.687 |
| fast_turbo2 | 285 | 0.554 | 0.923 | 0.879 | 0.686 |
| fast_turbo2_deep | 285 | 0.554 | 0.923 | 0.876 | 0.686 |
| fast_turbo4 | 285 | 0.568 | 0.915 | 0.872 | 0.693 |
| fast_turbo4_deep | 285 | 0.568 | 0.911 | 0.867 | 0.693 |
| nav_d3-lite-tuned-full_plusbase_sl10 | 285 | 0.720 | 0.970 | 0.947 | 0.814 |
| nav_d3-lite_plusbase_sl10_n60 | 60 | 0.411 | 0.946 | 0.863 | 0.591 |

Judges vs our baseline on the same questions (conditionalqa):

| judge run | n | hit@1 (base -> judge) | hit@10 (base -> judge) | char_recall@10 (base -> judge) | mrr@20 (base -> judge) |
|---|---|---|---|---|---|
| nav_d3-lite-tuned-full_plusbase_sl10 | 285 | 0.557 -> 0.720 | 0.919 -> 0.970 | 0.877 -> 0.947 | 0.688 -> 0.814 |
| nav_d3-lite_plusbase_sl10_n60 | 60 | 0.482 -> 0.411 | 0.929 -> 0.946 | 0.828 -> 0.863 | 0.628 -> 0.591 |

## legalbench

| run | n | hit@1 | hit@10 | char_recall@10 | mrr@20 |
|---|---|---|---|---|---|
| fast_exact | 776 | 0.075 | 0.338 | 0.265 | 0.154 |
| fast_exact_deep | 776 | 0.075 | 0.341 | 0.268 | 0.155 |
| fast_turbo2 | 776 | 0.075 | 0.336 | 0.266 | 0.154 |
| fast_turbo2_deep | 776 | 0.075 | 0.341 | 0.270 | 0.155 |
| fast_turbo4 | 776 | 0.076 | 0.339 | 0.265 | 0.156 |
| fast_turbo4_deep | 776 | 0.076 | 0.341 | 0.267 | 0.156 |
| nav_d3-lite-tuned-full_plusbase_sl10 | 776 | 0.213 | 0.527 | 0.456 | 0.319 |
| nav_d3-lite_plusbase_sl10_n60 | 60 | 0.167 | 0.483 | 0.380 | 0.280 |

LegalBench-RAG per sub-dataset (hit@10):

- fast_exact: contractnli 0.216, cuad 0.454, maud 0.062, privacy_qa 0.619
- fast_turbo2: contractnli 0.216, cuad 0.443, maud 0.067, privacy_qa 0.619
- fast_turbo4: contractnli 0.216, cuad 0.459, maud 0.062, privacy_qa 0.619
- nav_d3-lite-tuned-full_plusbase_sl10: contractnli 0.294, cuad 0.768, maud 0.227, privacy_qa 0.820
- nav_d3-lite_plusbase_sl10_n60: contractnli 0.333, cuad 0.867, maud 0.067, privacy_qa 0.667

Judges vs our baseline on the same questions (legalbench):

| judge run | n | hit@1 (base -> judge) | hit@10 (base -> judge) | char_recall@10 (base -> judge) | mrr@20 (base -> judge) |
|---|---|---|---|---|---|
| nav_d3-lite-tuned-full_plusbase_sl10 | 776 | 0.075 -> 0.213 | 0.338 -> 0.527 | 0.265 -> 0.456 | 0.154 -> 0.319 |
| nav_d3-lite_plusbase_sl10_n60 | 60 | 0.050 -> 0.167 | 0.233 -> 0.483 | 0.206 -> 0.380 | 0.110 -> 0.280 |

## bbsard

| run | n | recall@10 | recall@100 | ndcg@10 | mrr@20 |
|---|---|---|---|---|---|
| baseline_exact | 222 | 0.262 | 0.335 | 0.217 | 0.266 |
| fast_exact | 222 | 0.262 | 0.335 | 0.217 | 0.266 |
| fast_exact_deep | 222 | 0.271 | 0.524 | 0.219 | 0.262 |
| fast_turbo2 | 222 | 0.255 | 0.331 | 0.212 | 0.262 |
| fast_turbo2_deep | 222 | 0.278 | 0.517 | 0.219 | 0.260 |
| fast_turbo4 | 222 | 0.266 | 0.336 | 0.217 | 0.264 |
| fast_turbo4_deep | 222 | 0.279 | 0.523 | 0.221 | 0.260 |
| nav_d3-lite-tuned-full_plusbase_sl50 | 222 | 0.359 | 0.425 | 0.304 | 0.363 |
| nav_d3-lite_plusbase_sl50_n60 | 60 | 0.243 | 0.294 | 0.204 | 0.249 |

Judges vs our baseline on the same questions (bbsard):

| judge run | n | recall@10 (base -> judge) | recall@100 (base -> judge) | ndcg@10 (base -> judge) | mrr@20 (base -> judge) |
|---|---|---|---|---|---|
| nav_d3-lite-tuned-full_plusbase_sl50 | 222 | 0.262 -> 0.359 | 0.335 -> 0.425 | 0.217 -> 0.304 | 0.266 -> 0.363 |
| nav_d3-lite_plusbase_sl50_n60 | 60 | 0.163 -> 0.243 | 0.224 -> 0.294 | 0.122 -> 0.204 | 0.171 -> 0.249 |
