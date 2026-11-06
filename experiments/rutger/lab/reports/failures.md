# Failure analysis: d3-lite-tuned-full

## conditionalqa (285 questions)

Outcome per question (judge run):

| group | n | top10 | judge: gold ranked >10 | search: gold never judged |
|---|---|---|---|---|
| ALL | 285 | 0.92 | 0.01 | 0.06 |
| all | 285 | 0.92 | 0.01 | 0.06 |

Baseline top-10 hits the judge pushed out: 2; baseline misses the judge brought into the top 10: 16

Judge AUC, gold passages vs each kind of distractor in the same pools (mean P(yes) in brackets):

| group | gold P | other document | same document, other clause |
|---|---|---|---|
| ALL | 0.51 | 0.96 (0.03, n 14419) | 0.91 (0.08, n 1523) |
| all | 0.51 | 0.96 (0.03, n 14419) | 0.91 (0.08, n 1523) |

Judge AUC by passage length (characters):

| length | gold n | neg n | AUC | mean P gold | mean P neg |
|---|---|---|---|---|---|
| 0-800 | 15 | 1204 | 0.98 | 0.45 | 0.02 |
| 800-1600 | 452 | 14730 | 0.96 | 0.51 | 0.03 |
| 1600-3200 | 0 | 8 | nan | nan | 0.01 |

Top-10 hit rate by query length (characters x4 bucket):

- query ~0+ chars: 0.97 (n 58)
- query ~200+ chars: 0.90 (n 196)
- query ~400+ chars: 0.97 (n 31)

## legalbench (776 questions)

Outcome per question (judge run):

| group | n | top10 | judge: gold ranked >10 | search: gold never judged |
|---|---|---|---|---|
| ALL | 776 | 0.53 | 0.20 | 0.27 |
| contractnli | 194 | 0.29 | 0.44 | 0.27 |
| cuad | 194 | 0.77 | 0.13 | 0.10 |
| maud | 194 | 0.23 | 0.20 | 0.57 |
| privacy_qa | 194 | 0.82 | 0.05 | 0.13 |

Baseline top-10 hits the judge pushed out: 44; baseline misses the judge brought into the top 10: 191

Judge AUC, gold passages vs each kind of distractor in the same pools (mean P(yes) in brackets):

| group | gold P | other document type | other document, same type | same document, other clause |
|---|---|---|---|---|
| ALL | 0.38 | 0.80 (0.15, n 6808) | 0.84 (0.10, n 41619) | 0.88 (0.05, n 19035) |
| contractnli | 0.50 | 0.68 (0.31, n 2967) | 0.80 (0.17, n 12362) | 0.90 (0.07, n 1823) |
| cuad | 0.39 | 0.90 (0.06, n 428) | 0.92 (0.04, n 11184) | 0.91 (0.03, n 5485) |
| maud | 0.20 | 0.87 (0.03, n 333) | 0.68 (0.11, n 10115) | 0.76 (0.04, n 7923) |
| privacy_qa | 0.37 | 0.96 (0.02, n 3080) | 0.91 (0.06, n 7958) | 0.84 (0.10, n 3804) |

Judge AUC by passage length (characters):

| length | gold n | neg n | AUC | mean P gold | mean P neg |
|---|---|---|---|---|---|
| 0-800 | 132 | 23226 | 0.84 | 0.44 | 0.08 |
| 800-1600 | 716 | 35784 | 0.85 | 0.37 | 0.10 |
| 1600-3200 | 94 | 5813 | 0.81 | 0.35 | 0.11 |
| 3200-inf | 71 | 2639 | 0.86 | 0.31 | 0.05 |

Top-10 hit rate by query length (characters x4 bucket):

- query ~0+ chars: 0.55 (n 677)
- query ~200+ chars: 0.40 (n 99)

## bbsard (222 questions)

Outcome per question (judge run):

| group | n | top10 | judge: gold ranked >10 | search: gold never judged |
|---|---|---|---|---|
| ALL | 222 | 0.55 | 0.06 | 0.40 |
| Familie | 67 | 0.57 | 0.09 | 0.34 |
| Geld | 36 | 0.47 | 0.03 | 0.50 |
| Justitie | 30 | 0.63 | 0.00 | 0.37 |
| Sociale bescherming | 4 | 0.25 | 0.00 | 0.75 |
| Vreemdelingen | 13 | 0.77 | 0.15 | 0.08 |
| Werk | 6 | 0.33 | 0.00 | 0.67 |
| Wonen | 66 | 0.52 | 0.06 | 0.42 |

Baseline top-10 hits the judge pushed out: 4; baseline misses the judge brought into the top 10: 30

Judge AUC, gold passages vs each kind of distractor in the same pools (mean P(yes) in brackets):

| group | gold P | other document |
|---|---|---|
| ALL | 0.15 | 0.81 (0.04, n 6013) |
| Familie | 0.20 | 0.85 (0.04, n 1829) |
| Geld | 0.14 | 0.82 (0.04, n 1017) |
| Justitie | 0.10 | 0.75 (0.04, n 680) |
| Sociale bescherming | 0.02 | 0.78 (0.03, n 104) |
| Vreemdelingen | 0.13 | 0.86 (0.04, n 368) |
| Werk | 0.29 | 0.96 (0.06, n 166) |
| Wonen | 0.18 | 0.83 (0.03, n 1849) |

Judge AUC by passage length (characters):

| length | gold n | neg n | AUC | mean P gold | mean P neg |
|---|---|---|---|---|---|
| 0-800 | 89 | 2608 | 0.80 | 0.15 | 0.04 |
| 800-1600 | 98 | 1664 | 0.83 | 0.13 | 0.03 |
| 1600-3200 | 75 | 1100 | 0.81 | 0.14 | 0.03 |
| 3200-inf | 55 | 641 | 0.83 | 0.19 | 0.06 |

Top-10 hit rate by query length (characters x4 bucket):

- query ~0+ chars: 0.55 (n 217)
- query ~200+ chars: 0.40 (n 5)
