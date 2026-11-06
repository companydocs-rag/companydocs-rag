# Mini comparison on the same questions (untouched half)

## conditionalqa (100 questions)

| setup | hit@1 | hit@10 | mrr@20 |
|---|---|---|---|
| pipeline alone | 0.474 | 0.907 | 0.632 |
| 1 old pipeline + fine-tuned | 0.701 | 0.979 | 0.807 |
| 2 old pipeline + edited | 0.680 | 0.979 | 0.798 |
| 3 new pipeline + fine-tuned | 0.763 | 0.969 | 0.851 |
| 4 new pipeline + edited | 0.753 | 0.979 | 0.842 |

Rank of the first gold result per question (- = not in the list):

| question | pipeline alone | 1 old pipeline + fine-tuned | 2 old pipeline + edited | 3 new pipeline + fine-tuned | 4 new pipeline + edited |
|---|---|---|---|---|---|
| dev-0 | - | 10 | 9 | 1 | 1 |
| dev-1 | 1 | 1 | 1 | 1 | 1 |
| dev-101 | 1 | 1 | 1 | 1 | 1 |
| dev-103 | 1 | 4 | 4 | 2 | 3 |
| dev-105 | 1 | 1 | 1 | 1 | 1 |
| dev-107 | 2 | 1 | 1 | 1 | 1 |
| dev-112 | 4 | 2 | 1 | 3 | 3 |
| dev-113 | 1 | 1 | 1 | 1 | 1 |
| dev-114 | 2 | 1 | 1 | 1 | 1 |
| dev-117 | 9 | 1 | 1 | 2 | 2 |
| dev-118 | - | - | - | - | - |
| dev-119 | 18 | 4 | 6 | 5 | 6 |
| dev-121 | 2 | 1 | 1 | 1 | 1 |
| dev-122 | 6 | 2 | 2 | 2 | 2 |
| dev-123 | 1 | 1 | 1 | 1 | 1 |
| dev-125 | 1 | 3 | 3 | 2 | 2 |
| dev-127 | - | - | - | - | - |
| dev-128 | 1 | 1 | 1 | 1 | 2 |
| dev-129 | 1 | 1 | 1 | 1 | 1 |
| dev-13 | 1 | 1 | 1 | 1 | 1 |
| dev-130 | 15 | 1 | 1 | 2 | 1 |
| dev-131 | 1 | 1 | 1 | 1 | 1 |
| dev-133 | 3 | 1 | 1 | 1 | 1 |
| dev-134 | 2 | 2 | 2 | 2 | 2 |
| dev-135 | 2 | 1 | 2 | 1 | 1 |
| dev-136 | 2 | 1 | 2 | 1 | 2 |
| dev-138 | 1 | 1 | 1 | 1 | 1 |
| dev-14 | 1 | 1 | 1 | 1 | 1 |
| dev-140 | 1 | 1 | 1 | 1 | 1 |
| dev-142 | 4 | 5 | 5 | 1 | 1 |
| dev-143 | 4 | 1 | 1 | 1 | 1 |
| dev-146 | 1 | 6 | 6 | 3 | 2 |
| dev-147 | 1 | 1 | 1 | 1 | 1 |
| dev-148 | 1 | 1 | 1 | 1 | 1 |
| dev-149 | 2 | 1 | 2 | 1 | 1 |
| dev-150 | 4 | 1 | 1 | 1 | 1 |
| dev-154 | 1 | 1 | 1 | 1 | 1 |
| dev-160 | 1 | 1 | 1 | 1 | 1 |
| dev-161 | 2 | 1 | 1 | 1 | 1 |
| dev-165 | 1 | 1 | 1 | 1 | 1 |
| dev-166 | 2 | 2 | 2 | 1 | 1 |
| dev-167 | 16 | 1 | 1 | 1 | 1 |
| dev-168 | 1 | 1 | 1 | 1 | 1 |
| dev-169 | 2 | 1 | 1 | 1 | 1 |
| dev-170 | 1 | 1 | 1 | 1 | 1 |
| dev-171 | 1 | 1 | 1 | 1 | 1 |
| dev-174 | 3 | 2 | 2 | 2 | 2 |
| dev-175 | 1 | 1 | 1 | 1 | 1 |
| dev-178 | 1 | 1 | 1 | 1 | 1 |
| dev-179 | 2 | 1 | 1 | 1 | 1 |
| dev-18 | 1 | 1 | 1 | 1 | 1 |
| dev-19 | 4 | 1 | 1 | 1 | 1 |
| dev-20 | 12 | 3 | 2 | 1 | 2 |
| dev-22 | 4 | 8 | 8 | 19 | 17 |
| dev-24 | 1 | 1 | 1 | 1 | 1 |
| dev-3 | 1 | 1 | 1 | 1 | 1 |
| dev-30 | 1 | 1 | 1 | 1 | 1 |
| dev-33 | 3 | 4 | 3 | 4 | 4 |
| dev-34 | - | - | - | - | - |
| dev-35 | - | 1 | 1 | 1 | 1 |
| dev-37 | 1 | 2 | 2 | 2 | 1 |
| dev-39 | 1 | 1 | 1 | 1 | 1 |
| dev-4 | 1 | 1 | 1 | 1 | 1 |
| dev-40 | 3 | 12 | 11 | 16 | 10 |
| dev-42 | 9 | 4 | 4 | 3 | 4 |
| dev-43 | 6 | 2 | 2 | 3 | 3 |
| dev-44 | 1 | 1 | 1 | 1 | 1 |
| dev-46 | 4 | 1 | 1 | 1 | 1 |
| dev-47 | 3 | 1 | 1 | 1 | 1 |
| dev-48 | 1 | 1 | 1 | 1 | 1 |
| dev-5 | 5 | 1 | 1 | 1 | 1 |
| dev-50 | 3 | 2 | 2 | 2 | 2 |
| dev-51 | 3 | 1 | 1 | 1 | 1 |
| dev-52 | 1 | 4 | 4 | 1 | 1 |
| dev-56 | 2 | 1 | 1 | 2 | 4 |
| dev-57 | 3 | 1 | 1 | 1 | 1 |
| dev-58 | 2 | 2 | 2 | 2 | 2 |
| dev-59 | 1 | 2 | 2 | 1 | 1 |
| dev-6 | 3 | 1 | 1 | 1 | 1 |
| dev-60 | 3 | 1 | 1 | 1 | 1 |
| dev-63 | 1 | 1 | 1 | 1 | 1 |
| dev-65 | 2 | 1 | 1 | 1 | 1 |
| dev-68 | 1 | 1 | 1 | 1 | 1 |
| dev-74 | 2 | 1 | 1 | 1 | 1 |
| dev-79 | 1 | 1 | 1 | 1 | 1 |
| dev-80 | 12 | 2 | 1 | 1 | 1 |
| dev-82 | 3 | 1 | 1 | 1 | 1 |
| dev-84 | 2 | 2 | 2 | 1 | 1 |
| dev-85 | 1 | 1 | 1 | 1 | 1 |
| dev-86 | - | 1 | 1 | 1 | 1 |
| dev-88 | 1 | 1 | 2 | 1 | 1 |
| dev-89 | - | - | - | - | - |
| dev-9 | 1 | 2 | 2 | 2 | 2 |
| dev-90 | 1 | 1 | 1 | 1 | 1 |
| dev-91 | 1 | 2 | 2 | 2 | 2 |
| dev-94 | 1 | 1 | 1 | 1 | 1 |
| dev-95 | 3 | 1 | 1 | 1 | 1 |
| dev-96 | 3 | 2 | 2 | 1 | 1 |
| dev-98 | 1 | 1 | 1 | 1 | 1 |
| dev-99 | 7 | 7 | 7 | 7 | 7 |

## bbsard (16 questions)

| setup | recall@10 | ndcg@10 | mrr@20 |
|---|---|---|---|
| pipeline alone | 0.098 | 0.081 | 0.143 |
| 1 old pipeline + fine-tuned | 0.094 | 0.096 | 0.120 |
| 2 old pipeline + edited | 0.146 | 0.126 | 0.131 |
| 3 new pipeline + fine-tuned | 0.121 | 0.118 | 0.129 |
| 4 new pipeline + edited | 0.121 | 0.117 | 0.136 |

Rank of the first gold result per question (- = not in the list):

| question | pipeline alone | 1 old pipeline + fine-tuned | 2 old pipeline + edited | 3 new pipeline + fine-tuned | 4 new pipeline + edited |
|---|---|---|---|---|---|
| bbsard-1053 | - | - | - | - | - |
| bbsard-146 | - | - | - | - | - |
| bbsard-358 | - | - | - | - | - |
| bbsard-40 | - | - | - | - | - |
| bbsard-471 | 2 | 1 | 1 | 1 | 1 |
| bbsard-499 | 17 | 19 | 17 | 17 | 19 |
| bbsard-506 | - | - | - | - | - |
| bbsard-554 | 1 | 2 | 2 | 2 | 2 |
| bbsard-562 | 7 | 4 | 3 | 2 | 2 |
| bbsard-601 | - | - | - | - | - |
| bbsard-72 | - | - | - | - | - |
| bbsard-724 | - | - | - | - | - |
| bbsard-775 | - | - | - | - | - |
| bbsard-786 | 2 | 20 | 17 | 41 | 19 |
| bbsard-79 | - | - | - | - | - |
| bbsard-93 | 11 | 15 | 7 | 31 | 15 |

## legalbench (4 questions)

| setup | hit@1 | hit@10 | mrr@20 |
|---|---|---|---|
| pipeline alone | 0.250 | 0.500 | 0.375 |
| 1 old pipeline + fine-tuned | 0.000 | 0.250 | 0.162 |
| 2 old pipeline + edited | 0.250 | 0.250 | 0.266 |
| 3 new pipeline + fine-tuned | 0.250 | 1.000 | 0.542 |
| 4 new pipeline + edited | 0.250 | 0.250 | 0.283 |

Rank of the first gold result per question (- = not in the list):

| question | pipeline alone | 1 old pipeline + fine-tuned | 2 old pipeline + edited | 3 new pipeline + fine-tuned | 4 new pipeline + edited |
|---|---|---|---|---|---|
| contractnli-0 | - | 11 | - | 2 | - |
| contractnli-2 | - | - | - | 2 | 18 |
| contractnli-4 | 1 | 18 | 16 | 6 | 13 |
| cuad-0 | 2 | 2 | 1 | 1 | 1 |
