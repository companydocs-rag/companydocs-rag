# Shortcomings of the fine-tuned d3 judge: where it lacks the ability, where it has it

Sources: `runs/analysis/failures.md` (bench.failures over the full benchmark runs) and the probe test of
`runs/wsp/log.md` turn 29. Probe test: on the diagnosis half of the benchmark questions (split by a hash
of the question id; the other half is untouched for measuring later edits), a linear probe on the
judge's residual stream (5-fold cross-validation grouped by question) against the judge's own P(yes);
metric: mean per-question AUC, gold vs distractors of the same question. Probe clearly above the judge =
the information is inside the model but its decision does not use it (an edit can unlock it); probe
not above = the model lacks it (data or pipeline).

| # | shortcoming | evidence | judge | probe | verdict |
|---|---|---|---|---|---|
| 1 | Wrong document / lookalike clause among judged passages (LegalBench; ContractNLI worst) | ContractNLI: gold judged but ranked > 10 in 44% of questions; passages of another document type get mean P(yes) 0.31 (AUC 0.68) | 0.663 | 0.830 (layer 12) | has the ability (+0.17) |
| 2 | Dutch statute articles (bBSARD) | gold mean P(yes) 0.15; near chance against the hardest distractors | 0.529 | 0.663 (layer 15) | partly has (+0.13); mostly lacks |
| 3 | Picking the right article inside the right document (LegalBench, 121 of 208 search failures) | gold passage pruned after its document was reached | 0.825 | 0.797 | lacks (no gain in the probe) |
| 4 | Picking the right document (LegalBench, 87 of 208 search failures) | among 16 sibling documents the judge is near perfect | 0.987 | 0.999 | has it; the loss is in the search (beam 6 among ~95-400 documents, first level) |
| 5 | Gold outside the judged pool (bBSARD 40% of questions; 21 of 88 had it at baseline rank 21-100) | pool = 50 shortlist articles + baseline top 20 | - | - | pipeline (pool size), not the model |
| 6 | Long passages | real benchmarks: AUC flat over length (LegalBench 0.84 / 0.85 / 0.81 / 0.86; bBSARD 0.80-0.83); the synthetic filler probe exaggerated it | - | - | small on real data; the evidence is weaker (lacks) |
| 7 | MAUD (merger agreements): long, complex questions | 57% search failures; same-type distractors AUC 0.68; gold mean P(yes) 0.20 | (in #1) | (in #1) | not separately probed |
| 8 | Judge demotes good baseline hits | LegalBench 44 baseline top-10 hits pushed out vs 191 rescued; bBSARD 4 vs 30 | - | - | follows from #1 and #2 |
