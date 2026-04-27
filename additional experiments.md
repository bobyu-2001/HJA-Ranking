# Additional Real-Data Experiments

This note summarizes three real-data stress tests for the heterogeneous judge-aware ranking model.  The experiments use the four real-data datasets already used in the main benchmark suite: Chatbot Arena, MT-Bench, UltraFeedback, and the in-house dataset.  The compared methods are the proposed HJA model, the judge-scale baseline adapted from Xu and Zhou's implementation, and pooled standard BTL.

The goal is not to replace the main stability and bootstrap experiments, but to probe three practically important regimes: near ties, dependence on judge families, and structured biased judges.  All reported HJA ranks are selected by cross-validation on the relevant training or base sample unless otherwise stated.

## Real Near-Tie Slice

### Design

For each dataset, we split processed pairwise comparisons into train and test records.  Near-tie item pairs are selected from the training split only: for each item pair, we compute the empirical win rate and choose pairs closest to 0.5, subject to a minimum pair-count threshold.  We then fit each method on the training split and evaluate separately on held-out near-tie records and non-near-tie records.

The main metric is held-out pairwise negative log likelihood; lower is better.  We also report winner-prediction accuracy after excluding tie labels.

### Results

| Dataset | Near-tie test records | Method | Near-tie log loss | Other log loss | Near-tie accuracy | Other accuracy |
|---|---:|---|---:|---:|---:|---:|
| Chatbot Arena | 253 | Proposed | 0.562 | 0.528 | 0.724 | 0.754 |
| Chatbot Arena | 253 | Zhou | 0.692 | 0.615 | 0.529 | 0.687 |
| Chatbot Arena | 253 | BTL | 0.696 | 0.618 | 0.516 | 0.687 |
| MT-Bench | 1941 | Proposed | 0.450 | NA | 0.819 | NA |
| MT-Bench | 1941 | Zhou | 0.557 | NA | 0.752 | NA |
| MT-Bench | 1941 | BTL | 0.558 | NA | 0.752 | NA |
| UltraFeedback | 299 | Proposed | 0.578 | 0.525 | 0.706 | 0.759 |
| UltraFeedback | 299 | Zhou | 0.676 | 0.613 | 0.560 | 0.687 |
| UltraFeedback | 299 | BTL | 0.674 | 0.614 | 0.560 | 0.680 |
| In-house | 145 | Proposed | 0.708 | 0.615 | 0.508 | 0.651 |
| In-house | 145 | Zhou | 0.698 | 0.639 | 0.500 | 0.639 |
| In-house | 145 | BTL | 0.701 | 0.644 | 0.500 | 0.640 |

### Conclusion

The near-tie slice is the strongest positive real-data result for HJA.  The proposed model substantially improves near-tie log loss and tie-excluded accuracy on Chatbot Arena, MT-Bench, and UltraFeedback.  On the in-house dataset, the proposed model is slightly worse than Zhou in near-tie log loss, but it remains better on the non-near-tie slice and slightly better in near-tie accuracy.  Overall, the result supports the claim that modeling judge-specific heterogeneity is most useful when item pairs are hard to separate.

## Leave-One-Family-Out Stability

### Design

We group judge names by provider-style prefix, remove one judge family at a time, and refit each method on the remaining records.  For each method, we compare the refit ranking to that method's own full-data base ranking using Spearman correlation.  This is a non-competitive diagnostic: it measures whether conclusions are dominated by one judge family, not whether HJA is better than the baselines.

### Results

The table reports the worst-case family removal for each method.

| Dataset | Method | Worst removed family | Worst Spearman vs base |
|---|---|---|---:|
| Chatbot Arena | Proposed | zai-org | 0.665 |
| Chatbot Arena | Zhou | deepseek | 0.917 |
| Chatbot Arena | BTL | deepseek | 0.941 |
| MT-Bench | Proposed | zai-org | 0.886 |
| MT-Bench | Zhou | arcee_ai | 0.943 |
| MT-Bench | BTL | arcee_ai | 0.943 |
| UltraFeedback | Proposed | zai-org | 0.887 |
| UltraFeedback | Zhou | meta-llama | 0.968 |
| UltraFeedback | BTL | meta-llama | 0.953 |
| In-house | Proposed | qwen | 0.990 |
| In-house | Zhou | openai | 0.992 |
| In-house | BTL | openai | 0.992 |

### Conclusion

The family-ablation results show that rankings are generally stable to removing a judge family, especially on MT-Bench, UltraFeedback, and the in-house dataset.  Chatbot Arena is more sensitive for HJA, with the worst removal giving Spearman 0.665.  Since this experiment is meant as a dependence diagnostic rather than a head-to-head benchmark, the main conclusion is that no single family completely determines the ranking in most datasets, while Chatbot Arena deserves additional inspection.

## Structured Cluster-Bias Judge

### Design

We inject one synthetic biased judge into each dataset.  In the cluster-bias setting, the synthetic judge favors a low-consensus item cluster: the bottom third of items according to the base HJA consensus ranking.  If exactly one item in a comparison belongs to the favored cluster, the synthetic judge chooses that item.  If both or neither items belong to the cluster, the synthetic judge chooses randomly.

This experiment uses the original whole-scale injection design: the synthetic judge receives one rewritten judgment for every raw record in the dataset.  This intentionally creates a strong structured stress test.  We compare each method's ranking after injection to its own base ranking using Spearman correlation; higher means more stable.

### Results

| Dataset | Added synthetic records | Proposed | Zhou | BTL |
|---|---:|---:|---:|---:|
| Chatbot Arena | 10000 | 0.152 | 0.054 | 0.032 |
| MT-Bench | 10000 | 0.257 | -0.086 | -0.086 |
| UltraFeedback | 10000 | 0.752 | -0.267 | -0.265 |
| In-house | 36000 | -0.329 | -0.331 | -0.329 |

### Conclusion

The cluster-bias stress test is a useful example where low-rank heterogeneity helps.  HJA is more stable than both scale-only Zhou and pooled BTL on Chatbot Arena, MT-Bench, and UltraFeedback, with the clearest gain on UltraFeedback.  The in-house dataset is a hard case: all methods are similarly disrupted.  This supports the paper's main distinction between scalar judge reliability and structured preference heterogeneity: a judge who systematically favors a subset of items cannot be fully represented by a scale-only correction.

## Whole-Scale Anti-Consensus Judge

### Design

We also ran an intentionally extreme anti-consensus stress test.  The injected judge always prefers the lower-scored item under the base HJA consensus ranking.  As with the cluster-bias stress test, this section reports the first whole-scale injection design, where the synthetic judge receives one rewritten judgment for every raw record in the dataset.

This design should not be interpreted as adding a single ordinary adversarial judge.  Because the synthetic judge clones every raw record, it can have weight comparable to an entire judge panel rather than a typical individual judge.  The experiment is therefore best viewed as a failure-mode or stress-test result: what happens if a panel-scale adversarial source is added to the data?

### Results

| Dataset | Added synthetic records | Proposed | Zhou | BTL |
|---|---:|---:|---:|---:|
| Chatbot Arena | 10000 | -0.886 | -0.650 | -0.555 |
| MT-Bench | 10000 | -0.543 | -0.829 | -0.714 |
| UltraFeedback | 10000 | -0.865 | -0.838 | -0.816 |
| In-house | 36000 | -0.367 | -0.994 | -0.991 |

### Conclusion

All methods can be severely disrupted by a whole-scale anti-consensus source.  The proposed HJA model learns the synthetic judge's anti-consensus behavior, but the consensus ranking can still move substantially.  This reflects a limitation of the current consensus geometry: under the identifying constraints, the consensus direction is tied to the unweighted average of fitted judge score rows, so a coherent panel-sized adversarial row can distort the consensus itself.  Thus, this experiment should be framed as evidence for the need to distinguish ordinary heterogeneity from malicious or adversarial judging.  A robust extension could downweight or trim detected anti-consensus judges before refitting the consensus ranking.

## Overall Takeaway

The real near-tie experiment gives positive predictive evidence for HJA, especially in the regime where leaderboard conclusions are intrinsically unstable.  Leave-one-family-out is mostly reassuring as a non-competitive dependence check.  The cluster-bias experiment supports the value of modeling structured judge-item interactions beyond scalar reliability.  The whole-scale anti-consensus experiment exposes a limitation: HJA detects adversarial behavior, but the current consensus target is not robust to a panel-sized malicious source.  This limitation is conceptually useful for the manuscript because it separates structured heterogeneity, where HJA helps, from malicious anti-consensus judging, where additional robustification is needed.
