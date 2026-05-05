# HJA-Ranking

## Overview

This repository contains the official code for *Heterogeneous Judge-Aware Ranking with Sensitivity, Disagreement, and Confidence* (submitted to NeurIPS 2026). HJA is a structured multi-judge ranking framework that decomposes pairwise comparisons into a consensus ranking shared across judges, judge-specific sensitivity to consensus, and structured residual disagreement. The codebase supports synthetic data generation, model estimation (HJA, JA-Ranking, standard BTL, and unstructured baselines), uncertainty quantification, evaluation metrics, and reproduction of all experiments and figures in the paper.

---

## Repository layout

```
README.md
data/                                  # Pairwise comparison datasets (JSON)
src/                                   # Core library
    models.py                          # HJA model, alternating MLE, ReAnchor, UQ, BIC
    benchmarks.py                      # Unified fitting interface for all compared methods
    evaluate.py                        # Evaluation metrics (MSE, Spearman, NDCG, sign accuracy)
    uq.py                              # Delta-method UQ for JA-Ranking and pooled BTL baselines
    generate_data.py                   # Real-data loading, train/test split, bootstrapping
    generate_simulation_data.py        # Synthetic DGP generation and comparison design
run_simulation.py                      # Entry point for all synthetic experiments
run_real_data.py                       # Entry point for all real-data experiments
paper_ready_plots.ipynb                # Notebook that generates all camera-ready paper figures
```

### `data/`

Contains four LLM pairwise comparison datasets sourced from [the JA-Ranking benchmark](https://github.com/TanXZfra/A-Judge-Aware-Ranking-Framework-for-Evaluating-Large-Language-Models-without-Ground-Truth/tree/main/data). Each file is a JSON array of pairwise preference records annotated by judge LLMs. Raw records with unknown labels are dropped during preprocessing (handled by `src/generate_data.py`).

| File | Candidates (N) | Judges (K) |
|------|----------------|------------|
| `judge_results_10k_chatbot_arena.json` | 20 | 10 |
| `judge_results_10k_mtbench.json` | 6 | 20 |
| `judge_results_10k_ultrafeedback.json` | 17 | 20 |
| `in_house_data.json` | 45 | 18 |

The processed versions are constructed at runtime by `src/generate_data.py`, so no pre-processing step is needed.

### `src/`

The core library. Key modules:

- **`models.py`** — Implements the HJA decomposition `S = gamma * mu.T + U @ V.T` with identifiability constraints, initialization via pooled and judge-wise centered BTL, the anchored alternating MLE algorithm (Algorithm 1 in the paper), ReAnchor canonicalization, plug-in Fisher information, delta-method uncertainty quantification, and BIC-based rank selection.
- **`benchmarks.py`** — Provides a unified `fit_*` interface for all methods compared in the paper: HJA (`fit_proposed`), JA-Ranking (`fit_zhou_github`), standard pooled BTL (`fit_standard_btl`), unstructured BTL + truncated SVD (`fit_unstructured_btl_svd`), direct score MLE + SVD (`fit_direct_score_svd`), and a no-ReAnchor variant (`fit_proposed_no_reanchor`).
- **`evaluate.py`** — Computes parameter-level and score-level errors, ranking metrics (Spearman, NDCG@N, pairwise accuracy, judge-specific sign accuracy), and near-tie metrics.
- **`uq.py`** — Provides asymptotic delta-method confidence intervals for JA-Ranking (`zhou_uq_summary`) and pooled BTL (`pooled_btl_uq_summary`), complementing the HJA UQ routines in `models.py`.
- **`generate_data.py`** — Loads and preprocesses real pairwise comparison records, builds judge/item index mappings, performs record-level train/test splits and bootstrapping, injects noisy judges for robustness experiments, and aggregates raw comparisons into upper-triangular count matrices.
- **`generate_simulation_data.py`** — Generates true parameters under the paper's heterogeneous DGP (centered mu, Dirichlet gamma, orthogonalized U and V with descending singular strengths), constructs near-tie parameters, and allocates comparisons via balanced or unbalanced designs.

### `run_simulation.py`

Entry point for all synthetic experiments. Supports:

- `--experiment presentation` — Main benchmark under both the HJA DGP (Fig 2) and the JA-Ranking DGP (Fig A3).
- `--experiment convergence_likelihood` — Convergence trace analysis (Fig A2).
- `--experiment reanchor_ablation` — ReAnchor ablation study (Fig A4).

All experiments save summary JSON files under `results/` for later consumption by `paper_ready_plots.ipynb`.

### `run_real_data.py`

Entry point for all real-data experiments. Supports:

- `--experiment stability` — Record-level train/test split with held-out pairwise prediction accuracy (Table 1, hold-out column).
- `--experiment robustness` — Sequential noisy-judge injection with ranking stability evaluation (Table 1, robustness columns).
- `--experiment near_tie_slice` — Near-tie item-pair accuracy stratified by tertile (Table 1, near-tie columns).

Each experiment accepts `--dataset` to select a specific dataset and `--random-seed` for reproducibility.

### `paper_ready_plots.ipynb`

A Jupyter notebook that generates all camera-ready figures in the paper. It loads summary JSON files produced by `run_simulation.py` for synthetic figures, and directly fits HJA on each real dataset for the diagnostic figures. Output figures include Fig 2–3 and Appendix Figs A2–A10.

---

## Requirements

```
numpy
scipy
matplotlib
seaborn
```

Install with:

```bash
pip install numpy scipy matplotlib seaborn
```

---

## Reproducing paper experiments

### Synthetic experiments

All synthetic experiments use N=8 items, K=4 judges, and true heterogeneity rank r=1 by default.

**Fig 2 & Fig A3 — Main benchmark**

```bash
# Full run (50 repetitions)
python run_simulation.py --experiment presentation

# Quick smoke test (2 repetitions)
python run_simulation.py --experiment presentation --presentation-smoke
```

**Fig A2 — Convergence analysis**

```bash
python run_simulation.py --experiment convergence_likelihood
```

**Fig A4 — ReAnchor ablation**

```bash
# Full run
python run_simulation.py --experiment reanchor_ablation

# Quick smoke test
python run_simulation.py --experiment reanchor_ablation --ablation-smoke
```

**Fig A1 — BIC rank selection**

```bash
python -c "
from run_simulation import run_bic_experiment
run_bic_experiment(N=8, K=4, r_true=1,
                   T_list=[100, 150, 200, 250, 300, 400, 500, 600],
                   n_repeats=50)
# Figure saved to results/bic_rank_recovery.pdf
"
```

### Real-data experiments (Table 1)

Replace `--dataset` with `chatbot_arena`, `mtbench`, `ultrafeedback`, or `in_house`.

**Hold-out prediction**

```bash
python run_real_data.py --experiment stability --dataset mtbench
```

**Noisy-judge robustness**

```bash
python run_real_data.py --experiment robustness --dataset mtbench --max-noisy-step 10
```

**Near-tie accuracy**

```bash
python run_real_data.py --experiment near_tie_slice --dataset mtbench
```

### Paper figures (Fig 3 & Appendix Figs A5–A10)

Open `paper_ready_plots.ipynb` and run cells in order. The notebook expects simulation summary JSONs in `results/` (produced by the synthetic experiments above) and the four datasets in `data/`.
