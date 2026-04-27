# real_data README (2026-04-25)

This directory contains the real-data experiments for the heterogeneous judge-aware ranking model.  The main entry point is `run_real_data.py`; helper scripts redraw saved summaries without refitting.

## 1. Methods

The experiments compare:

- `proposed`: the HJA model, `S = gamma * mu^T + U V^T`.
- `zhou_github`: the judge-scale weighted ranking baseline adapted from Zhou's implementation.
- `standard_btl`: pooled Bradley-Terry-Luce with no judge-specific reliability or heterogeneity.

For `stability`, `bootstrap`, and `robustness`, the reported `proposed` model now selects its rank by 5-fold cross validation.  Candidate ranks are

```text
0, 1, ..., min(K - 1, N - 2)
```

where `K` is the number of judges and `N` is the number of items.  Rank is selected by validation negative log-likelihood.  Failed candidate ranks are recorded and skipped; the best successful candidate is used.

In the bootstrap experiment only, an additional reference method is included:

- `proposed_full_rank`: HJA fitted at `min(K - 1, N - 2)`.

This is kept as an overparameterized reference, not as the main proposed model.

## 2. Data

Datasets are configured in `run_real_data.py`:

- `chatbot_arena`: `data/judge_results_10k_chatbot_arena.json`
- `mtbench`: `data/judge_results_10k_mtbench.json`
- `ultrafeedback`: `data/judge_results_10k_ultrafeedback.json`
- `in_house`: `data/in_house_data.json`

`src/generate_data.py` converts raw JSON records into processed pairwise comparisons with fields such as `k`, `i`, `j`, and `y`.

## 3. Experiments

### Stability

Purpose: measure test-set prediction accuracy for each `(judge k, item i, item j)` comparison.

Pipeline:

1. Load one real dataset.
2. Split processed comparisons into train/test records.
3. Select the `proposed` rank by 5-fold CV on the training split.
4. Fit `proposed`, `zhou_github`, and `standard_btl` on the training split.
5. Evaluate pairwise winner prediction accuracy on the test split.
6. Save fitted parameters and a Proposed `U @ V.T` heatmap.

Main outputs:

- `results/stability_summary.json`
- `results/stability_accuracy.png`
- `results/proposed_uvt_heatmaps/<dataset>.png`

### Bootstrap

Purpose: measure ranking stability under bootstrap resampling.

Pipeline:

1. Load one real dataset.
2. Select the `proposed` rank once by 5-fold CV on the full sample.
3. Fit baseline rankings on the full sample for:
   - `proposed`
   - `proposed_full_rank`
   - `zhou_github`
   - `standard_btl`
4. Draw bootstrap samples from processed records.
5. Refit all four methods on each bootstrap sample.
6. Report only top-3 and top-5 stability.

Top-k exact match is order-sensitive.  For example, baseline top-3

```text
A, B, C
```

does not exactly match bootstrap top-3

```text
B, A, C
```

even though the unordered set is the same.  The summary also keeps Jaccard overlap as a secondary unordered overlap statistic.

The bootstrap runner no longer accepts arbitrary `top_k` as an experiment argument.  It always reports top-3 and top-5, clipped to the number of items when a dataset has fewer than 5 items.

Main outputs:

- `results/bootstrap_summary.json`
- `results/bootstrap_summaries/bootstrap_summary_<dataset>_samples_<n>.json`
- `results/bootstrap_rank_distribution/*.png`
- `results/bootstrap_topk_stability_top_3_samples_<n>.png`
- `results/bootstrap_topk_stability_top_5_samples_<n>.png`

The old curve plotting helper `plot_bootstrap_top_k_curve(...)` is still present in `run_real_data.py` for future use, but the current bootstrap experiment does not call it.

### Robustness / Noisy Judges

Purpose: test how rankings change as synthetic noisy judges are added.

Pipeline:

1. Select the base `proposed` rank by 5-fold CV on the original dataset.
2. Fit base methods on the original dataset.
3. Add noisy judges one step at a time.
4. Refit methods after each noisy-judge step, keeping the `proposed` rank fixed at the base CV-selected rank.
5. Compare each new ranking to the base ranking.
6. Save noisy datasets and per-dataset noisy summaries.

Main outputs:

- `results/noisy_summaries/noisy_judge_summary_<dataset>.json`
- `results/noisy_datasets/<dataset>_plus_<step>_noisy_judges.json`
- `results/noisy_rank_shift.png`
- custom noisy-judge detection plots under `results/noisy_judge_detection/`

### Structured Bias Injection

Purpose: stress-test the methods with biased synthetic judges whose errors are structured rather than uniformly random.

Current variants:

- `anti_consensus`: the injected judge systematically prefers the lower-ranked item under the base consensus.
- `family_bias`: the injected judge favors the largest detected item provider/family when exactly one side of a comparison belongs to that family.
- `cluster_bias`: the injected judge favors a low-consensus item cluster.

Main outputs:

- `results/structured_bias_summary.json`
- `results/structured_bias_datasets/<dataset>_<variant>.json`

### Leave-One-Family-Out

Purpose: measure whether rankings are driven by one provider/judge family.

Pipeline:

1. Group judge names by provider-style prefix.
2. Fit base methods on the full dataset.
3. Remove one judge family at a time.
4. Refit methods with the base CV-selected HJA rank, clamped to the feasible rank after removal.
5. Report rank shifts versus each method's own full-data baseline.

Main output:

- `results/leave_one_family_out_summary.json`

### Real Near-Tie Slice

Purpose: evaluate methods on real item pairs whose empirical training win rate is closest to 0.5.

Pipeline:

1. Split processed records into train/test.
2. Select near-tie item pairs from the training split only.
3. Fit all methods on the training split.
4. Report held-out log loss and tie-skipping accuracy separately for near-tie and non-near-tie test records.

Main output:

- `results/near_tie_slice_summary.json`

## 4. Running Experiments

Run commands from inside `real_data/`:

```bash
cd real_data
```

or from the repository root with `python real_data/run_real_data.py`.

### Stability

All datasets:

```bash
python run_real_data.py --experiment stability
```

Single dataset:

```bash
python run_real_data.py --experiment stability --dataset mtbench
```

### Bootstrap

Single dataset:

```bash
python run_real_data.py --experiment bootstrap --dataset mtbench --bootstrap-samples 100
```

With explicit worker count:

```bash
python run_real_data.py --experiment bootstrap --dataset mtbench --bootstrap-samples 100 --bootstrap-workers 4
```

Notes:

- `--bootstrap-samples` must be at least `1`.
- `--bootstrap-seed` controls bootstrap sampling and CV fold shuffling for bootstrap rank selection.
- `--bootstrap-workers` controls multiprocessing.  If it is `1`, samples run serially in the current process.  If it is greater than `1`, samples run through a spawned `ProcessPoolExecutor`.
- No `--bootstrap-top-k` argument is used; top-3 and top-5 are always reported.

### Robustness

Single dataset through all noisy-judge steps:

```bash
python run_real_data.py --experiment robustness --dataset mtbench
```

Limit noisy steps:

```bash
python run_real_data.py --experiment robustness --dataset mtbench --max-noisy-step 5
```

### Structured Bias Injection

```bash
python run_real_data.py --experiment structured_bias --dataset mtbench
```

### Leave-One-Family-Out

```bash
python run_real_data.py --experiment leave_one_family_out --dataset mtbench
python run_real_data.py --experiment leave_one_family_out --dataset mtbench --min-family-judges 2
```

### Real Near-Tie Slice

```bash
python run_real_data.py --experiment near_tie_slice --dataset mtbench
python run_real_data.py --experiment near_tie_slice --dataset mtbench --near-tie-min-pair-records 20 --near-tie-max-pairs 20
```

### Seed Sweeps for Error Bars

`run_seed_sweep.py` runs repeated seeds and writes raw per-seed JSON plus an aggregate summary under `results/seed_sweeps/`.

Default protocol:

- `stability`: 20 train/test split seeds.
- `near_tie_slice`: 20 train/test split seeds.
- `bootstrap`: 5 bootstrap RNG seeds, with 500 bootstrap samples per seed.

All three experiments:

```bash
python run_seed_sweep.py --experiment all --bootstrap-samples 500
```

Single experiment:

```bash
python run_seed_sweep.py --experiment stability
python run_seed_sweep.py --experiment near_tie_slice
python run_seed_sweep.py --experiment bootstrap --bootstrap-samples 500
```

Use `--reuse` to aggregate from existing per-seed raw JSON without rerunning completed seeds.

### All

Run stability, robustness, and optionally bootstrap:

```bash
python run_real_data.py --experiment all --dataset mtbench --bootstrap-samples 20
```

If `--bootstrap-samples` is omitted or set to `0`, the `all` run skips bootstrap.

## 5. Redrawing Saved Summaries

These scripts read saved JSON summaries and redraw figures without refitting.

### Stability Rank Panels

```bash
python plot_stability_accuracy_panels.py
python plot_stability_accuracy_panels.py --dataset mtbench
```

Reads:

- `results/stability_summary.json`

Writes:

- `results/stability_accuracy_panels/<dataset>_stability_accuracy_panel.png`

### Bootstrap Summary Bars

```bash
python plot_bootstrap_summary_bars.py --bootstrap-samples 100
```

Reads archived summaries from:

- `results/bootstrap_summaries/`

Writes:

- `results/bootstrap_topk_stability_top_3_samples_<n>_custom.png`
- `results/bootstrap_topk_stability_top_5_samples_<n>_custom.png`

The script includes `proposed_full_rank` when present in the saved summary and leaves missing values blank for older summaries.

### Bootstrap Rank Distribution Panels

```bash
python plot_bootstrap_rank_distribution_panels.py
python plot_bootstrap_rank_distribution_panels.py --dataset mtbench --bootstrap-samples 100
```

Reads archived summaries from:

- `results/bootstrap_summaries/`

Writes:

- `results/bootstrap_rank_distribution_panels/<dataset>_samples_<n>_rank_distribution_panel.png`

### Noisy Judge Rank Shift

```bash
python plot_noisy_judge_rank_shift.py --use-existing-summary
python plot_noisy_judge_rank_shift.py --dataset mtbench --use-existing-summary
```

### Noisy Judge Detection

```bash
python plot_noisy_judge_detection.py
```

## 6. Code Map

Main files:

- `run_real_data.py`: main experiment runner, plotting for core summaries, rank CV helper.
- `src/models.py`: HJA likelihood, constraints, alternating MLE, rank-0 fit, BIC helper.
- `src/benchmarks.py`: wrappers for Proposed, Zhou, and standard BTL fits.
- `src/generate_data.py`: data processing, train/test split, bootstrap resampling, rank summaries, top-k stability.
- `plot_stability_accuracy_panels.py`: redraw item-level stability panels.
- `plot_bootstrap_summary_bars.py`: redraw top-3/top-5 bootstrap exact-match bars.
- `plot_bootstrap_rank_distribution_panels.py`: redraw bootstrap rank distribution panels.
- `plot_noisy_judge_rank_shift.py`: redraw noisy-judge exact rank-match plots.
- `plot_noisy_judge_detection.py`: analyze noisy-judge separation.
- `analyze_bootstrap_summary.ipynb`: notebook for inspecting bootstrap summaries.

## 7. Practical Notes

- `proposed` can be expensive on larger datasets, especially during CV because every candidate rank is fitted across folds.
- `proposed_full_rank` in bootstrap is a reference and can fail or be slow on difficult datasets.
- Bootstrap multiprocessing uses spawned processes when `--bootstrap-workers > 1`; use `--bootstrap-workers 1` for debugging or notebook-style execution.
- Existing result files may reflect older code versions.  Rerun the relevant experiment before using a figure or JSON summary in the manuscript.
