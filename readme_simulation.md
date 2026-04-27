# `simulation_0420` benchmark summary

结果来源：`results/summary.json`

## 实验设置

| 项目 | 数值 |
|---|---:|
| N | 8 |
| K / judges | 4 |
| r_true | 1 |
| n_repeats | 3 |
| T_list | 400, 800, 1200, 1600, 2000 |
| scale_list | 0.0, 0.5, 1.0, 2.0 |
| T_fixed | 800 |

## 收敛与 rank 选择

| 指标 | 结果 |
|---|---|
| convergence vs T | 63.67, 15.33, 10.00, 9.00, 8.67 |
| BIC rank recovery | 1.00, 1.00, 1.00, 1.00, 1.00 |
| reference run converged | true |
| reference run n_iter | 19 |
| reference run nll | 337.7489 |

## Sample size / design：final point at `T=2000`

| Design | Method | score_error | spearman |
|---|---|---:|---:|
| balanced | proposed | 1.2688 | 0.9762 |
| balanced | zhou_github | 5.8242 | 0.8571 |
| balanced | standard_btl | 7.4902 | 0.9444 |
| unbalanced_uniform | proposed | 1.5796 | 0.9921 |
| unbalanced_uniform | zhou_github | 5.7292 | 0.8651 |
| unbalanced_uniform | standard_btl | 7.4955 | 0.9603 |
| unbalanced_biased | proposed | 1.1709 | 0.9841 |
| unbalanced_biased | zhou_github | 5.7562 | 0.9365 |
| unbalanced_biased | standard_btl | 7.4463 | 0.9365 |

## Heterogeneity experiment

| scale | proposed score_error | proposed spearman | zhou_github score_error | zhou_github spearman | standard_btl score_error | standard_btl spearman |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 2.0898 | 0.9444 | 1.5647 | 0.9286 | 4.5937 | 0.9365 |
| 0.5 | 1.5456 | 0.9762 | 2.9596 | 0.9524 | 5.4325 | 0.9524 |
| 1.0 | 1.9515 | 0.9683 | 5.5106 | 0.8651 | 7.4483 | 0.9762 |
| 2.0 | 4.9359 | 0.9762 | 10.2380 | 0.5159 | 12.5962 | 0.8889 |

## Near-tie experiment

| Method | sign_accuracy | probability_error | reordering_risk |
|---|---:|---:|---:|
| proposed | 0.9167 | 0.0435 | 0.6111 |
| zhou_github | 0.4444 | 0.1251 | 0.2778 |
| standard_btl | 0.5000 | 0.1591 | 0.3333 |

## 简短结论

| 观察 | 结论 |
|---|---|
| sample size 增大 | proposed 收敛更快，`score_error` 明显下降 |
| BIC | 本组实验里始终正确选中 `r_true = 1` |
| design robustness | proposed 在 balanced / unbalanced 两类设计下都最好 |
| heterogeneity | proposed 最稳；`scale=2.0` 时 zhou_github 明显退化 |
| near-tie | proposed 在 sign / probability error 上最好，但 reordering_risk 更高 |
| gamma export scope | 前面所有正式实验都导出 proposed `gamma`；`final reference run` 额外导出完整 `gamma/mu/U/V` |
| gamma sign | `final reference run` 里 `gamma` 有 `1 / 4` 个负值 |

## Proposed gamma negative-judge scan across all formal experiments

| experiment | condition | rep | seed | negative judges (1-based) | negative count |
|---|---|---:|---:|---|---:|
| convergence_vs_T | T=400 | 0 | 79 | 2 | 1 |
| convergence_vs_T | T=400 | 1 | 1079 | none | 0 |
| convergence_vs_T | T=400 | 2 | 2079 | none | 0 |
| convergence_vs_T | T=800 | 0 | 79 | 2 | 1 |
| convergence_vs_T | T=800 | 1 | 1079 | none | 0 |
| convergence_vs_T | T=800 | 2 | 2079 | none | 0 |
| convergence_vs_T | T=1200 | 0 | 79 | none | 0 |
| convergence_vs_T | T=1200 | 1 | 1079 | none | 0 |
| convergence_vs_T | T=1200 | 2 | 2079 | none | 0 |
| convergence_vs_T | T=1600 | 0 | 79 | 2 | 1 |
| convergence_vs_T | T=1600 | 1 | 1079 | none | 0 |
| convergence_vs_T | T=1600 | 2 | 2079 | none | 0 |
| convergence_vs_T | T=2000 | 0 | 79 | none | 0 |
| convergence_vs_T | T=2000 | 1 | 1079 | none | 0 |
| convergence_vs_T | T=2000 | 2 | 2079 | none | 0 |
| bic_rank_recovery | T=400 | 0 | 79 | 2 | 1 |
| bic_rank_recovery | T=400 | 1 | 2079 | none | 0 |
| bic_rank_recovery | T=400 | 2 | 4079 | none | 0 |
| bic_rank_recovery | T=800 | 0 | 79 | 2 | 1 |
| bic_rank_recovery | T=800 | 1 | 2079 | none | 0 |
| bic_rank_recovery | T=800 | 2 | 4079 | none | 0 |
| bic_rank_recovery | T=1200 | 0 | 79 | none | 0 |
| bic_rank_recovery | T=1200 | 1 | 2079 | none | 0 |
| bic_rank_recovery | T=1200 | 2 | 4079 | none | 0 |
| bic_rank_recovery | T=1600 | 0 | 79 | 2 | 1 |
| bic_rank_recovery | T=1600 | 1 | 2079 | none | 0 |
| bic_rank_recovery | T=1600 | 2 | 4079 | none | 0 |
| bic_rank_recovery | T=2000 | 0 | 79 | none | 0 |
| bic_rank_recovery | T=2000 | 1 | 2079 | none | 0 |
| bic_rank_recovery | T=2000 | 2 | 4079 | none | 0 |
| sample_size_designs | balanced, T=400 | 0 | 79 | 2 | 1 |
| sample_size_designs | balanced, T=400 | 1 | 3079 | none | 0 |
| sample_size_designs | balanced, T=400 | 2 | 6079 | none | 0 |
| sample_size_designs | balanced, T=800 | 0 | 79 | 2 | 1 |
| sample_size_designs | balanced, T=800 | 1 | 3079 | none | 0 |
| sample_size_designs | balanced, T=800 | 2 | 6079 | none | 0 |
| sample_size_designs | balanced, T=1200 | 0 | 79 | none | 0 |
| sample_size_designs | balanced, T=1200 | 1 | 3079 | none | 0 |
| sample_size_designs | balanced, T=1200 | 2 | 6079 | none | 0 |
| sample_size_designs | balanced, T=1600 | 0 | 79 | 2 | 1 |
| sample_size_designs | balanced, T=1600 | 1 | 3079 | none | 0 |
| sample_size_designs | balanced, T=1600 | 2 | 6079 | none | 0 |
| sample_size_designs | balanced, T=2000 | 0 | 79 | none | 0 |
| sample_size_designs | balanced, T=2000 | 1 | 3079 | none | 0 |
| sample_size_designs | balanced, T=2000 | 2 | 6079 | none | 0 |
| sample_size_designs | unbalanced_uniform, T=400 | 0 | 79 | 2 | 1 |
| sample_size_designs | unbalanced_uniform, T=400 | 1 | 3079 | none | 0 |
| sample_size_designs | unbalanced_uniform, T=400 | 2 | 6079 | 1 | 1 |
| sample_size_designs | unbalanced_uniform, T=800 | 0 | 79 | 2,3 | 2 |
| sample_size_designs | unbalanced_uniform, T=800 | 1 | 3079 | none | 0 |
| sample_size_designs | unbalanced_uniform, T=800 | 2 | 6079 | 1 | 1 |
| sample_size_designs | unbalanced_uniform, T=1200 | 0 | 79 | 2 | 1 |
| sample_size_designs | unbalanced_uniform, T=1200 | 1 | 3079 | none | 0 |
| sample_size_designs | unbalanced_uniform, T=1200 | 2 | 6079 | none | 0 |
| sample_size_designs | unbalanced_uniform, T=1600 | 0 | 79 | none | 0 |
| sample_size_designs | unbalanced_uniform, T=1600 | 1 | 3079 | none | 0 |
| sample_size_designs | unbalanced_uniform, T=1600 | 2 | 6079 | none | 0 |
| sample_size_designs | unbalanced_uniform, T=2000 | 0 | 79 | none | 0 |
| sample_size_designs | unbalanced_uniform, T=2000 | 1 | 3079 | none | 0 |
| sample_size_designs | unbalanced_uniform, T=2000 | 2 | 6079 | none | 0 |
| sample_size_designs | unbalanced_biased, T=400 | 0 | 79 | 2 | 1 |
| sample_size_designs | unbalanced_biased, T=400 | 1 | 3079 | none | 0 |
| sample_size_designs | unbalanced_biased, T=400 | 2 | 6079 | 1 | 1 |
| sample_size_designs | unbalanced_biased, T=800 | 0 | 79 | 2 | 1 |
| sample_size_designs | unbalanced_biased, T=800 | 1 | 3079 | none | 0 |
| sample_size_designs | unbalanced_biased, T=800 | 2 | 6079 | none | 0 |
| sample_size_designs | unbalanced_biased, T=1200 | 0 | 79 | 2 | 1 |
| sample_size_designs | unbalanced_biased, T=1200 | 1 | 3079 | none | 0 |
| sample_size_designs | unbalanced_biased, T=1200 | 2 | 6079 | none | 0 |
| sample_size_designs | unbalanced_biased, T=1600 | 0 | 79 | 2 | 1 |
| sample_size_designs | unbalanced_biased, T=1600 | 1 | 3079 | none | 0 |
| sample_size_designs | unbalanced_biased, T=1600 | 2 | 6079 | none | 0 |
| sample_size_designs | unbalanced_biased, T=2000 | 0 | 79 | 2,3 | 2 |
| sample_size_designs | unbalanced_biased, T=2000 | 1 | 3079 | none | 0 |
| sample_size_designs | unbalanced_biased, T=2000 | 2 | 6079 | none | 0 |
| heterogeneity_levels | scale=0.0 | 0 | 79 | none | 0 |
| heterogeneity_levels | scale=0.0 | 1 | 4079 | none | 0 |
| heterogeneity_levels | scale=0.0 | 2 | 8079 | none | 0 |
| heterogeneity_levels | scale=0.5 | 0 | 79 | 2 | 1 |
| heterogeneity_levels | scale=0.5 | 1 | 4079 | none | 0 |
| heterogeneity_levels | scale=0.5 | 2 | 8079 | none | 0 |
| heterogeneity_levels | scale=1.0 | 0 | 79 | 2 | 1 |
| heterogeneity_levels | scale=1.0 | 1 | 4079 | none | 0 |
| heterogeneity_levels | scale=1.0 | 2 | 8079 | none | 0 |
| heterogeneity_levels | scale=2.0 | 0 | 79 | 2 | 1 |
| heterogeneity_levels | scale=2.0 | 1 | 4079 | none | 0 |
| heterogeneity_levels | scale=2.0 | 2 | 8079 | none | 0 |
| near_tie | T_fixed=800 | 0 | 79 | 2,3 | 2 |
| near_tie | T_fixed=800 | 1 | 5079 | none | 0 |
| near_tie | T_fixed=800 | 2 | 10079 | none | 0 |

**Quick pattern:** most negative `gamma` cases come from `seed=79`; repeated negative judge is judge 2, with a few cases involving judge 1 or judge 3 under unbalanced / near-tie settings.

## 文件与函数说明

下面按 `neurips_2026.tex` 里的术语理解：核心模型是 judge-aware Bradley--Terry--Luce，分解为
`S = gamma * mu^T + U V^T`，并通过 `Initialize`、`ReAnchor`、proximal alternating MLE、BIC rank selection 来估计。

### `run_simulation.py`
- `to_jsonable`：把 `numpy` 结果转成可写入 `summary.json` 的 Python 标量 / 列表。
- `ensure_results_dir`：确保 `results/` 存在。
- `fit_all_methods`：统一跑 proposed / zhou_github / standard_btl 三个方法。
- `evaluate_methods`：统一计算参数误差、ranking 指标、near-tie 指标。
- `_plot_metric_grid`：画实验曲线图。
- `run_convergence_experiment`：做 sample size 对收敛迭代数影响实验。
- `run_bic_experiment`：做 BIC rank recovery 实验。
- `run_sample_size_design_experiments`：比较 balanced / unbalanced 设计下三种方法表现。
- `run_heterogeneity_experiment`：比较不同 heterogeneity scale 下表现。
- `run_near_tie_experiment`：做 near-tie stress test。
- `run_final_parameter_export`：导出一组 reference run 的完整 `gamma/mu/U/V`。
- `run_benchmark`：串起全部实验，保存图和 `summary.json`。

### `src/generate_data.py`
- `validate_rank`：检查 rank 是否满足可识别性上界 `r <= min(K-1, N-2)`。
- `zero_sum_basis`：构造零和子空间基。
- `_canonicalize_columns`：规范 latent factor 列符号，避免等价解符号翻转。
- `generate_true_parameters`：生成论文模型下真值参数 `mu, gamma, U, V`。
- `compute_score_matrix`：按 `S = gamma * mu^T + U V^T` 组装 judge-item score matrix。
- `build_near_tie_parameters`：把指定 item pair 改造成 near-tie，同时保持识别约束。
- `_logistic`：logistic link。
- `generate_balanced_comparisons`：生成近似 balanced 设计的 pairwise comparisons。
- `generate_unbalanced_comparisons`：生成 uniform / biased 两种 unbalanced 设计。
- `comparisons_to_aggregated`：把逐条比较转成 `n_ijk` 和 `y_ijk` 聚合张量。
- `identify_near_tie_pairs`：找最接近 0.5 胜率的 item pairs。

### `src/models.py`
- `validate_rank`：再次检查估计阶段 rank 可行性。
- `sigmoid`：调用 `expit`。
- `project_zero_sum`：把向量投影到零和约束空间。
- `canonicalize_columns`：同时规范 `U`、`V` 列符号。
- `negative_log_likelihood`：计算论文目标函数 negative log-likelihood。
- `aggregate_judge_pairs`：取 pooled 或单 judge 的 pairwise 聚合数据。
- `fit_centered_btl_from_pairs`：初始化阶段用 centered BTL 拟合 item scores。
- `reanchor`：实现论文 Appendix 里的 `ReAnchor`，恢复 identified representation。
- `initialize_parameters`：实现 `Initialize`，从 pooled / judgewise BTL 和 residual SVD 构造初值。
- `_pack_params`：把参数块拼成单向量，便于优化和收敛监控。
- `make_judge_constraints`：构造 judge block 的约束，主要是 `sum(gamma)=K` 和 `1^T U=0`。
- `make_item_constraints`：构造 item block 的约束，主要是 `1^T mu=0` 和 `1^T V=0`。
- `alternating_mle`：实现 proximal anchored alternating MLE 主算法。
- `fit_rank0_model`：实现 `r=0` 特例估计。
- `estimate_parameters`：估计入口，分派到 rank-0 或一般 rank 算法。
- `select_rank_by_bic`：按论文里的 BIC 公式选 rank。

### `src/benchmarks.py`
- `fit_proposed`：跑本文 proposed estimator，返回完整参数块和 score matrix。
- `fit_standard_btl`：跑 pooled standard BTL baseline。
- `fit_zhou_github`：跑 Zhou github baseline，把 judge effect 写成 `exp(alpha_k)`。

### `src/evaluate.py`
- `_canonicalize_columns`：评估前先统一 factor 列符号。
- `align_mu`：把估计 `mu` 对齐到真值尺度。
- `align_factors_with_reanchor`：按 `ReAnchor` 风格对齐 `U, V`。
- `compute_parameter_errors`：算 `mu/gamma/UV/score` 误差。
- `compute_ranking_metrics`：算 Spearman、pairwise accuracy、score sign accuracy。
- `compute_near_tie_metrics`：算 near-tie sign accuracy、probability error、reordering risk。

### 与 `neurips_2026.tex` 的对应关系
- 模型定义：`src/generate_data.py` 和 `src/models.py` 对应论文里的 judge-aware BTL 分解 `S = gamma mu^T + UV^T`。
- 初始化与重锚定：`initialize_parameters` / `reanchor` 对应 appendix 里的 `Initialize` / `ReAnchor` 算法。
- 主估计器：`alternating_mle` 对应论文主算法里的 proximal alternating updates。
- rank 选择：`select_rank_by_bic` 对应 tex 里的 BIC section。
- 实验块：`run_simulation.py` 里的 convergence / BIC / heterogeneity / near-tie，对应 tex 里 synthetic recovery、heterogeneity、near-tie 这类实验思路。
