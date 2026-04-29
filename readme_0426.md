# real_data README (2026-04-26)

今日三项修改。

---

## 1. Stability 实验（排除 zai-org/GLM-4.5-Air-FP8）

### 动机

Leave-One-Family-Out 实验中，zai-org 家族在 chatbot_arena / mtbench / ultrafeedback 上都是 Proposed 的最差 family removal（chatbot_arena Spearman 降到 0.665）。此项直接移除该 judge，量化其对 accuracy 和 heterogeneity 结构的影响。

### 代码改动

- `run_real_data.py`
  - 新增 `run_stability_excluding_judge(...)`：过滤指定 judge 后跑完整 stability 流程（train/test split → CV 选 rank → fit 三种方法 → heatmap）
  - 新增 `ensure_stability_excluding_dir()` / `ensure_stability_excluding_heatmap_dir()`
  - 修改 `plot_proposed_uvt_heatmap(...)`：增加可选 `output_dir=None`、`sort_rows=True`、`sort_cols=True` 参数，行列按 `U @ V.T` 自身均值降序排列（暖色聚左上）
  - 修改 `plot_real_data_accuracy(...)`：增加可选 `output_path=None`、`title_suffix=None` 参数
  - CLI 新增 `--experiment stability_excluding_zai_org`

### 运行

```bash
python run_real_data.py --experiment stability_excluding_zai_org
```

### 结果

| Dataset | Proposed (含) | Proposed (不含) | Δ |
|---|---:|---:|---:|
| chatbot_arena | 0.7413 | 0.6072 | -0.1341 |
| mtbench | 0.8218 | 0.7579 | -0.0639 |
| ultrafeedback | 0.7602 | 0.6712 | -0.0890 |
| in_house | 0.6567 | 0.6238 | -0.0329 |

- 移除后 Proposed 仍是最优方法，但所有方法 accuracy 均下降
- chatbot_arena 降幅最大，与 LOF-O 结论一致
- 所有数据集 CV 均选 rank=1

### 输出

```
results/stability_excluding_zai_org/
├── stability_summary.json
├── stability_accuracy.png
└── proposed_uvt_heatmaps/
    ├── chatbot_arena.png
    ├── mtbench.png
    ├── ultrafeedback.png
    └── in_house.png
```

---

## 2. In-House Mu 95% CI 表格

### 动机

为 Proposed 方法在 in_house 数据上的 mu（consensus score）提供 95% bootstrap 置信区间。

### 代码改动

- `plot_in_house_mu_ci_table.py`（新建，独立脚本）
  - 加载 in_house 数据 → 5-fold CV 选 rank → 全量拟合 baseline → 200 次 bootstrap → percentile CI → 表格输出

### 运行

```bash
python plot_in_house_mu_ci_table.py
```

### 结果

- CV 选 rank=1，199/200 bootstrap 成功，耗时 ~176s
- mu 最高：Qwen/Qwen3-Next-80B-A3B-Instruct（1.0060），最低：arize-ai/qwen-2-1.5b-instruct（-1.4571）
- 详细结果见 `results/in_house_mu_ci_table.csv`

### 输出

```
results/
├── in_house_mu_ci_table.csv
└── in_house_mu_ci_summary.json
```

---

## 3. Heatmap 行列排序

### 代码改动

- `run_real_data.py`
  - `plot_proposed_uvt_heatmap(...)`：新增可选参数 `sort_rows=True`、`sort_cols=True`
    - 列按 `U @ V.T` 列均值降序 → 正 heterogeneity 高的 item 靠左
    - 行按 `U @ V.T` 行均值降序 → 正 heterogeneity 高的 judge 靠上
    - 不影响原有调用（默认开启），可通过 `sort_rows=False` / `sort_cols=False` 关闭

### 影响范围

主 stability 和 stability_excluding_zai_org 的 8 张 heatmap 均已按新规则重新生成。
