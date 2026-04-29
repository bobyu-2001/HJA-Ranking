# real_data README (2026-04-26)

## 今日更新

新增 `stability_excluding_zai_org` 实验：在所有四个真实数据集上，移除 `zai-org/GLM-4.5-Air-FP8` 这个 judge 后重新跑 stability 实验（train/test split + accuracy + heatmap）。

动机：之前的 Leave-One-Family-Out 实验中，zai-org 家族在 chatbot_arena / mtbench / ultrafeedback 上都是 Proposed 的最差 family removal，尤其是 chatbot_arena 上 Spearman 降到 0.665。这个实验对此做更直接的量化 —— 直接移除该 judge，看 accuracy 和 heterogeneity 结构的变化。

---

## 代码改动

### 新增函数

1. **`run_stability_excluding_judge(...)`**（`run_real_data.py`）
   - 参数：`test_ratio`、`random_seed`、`dataset_name`、`exclude_judges`（默认 `['zai-org/GLM-4.5-Air-FP8']`）
   - 流程：加载数据 → 过滤掉排除 judge 的记录 → 用 `build_real_dataset_from_records` 重建 → train/test split → 5-fold CV 选 rank → fit 三种方法 → 生成 heatmap → 落盘
   - 输出目录：`results/stability_excluding_zai_org/`

2. **`ensure_stability_excluding_dir()` / `ensure_stability_excluding_heatmap_dir()`**（`run_real_data.py`）
   - 创建新实验对应的输出目录

### 修改的函数（仅增加可选参数，不影响原有调用）

- **`plot_proposed_uvt_heatmap(...)`**：新增 `output_dir=None` 参数，允许指定 heatmap 输出目录
- **`plot_real_data_accuracy(...)`**：新增 `output_path=None` 和 `title_suffix=None` 参数，允许自定义输出路径和标题后缀

### CLI 入口

```bash
# 全量（四个数据集）
python run_real_data.py --experiment stability_excluding_zai_org

# 单数据集
python run_real_data.py --experiment stability_excluding_zai_org --dataset mtbench
```

---

## 实验结果

### 移除 judge 后各数据集规模

| Dataset | 原始 records | 移除 records | 保留 records | 原始 K | 保留 K |
|---|---:|---:|---:|---:|---:|
| chatbot_arena | 10000 | 1027 | 8973 | 10 | 9 |
| mtbench | 10000 | 498 | 9502 | 20 | 19 |
| ultrafeedback | 10000 | 467 | 9533 | 20 | 19 |
| in_house | 36000 | 1973 | 34027 | 18 | 17 |

### Test accuracy（不含 zai-org/GLM-4.5-Air-FP8）

| Dataset | Proposed | Zhou github | Standard BTL |
|---|---:|---:|---:|
| chatbot_arena | 0.6072 | 0.5932 | 0.5926 |
| mtbench | 0.7579 | 0.7085 | 0.7085 |
| ultrafeedback | 0.6712 | 0.6102 | 0.6112 |
| in_house | 0.6238 | 0.6210 | 0.6203 |

### 与原始 stability（含该 judge）对比

| Dataset | Proposed (含) | Proposed (不含) | Δ |
|---|---:|---:|---:|
| chatbot_arena | 0.7413 | 0.6072 | -0.1341 |
| mtbench | 0.8218 | 0.7579 | -0.0639 |
| ultrafeedback | 0.7602 | 0.6712 | -0.0890 |
| in_house | 0.6567 | 0.6238 | -0.0329 |

### Proposed 的 CV 选 rank

| Dataset | 候选 rank 范围 | 选中 rank |
|---|---|---|
| chatbot_arena | 0..8 | 1 |
| mtbench | 0..4 | 1 |
| ultrafeedback | 0..15 | 1（部分高 rank 在 CV 中 fold 内不收敛） |
| in_house | 0..16 | 1 |

---

## 输出文件

```
results/stability_excluding_zai_org/
├── stability_summary.json          # 主结果
├── stability_accuracy.png          # accuracy 柱状图
└── proposed_uvt_heatmaps/
    ├── chatbot_arena.png           # 9 judges × 20 items
    ├── mtbench.png                 # 19 judges × 6 items
    ├── ultrafeedback.png           # 19 judges × 17 items
    └── in_house.png                # 17 judges × 45 items
```

---

## 结论

1. 移除 `zai-org/GLM-4.5-Air-FP8` 后，**Proposed 仍然是三种方法中 test accuracy 最高的**，但所有方法的 absolute accuracy 均下降。
2. **chatbot_arena 降幅最大（-0.1341）**，这与 Leave-One-Family-Out 实验中 zai-org 是该数据集最敏感 family 的结论完全一致。
3. **in_house 降幅最小（-0.0329）**，说明该 judge 在 in_house 上的影响相对有限。
4. 所有四个数据集中，Proposed 的 CV 都稳定选择了 `rank=1`，即低秩 heterogeneity 结构在不同 judge panel 下保持一致。
