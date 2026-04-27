# real_data_0422

## 1. 当前结论

- `stability`：当前版本下，`proposed` 在四个真实数据集上的 overall test accuracy 整体最好
  - `stability ranking`：四个 dataset 都已可直接重画 `3x1` rank panel；y 轴是各方法的 `mu ranking`，点颜色表示该 item 在 test split 上的 prediction accuracy，结果见 `results/stability_accuracy_panels/` --> proposed ranking与其他两个模型**有一定差距**
- `bootstrap`：`sample=20/50/100` 的 `top-3` / `top-5` bootstrap summary 已有，三个模型**无明显差距**，结果见 `results/bootstrap_topk_stability_*.png`
- `robustness`：`chatbot_arena` 上 `proposed` 也很强但仍弱于 `zhou_github`；`mtbench` / `ultrafeedback` 上 `zhou_github` 明显更强；`in_house` 上两者都不算强，没有明显赢家；结果见 `results/noisy_rank_shift_custom.png`，横轴 `steps` 代表增加了多少个noisy judge
  - 若把 **gamma 分离 noisy judge** 作为 robustness 补充评估，当前结果支持：`zhou_github > proposed >> standard_btl`，结果见 `results/noisy_judge_detection`

---

当前目录只保留真实数据实验。核心入口是 `run_real_data.py`，包含三类实验：
- `stability`：train/test split 后比较 test accuracy
- `bootstrap`：对 processed records 做 bootstrap 重采样，比较 ranking 稳定性
- `robustness`：逐步加入 noisy judge，比较 ranking shift 与 judge-level 分离能力

使用的三种方法：
- `proposed`
- `zhou_github`
- `standard_btl`

---

## 2. 数据集与默认设定

数据集定义在 `run_real_data.py`：
- `chatbot_arena`
- `mtbench`
- `ultrafeedback`
- `in_house`

当前关键默认设定：
- `stability` 默认做 record-level `train/test split`
- `stability` 默认 `test_ratio=0.2`、`random_seed=42`
- `bootstrap` 重采样随机种子默认：`42`
- `bootstrap` 汇总重画脚本默认 sample 数：`100`
- `robustness` 的 noisy judge step：`1..10`
- `proposed` 在真实数据实验里统一走当前 `run_real_data.py` 中的拟合配置

---

## 3. 三类实验在做什么

### 3.1 Stability
目的：比较三个方法在真实数据上的预测稳定性。

流程：
1. 读入单个 dataset 的 processed comparison records
2. 做 record-level train/test split
3. 在训练集上拟合三个方法
4. 在测试集上预测 pairwise winner
5. 汇总 overall test accuracy，并保存 Proposed 的 `U @ V.T` heatmap
6. 后处理阶段可基于已保存参数重建 test split 上的 rank panel

主输出：
- `results/stability_summary.json`
- `results/stability_accuracy.png`
- `results/stability_accuracy_panels/<dataset>_stability_accuracy_panel.png`
- `results/proposed_uvt_heatmaps/<dataset>.png`

当前结果结论：
- 当前版本下，`proposed` 在四个真实数据集上的 overall test accuracy 整体最好
- `standard_btl` 通常最快，但精度最低或并列最低
- `in_house` 最慢，主要是大规模下拟合更重
- 已新增每个 dataset 一张的 stability rank panel 图，共 `4` 张
- 每张 panel 图内部为 `3x1` 子图：`proposed` / `zhou_github` / `standard_btl`
- 三个子图共用同一套 x-axis item 顺序，只在最下面显示一次
- 图中额外画了：
  - y 轴：该方法的 `mu ranking`
  - 灰色虚线 `y=x`：参考排序线
  - 点颜色：该 item 在 test split 上的 accuracy
  - 点大小：该 item 在 test split 中参与比较的次数
  - 子图标题里的 `acc=...`：该方法在该 dataset 上的 overall test accuracy

### 3.2 Bootstrap
目的：比较三个方法在重复重采样下，item ranking 尤其是 top-k 是否稳定。

流程：
1. 用全量 processed records 先拟合 baseline ranking
2. 对同一 dataset 做 bootstrap resampling
3. 每个 sample 都重新拟合三个方法
4. 汇总每个 item 的 bootstrap rank 分布
5. 比较 bootstrap ranking 和 baseline ranking 的 top-k 一致性

当前保存与重画逻辑：
- 运行实验时会写 `results/bootstrap_summary.json`
- 同时按 `dataset + sample_count` 归档到 `results/bootstrap_summaries/`
- 因为归档 summary 已保存，所以后续重画不用重跑

主输出：
- `results/bootstrap_summary.json`
- `results/bootstrap_summaries/bootstrap_summary_<dataset>_samples_<n>.json`
- `results/bootstrap_topk_curves/*.png`
- `results/bootstrap_rank_distribution/*.png`
- `results/bootstrap_rank_distribution_panels/<dataset>_samples_<n>_rank_distribution_panel.png`
- `results/bootstrap_topk_stability_top_<k>_samples_<n>.png`
- `results/bootstrap_topk_stability_top_<k>_samples_<n>_custom.png`

当前结果状态：
- `sample=20/50/100` 的 bootstrap summary 已有
- 已可直接重画跨 dataset 的 `top-3` / `top-5` 汇总图
- 已新增按 `dataset × sample` 汇总的 rank distribution panel 图，共 `4 × 3 = 12` 张
- 每张 panel 图内部为 `3x1` 子图：`proposed` / `zhou_github` / `standard_btl`
- 三个子图共用同一套 x-axis item 顺序，只在最下面显示一次
- 图中额外画了：
  - 红色 `x`：true rank
  - 灰色虚线 `y=x`：左上到右下的参考线，用来看偏离 true rank 的程度
- 当前已生成的 sample=100 汇总图：
  - `results/bootstrap_topk_stability_top_3_samples_100_custom.png`
  - `results/bootstrap_topk_stability_top_5_samples_100_custom.png`
- 当前已生成的 panel 图目录：
  - `results/bootstrap_rank_distribution_panels/`

### 3.3 Robustness / noisy judge
目的：比较三个方法在逐步加入 noisy judge 后，item ranking 是否稳定，以及是否能把 noisy judge 从原 judge 中分离出来。

流程：
1. 先用原始 dataset 拟合 base methods
2. 逐步加入 `noisy_judge_1 ... noisy_judge_10`
3. 每个 step 都重新拟合三个方法
4. 比较新 ranking 与 baseline ranking 的 shift
5. 保存每个 step 的 judge-level `gamma / mu / ranking / rank_shift`
6. 后处理阶段再看 noisy judge detection / separation

当前 summary 设计：
- **只保留 per-dataset summary**
- 不再使用会被覆盖的旧总文件方案
- 每个 dataset 的 noisy summary 固定写到：
  - `results/noisy_summaries/noisy_judge_summary_<dataset>.json`

主输出：
- `results/noisy_summaries/noisy_judge_summary_<dataset>.json`
- `results/noisy_datasets/<dataset>_plus_<step>_noisy_judges.json`
- `results/noisy_rank_shift.png`
- `results/noisy_rank_shift_custom.png`
- `results/noisy_judge_detection/*.png`
- `results/noisy_judge_detection/*.json`

当前结果结论：
- 四个 dataset 的 noisy summary 已齐
- 最终 4-dataset rank-shift 图已重画：
  - `results/noisy_rank_shift_custom.png`
- 若把 **gamma 分离 noisy judge** 作为补充评估，当前结果支持：
  - **`zhou_github > proposed >> standard_btl`**
- 具体上：
  - `chatbot_arena`：`proposed` 也很强，但 `zhou_github` 分离更明显
  - `mtbench` / `ultrafeedback`：`zhou_github` 明显强于 `proposed`
  - `in_house`：两者都不算强，没有明显赢家

---

## 4. 运行方式

### 4.1 Stability
全量：

```bash
python run_real_data.py --experiment stability
```

单 dataset：

```bash
python run_real_data.py --experiment stability --dataset mtbench
```

### 4.2 Bootstrap
运行 bootstrap 实验：

```bash
python run_real_data.py --experiment bootstrap --dataset mtbench --bootstrap-samples 100 --bootstrap-top-k 3
```

参数：
- `--bootstrap-samples`：bootstrap 次数，必须 `>= 1`
- `--bootstrap-top-k`：主视图关注的 `top_k`
- `--bootstrap-seed`：随机种子

### 4.3 Robustness
运行 noisy judge robustness：

```bash
python run_real_data.py --experiment robustness --dataset mtbench --max-noisy-step 10
```

参数：
- `--max-noisy-step`：只跑到指定 noisy step

---

## 5. 独立重画 / 分析脚本

### 5.1 `plot_stability_accuracy_panels.py`
用途：
- 不重跑 stability 拟合
- 直接读取 `results/stability_summary.json`
- 用 summary 里已保存的 `mu / gamma / U / V` 重建 score matrix
- 按 stability 默认 `test_ratio=0.2`、`random_seed=42` 重新切 test split
- 为每个 dataset 生成一张 stability rank panel 图
- 每张图内部固定为 `3x1` 子图：`proposed` / `zhou_github` / `standard_btl`
- 三个子图共用同一套 x-axis item 顺序，只在最下面显示一次
- y 轴画该方法的 `mu ranking`，accuracy 用颜色和标题补充

常用命令：

```bash
python plot_stability_accuracy_panels.py
python plot_stability_accuracy_panels.py --dataset mtbench
```

输出：
- `results/stability_accuracy_panels/<dataset>_stability_accuracy_panel.png`

图怎么读：
- 一张图对应一个 `dataset`
- 三个子图分别对应三个方法
- x 轴：固定 reference order 下的 items，只在最下面显示一次
- y 轴：该方法的 `mu ranking`
- 灰色虚线 `y=x`：参考排序线；越贴近表示越接近 reference order
- 点颜色：该 item 在 test split 上的 pairwise prediction accuracy
- 点大小：该 item 在 test split 中出现次数越多
- 子图标题里的 `acc=...`：该方法在该 dataset 上的 overall test accuracy

### 5.2 `plot_bootstrap_summary_bars.py`
用途：
- 不重跑 bootstrap
- 直接读取 `results/bootstrap_summaries/` 下已有归档 summary
- 重画跨 dataset 的 `top-3` / `top-5` 稳定性汇总柱状图

常用命令：

```bash
python plot_bootstrap_summary_bars.py --bootstrap-samples 20
python plot_bootstrap_summary_bars.py --bootstrap-samples 50
python plot_bootstrap_summary_bars.py --bootstrap-samples 100
```

输出：
- `results/bootstrap_topk_stability_top_3_samples_<n>_custom.png`
- `results/bootstrap_topk_stability_top_5_samples_<n>_custom.png`

### 5.3 `plot_bootstrap_rank_distribution_panels.py`
用途：
- 不重跑 bootstrap
- 直接读取 `results/bootstrap_summaries/` 下已有归档 summary
- 为每个 `dataset × sample_count` 生成一张 rank distribution panel 图
- 每张图内部固定为 `3x1` 子图：`proposed` / `zhou_github` / `standard_btl`
- 三个子图共用同一套 x-axis item 顺序，只在最下面显示一次
- 图中同时叠加 true rank 和对角参考线，便于直接比较不同方法相对 true 的偏离

常用命令：

```bash
python plot_bootstrap_rank_distribution_panels.py
python plot_bootstrap_rank_distribution_panels.py --dataset mtbench --bootstrap-samples 100
```

输出：
- `results/bootstrap_rank_distribution_panels/<dataset>_samples_<n>_rank_distribution_panel.png`

图怎么读：
- 一张图对应一个 `dataset` 和一个 `sample_count`
- 三个子图分别对应三个方法
- x 轴：固定 true order 下的 items，只在最下面显示一次
- y 轴：rank
- 蓝色点和误差棒：bootstrap rank 的中位数及区间
- 红色 `x`：true rank
- 灰色虚线 `y=x`：左上到右下的参考线；越贴近表示越接近 true rank

### 5.4 `plot_noisy_judge_rank_shift.py`
用途：
- 不重跑实验，直接读取 `results/noisy_summaries/` 下已有 summary
- 重画 noisy judge exact rank-match 图
- 支持单 dataset，也支持自动合并多个 dataset summary

常用命令：

```bash
python plot_noisy_judge_rank_shift.py --use-existing-summary
python plot_noisy_judge_rank_shift.py --dataset mtbench --use-existing-summary
python plot_noisy_judge_rank_shift.py --dataset mtbench --max-noisy-step 5 --use-existing-summary
```

输出：
- `results/noisy_exact_rank_match_rate_custom.png`
- 或 `results/noisy_exact_rank_match_rate_<dataset>_through_<step>_custom.png`

图怎么读：
- 子图：每个 dataset 一个格子
- x 轴：`# noisy judges`
- y 轴：`Exact rank-match rate vs base`
- 每个格子三条线：三个方法

### 5.5 `plot_noisy_judge_detection.py`
用途：
- 不重跑实验，直接读取 `results/noisy_summaries/`
- 检查三个方法是否能把 noisy judge 从原 judge 中分离出来
- 把 judge-level detection / separation 当作 robustness 补充衡量

常用命令：

```bash
python plot_noisy_judge_detection.py
python plot_noisy_judge_detection.py --dataset mtbench
python plot_noisy_judge_detection.py --dataset mtbench --max-noisy-step 5
```

输出目录：
- `results/noisy_judge_detection/`

每个 dataset 会生成：
- `noisy_judge_gamma_scatter_<dataset>.png`
- `noisy_judge_gamma_boxplot_<dataset>.png`
- `noisy_judge_hit_rate_<dataset>.png`
- `noisy_judge_detection_metrics_<dataset>.json`

图怎么读：

1. `gamma_scatter`
- 功能：看最后一个可用 noisy step 时，哪些 judge 被排到 gamma 异常端
- x 轴：judge 按 gamma 排序后的名次（从低到高）
- y 轴：judge `gamma`
- 红点：noisy judge；蓝点：original judge

2. `gamma_boxplot`
- 功能：看不同 noisy step 下，noisy judge 和 original judge 的 gamma 分布是否分离
- x 轴：`# noisy judges`
- y 轴：judge `gamma`
- 每个 step 两组箱线图：红色 noisy，蓝色 original

3. `hit_rate`
- 功能：看按 gamma 抓 noisy judge 的准确率
- x 轴：`# noisy judges`
- y 轴：hit rate
- 定义：取 gamma 最低的前 `k` 个 judge，`k = 当前 noisy judge 个数`；看其中有多少比例是真 noisy judge

### 5.6 `analyze_zhou_gamma.py`
用途：
- 快速读取 `results/noisy_summaries/`
- 打印 Zhou judge gamma 的数值概览
- 适合做命令行排查，不是主可视化脚本

常用命令：

```bash
python analyze_zhou_gamma.py
python analyze_zhou_gamma.py --dataset mtbench
```

---

## 6. 关键文件在做什么

### `run_real_data.py`
主实验入口。负责 stability / bootstrap / robustness 三类实验的拟合、summary 落盘和基础可视化。

### `plot_stability_accuracy_panels.py`
读取已有 `stability_summary.json`，重建 test split 上的 item-level accuracy，并按 dataset 重画 `3x1` panel 图。

### `plot_bootstrap_summary_bars.py`
读取已有 bootstrap 归档 summary，重画跨 dataset 汇总柱状图。

### `plot_bootstrap_rank_distribution_panels.py`
读取已有 bootstrap 归档 summary，按 `dataset × sample_count` 重画 `3x1` rank distribution panel 图，并叠加 true rank 与 `y=x` 参考线。

### `plot_noisy_judge_rank_shift.py`
读取已有 noisy summary，重画 noisy rank shift 图；也支持重新跑 robustness 后再画图。

### `plot_noisy_judge_detection.py`
读取已有 noisy summary，做 judge-level gamma 分离分析，输出 scatter / boxplot / hit-rate。

### `analyze_zhou_gamma.py`
读取已有 noisy summary，做 Zhou gamma 的数值检查。

### `run_in_house_robustness_step10.sh`
用于单独跑 `in_house` 的 robustness step=10，并归档日志与对应 per-dataset noisy summary。

---

## 7. 关键函数在做什么

以下只列当前最重要函数。

### `run_real_data.py`
- `fit_all_methods_safe(...)`
  - 统一调用三个方法拟合，并捕获异常、记录耗时
- `serialize_method_result(...)`
  - 把拟合结果序列化成可写入 summary 的字典
- `run_real_data_stability_experiment(...)`
  - 跑 stability，输出 test accuracy 与 Proposed heatmap
- `run_real_data_bootstrap_experiment(...)`
  - 跑 bootstrap，输出 baseline、bootstrap rankings、top-k stability、rank distribution
- `run_real_data_noisy_judge_experiment(...)`
  - 跑 noisy judge robustness，逐步加 noisy judge 并记录 step summary
- `plot_real_data_accuracy(...)`
  - 画跨 dataset 的 overall stability accuracy 汇总柱状图
- `plot_noisy_rank_shift(...)`
  - 画 noisy judge rank shift 图，当前布局是按 dataset 分子图
- `plot_bootstrap_top_k_curve(...)`
  - 画单 dataset 的 bootstrap top-k exact-match 曲线
- `plot_bootstrap_rank_distribution(...)`
  - 画单 dataset × method × top_k 的 bootstrap rank 分布图
- `plot_bootstrap_top_k_stability_summary(...)`
  - 画跨 dataset 的 bootstrap top-k 汇总柱状图
- `write_noisy_summary(...)`
  - 按 dataset 写 noisy summary 到 `results/noisy_summaries/`
- `write_bootstrap_summary(...)`
  - 写 bootstrap summary 到主文件和按 `dataset + samples` 归档文件

### `plot_stability_accuracy_panels.py`
- `rebuild_score_matrix(...)`
  - 用 summary 中已保存的 `mu / gamma / U / V` 重建方法 score matrix
- `compute_item_accuracy(...)`
  - 用 test split 统计每个 item 的预测 accuracy 和 support
- `build_dataset_method_results(...)`
  - 对单个 dataset 整理三个方法的 item-level accuracy
- `plot_method_accuracy(...)`
  - 画单个方法子图，并叠加 overall accuracy 参考线
- `plot_dataset(...)`
  - 画单个 dataset 的 `3x1` stability accuracy panel 图

### `plot_bootstrap_summary_bars.py`
- `collect_plot_values(...)`
  - 从归档 summary 提取各 dataset 的 exact-match rate
- `plot_summary_bar(...)`
  - 生成跨 dataset 的 top-k 汇总柱状图

### `plot_bootstrap_rank_distribution_panels.py`
- `get_reference_item_order(...)`
  - 取 panel 图统一使用的 item 顺序
- `build_rank_distribution(...)`
  - 从 bootstrap rankings 汇总每个 item 的 rank 中位数与区间
- `plot_method_distribution(...)`
  - 画单个方法子图，并叠加 true rank 与 `y=x` 参考线
- `plot_dataset_sample(...)`
  - 画单个 `dataset × sample_count` 的 `3x1` panel 图

### `plot_noisy_judge_rank_shift.py`
- `load_existing_noisy_summary(...)`
  - 读取一个 dataset 或合并多个 dataset 的 noisy summary
- `build_output_path(...)`
  - 生成 custom 图文件名

### `plot_noisy_judge_detection.py`
- `build_method_step_records(...)`
  - 把 summary 整理成按 method / step 的 gamma 记录
- `compute_hit_rate(...)`
  - 计算按低 gamma 抓 noisy judge 的命中率
- `plot_sorted_gamma_scatter(...)`
  - 画 gamma 排序散点图
- `plot_gamma_boxplot(...)`
  - 画 noisy / original judge 的 gamma 分布箱线图
- `plot_hit_rate(...)`
  - 画 hit-rate 曲线
- `save_metrics(...)`
  - 写出 detection metrics json

### `analyze_zhou_gamma.py`
- `load_summary(...)`
  - 读取 per-dataset noisy summary
- `summarize_gamma(...)`
  - 汇总 noisy / original gamma 的最小值、中位数、最大值
- `analyze_dataset(...)`
  - 打印某个 dataset 的 Zhou gamma 分析结果

---

## 8. 当前建议

- 做图前先查已有 summary，优先复用，不要先重跑
- `stability` 的 item-level accuracy panel 直接复用 `results/stability_summary.json` 重画
- noisy judge 结果统一以 `results/noisy_summaries/` 为准
- bootstrap 跨 dataset 汇总图统一从 `results/bootstrap_summaries/` 重画
- 想看 overall stability：先看 `results/stability_accuracy.png`
- 想看单 dataset 哪些 item 更稳：看 `results/stability_accuracy_panels/`
- 想看 overall robustness：先看 `results/noisy_rank_shift_custom.png`
- 想看 judge-level noisy detection：再看 `results/noisy_judge_detection/`
