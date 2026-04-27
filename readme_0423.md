## 整体框架

当前目录只保留真实数据实验。

包含三类拟合方法：
1. **Proposed**：`S = gamma * mu^T + U V^T`，使用论文 `neurips_2026.tex` 中对应的初始化、 alternating update 和 re-anchor 思路。
2. **Zhou github**：参考 `ZhouD_github/main/main.py` 的 judge-specific scale weighted MLE baseline。
3. **Standard BTL**：不考虑 judge reliability 和 heterogeneity，固定 `gamma = 1`。

当前真实数据实验已经拆分为三套可独立运行的流程：
1. **稳定性实验（stability / test accuracy）**：对每个真实数据集做 question-level train/test split，在测试集上预测 `(judge k, item i, item j)` 哪个回答胜出。
2. **稳健性实验（robustness / noisy judge）**：在原始数据集基础上逐步增加随机 noisy judge，当前固定数量范围为 `1..10`，保存增强后的数据，并比较 ranking 是否发生明显变化。
3. **Bootstrap 排名稳定性实验（bootstrap）**：按 `question_id` block 做有放回 resampling；每次 bootstrap sample 都重新估计三个模型，汇总 item rank 分布，并检查 top-k 是否稳定。
当前 `Proposed` 方法在真实数据实验里统一使用固定 `tau = 30.0`。

### 2026-04-23 更新

今天在 `real_data_0423/` 主要改了两件事：
1. **稳定性实验的 train/test split 改成 question-level split**。
   - 不再按单条 processed record 随机切分。
   - 现在同一个 `question_id` 下的所有记录会一起进 train 或一起进 test，避免同题目同时出现在训练集和测试集。
   - `in_house` 里的 `question_id` 是 list，代码里已统一转成 tuple 再做分组。
2. **Bootstrap 改成 question-block bootstrap**。
   - 不再直接对 processed records 做 row-level resampling。
   - 现在先按 `question_id` 分组，再对 question blocks 做有放回采样。
   - 每个 bootstrap replicate 抽取的 block 数 = 原始 dataset 的 question 数，而不是原始 `usable_records` 数。
   - 因为不同 question block 大小不同，所以每个 bootstrap replicate 的最终 record 数可能和原始 `usable_records` 不完全相同；这是 block bootstrap 的正常现象。

这样修改的原因是：
- 虽然 `question_id` 不进入当前 HJA 模型的 likelihood，
- 但它属于真实数据的采样与评测过程，evaluation 不能把同一 question 下的相关记录当成完全独立的 IID row 来处理。

### 2026-04-21 更新

今天修改三件事：
1. **把 `real_data_0421/src/models.py` 的 Proposed 实现完全同步到 `simulation_0420/src/models.py`**。
   - 初始化、`negative_log_likelihood(...)`、`reanchor(...)`、`alternating_mle(...)`、`fit_rank0_model(...)`、`select_rank_by_bic(...)` 现在和 simulation 版本保持一致。
   - judge-side / item-side update 统一改成 `trust-constr` + 显式等式约束写法。
2. **去掉 `gamma` 非负 / simplex 限制**。
   - 现在只保留 `sum(gamma) = K` 这个识别约束。
   - 不再做 `np.maximum(gamma, 0)`、不再要求 `gamma >= 0`、不再做旧版 gamma clamping / renormalization。
3. **修正真实数据路径和稳定性实验落盘时机**。
   - `run_real_data.py` 里的四个数据文件路径现在都通过 `BASE_DIR` 锚定到 `real_data_0421/data/`，不依赖运行时 cwd。
   - `run_real_data_stability_experiment(...)` 现在每跑完一个 dataset，立刻更新一次：
     - `results/stability_summary.json`
     - `results/stability_accuracy.png`
   - 因此四个数据集不用全跑完，就能先看到已完成部分结果。
4. **扩展稳定性实验输出**。
   - `stability_summary.json` 现在会额外保存最终参数 `gamma`、`mu`、`U`、`V`。
   - 每个 real-data 数据集会额外生成一张 Proposed 的 `U @ V.T` heatmap，输出到 `results/proposed_uvt_heatmaps/`。

补充：今天排查过 `in_house` 很慢原因。主瓶颈不是读数据，而是 Proposed 在 `N=45, K=18` 时 `trust-constr` 子问题很重，judge-side 单次内层优化就可能耗时很久。

`U @ V.T` heatmap 的含义：
- 行是 judge，列是 item。
- 正值（暖色）表示该 judge 对该 item 有相对正向异质性偏移。
- 负值（冷色）表示相对负向异质性偏移。
- 如果某个 judge 对一类 item 特别偏爱，对应 cell 会明显点亮。

## 稳定性实验结果（2026-04-21，当前版本）

下面这组数值来自当前代码版本，即：
- `src/models.py` 已同步到 `simulation_0420` 版本
- `negative_log_likelihood(...)` 已改成向量化实现
- Proposed 真实数据稳定性实验使用 `max_steps=30`, `tau=30.0`
- `stability_summary.json` 现在会保存最终参数 `gamma / mu / U / V`
- 每个 real-data 数据集会额外输出 Proposed 的 `U @ V.T` heatmap

四个 real-data 数据集总表如下。

补充：下表是 **2026-04-21 当前代码版本的最终完整重跑结果**。

| Dataset | Items | Judges | Usable pairs | Dataset total time (s) | Proposed acc | Proposed iter | Proposed time (s) | Zhou acc | Zhou iter | Zhou time (s) | BTL acc | BTL iter | BTL time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chatbot_arena | 20 | 10 | 8713 | 54.94 | 0.7413 | 11 | 22.96 | 0.6862 | 65 | 29.22 | 0.6776 | 12 | 1.55 |
| mtbench | 6 | 20 | 8950 | 11.63 | 0.8218 | 8 | 6.67 | 0.7525 | 54 | 3.77 | 0.7525 | 8 | 0.03 |
| ultrafeedback | 17 | 20 | 8965 | 99.69 | 0.7602 | 10 | 41.54 | 0.6509 | 73 | 57.13 | 0.6570 | 13 | 0.39 |
| in_house | 45 | 18 | 34057 | 1026.75 | 0.6567 | 14 | 386.74 | 0.6481 | 86 | 626.44 | 0.6469 | 7 | 11.38 |

四个 real-data 数据集的 pair 和 judge 规模如下：

| Dataset | Usable pairs | Judges | Items |
| --- | ---: | ---: | ---: |
| chatbot_arena | 8713 | 10 | 20 |
| mtbench | 8950 | 20 | 6 |
| ultrafeedback | 8965 | 20 | 17 |
| in_house | 34057 | 18 | 45 |

Proposed 最终 `gamma` 中负值个数如下：

| Dataset | Negative gamma count | Total judges |
| --- | ---: | ---: |
| chatbot_arena | 0 | 10 |
| mtbench | 1 | 20 |
| ultrafeedback | 2 | 20 |
| in_house | 1 | 18 |

四个 real-data 数据集的 test accuracy 对比如下：

| Dataset | Proposed | Zhou github | Standard BTL |
| --- | ---: | ---: | ---: |
| chatbot_arena | 0.7413 | 0.6862 | 0.6776 |
| mtbench | 0.8218 | 0.7525 | 0.7525 |
| ultrafeedback | 0.7602 | 0.6509 | 0.6570 |
| in_house | 0.6567 | 0.6481 | 0.6469 |

结论：
- 四个数据集上，**Proposed** 都成功收敛，且 test accuracy 都是三种方法里最高。
- `in_house` 上 `delta_grad == 0.0` 仍会出现，但最终可以收敛；主问题是 **item-side `trust-constr` 子问题很慢**，不是死循环。
- `Standard BTL` 始终最快，但 accuracy 最低或并列最低。
- `Zhou github` 在 `in_house` 上也很慢，说明大数据集下瓶颈不只在 Proposed。

### 2026-04-18 更新

本次修改做了两件事：
1. **把稳定性实验和稳健性实验拆成两套入口**，避免默认一条命令同时触发两个流程。
2. **重新跑了稳定性实验的四个真实数据集**，并把结果汇总到本文档。

当前 `run_real_data.py` 的 CLI 入口支持：
- `--experiment stability`：只跑稳定性实验
- `--experiment robustness`：只跑 noisy-judge 稳健性实验
- `--experiment all`：顺序跑两类实验

---

## 运行方式

### 1. 只跑稳定性实验

默认就是 stability，因此下面两种写法等价：

```bash
python run_real_data.py
```

```bash
python run_real_data.py --experiment stability
```

如果要只跑某一个数据集：

```bash
python run_real_data.py --experiment stability --dataset mtbench
```

### 2. 只跑稳健性实验

```bash
python run_real_data.py --experiment robustness
```

如果要做 smoke test：

```bash
python run_real_data.py --experiment robustness --dataset mtbench --max-noisy-step 2
```

### 3. Bootstrap 排名稳定性实验

```bash
python run_real_data.py --experiment bootstrap --dataset mtbench --bootstrap-samples 20 --bootstrap-top-k 3
```

参数：
- `--bootstrap-samples`：bootstrap 次数；`--experiment bootstrap` 时必须 `>= 1`
- `--bootstrap-top-k`：检查 top-k 稳定性的 `k`
- `--bootstrap-seed`：bootstrap resampling 随机种子

如果想把 bootstrap 也并到总入口里：

```bash
python run_real_data.py --experiment all --bootstrap-samples 20 --bootstrap-top-k 3
```

---

## Bootstrap 排名稳定性实验（2026-04-21）

Bootstrap 流程定义：
- 先按 `question_id` 把 `dataset["processed"]` 分成 question blocks
- 对这些 question blocks 做有放回采样
- 每个 bootstrap sample 的 block 数 = 原始 dataset 的 unique `question_id` 数
- 每个 bootstrap sample 的最终 record 数不一定等于原始 `usable_records`
- 每次 sample 都重新聚合成 `n_ijk / y_ijk`
- 每次 sample 都重新拟合 `Proposed` / `Zhou github` / `Standard BTL`
- baseline ranking 使用同一数据集 **全量 processed records** 拟合一次，不做 train/test split

这里：
- `5/5` 里的 `5` 指 **bootstrap sample 次数**
- `baseline` 指 **原始全量 processed records 拟合一次得到的参考 ranking**
- 后面每次 bootstrap sample 的 ranking，都和这个 baseline 比较 top-k 稳定性

当前 `bootstrap_samples` 参数表示：
- **bootstrap replications 次数**
- 不是单个 replicate 里的 row sample size

当前 `bootstrap_summary.json` 中每个方法会保存：
- `baseline`
- `successful_samples` / `failed_samples`
- `bootstrap_rankings`
- `top_k_summaries`
- `rank_distribution`
- `top_k_stability`
- `rank_distribution_plot`
- `top_k_stability_curve`

其中：
- `bootstrap_rankings` 会保存每次成功 bootstrap sample 的原始 ranking（含 `mu`、`ranking`、`item_to_rank`），因此后续想改看 `top_3` / `top_5` / 更大 `top_k` 时，不必重新拟合
- `top_k_summaries` 当前会至少保存 `top_3` 和 `top_5` 两套派生结果；每套里会带 `rank_distribution`、`top_k_stability`、`rank_distribution_plot`、`top_k_curve_plot`、`top_k_stability_curve`
- `rank_distribution` 会给每个 item 记录 `mean_rank`、`median_rank`、`rank_p05`、`rank_p95`、`top_k_frequency`
- `top_k_stability` 会记录 baseline top-k、每个 item 进入 top-k 的频率、`mean_jaccard_vs_baseline`、`exact_match_rate_vs_baseline_top_k`
- 顶层 `rank_distribution` / `top_k_stability` / `rank_distribution_plot` / `top_k_stability_curve` 仍保留，作为本次 `--bootstrap-top-k` 主参数对应的主视图
- 单 dataset 运行时，除了 `results/bootstrap_summary.json`，还会额外归档到 `results/bootstrap_summaries/bootstrap_summary_<dataset>_samples_<n>.json`，避免后续运行覆盖旧 summary

### Bootstrap 可视化

当前会额外输出三类图，并且**文件名同时体现 `top_{k}` 和 `samples_{n}`**，避免不同运行结果互相覆盖：
- `results/bootstrap_topk_curves/*.png`
  - 每个 dataset 对应 1 张折线图
  - 横轴是 `top_k`
  - 纵轴是 `exact_match_rate_vs_baseline_top_k`
  - 3 条线分别对应 `Proposed` / `Zhou github` / `Standard BTL`
  - 例子：
    - `results/bootstrap_topk_curves/mtbench_top_3_samples_5_exact_match_curve.png`
    - `results/bootstrap_topk_curves/mtbench_top_5_samples_5_exact_match_curve.png`
- `results/bootstrap_rank_distribution/*.png`
  - 每个 `dataset × method × top_k × sample_count` 一张图
  - 横轴是 item，纵轴是 bootstrap rank（数值越小越靠前，图里会反转 y 轴）
  - 点是 `median_rank`
  - 误差条对应 `rank_p05 ~ rank_p95`
  - 之所以不用 `mean_rank` 当中心点，是因为分位数区间不一定围绕均值对称；之前这里会触发 matplotlib 的 `'yerr' must not contain negative values'` 报错
  - 例子：
    - `results/bootstrap_rank_distribution/mtbench_proposed_top_3_samples_5_rank_distribution.png`
    - `results/bootstrap_rank_distribution/mtbench_proposed_top_5_samples_5_rank_distribution.png`
- `results/bootstrap_topk_stability_top_<k>_samples_<n>.png`
  - 跨 dataset 汇总柱状图
  - 横轴是 dataset，纵轴是 `exact_match_rate_vs_baseline_top_k`
  - 每张图固定一个 `top_k`，3 组柱子对应 3 个方法
  - 例子：
    - `results/bootstrap_topk_stability_top_3_samples_50.png`
    - `results/bootstrap_topk_stability_top_5_samples_50.png`

补充：
- 单 dataset 的 curve 图保存在 `results/bootstrap_topk_curves/`
- 跨 dataset 的 summary bar 图保存在 `results/` 根目录
- 两类图会同时保留
- `top_3` 和 `top_5` 会在一次运行后同时生成，不需要分别重跑
- 如果是单 dataset 运行，curve 图一定会生成；跨 dataset 汇总图只会汇总当前 `bootstrap_summary.json` 里同 sample 数、已存在的 dataset

### 独立绘制 sample=50 汇总柱状图（2026-04-22）

为了单独重画不同 `bootstrap sample` 下 4 个 dataset × 3 个 model 的 summary bar 图，当前额外增加了独立脚本：
- `plot_bootstrap_summary_bars.py`

用途：
- 不重新拟合模型
- 直接读取 `results/bootstrap_summaries/` 里的归档 summary
- 默认读取 `samples_100`
- 同时输出 `top_3` 和 `top_5` 两张跨 dataset 柱状图
- 也可通过参数切换到别的 sample 数

脚本默认读取：
- `results/bootstrap_summaries/bootstrap_summary_chatbot_arena_samples_100.json`
- `results/bootstrap_summaries/bootstrap_summary_mtbench_samples_100.json`
- `results/bootstrap_summaries/bootstrap_summary_ultrafeedback_samples_100.json`
- 如果存在，也会读取 `results/bootstrap_summaries/bootstrap_summary_in_house_samples_100.json`

运行方式：

```bash
python plot_bootstrap_summary_bars.py
python plot_bootstrap_summary_bars.py --bootstrap-samples 100
```

输出文件：
- `results/bootstrap_topk_stability_top_3_samples_100_custom.png`
- `results/bootstrap_topk_stability_top_5_samples_100_custom.png`

说明：
- 该脚本只负责重画跨 dataset 汇总柱状图
- 当前已把配色、`figsize=(12, 5)`、网格样式、柱宽 `width=0.25` 对齐到原始 summary bar 图风格
- 为避免覆盖原图，输出文件名加了 `_custom`
- 若某个 dataset 缺少对应 `samples_<n>` 归档 summary，会自动跳过

### noisy judge rank shift 独立脚本（2026-04-22）

新增脚本：
- `plot_noisy_judge_rank_shift.py`

用途：
- 单独重画 noisy judge rank shift 图
- 可直接读取已有 `results/noisy_summaries/noisy_judge_summary_<dataset>.json`
- 也可重新跑 noisy judge 实验后画图

当前 summary 设计：
- 只保留按 dataset 保存的 summary 文件
- 重画总图时自动合并 `results/noisy_summaries/` 下已有 dataset summary

图布局：
- `2x2` 子图
- 每格 1 个 dataset
- 每格 3 条线，分别对应 `Proposed` / `Zhou github` / `Standard BTL`

常用命令：

```bash
python plot_noisy_judge_rank_shift.py --use-existing-summary
python plot_noisy_judge_rank_shift.py --dataset mtbench --use-existing-summary
python plot_noisy_judge_rank_shift.py --dataset mtbench --max-noisy-step 5
python plot_noisy_judge_rank_shift.py --dataset mtbench --max-noisy-step 5 --use-existing-summary
```

输出文件：
- `results/noisy_rank_shift_custom.png`
- 或按参数输出 `results/noisy_rank_shift_<dataset>_through_<step>_custom.png`

说明：
- 画图函数仍复用 `run_real_data.py` 中 `plot_noisy_rank_shift(...)`
- `--use-existing-summary` 模式不重跑，只读已有 summary
- 不加 `--use-existing-summary` 时，会执行和 `run_real_data.py --experiment robustness` 一致的流程
- 输出文件名加 `_custom`，避免覆盖原始 `results/noisy_rank_shift.png`
- 已有 summary 时，优先重画，不要重跑

生成单 dataset summary：

```bash
python run_real_data.py --experiment robustness --dataset mtbench
```

会写入：
- `results/noisy_summaries/noisy_judge_summary_mtbench.json`

### noisy judge 检测分离：作为 robustness 的补充衡量（2026-04-22）

**一句话结论：若把 gamma 分离 noisy judge 作为补充评估，现有结果支持 `zhou_github > proposed >> standard_btl`。**

除了看 rank shift，还可以补充看：**方法是否能把 noisy judges 从原 judges 中分离出来**。

这里把 judge-level `gamma` 当作检测信号。
- 若 noisy judge 的 `gamma` 长期落在异常端，说明该方法有 judge-level noisy detection 能力
- 若 noisy judge 与原 judge 的 `gamma` 分布明显分开，说明分离更强
- 若按 `gamma` 排序后能优先抓出 noisy judge，说明检测更准确

新增脚本：
- `plot_noisy_judge_detection.py`

用途：
- 不重跑实验，直接读取 `results/noisy_summaries/` 下已有 summary
- 检查三个方法是否把 noisy judges 从原 judges 里分离出来
- 把“检测分离能力”作为 noisy judge robustness 的补充衡量
- 输出三类图和一份指标 json

命令：

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

#### 图 1：`noisy_judge_gamma_scatter_<dataset>.png`
功能：看最后一个可用 noisy step 时，哪些 judges 被排到 gamma 异常端。
- x 轴：judge 按 gamma 排序后的名次（从低到高）
- y 轴：judge `gamma`
- 颜色：红点 = noisy judge；蓝点 = original judge

判读：
- 红点若集中在左侧低 gamma 区域，说明方法能把 noisy judge 排到异常端
- 红蓝两类若明显分开，说明检测分离效果强

#### 图 2：`noisy_judge_gamma_boxplot_<dataset>.png`
功能：看不同 noisy step 下，noisy judge 与 original judge 的 gamma 分布是否分离。
- x 轴：`# noisy judges`（step）
- y 轴：judge `gamma`
- 每个 step 两组箱线图：蓝色 = original judge；红色 = noisy judge

判读：
- 红色箱线图若整体明显低于蓝色，说明 gamma 有分离能力
- 两组 overlap 越小，说明 noisy judge 与原 judge 越容易区分

#### 图 3：`noisy_judge_hit_rate_<dataset>.png`
功能：看按 gamma 抓 noisy judge 的准确率。
- x 轴：`# noisy judges`（step）
- y 轴：hit rate
- 定义：每个 step 里，取 gamma 最低的前 `k` 个 judge，其中 `k = 当前 noisy judge 个数`；hit rate = 这 `k` 个 judge 里真 noisy judge 的比例

判读：
- `hit_rate = 1.0` 表示该 step 下按 gamma 能把 noisy judge 全部抓出来
- 越接近 1，说明检测越准；接近 0，说明几乎抓不出来

#### 指标文件：`noisy_judge_detection_metrics_<dataset>.json`
包含：
- 每个方法、每个 step 的 `hit_rate`
- `noisy_median_gamma`
- `original_median_gamma`
- `median_gap_noisy_minus_original`

其中：
- `median_gap_noisy_minus_original < 0`：说明 noisy judge 的 gamma 中位数低于 original judge
- 数值越负，说明分离越明显

#### 当前结论（基于 2026-04-22 已生成结果）
把“检测分离 noisy judge”作为补充衡量时，目前结论是：
- **总体上 `zhou_github` 最强**：在多个 dataset 上，`hit_rate` 更高，gamma 分离也最明显
- **`proposed` 次之**：有检测能力，但整体弱于 `zhou_github`
- **`standard_btl` 最弱**：当前实现里 gamma 基本不提供 judge-level noisy detection 信号

按 dataset 看：
- `chatbot_arena`：`proposed` 也很强，但 `zhou_github` 的 gamma 分离幅度仍更大
- `mtbench`：`zhou_github` 明显强于 `proposed`
- `ultrafeedback`：`zhou_github` 明显强于 `proposed`
- `in_house`：两者都不算强，当前指标下没有明显赢家

因此当前更准确说法是：
- **若把 gamma 分离 noisy judge 作为补充评估，现有结果支持 `zhou_github > proposed >> standard_btl`**
- 但这只是在比较“judge-level noisy detection / separation”能力，**不等于**最终 item ranking 准确率或整体 robustness 的唯一结论

判读建议：
- 先看 `gamma_scatter`：红点是否落到低 gamma 一侧
- 再看 `gamma_boxplot`：红蓝两组分布是否分开
- 最后看 `hit_rate`：按 gamma 排序能否稳定抓出 noisy judge
- 三张图一起看，比只看单一指标更稳妥

```bash
# 产物
results/bootstrap_topk_stability_top_3_samples_100_custom.png
results/bootstrap_topk_stability_top_5_samples_100_custom.png
results/noisy_rank_shift_custom.png
```

```bash
# 脚本
plot_bootstrap_summary_bars.py
plot_noisy_judge_rank_shift.py
```

```bash
# 默认值
plot_bootstrap_summary_bars.py: DEFAULT_BOOTSTRAP_SAMPLES = 100
```

```bash
# 保持原逻辑来源
plot_noisy_judge_rank_shift.py -> run_real_data.plot_noisy_rank_shift(...)
plot_noisy_judge_rank_shift.py -> run_real_data.run_real_data_noisy_judge_experiment(...)
```

```bash
# 兼容直接重画与重新跑实验
--use-existing-summary
--dataset
--max-noisy-step
--random-seed
```

```bash
# 结果目录
results/
results/bootstrap_summaries/
```

```bash
# 目的
bootstrap 图单独重画
noisy judge rank shift 单独运行
```

```bash
# 完成时间
2026-04-22
```

```bash
# 备注
此处记录当前实际代码状态，后续若默认 sample 数再改，readme 也要同步更新。
```

```bash
# 额外提醒
若要看原始 noisy 图，文件仍是 results/noisy_rank_shift.png
```

```bash
# 额外提醒
若要看原始 bootstrap 汇总图，非 custom 文件仍由 run_real_data.py 生成
```

```bash
# 额外提醒
custom 图只负责重画，不改 summary 内容
```

```bash
# 额外提醒
run_real_data.py CLI 入口保持不变
```

```bash
# 额外提醒
新增脚本只做拆分，不改核心实验逻辑
```

```bash
# 额外提醒
plot_bootstrap_summary_bars.py 仍按现有 summary 结构读取 top_k_summaries
```

```bash
# 额外提醒
若 summary 缺键，脚本会在读取阶段报错；这类情况需先检查归档 summary 是否完整
```

```bash
# 额外提醒
当前 noisy 独立脚本默认输出总图；按 dataset/step 参数会输出更细粒度 custom 图
```

```bash
# 额外提醒
若想只改文件名风格或配色，可后续继续在独立脚本里调，不影响主实验入口
```

```bash
# 额外提醒
本节覆盖 2026-04-22 新增修改
```

```bash
# 额外提醒
旧 sample=50 说明不再作为默认值
```

```bash
# 额外提醒
但 `--bootstrap-samples 50` 仍完全可用
```

```bash
# 额外提醒
旧 20/50 custom 图文件仍保留在 results/
```

```bash
# 额外提醒
noisy judge summary 现在只写到 results/noisy_summaries/noisy_judge_summary_<dataset>.json
```
### mtbench smoke test（2026-04-21）

运行命令：

```bash
python run_real_data.py --experiment bootstrap --dataset mtbench --bootstrap-samples 5 --bootstrap-top-k 3 --bootstrap-seed 42
```

本次 smoke test 已确认：
- `bootstrap_summary.json` 现在会保存每次成功 sample 的原始 ranking
- 新增/确认生成的图包括：
  - `results/bootstrap_topk_curves/mtbench_top_3_samples_5_exact_match_curve.png`
  - `results/bootstrap_topk_curves/mtbench_top_5_samples_5_exact_match_curve.png`
  - `results/bootstrap_rank_distribution/mtbench_proposed_top_3_samples_5_rank_distribution.png`
  - `results/bootstrap_rank_distribution/mtbench_proposed_top_5_samples_5_rank_distribution.png`

### 三个 real-data dataset 的 50-sample bootstrap（2026-04-22，当前结果）

运行命令：

```bash
python run_real_data.py --experiment bootstrap --dataset chatbot_arena --bootstrap-samples 50 --bootstrap-top-k 3 --bootstrap-seed 42
python run_real_data.py --experiment bootstrap --dataset mtbench --bootstrap-samples 50 --bootstrap-top-k 3 --bootstrap-seed 42
python run_real_data.py --experiment bootstrap --dataset ultrafeedback --bootstrap-samples 50 --bootstrap-top-k 3 --bootstrap-seed 42
```

生成/确认的归档 summary：
- `results/bootstrap_summaries/bootstrap_summary_chatbot_arena_samples_50.json`
- `results/bootstrap_summaries/bootstrap_summary_mtbench_samples_50.json`
- `results/bootstrap_summaries/bootstrap_summary_ultrafeedback_samples_50.json`

对应汇总柱状图可由：

```bash
python plot_bootstrap_summary_bars.py --bootstrap-samples 50
```

重画得到：
- `results/bootstrap_topk_stability_top_3_samples_50_custom.png`
- `results/bootstrap_topk_stability_top_5_samples_50_custom.png`

### mtbench robustness smoke test（2026-04-22）

运行命令：

```bash
python run_real_data.py --experiment robustness --dataset mtbench --max-noisy-step 2
```

用于确认：
- `noisy_datasets/mtbench_plus_1_noisy_judges.json`
- `noisy_datasets/mtbench_plus_2_noisy_judges.json`
- `results/noisy_rank_shift.png`
- `results/noisy_summaries/noisy_judge_summary_mtbench.json`

其中最后一个文件现在是 noisy judge 的标准 summary 存储位置。

如果后续只想重画 mtbench 的图：

```bash
python plot_noisy_judge_rank_shift.py --dataset mtbench --use-existing-summary
```

如果想重画所有已完成 dataset 的总图：

```bash
python plot_noisy_judge_rank_shift.py --use-existing-summary
```

这要求 `results/noisy_summaries/` 下已有对应 dataset 的 summary 文件。

### 结果目录状态（2026-04-22）

当前与绘图/summary 直接相关的目录：
- `results/bootstrap_summaries/`
- `results/noisy_summaries/`
- `results/bootstrap_topk_curves/`
- `results/bootstrap_rank_distribution/`
- `results/noisy_datasets/`

其中 noisy judge 部分已经完全切到 per-dataset summary 设计。

### 后续维护提醒

如果想保持可重画、可复用、避免重跑：
- bootstrap：保留 `results/bootstrap_summaries/`
- noisy judge：保留 `results/noisy_summaries/`
- 重画前优先检查现有 summary 是否已齐，不要先重跑

### 当前默认行为

- 单 dataset robustness 运行：只更新对应 dataset 的 summary 文件
- 总图重画：扫描 `results/noisy_summaries/` 中已有 dataset
- 缺哪个 dataset，就少哪个格子

### noisy 图结构

- 2 行 2 列
- 每格 1 个 dataset
- 每格 3 条 model 曲线
- 横轴 `# noisy judges`
- 纵轴 `Ranking Spearman vs base`

### mtbench smoke test（2026-04-21）

运行命令：

```bash
python run_real_data.py --experiment bootstrap --dataset mtbench --bootstrap-samples 5 --bootstrap-top-k 3 --bootstrap-seed 42
```

本次 smoke test 已确认：

直接用已有 summary 重画：

```bash
python plot_noisy_judge_rank_shift.py --use-existing-summary
```

重新跑实验并画图：

```bash
python plot_noisy_judge_rank_shift.py
```

可选参数：

```bash
python plot_noisy_judge_rank_shift.py --dataset mtbench --max-noisy-step 5
python plot_noisy_judge_rank_shift.py --dataset mtbench --max-noisy-step 5 --use-existing-summary
```

输出文件：
- `results/noisy_rank_shift_custom.png`
- 或按参数输出 `results/noisy_rank_shift_<dataset>_through_<step>_custom.png`

说明：
- 画图函数仍复用 `run_real_data.py` 中 `plot_noisy_rank_shift(...)`
- `--use-existing-summary` 模式下不重新拟合模型，只读取已有 summary
- 不加 `--use-existing-summary` 时，会执行和 `run_real_data.py --experiment robustness` 一致的流程
- 输出文件名加 `_custom`，避免覆盖原始 `results/noisy_rank_shift.png`

当前已生成：
- `results/noisy_rank_shift_custom.png`
- `results/bootstrap_topk_stability_top_3_samples_100_custom.png`
- `results/bootstrap_topk_stability_top_5_samples_100_custom.png`

以下 3 个 100-sample 归档 summary 当前已存在：
- `results/bootstrap_summaries/bootstrap_summary_chatbot_arena_samples_100.json`
- `results/bootstrap_summaries/bootstrap_summary_mtbench_samples_100.json`
- `results/bootstrap_summaries/bootstrap_summary_ultrafeedback_samples_100.json`

`in_house` 的 `samples_100` 归档目前未看到；如后续生成，`plot_bootstrap_summary_bars.py --bootstrap-samples 100` 会自动纳入。

以下命令已验证可用：

```bash
python plot_bootstrap_summary_bars.py --bootstrap-samples 100
python plot_noisy_judge_rank_shift.py --use-existing-summary
```

请注意：`plot_noisy_judge_rank_shift.py` 当前使用 `typing.Optional[...]` 类型注解，以兼容较老 Python 版本。

```bash
# 产物
results/bootstrap_topk_stability_top_3_samples_100_custom.png
results/bootstrap_topk_stability_top_5_samples_100_custom.png
results/noisy_rank_shift_custom.png
```

```bash
# 脚本
plot_bootstrap_summary_bars.py
plot_noisy_judge_rank_shift.py
```

```bash
# 默认值
plot_bootstrap_summary_bars.py: DEFAULT_BOOTSTRAP_SAMPLES = 100
```

```bash
# 保持原逻辑来源
plot_noisy_judge_rank_shift.py -> run_real_data.plot_noisy_rank_shift(...)
plot_noisy_judge_rank_shift.py -> run_real_data.run_real_data_noisy_judge_experiment(...)
```

```bash
# 兼容直接重画与重新跑实验
--use-existing-summary
--dataset
--max-noisy-step
--random-seed
```

```bash
# 结果目录
results/
results/bootstrap_summaries/
```

```bash
# 目的
bootstrap 图单独重画
noisy judge rank shift 单独运行
```

```bash
# 完成时间
2026-04-22
```

```bash
# 备注
此处记录当前实际代码状态，后续若默认 sample 数再改，readme 也要同步更新。
```

```bash
# 额外提醒
若要看原始 noisy 图，文件仍是 results/noisy_rank_shift.png
```

```bash
# 额外提醒
若要看原始 bootstrap 汇总图，非 custom 文件仍由 run_real_data.py 生成
```

```bash
# 额外提醒
custom 图只负责重画，不改 summary 内容
```

```bash
# 额外提醒
run_real_data.py CLI 入口保持不变
```

```bash
# 额外提醒
新增脚本只做拆分，不改核心实验逻辑
```

```bash
# 额外提醒
plot_bootstrap_summary_bars.py 仍按现有 summary 结构读取 top_k_summaries
```

```bash
# 额外提醒
若 summary 缺键，脚本会在读取阶段报错；这类情况需先检查归档 summary 是否完整
```

```bash
# 额外提醒
当前 noisy 独立脚本默认输出总图；按 dataset/step 参数会输出更细粒度 custom 图
```

```bash
# 额外提醒
若想只改文件名风格或配色，可后续继续在独立脚本里调，不影响主实验入口
```

```bash
# 额外提醒
本节覆盖 2026-04-22 新增修改
```

```bash
# 额外提醒
旧 sample=50 说明不再作为默认值
```

```bash
# 额外提醒
但 `--bootstrap-samples 50` 仍完全可用
```

```bash
# 额外提醒
旧 20/50 custom 图文件仍保留在 results/
```

```bash
# 额外提醒
noisy judge summary 现在只写到 results/noisy_summaries/noisy_judge_summary_<dataset>.json
```
### mtbench smoke test（2026-04-21）

运行命令：

```bash
python run_real_data.py --experiment bootstrap --dataset mtbench --bootstrap-samples 5 --bootstrap-top-k 3 --bootstrap-seed 42
```

本次 smoke test 已确认：
- `bootstrap_summary.json` 现在会保存每次成功 sample 的原始 ranking
- 一次运行后会同时产出 `top_3` 和 `top_5` 两套派生结果
- 输出文件名里会同时带 `top_{k}` 和 `samples_{n}`
- `mtbench` 当前已生成：
  - `results/bootstrap_topk_curves/mtbench_top_3_samples_5_exact_match_curve.png`
  - `results/bootstrap_topk_curves/mtbench_top_5_samples_5_exact_match_curve.png`
  - `results/bootstrap_rank_distribution/mtbench_proposed_top_3_samples_5_rank_distribution.png`
  - `results/bootstrap_rank_distribution/mtbench_proposed_top_5_samples_5_rank_distribution.png`
  - 以及另外两个 baseline 方法对应的同名风格图

### 三个 real-data dataset 的 50-sample bootstrap（2026-04-22，当前结果）

运行命令：

```bash
python run_real_data.py --experiment bootstrap --dataset chatbot_arena --bootstrap-samples 50 --bootstrap-top-k 3 --bootstrap-seed 42
python run_real_data.py --experiment bootstrap --dataset mtbench --bootstrap-samples 50 --bootstrap-top-k 3 --bootstrap-seed 42
python run_real_data.py --experiment bootstrap --dataset ultrafeedback --bootstrap-samples 50 --bootstrap-top-k 3 --bootstrap-seed 42
```

当前 3 个 dataset 的 50-sample 归档 summary 在：
- `results/bootstrap_summaries/bootstrap_summary_chatbot_arena_samples_50.json`
- `results/bootstrap_summaries/bootstrap_summary_mtbench_samples_50.json`
- `results/bootstrap_summaries/bootstrap_summary_ultrafeedback_samples_50.json`

`top_3` 指标如下：

| Dataset | Method | Success | Exact match rate | Mean Jaccard |
| --- | --- | ---: | ---: | ---: |
| chatbot_arena | Proposed | 49/50 | 0.9388 | 0.9694 |
| chatbot_arena | Zhou github | 50/50 | 1.0000 | 1.0000 |
| chatbot_arena | Standard BTL | 50/50 | 1.0000 | 1.0000 |
| mtbench | Proposed | 49/50 | 1.0000 | 1.0000 |
| mtbench | Zhou github | 50/50 | 1.0000 | 1.0000 |
| mtbench | Standard BTL | 50/50 | 1.0000 | 1.0000 |
| ultrafeedback | Proposed | 50/50 | 1.0000 | 1.0000 |
| ultrafeedback | Zhou github | 50/50 | 0.9800 | 0.9900 |
| ultrafeedback | Standard BTL | 50/50 | 0.9400 | 0.9700 |

`top_5` 指标如下：

| Dataset | Method | Success | Exact match rate | Mean Jaccard |
| --- | --- | ---: | ---: | ---: |
| chatbot_arena | Proposed | 49/50 | 0.7347 | 0.9116 |
| chatbot_arena | Zhou github | 50/50 | 0.4400 | 0.8133 |
| chatbot_arena | Standard BTL | 50/50 | 0.8600 | 0.9533 |
| mtbench | Proposed | 49/50 | 1.0000 | 1.0000 |
| mtbench | Zhou github | 50/50 | 1.0000 | 1.0000 |
| mtbench | Standard BTL | 50/50 | 1.0000 | 1.0000 |
| ultrafeedback | Proposed | 50/50 | 0.9800 | 0.9933 |
| ultrafeedback | Zhou github | 50/50 | 1.0000 | 1.0000 |
| ultrafeedback | Standard BTL | 50/50 | 0.3400 | 0.7705 |

结论：
- `mtbench` 上，三种方法在 `top_3` 和 `top_5` 下都完全稳定。
- `ultrafeedback` 上，`Proposed` 的 `top_3` 完全稳定；`top_5` 下 `Proposed` 和 `Zhou github` 都非常稳定，但 `Standard BTL` 明显更差。
- `chatbot_arena` 上，`top_3` 下 `Zhou github` 和 `Standard BTL` 比 `Proposed` 更稳；`top_5` 下 `Standard BTL` 最稳，`Proposed` 次之，`Zhou github` 最弱。
- `Proposed` 在 `chatbot_arena` 和 `mtbench` 的 50-sample 运行里各有 1 次 sample 未计入成功样本，因此 success 分别是 `49/50`。

### 四个 real-data dataset 全量 bootstrap（2026-04-21，历史结果）

下面这组数值来自**旧版 bootstrap summary / 旧版可视化阶段**，当时主汇总还是 `top_k=3`，尚未保存每次 sample 的原始 ranking，也还没有现在这版带 `top_{k}`、`samples_{n}` 的新文件命名。

运行命令：

```bash
python run_real_data.py --experiment bootstrap --bootstrap-samples 5 --bootstrap-top-k 3 --bootstrap-seed 42
```

四个数据集总耗时如下：

| Dataset | Bootstrap samples | Top-k | Total time (s) |
| --- | ---: | ---: | ---: |
| chatbot_arena | 5 | 3 | 217.67 |
| mtbench | 5 | 3 | 84.18 |
| ultrafeedback | 5 | 3 | 820.82 |
| in_house | 5 | 3 | 12398.40 |

三种方法的 bootstrap 成功率与 top-k 稳定性如下：

| Dataset | Method | Success | Mean Jaccard vs baseline | Exact match rate |
| --- | --- | ---: | ---: | ---: |
| chatbot_arena | Proposed | 5/5 | 1.00 | 1.00 |
| chatbot_arena | Zhou github | 5/5 | 1.00 | 1.00 |
| chatbot_arena | Standard BTL | 5/5 | 1.00 | 1.00 |
| mtbench | Proposed | 5/5 | 1.00 | 1.00 |
| mtbench | Zhou github | 5/5 | 1.00 | 1.00 |
| mtbench | Standard BTL | 5/5 | 1.00 | 1.00 |
| ultrafeedback | Proposed | 5/5 | 1.00 | 1.00 |
| ultrafeedback | Zhou github | 5/5 | 1.00 | 1.00 |
| ultrafeedback | Standard BTL | 5/5 | 1.00 | 1.00 |
| in_house | Proposed | 4/5 | 0.875 | 0.75 |
| in_house | Zhou github | 5/5 | 0.90 | 0.80 |
| in_house | Standard BTL | 5/5 | 0.60 | 0.20 |

结论：
- `chatbot_arena`、`mtbench`、`ultrafeedback` 上，三种方法的 baseline top-3 在这 5 次 bootstrap 中都完全稳定。
- `in_house` 明显更难，三种方法的 top-3 都会波动，其中 `Proposed` 有 1 次 sample 没有在 `max_steps` 内收敛。
- `in_house` 上 `Proposed` 的 top-k 稳定性仍然最好，但成本也最高；`Standard BTL` 最快，但 top-3 稳定性最差。

补充：
- `chatbot_arena` 上，`Proposed` baseline top-3 是 `claude-instant-v1`, `claude-v1`, `gpt-4`，5 次 bootstrap 全部一致。
- `mtbench` 上，`Proposed` 的 6 个 item rank 完全不变；`Zhou github` 和 `Standard BTL` 的 top-3 集合稳定，但 `gpt-4` 和 `claude-v1` 的前两名次序会互换。
- `ultrafeedback` 上，`Proposed` baseline top-3 是 `gpt-3.5-turbo`, `gpt-4`, `bard`，5 次 bootstrap 全部一致。
- `in_house` 上，`Proposed` baseline top-3 是 `Qwen/Qwen3-Next-80B-A3B-Instruct`, `openai/gpt-oss-120b`, `moonshotai/Kimi-K2-Instruct-0905`；4 次成功 bootstrap 中 top-3 不是每次都和 baseline 完全一致。
- 旧版结果里 `results/bootstrap_rank_distribution/` 下当时共生成 12 张 `dataset × method` 图。

---

## 稳定性实验结果（2026-04-18，历史记录）

这部分只保留**旧版 Proposed 实现**的历史结论，供和 `2026-04-21` 当前版本做粗略对照，不再重复展开完整结果表。

历史结论：
- `chatbot_arena`、`mtbench` 上，旧版 Proposed 可以收敛。
- `ultrafeedback`、`in_house` 上，旧版 Proposed 当时未能在 `max_steps` 内收敛。
- `2026-04-21` 当前版本已同步 `simulation_0420` 实现、去掉旧版 `gamma` 非负限制，并完成向量化；四个数据集的最新结果以上一节总表为准。

如果要看最新全量结果，直接读取：
- `results/stability_summary.json`
- `results/stability_accuracy.png`

如果只想快速检查代码和输出链路，建议先用：

```bash
python run_real_data.py --experiment stability --dataset mtbench
```

---

## 稳健性实验说明

稳健性实验会在原始数据集基础上逐步增加随机 noisy judge，并比较 ranking shift。

### 输出指标

当前会输出：
- Spearman correlation
- pairwise agreement
- top-1 是否变化
- 每个 item 的 rank shift

若 `Spearman < 0.95` 或 top-1 发生变化，则记为 `significant_change = true`。

### 关于 Proposed 的慢收敛问题

之前排查确认：
- noisy-judge 阶段真正的主瓶颈不是读数据，而是 `Proposed` 在加入 noisy judge 后明显更难收敛。
- 这里的 `failed` / `failed to converge` **不是整个实验失败**，而是指某一个方法在某一个 step 上没有在 `max_steps` 内收敛；其他方法和整个实验流程仍可继续。
- 本质上都是 judge-side 异质性 / scale imbalance 变强后，`alternating_mle(...)` 里的约束优化更难做，因此 Proposed 更容易慢收敛或在 `max_steps` 内不收敛。

当前 robustness 流程已经包含保护逻辑：
1. 增加 dataset / step / method 级日志和计时，便于判断卡在哪个阶段。
2. noisy-judge 阶段从磁盘回读改成内存继续构造，减少每个 step 的重复 IO。
3. 不再为每个 noisy step 单独画图，只保留最终汇总图，减少额外开销。
4. 一旦 `Proposed` 在某个 noisy step 上 non-convergence，后续 noisy step 直接跳过它，避免整晚卡死。

当前保护逻辑的含义：
- 第一次 noisy-step non-convergence 会被如实记录在对应 dataset 的 `noisy_summaries/noisy_judge_summary_<dataset>.json` 中。
- 后续 step 中，`Proposed` 会被标记为 `skipped after earlier noisy-step non-convergence`。
- 这表示该方法在后续 step 没有再次尝试拟合，不是说其他 baseline 或整个实验失败。

### 为什么 `Zhou github` 的结果看起来这么好

核心原因大概有 4 个：

1. 当前加进去的 `noisy judge` 很“纯随机”。
   - 现在的 noisy judge 是对每条记录独立做 `50/50` 的 `a/b` 随机选择。
   - 不带系统偏差，不偏爱某些 model，也不跟题型、judge、item 结构绑定。
   - 这类噪声对大样本 MLE 来说，常常更像均值为 0 的白噪声；会拉低信号，但不一定会强烈扭曲整体排序。

2. `Zhou github` baseline 本来就有 judge-specific scale。
   - 它不是纯 `Standard BTL`，而是允许不同 judge 有不同 scale / reliability。
   - 所以当加入很多随机 noisy judge 时，这类模型有机会把这些 judge 吸收到“低信息量 / 低可信度 / 接近均匀噪声”的那一侧参数里。
   - 一句话：`Zhou github` 比标准 BTL 更有能力“吃掉”坏 judge。

3. 当前 robustness 指标主要看 ranking，不直接看 likelihood 或参数误差。
   - 现在最终比较的是 `Spearman`、`pairwise agreement`、`top-1` 是否变化、以及每个 item 的 `rank shift`。
   - 这些指标更偏排序层面。
   - 所以即使 noisy judge 让参数数值变了不少，只要 item 相对顺序没怎么变、top few 没翻，看起来就会“结果很好”。
   - 换句话说：这里夸的是 rank 稳，不是说模型一定把噪声学得特别准。

4. 原始真实数据的信号本来可能就很强。
   - 当前实验里 noisy judge 不是替换原 judge，而是在保留原始 judge 的基础上继续叠加。
   - 如果原始数据里样本很多、强模型和弱模型差距明显、排名 margin 较大，那么随机噪声即使很多，也未必足够把排序掀翻。
   - 这时 `Zhou github` 只要别被带偏太狠，结果就会很好。

为什么它有时甚至看起来比 `Proposed` 还稳，也不奇怪：
- `Proposed` 更灵活，参数更多，能够表达更强的 judge/item heterogeneity。
- 但灵活也有代价：加入 noisy judge 后更难优化，更容易慢收敛，甚至 non-convergence。
- 所以在这种“加随机 judge、看 ranking shift”的实验里，可能出现 `Zhou github` 更稳的现象；这不一定说明它整体更强，也可能只是因为它更简单、更硬、更容易把随机噪声平均掉。

当前如果只看这版 noisy judge 设计，更可能的解释是：
- `Zhou github` 结果好，主要不是因为它特别“懂噪声”，而是因为当前噪声太无结构；再加上它有 judge-specific scale，因此足够把这类随机噪声平均或钝化掉。

如果后面要进一步验证是不是这个原因，最有用的方向有 3 个：
1. 不只看 rank，也看每步参数；例如比较 `Zhou github` 每步 judge-scale 分布，看 noisy judges 是否被压到低权重区。
2. 把 noisy judge 改成“有结构噪声”；例如固定偏爱某些 model、固定偏向 `model_a`、对 top models 反向投票、或只在某类题上更容易出错。
3. 增加更敏感的指标；例如除了 rank shift，还看 `mu` 偏移大小、top-k margin 变化、或 baseline vs noisy 的 score correlation。

---

## 输出文件

运行后会在 `results/` 下生成：

### 稳定性实验输出
- `stability_summary.json`（包含 `gamma`、`mu`、`U`、`V` 等最终参数）
- `stability_accuracy.png`
- `proposed_uvt_heatmaps/*.png`（每个 real-data 数据集一张 Proposed `U @ V.T` heatmap）

### Robustness / noisy judge 输出
- `noisy_summaries/noisy_judge_summary_<dataset>.json`（按 dataset 单独保存 summary，避免不同 dataset 互相覆盖）
- `noisy_datasets/<dataset>_plus_<step>_noisy_judges.json`（每个 step 的增强后原始 records）
- `noisy_rank_shift.png`（当前运行返回的 dataset 集合对应的汇总 Spearman 曲线图）
- `noisy_rank_shift_custom.png`（独立脚本按已有 per-dataset summary 重画的 custom 图）

总图重画来源：
- `plot_noisy_judge_rank_shift.py --use-existing-summary` 会扫描 `results/noisy_summaries/` 下已有 dataset summary
- 缺哪个 dataset summary，总图就少哪个格子

当前标准做法：
- 运行单 dataset robustness 时，只写对应 dataset 的 summary 文件
- 之后需要汇总总图时，再从 `noisy_summaries/` 合并重画

这样不会再出现单 dataset 运行覆盖总 summary 的问题。

### Robustness / noisy judge 输出（补充说明）
- 单 dataset summary 文件名固定为 `results/noisy_summaries/noisy_judge_summary_<dataset>.json`
- 例如：`results/noisy_summaries/noisy_judge_summary_mtbench.json`
- 独立脚本支持只画一个 dataset，也支持合并所有已存在 dataset summary 画 2x2 总图
- 总图每个子图 1 个 dataset，子图内 3 条线分别对应三个方法

保留这些 summary 文件，后续就能直接重画，避免重跑。

### Robustness / noisy judge 输出
- `noisy_summaries/noisy_judge_summary_<dataset>.json`（按 dataset 单独保存 summary，避免不同 dataset 互相覆盖）
- `noisy_datasets/<dataset>_plus_<step>_noisy_judges.json`（每个 step 的增强后原始 records）
- `noisy_rank_shift.png`（当前运行返回的 dataset 集合对应的汇总 Spearman 曲线图）
- `noisy_rank_shift_custom.png`（独立脚本按已有 per-dataset summary 重画的 custom 图）

其中 noisy judge summary 会保存 baseline / step 级 `judge_names`、`item_names`、各方法参数、rank shift 指标，以及增强数据文件路径，便于后续直接分析 `gamma` 与 noisy judges 的对应关系。

其中 summary 的标准持久化位置现在只有 `results/noisy_summaries/`。

### 运维建议
- 不要把 `noisy_summaries/` 当缓存删掉
- 这就是后续重画总图的正式输入
- 若想省时间，优先检查这里是否已有可复用 summary

### 结果复用顺序
1. 先看 `results/noisy_summaries/` 是否已有对应 dataset summary
2. 有则直接 `--use-existing-summary` 重画
3. 没有才重跑对应 dataset

### 图结构提醒
- 总图不是“每个方法一个子图”
- 总图现在是“每个 dataset 一个子图”
- 每个 dataset 子图里才是三条方法线

### 变更原因
- 之前 noisy summary 容易被单 dataset 运行覆盖
- 覆盖后总图会缺线/缺格子
- 改成 per-dataset summary 后，这个问题消失

### 实际代码状态
- `run_real_data.py`：只写 per-dataset noisy summary
- `plot_noisy_judge_rank_shift.py`：只读 `results/noisy_summaries/`

### Bootstrap 实验输出
- `bootstrap_summary.json`（当前最新主汇总）
- `bootstrap_summaries/bootstrap_summary_<dataset>_samples_<n>.json`（按 dataset + sample 数归档，避免覆盖）
- `bootstrap_topk_curves/*.png`（单 dataset 的 top-k exact-match curve）
- `bootstrap_rank_distribution/*.png`（`dataset × method × top_k × sample_count` rank distribution 图）
- `bootstrap_topk_stability_top_<k>_samples_<n>.png`（跨 dataset 的 top-k stability 汇总柱状图）

其中 bootstrap summary 会保存 baseline ranking、bootstrap rank distribution、top-k stability、top-k curve，以及每次成功 sample 的原始 ranking。

---

## 真实数据实验说明

当前实验默认读取本目录 `data/` 下四个文件：
- `data/judge_results_10k_chatbot_arena.json`
- `data/judge_results_10k_mtbench.json`
- `data/judge_results_10k_ultrafeedback.json`
- `data/in_house_data.json`

这些路径在 `run_real_data.py` 里通过 `BASE_DIR` 锚定，因此不依赖命令执行时所在 cwd。

每条记录包含：
- `question_id`
- `model_a`, `model_b`
- `judge_model`
- `judge_preferred_model`
- `judge_confidence`

### 标签处理规则

- `judge_preferred_model == "a"`：记为 `model_a` 胜
- `judge_preferred_model == "b"`：记为 `model_b` 胜
- `judge_preferred_model == "c"`：视为平局，只统计数量，不进入当前二元拟合与 accuracy 计算
- `judge_preferred_model == "unknown"`：统计数量后剔除

当前实现仍然使用二元 Bradley-Terry / logistic 风格拟合，因此 `c` 不会进入训练和测试 accuracy 的分母。

一句话总结：**truth = 单条测试记录里原始 judge 的 winner label，经 `(model_a, model_b)` 排序后转成二元 `y`。**
- 不是全局真值。
- 是 judge-specific, record-level ground truth。

稳定性实验输出里：
- `methods.proposed.gamma / mu / U / V` 是最终估计参数。
- `results/proposed_uvt_heatmaps/<dataset>.png` 可直接查看 `U @ V.T` 的 judge-item 异质性结构。
- 热图更适合看“某个 judge 对某类 item 是否系统性偏高/偏低”，不是看总体 ranking。

---

## 文件和代码说明

### 1. `run_real_data.py`
主入口，只负责组织真实数据实验、结果汇总与画图。

主要逻辑：
- `run_real_data_stability_experiment(...)`：对真实数据集做 train/test split，并汇总 test accuracy。
- `run_real_data_noisy_judge_experiment(...)`：逐步加入 noisy judge，保存增强数据并比较 ranking shift。
- `run_real_data_bootstrap_experiment(...)`：对 processed comparison records 做 bootstrap resampling，保存每次成功 sample 的原始 ranking，并汇总 rank distribution、top-k stability、top-k curve 与归档 summary。
- `run_real_data_benchmarks(...)`：顺序调度多类真实数据实验。

CLI 入口：
- `--experiment stability|robustness|bootstrap|all`
- `--dataset`
- `--max-noisy-step`
- `--bootstrap-samples`
- `--bootstrap-top-k`
- `--bootstrap-seed`

### 2. `src/generate_data.py`
负责真实数据预处理、聚合、 noisy judge 构造和 ranking shift 统计。

核心函数：
- `comparisons_to_aggregated(...)`：把比较记录转成 `n_ijk` 和 `y_ijk`
- `load_real_dataset(...)`：读取真实 JSON，统计 `a/b/c/unknown`，并把可用记录转成 `(k, i, j, y)`
- `split_real_dataset(...)`：对可用真实记录做 record-level train/test split
- `bootstrap_processed_records(...)`：对 processed comparison records 做有放回 bootstrap resampling
- `processed_records_to_aggregated(...)`：把处理后的记录直接转成聚合张量
- `make_noisy_judge_records(...)`：构造 `noisy_judge_1`, `noisy_judge_2`, ... 的记录
- `rank_items_from_mu(...)`：把 `mu` 转成 ranking
- `summarize_rank_distribution(...)`：汇总 bootstrap rank 分布
- `compute_top_k_stability(...)`：统计 bootstrap top-k 稳定性
- `compute_rank_shift(...)`：比较 noisy judge 前后的 ranking 变化

### 3. `src/models.py`
实现 proposed estimator，对应论文中的初始化、交替优化与 re-anchor。

### 4. `src/benchmarks.py`
统一封装三个拟合方法：
- `fit_proposed(...)`
- `fit_zhou_github(...)`
- `fit_standard_btl(...)`

其中 `fit_zhou_github(...)` 参考 `ZhouD_github/main/main.py` 的 weighted MLE + judge-specific scale。

### 5. ranking 记录方式
当前 ranking 统一由 `mu` 从高到低排序得到。summary 中会记录：
- 每个 item 的 `mu`
- 排序后的 item 列表
- `item -> rank` 映射
