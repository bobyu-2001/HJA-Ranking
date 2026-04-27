# Table 1 基线复现

这个文件夹包含用于复现论文 *A Judge-Aware Ranking Framework for Evaluating Large Language Models without Ground Truth* 中 Table 1 的代码，使用的数据是 MT-Bench，对应文件 `../data/judge_results_10k_mtbench.json`。

## 脚本做什么

`run_table1_baseline.py` 会重建 Table 1 中 5 个模型的分数：

- LLaMA-13B
- Alpaca-13B
- Vicuna-13B (all)
- GPT-3.5
- GPT-4

脚本会在 MT-Bench 两两比较数据上拟合 2 个估计器：

- 非加权 BTL，对应 `s^u`，通过 `fit_standard_btl(...)`
- 带 judge-specific scale 的加权 MLE，对应 `s^w`，通过 `fit_zhou_github(...)`

然后把论文中的数值、复现得到的数值以及二者差值并排写出。

## 数据与预处理

输入数据：

- `../data/judge_results_10k_mtbench.json`

只保留比较双方都属于论文 Table 1 模型集合的记录。

标签处理方式与论文表述一致：

- `judge_preferred_model == "a"` -> `model_a` 获胜
- `judge_preferred_model == "b"` -> `model_b` 获胜
- `judge_preferred_model == "c"` -> 平局 / 无偏好，编码为 `y = 0.5`
- `judge_preferred_model == "unknown"` -> 丢弃

这点和当前 `real_data_0420/src/generate_data.py` 里的 benchmark 流程不同：这里不会丢弃 tie。

## 复用代码

这份实现复用了当前项目里的 benchmark 代码：

- `../src/benchmarks.py`
  - `fit_standard_btl(...)`
  - `fit_zhou_github(...)`

脚本中还内置了论文 Table 1 的元数据：

- token 数
- MMLU
- TruthfulQA
- MT-Bench score
- 论文报告的 `s^u` 和 `s^w`

## 运行方式

在 `real_data_0420/` 目录下执行：

```bash
python baseline/run_table1_baseline.py --output-dir baseline/results_table1
```

## 输出文件

脚本会写出：

- `baseline/results_table1/table1_mtbench_reproduction.csv`
- `baseline/results_table1/table1_mtbench_reproduction.json`

### CSV 列含义

每一行对应 Table 1 中一个模型。

列包括：

- `model_key`
- `model`
- `tokens`
- `mmlu`
- `truthfulqa`
- `mt_bench_score`
- `paper_s_u`
- `paper_s_w`
- `reproduced_s_u`
- `reproduced_s_w`
- `delta_s_u`
- `delta_s_w`

### JSON 内容

JSON 中包括：

- 数据集摘要
- 可用记录数
- judge 数量和 item 数量
- 两个方法的拟合元信息
- 复现得到的原始 `mu` 和 `gamma`
- 最终输出行

## 当前复现结果

当前输出文件 `baseline/results_table1/table1_mtbench_reproduction.csv` 已经能复现论文中的整体趋势，数值也比较接近，但还没有做到与论文逐项完全一致。

与论文对比如下：

| model | paper s^u | repro s^u | delta | paper s^w | repro s^w | delta |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA-13B | -1.27 | -1.16 | +0.11 | -1.12 | -1.00 | +0.12 |
| Alpaca-13B | -0.62 | -0.51 | +0.11 | -0.54 | -0.53 | +0.01 |
| Vicuna-13B (all) | -0.29 | -0.10 | +0.19 | -0.25 | -0.05 | +0.20 |
| GPT-3.5 | 0.51 | 0.77 | +0.26 | 0.43 | 0.65 | +0.22 |
| GPT-4 | 0.83 | 1.01 | +0.18 | 0.73 | 0.92 | +0.19 |

总结：

- 排序趋势和论文一致
- 分数量级比较接近
- 还没有做到精确数值复现
- 当前最大观测偏差约为 `0.26`

目前可以这样评价：

- 定性复现：成功
- 近似定量复现：成功
- 严格数值复现：尚未完成

## 为什么还会和论文有差异

最可能原因：

- 本地 `judge_results_10k_mtbench.json` 和论文当时使用的数据快照不完全一致
- 官方代码里可能还有少量预处理细节没有被完全还原
- 优化设置、停止阈值或归一化细节可能和原始实验不同
- 论文表格在拟合前可能还做了轻微额外筛选

## 参考文件

本次复现主要参考了：

- `../../ZhouD_github/main/main.py`
- `../src/benchmarks.py`
- `../data/judge_results_10k_mtbench.json`
- 论文 PDF：`../../2601.21817v1.pdf`
