import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_noisy_judge_rank_shift import DATASET_NAMES, load_existing_noisy_summary
from run_real_data import METHOD_LABELS


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "noisy_judge_detection"
NOISY_PREFIX = "noisy_judge_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_NAMES)
    parser.add_argument("--max-noisy-step", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def is_noisy_judge(judge_name: str) -> bool:
    return str(judge_name).startswith(NOISY_PREFIX)


def split_gamma_groups(judge_names: List[str], gamma_values: List[float]) -> Tuple[List[float], List[float], List[Tuple[str, float, bool]]]:
    triples = []
    original = []
    noisy = []
    for judge_name, gamma in zip(judge_names, gamma_values):
        gamma_value = float(gamma)
        judge_is_noisy = is_noisy_judge(judge_name)
        triples.append((judge_name, gamma_value, judge_is_noisy))
        if judge_is_noisy:
            noisy.append(gamma_value)
        else:
            original.append(gamma_value)
    return original, noisy, triples


def get_step_summaries(dataset_summary: dict, max_step: Optional[int]) -> List[dict]:
    steps = []
    for step_summary in dataset_summary.get("steps", []):
        step = int(step_summary["step"])
        if max_step is not None and step > max_step:
            continue
        steps.append(step_summary)
    return steps


def build_method_step_records(dataset_summary: dict, max_step: Optional[int]) -> Dict[str, List[dict]]:
    records = {method_name: [] for method_name in METHOD_LABELS}
    for step_summary in get_step_summaries(dataset_summary, max_step=max_step):
        judge_names = step_summary.get("judge_names", [])
        step = int(step_summary["step"])
        for method_name in METHOD_LABELS:
            method_summary = step_summary.get("methods", {}).get(method_name, {})
            gamma_values = method_summary.get("gamma")
            if method_summary.get("error") is not None or not gamma_values or len(gamma_values) != len(judge_names):
                continue
            original, noisy, triples = split_gamma_groups(judge_names, gamma_values)
            records[method_name].append(
                {
                    "step": step,
                    "judge_names": judge_names,
                    "gamma": [float(value) for value in gamma_values],
                    "original": original,
                    "noisy": noisy,
                    "triples": triples,
                }
            )
    return records


def compute_hit_rate(triples: List[Tuple[str, float, bool]]) -> Optional[float]:
    noisy_count = sum(1 for _, _, is_noisy in triples if is_noisy)
    if noisy_count == 0:
        return None
    ranked = sorted(triples, key=lambda row: row[1])
    top_k = ranked[:noisy_count]
    hits = sum(1 for _, _, is_noisy in top_k if is_noisy)
    return float(hits / noisy_count)


def compute_metrics(dataset_name: str, method_records: Dict[str, List[dict]]) -> dict:
    metrics = {"dataset": dataset_name, "methods": {}}
    for method_name, records in method_records.items():
        method_metrics = []
        for record in records:
            noisy_median = float(np.median(record["noisy"])) if record["noisy"] else None
            original_median = float(np.median(record["original"])) if record["original"] else None
            method_metrics.append(
                {
                    "step": record["step"],
                    "num_noisy": len(record["noisy"]),
                    "num_original": len(record["original"]),
                    "hit_rate": compute_hit_rate(record["triples"]),
                    "noisy_median_gamma": noisy_median,
                    "original_median_gamma": original_median,
                    "median_gap_noisy_minus_original": None
                    if noisy_median is None or original_median is None
                    else float(noisy_median - original_median),
                }
            )
        metrics["methods"][method_name] = method_metrics
    return metrics


def plot_sorted_gamma_scatter(dataset_name: str, method_records: Dict[str, List[dict]], output_dir: Path) -> Optional[Path]:
    fig, axes = plt.subplots(1, len(METHOD_LABELS), figsize=(6 * len(METHOD_LABELS), 5), squeeze=False)
    axes_flat = axes.flatten()
    used_any = False

    for ax, method_name in zip(axes_flat, METHOD_LABELS):
        records = method_records.get(method_name, [])
        if not records:
            ax.axis("off")
            continue
        record = records[-1]
        ranked = sorted(record["triples"], key=lambda row: row[1])
        x_noisy = [idx + 1 for idx, (_, _, is_noisy) in enumerate(ranked) if is_noisy]
        y_noisy = [gamma for _, gamma, is_noisy in ranked if is_noisy]
        x_original = [idx + 1 for idx, (_, _, is_noisy) in enumerate(ranked) if not is_noisy]
        y_original = [gamma for _, gamma, is_noisy in ranked if not is_noisy]
        if x_original:
            ax.scatter(x_original, y_original, s=28, alpha=0.8, label="original", color="#4c78a8")
            used_any = True
        if x_noisy:
            ax.scatter(x_noisy, y_noisy, s=36, alpha=0.9, label="noisy", color="#e45756")
            used_any = True
        ax.set_title(f"{METHOD_LABELS[method_name]} (step {record['step']})")
        ax.set_xlabel("Judge rank by gamma (low to high)")
        ax.set_ylabel("gamma")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    if not used_any:
        plt.close(fig)
        return None

    fig.suptitle(f"{dataset_name}: sorted gamma scatter", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / f"noisy_judge_gamma_scatter_{dataset_name}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_gamma_boxplot(dataset_name: str, method_records: Dict[str, List[dict]], output_dir: Path) -> Optional[Path]:
    fig, axes = plt.subplots(1, len(METHOD_LABELS), figsize=(6 * len(METHOD_LABELS), 5), squeeze=False)
    axes_flat = axes.flatten()
    used_any = False
    legend_handles = [
        Patch(facecolor="#4c78a8", alpha=0.5, label="original"),
        Patch(facecolor="#e45756", alpha=0.5, label="noisy"),
    ]

    for ax, method_name in zip(axes_flat, METHOD_LABELS):
        records = method_records.get(method_name, [])
        if not records:
            ax.axis("off")
            continue

        positions_original = []
        positions_noisy = []
        original_data = []
        noisy_data = []
        xticks = []
        xticklabels = []

        for idx, record in enumerate(records, start=1):
            if not record["original"] or not record["noisy"]:
                continue
            positions_original.append(idx - 0.18)
            positions_noisy.append(idx + 0.18)
            original_data.append(record["original"])
            noisy_data.append(record["noisy"])
            xticks.append(idx)
            xticklabels.append(str(record["step"]))

        if not original_data or not noisy_data:
            ax.axis("off")
            continue

        original_box = ax.boxplot(original_data, positions=positions_original, widths=0.3, patch_artist=True, manage_ticks=False)
        noisy_box = ax.boxplot(noisy_data, positions=positions_noisy, widths=0.3, patch_artist=True, manage_ticks=False)
        for patch in original_box["boxes"]:
            patch.set_facecolor("#4c78a8")
            patch.set_alpha(0.5)
        for patch in noisy_box["boxes"]:
            patch.set_facecolor("#e45756")
            patch.set_alpha(0.5)

        ax.set_title(METHOD_LABELS[method_name])
        ax.set_xlabel("# noisy judges")
        ax.set_ylabel("gamma")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.grid(True, axis="y", alpha=0.3)
        used_any = True

    if not used_any:
        plt.close(fig)
        return None

    fig.suptitle(f"{dataset_name}: gamma boxplot by noisy step", fontsize=14)
    fig.legend(handles=legend_handles, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path = output_dir / f"noisy_judge_gamma_boxplot_{dataset_name}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_hit_rate(dataset_name: str, method_records: Dict[str, List[dict]], output_dir: Path) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(7, 5))
    used_any = False

    for method_name in METHOD_LABELS:
        records = method_records.get(method_name, [])
        if not records:
            continue
        steps = []
        hit_rates = []
        for record in records:
            hit_rate = compute_hit_rate(record["triples"])
            if hit_rate is None:
                continue
            steps.append(record["step"])
            hit_rates.append(hit_rate)
        if not steps:
            continue
        ax.plot(steps, hit_rates, marker="o", label=METHOD_LABELS[method_name])
        used_any = True

    if not used_any:
        plt.close(fig)
        return None

    ax.set_title(f"{dataset_name}: noisy judge hit@k by gamma")
    ax.set_xlabel("# noisy judges")
    ax.set_ylabel("hit rate among lowest-gamma top-k")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / f"noisy_judge_hit_rate_{dataset_name}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_metrics(dataset_name: str, method_records: Dict[str, List[dict]], output_dir: Path) -> Path:
    output_path = output_dir / f"noisy_judge_detection_metrics_{dataset_name}.json"
    metrics = compute_metrics(dataset_name, method_records)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return output_path


def main() -> None:
    args = parse_args()
    noisy_summary = load_existing_noisy_summary(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for dataset_name, dataset_summary in noisy_summary.items():
        method_records = build_method_step_records(dataset_summary, max_step=args.max_noisy_step)
        scatter_path = plot_sorted_gamma_scatter(dataset_name, method_records, args.output_dir)
        boxplot_path = plot_gamma_boxplot(dataset_name, method_records, args.output_dir)
        hit_rate_path = plot_hit_rate(dataset_name, method_records, args.output_dir)
        metrics_path = save_metrics(dataset_name, method_records, args.output_dir)
        for path in (scatter_path, boxplot_path, hit_rate_path, metrics_path):
            if path is not None:
                output_paths.append(str(path))

    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
