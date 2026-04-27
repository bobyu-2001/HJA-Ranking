import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_DIR = BASE_DIR / "results" / "bootstrap_summaries"
OUTPUT_DIR = BASE_DIR / "results" / "bootstrap_rank_distribution_panels"
DATASET_ORDER = ["chatbot_arena", "mtbench", "ultrafeedback", "in_house"]
SAMPLE_ORDER = [20, 50, 100]
METHOD_ORDER = ["proposed", "proposed_full_rank", "zhou_github", "standard_btl"]
METHOD_LABELS = {
    "proposed": "Proposed",
    "proposed_full_rank": "Proposed (full rank)",
    "zhou_github": "Zhou github",
    "standard_btl": "Standard BTL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_ORDER)
    parser.add_argument("--bootstrap-samples", type=int, choices=SAMPLE_ORDER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_summary(dataset_name: str, bootstrap_samples: int) -> dict:
    summary_path = SUMMARY_DIR / f"bootstrap_summary_{dataset_name}_samples_{int(bootstrap_samples)}.json"
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if dataset_name not in payload:
        raise KeyError(f"dataset '{dataset_name}' not found in {summary_path}")
    return payload[dataset_name]


def get_reference_item_order(dataset_summary: dict) -> list:
    for method_name in METHOD_ORDER:
        baseline = dataset_summary.get("methods", {}).get(method_name, {}).get("baseline", {}).get("ranking", {}).get("ranking", [])
        if baseline:
            return list(baseline)
    return []


def build_rank_distribution(method_summary: dict, item_order: list) -> tuple:
    rankings = method_summary.get("bootstrap_rankings", [])
    baseline_item_to_rank = method_summary.get("baseline", {}).get("ranking", {}).get("item_to_rank", {})
    if not item_order or not rankings:
        return {}, baseline_item_to_rank

    item_to_ranks = {item_name: [] for item_name in item_order}
    for ranking_payload in rankings:
        item_to_rank = ranking_payload.get("item_to_rank", {})
        for item_name in item_order:
            rank_value = item_to_rank.get(item_name)
            if rank_value is not None:
                item_to_ranks[item_name].append(float(rank_value))

    rank_distribution = {}
    for item_name in item_order:
        ranks = item_to_ranks[item_name]
        if not ranks:
            continue
        rank_distribution[item_name] = {
            "median_rank": float(np.median(ranks)),
            "rank_p05": float(np.percentile(ranks, 5)),
            "rank_p95": float(np.percentile(ranks, 95)),
        }
    return rank_distribution, baseline_item_to_rank


def plot_method_distribution(ax, method_name: str, method_summary: dict, item_order: list, show_x_axis: bool) -> bool:
    rank_distribution, baseline_item_to_rank = build_rank_distribution(method_summary, item_order)
    if not rank_distribution:
        ax.axis("off")
        return False

    item_names = [item_name for item_name in item_order if item_name in rank_distribution]
    median_ranks = [rank_distribution[item_name]["median_rank"] for item_name in item_names]
    low_err = [
        max(0.0, rank_distribution[item_name]["median_rank"] - rank_distribution[item_name]["rank_p05"])
        for item_name in item_names
    ]
    high_err = [
        max(0.0, rank_distribution[item_name]["rank_p95"] - rank_distribution[item_name]["median_rank"])
        for item_name in item_names
    ]
    true_ranks = [baseline_item_to_rank.get(item_name) for item_name in item_names]

    x = np.arange(len(item_names))
    diagonal_ranks = np.arange(1, len(item_names) + 1)
    ax.plot(x, diagonal_ranks, linestyle="--", linewidth=1.2, color="#9c9c9c", label="y=x")
    ax.errorbar(x, median_ranks, yerr=[low_err, high_err], fmt="o", capsize=4, label="bootstrap")
    ax.scatter(x, true_ranks, marker="x", s=42, color="#e45756", label="true")
    ax.set_xticks(x)
    ax.set_axisbelow(True)
    if show_x_axis:
        ax.set_xticklabels(item_names, rotation=90, fontsize=9)
        ax.set_xlabel("Items (fixed true order)")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Rank")
    ax.set_title(METHOD_LABELS[method_name])
    ax.invert_yaxis()
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    return True


def build_output_path(output_dir: Path, dataset_name: str, bootstrap_samples: int) -> Path:
    return output_dir / f"{dataset_name}_samples_{int(bootstrap_samples)}_rank_distribution_panel.png"


def plot_dataset_sample(dataset_name: str, bootstrap_samples: int, output_dir: Path) -> Optional[Path]:
    dataset_summary = load_summary(dataset_name, bootstrap_samples)
    item_order = get_reference_item_order(dataset_summary)
    fig, axes = plt.subplots(
        len(METHOD_ORDER),
        1,
        figsize=(max(12, 0.55 * dataset_summary["num_items"]), 16),
        squeeze=False,
        sharex=True,
    )
    axes_flat = axes.flatten()
    used_any = False

    for idx, (ax, method_name) in enumerate(zip(axes_flat, METHOD_ORDER)):
        method_summary = dataset_summary.get("methods", {}).get(method_name, {})
        show_x_axis = idx == len(METHOD_ORDER) - 1
        used_any = plot_method_distribution(ax, method_name, method_summary, item_order, show_x_axis=show_x_axis) or used_any
        ax.set_ylim(dataset_summary["num_items"] + 0.5, 0.5)
        ax.set_yticks(np.arange(1, dataset_summary["num_items"] + 1))
        ax.set_xlim(-0.5, len(item_order) - 0.5)

    if item_order:
        axes_flat[-1].set_xticks(np.arange(len(item_order)))
        axes_flat[-1].set_xticklabels(item_order, rotation=90, fontsize=9)

    fig.subplots_adjust(hspace=0.22)
    if not used_any:
        plt.close(fig)
        return None

    fig.suptitle(f"{dataset_name} bootstrap rank distribution (samples={int(bootstrap_samples)})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = build_output_path(output_dir, dataset_name, bootstrap_samples)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = [args.dataset] if args.dataset else DATASET_ORDER
    sample_counts = [args.bootstrap_samples] if args.bootstrap_samples else SAMPLE_ORDER

    output_paths = []
    for dataset_name in dataset_names:
        for bootstrap_samples in sample_counts:
            output_path = plot_dataset_sample(dataset_name, bootstrap_samples, args.output_dir)
            if output_path is not None:
                output_paths.append(str(output_path))

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
