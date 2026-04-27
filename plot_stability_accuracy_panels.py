import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.benchmarks import compute_score_matrix
from src.generate_data import load_real_dataset, split_real_dataset
from run_real_data import METHOD_LABELS, REAL_DATA_FILES


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = BASE_DIR / "results" / "stability_summary.json"
OUTPUT_DIR = BASE_DIR / "results" / "stability_accuracy_panels"
DATASET_ORDER = ["chatbot_arena", "mtbench", "ultrafeedback", "in_house"]
METHOD_ORDER = ["proposed", "zhou_github", "standard_btl"]
TEST_RATIO = 0.2
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_ORDER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_stability_summary() -> dict:
    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_reference_item_order(dataset_summary: dict) -> List[str]:
    for method_name in METHOD_ORDER:
        ranking = dataset_summary.get("methods", {}).get(method_name, {}).get("ranking", {}).get("ranking", [])
        if ranking:
            return list(ranking)
    return []


def rebuild_score_matrix(method_summary: dict) -> Optional[np.ndarray]:
    if method_summary.get("error") is not None:
        return None
    mu = np.asarray(method_summary.get("mu", []), dtype=float)
    gamma = np.asarray(method_summary.get("gamma", []), dtype=float)
    U = np.asarray(method_summary.get("U", []), dtype=float)
    V = np.asarray(method_summary.get("V", []), dtype=float)
    if mu.ndim != 1 or gamma.ndim != 1:
        return None
    if U.ndim != 2 or V.ndim != 2:
        return None
    return compute_score_matrix(mu, gamma, U, V)


def compute_item_accuracy(score_matrix: np.ndarray, test_records: List[dict], item_names: List[str]) -> Tuple[Dict[str, dict], Optional[float]]:
    item_stats = {
        item_name: {"correct": 0, "total": 0}
        for item_name in item_names
    }
    correct = 0
    total = 0

    for entry in test_records:
        pred = int(score_matrix[entry["k"], entry["i"]] > score_matrix[entry["k"], entry["j"]])
        is_correct = int(pred == entry["y"])
        correct += is_correct
        total += 1

        item_i = item_names[entry["i"]]
        item_j = item_names[entry["j"]]
        item_stats[item_i]["correct"] += is_correct
        item_stats[item_i]["total"] += 1
        item_stats[item_j]["correct"] += is_correct
        item_stats[item_j]["total"] += 1

    out = {}
    for item_name, stats in item_stats.items():
        item_total = int(stats["total"])
        out[item_name] = {
            "accuracy": None if item_total == 0 else float(stats["correct"] / item_total),
            "support": item_total,
        }
    overall_accuracy = None if total == 0 else float(correct / total)
    return out, overall_accuracy


def build_dataset_method_results(dataset_name: str, dataset_summary: dict) -> Dict[str, dict]:
    dataset = load_real_dataset(REAL_DATA_FILES[dataset_name])
    _, test_records = split_real_dataset(dataset["processed"], test_ratio=TEST_RATIO, random_seed=RANDOM_SEED)
    item_names = dataset["item_names"]

    method_results = {}
    for method_name in METHOD_ORDER:
        method_summary = dataset_summary.get("methods", {}).get(method_name, {})
        score_matrix = rebuild_score_matrix(method_summary)
        ranking_map = method_summary.get("ranking", {}).get("item_to_rank", {})
        if score_matrix is None:
            method_results[method_name] = {
                "item_accuracy": {},
                "item_to_rank": ranking_map,
                "overall_accuracy": None,
            }
            continue
        item_accuracy, overall_accuracy = compute_item_accuracy(score_matrix, test_records, item_names)
        method_results[method_name] = {
            "item_accuracy": item_accuracy,
            "item_to_rank": ranking_map,
            "overall_accuracy": overall_accuracy,
        }
    return method_results


def plot_method_accuracy(ax, method_name: str, method_result: dict, item_order: List[str], show_x_axis: bool) -> bool:
    item_accuracy = method_result.get("item_accuracy", {})
    item_to_rank = method_result.get("item_to_rank", {})
    item_names = [
        item_name
        for item_name in item_order
        if item_name in item_accuracy and item_accuracy[item_name].get("accuracy") is not None and item_name in item_to_rank
    ]
    if not item_names:
        ax.axis("off")
        return False

    x = np.arange(len(item_names))
    y = [item_to_rank[item_name] for item_name in item_names]
    accuracy_values = np.asarray([item_accuracy[item_name]["accuracy"] for item_name in item_names], dtype=float)
    support = np.asarray([item_accuracy[item_name]["support"] for item_name in item_names], dtype=float)
    max_support = float(np.max(support)) if np.max(support) > 0 else 1.0
    marker_sizes = 30.0 + 70.0 * support / max_support
    overall_accuracy = method_result.get("overall_accuracy")
    diagonal_ranks = np.arange(1, len(item_names) + 1)

    ax.plot(x, diagonal_ranks, linestyle="--", linewidth=1.2, color="#9c9c9c", label="y=x")
    scatter = ax.scatter(x, y, s=marker_sizes, c=accuracy_values, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.95)
    ax.plot(x, y, linewidth=1.1, color="#4c78a8", alpha=0.55)
    ax.set_xticks(x)
    ax.set_axisbelow(True)
    if show_x_axis:
        ax.set_xticklabels(item_names, rotation=90, fontsize=9)
        ax.set_xlabel("Items (fixed reference order)")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Rank")
    title = METHOD_LABELS[method_name]
    if overall_accuracy is not None:
        title += f" (acc={overall_accuracy:.3f})"
    ax.set_title(title)
    ax.set_ylim(len(item_names) + 0.5, 0.5)
    ax.set_yticks(np.arange(1, len(item_names) + 1))
    ax.grid(True, axis="y", alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Item accuracy")
    return True


def build_output_path(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / f"{dataset_name}_stability_accuracy_panel.png"


def plot_dataset(dataset_name: str, dataset_summary: dict, output_dir: Path) -> Optional[Path]:
    item_order = get_reference_item_order(dataset_summary)
    method_results = build_dataset_method_results(dataset_name, dataset_summary)
    fig, axes = plt.subplots(
        len(METHOD_ORDER),
        1,
        figsize=(max(12, 0.55 * dataset_summary["num_items"]), 14),
        squeeze=False,
        sharex=True,
    )
    axes_flat = axes.flatten()
    used_any = False

    for idx, (ax, method_name) in enumerate(zip(axes_flat, METHOD_ORDER)):
        show_x_axis = idx == len(METHOD_ORDER) - 1
        used_any = plot_method_accuracy(ax, method_name, method_results.get(method_name, {}), item_order, show_x_axis) or used_any
        ax.set_xlim(-0.5, len(item_order) - 0.5)

    if item_order:
        axes_flat[-1].set_xticks(np.arange(len(item_order)))
        axes_flat[-1].set_xticklabels(item_order, rotation=90, fontsize=9)

    fig.subplots_adjust(hspace=0.18)
    if not used_any:
        plt.close(fig)
        return None

    fig.suptitle(f"{dataset_name} stability rank panel (test split)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = build_output_path(output_dir, dataset_name)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stability_summary = load_stability_summary()

    dataset_names = [args.dataset] if args.dataset else DATASET_ORDER
    output_paths = []
    for dataset_name in dataset_names:
        dataset_summary = stability_summary.get(dataset_name)
        if dataset_summary is None:
            continue
        output_path = plot_dataset(dataset_name, dataset_summary, args.output_dir)
        if output_path is not None:
            output_paths.append(str(output_path))

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
