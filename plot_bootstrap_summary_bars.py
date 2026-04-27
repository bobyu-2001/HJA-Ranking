import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_DIR = BASE_DIR / "results" / "bootstrap_summaries"
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_BOOTSTRAP_SAMPLES = 100
DATASET_ORDER = ["chatbot_arena", "mtbench", "ultrafeedback", "in_house"]
METHOD_ORDER = ["proposed", "proposed_full_rank", "zhou_github", "standard_btl"]
METHOD_LABELS = {
    "proposed": "Proposed",
    "proposed_full_rank": "Proposed (full rank)",
    "zhou_github": "Zhou github",
    "standard_btl": "Standard BTL",
}
DATASET_LABELS = {
    "chatbot_arena": "chatbot_arena",
    "mtbench": "mtbench",
    "ultrafeedback": "ultrafeedback",
    "in_house": "in_house",
}


def load_summary(dataset_name: str, bootstrap_samples: int) -> dict:
    summary_path = SUMMARY_DIR / f"bootstrap_summary_{dataset_name}_samples_{bootstrap_samples}.json"
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if dataset_name not in payload:
        raise KeyError(f"dataset '{dataset_name}' not found in {summary_path}")
    return payload[dataset_name]


def get_exact_match_rate(dataset_summary: dict, method_name: str, top_k: int) -> float:
    top_k_key = f"top_{int(top_k)}"
    method_summary = dataset_summary.get("methods", {}).get(method_name)
    if method_summary is None:
        return np.nan
    top_k_summary = method_summary.get("top_k_summaries", {}).get(top_k_key)
    if top_k_summary is None:
        return np.nan
    stability = top_k_summary["top_k_stability"]
    return float(stability["exact_match_rate_vs_baseline_top_k"]) if stability is not None else np.nan


def collect_plot_values(top_k: int, bootstrap_samples: int):
    dataset_names = []
    values_by_method = {method_name: [] for method_name in METHOD_ORDER}

    for dataset_name in DATASET_ORDER:
        summary_path = SUMMARY_DIR / f"bootstrap_summary_{dataset_name}_samples_{bootstrap_samples}.json"
        if not summary_path.exists():
            continue
        dataset_summary = load_summary(dataset_name, bootstrap_samples)
        dataset_names.append(dataset_name)
        for method_name in METHOD_ORDER:
            values_by_method[method_name].append(
                get_exact_match_rate(dataset_summary, method_name, top_k)
            )

    if not dataset_names:
        raise FileNotFoundError(
            f"no bootstrap summary files found in {SUMMARY_DIR} for samples={bootstrap_samples}"
        )

    return dataset_names, values_by_method


def plot_summary_bar(top_k: int, bootstrap_samples: int) -> Path:
    dataset_names, values_by_method = collect_plot_values(top_k, bootstrap_samples)

    x = np.arange(len(dataset_names))
    width = 0.8 / len(METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(12, 5))

    for offset, method_name in enumerate(METHOD_ORDER):
        values = values_by_method[method_name]
        bars = ax.bar(
            x + (offset - (len(METHOD_ORDER) - 1) / 2.0) * width,
            values,
            width=width,
            label=METHOD_LABELS[method_name],
        )
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.015, 0.995),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[name] for name in dataset_names], rotation=15)
    ax.set_ylabel("Exact-match rate vs baseline top-k")
    ax.set_xlabel("Dataset")
    ax.set_title(f"Bootstrap top-k stability (top-{top_k}, samples={bootstrap_samples})")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y")
    ax.legend()

    fig.tight_layout()
    output_path = RESULTS_DIR / f"bootstrap_topk_stability_top_{int(top_k)}_samples_{int(bootstrap_samples)}_custom.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap sample count used in archived summary filenames.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for top_k in (3, 5):
        output_path = plot_summary_bar(top_k=top_k, bootstrap_samples=args.bootstrap_samples)
        print(output_path)
