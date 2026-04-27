import argparse
import json
import os
import statistics
import time

from run_real_data import (
    REAL_DATA_FILES,
    run_real_data_bootstrap_experiment,
    run_real_data_stability_experiment,
    run_real_near_tie_slice_experiment,
)


DEFAULT_SEEDS = [
    11,
    17,
    23,
    29,
    31,
    37,
    41,
    42,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
]
DEFAULT_BOOTSTRAP_SEEDS = DEFAULT_SEEDS[:5]


def ensure_seed_sweep_dir():
    output_dir = os.path.join(os.path.dirname(__file__), "results", "seed_sweeps")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def parse_seeds(seed_text):
    if seed_text is None:
        return list(DEFAULT_SEEDS)
    seeds = [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def default_seeds_for_experiment(experiment):
    if experiment == "bootstrap":
        return list(DEFAULT_BOOTSTRAP_SEEDS)
    return list(DEFAULT_SEEDS)


def mean_std(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "stderr": None, "min": None, "max": None}
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "std": std,
        "stderr": std / (len(clean) ** 0.5),
        "min": min(clean),
        "max": max(clean),
    }


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_existing_raw(output_dir, experiment, seed):
    path = os.path.join(output_dir, f"{experiment}_seed_{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_raw(output_dir, experiment, seed, summary):
    path = os.path.join(output_dir, f"{experiment}_seed_{seed}.json")
    write_json(path, summary)
    return path


def iter_dataset_names(raw_by_seed):
    names = set()
    for summary in raw_by_seed.values():
        for dataset, dataset_summary in summary.items():
            if dataset.startswith("_") or not isinstance(dataset_summary, dict):
                continue
            if "methods" in dataset_summary or "base_methods" in dataset_summary:
                names.add(dataset)
    return sorted(names)


def run_one_experiment(experiment, seed, args):
    if experiment == "stability":
        return run_real_data_stability_experiment(
            dataset_name=args.dataset,
            test_ratio=args.test_ratio,
            random_seed=seed,
        )
    if experiment == "near_tie_slice":
        return run_real_near_tie_slice_experiment(
            dataset_name=args.dataset,
            test_ratio=args.test_ratio,
            min_pair_records=args.near_tie_min_pair_records,
            max_pairs=args.near_tie_max_pairs,
            random_seed=seed,
        )
    if experiment == "bootstrap":
        return run_real_data_bootstrap_experiment(
            dataset_name=args.dataset,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=seed,
            bootstrap_workers=args.bootstrap_workers,
        )
    raise ValueError(f"unknown experiment '{experiment}'")


def aggregate_stability(raw_by_seed):
    datasets = iter_dataset_names(raw_by_seed)
    out = {}
    for dataset in datasets:
        method_names = sorted({
            method
            for summary in raw_by_seed.values()
            if dataset in summary
            for method in summary[dataset].get("methods", {})
        })
        out[dataset] = {
            "train_size": next((summary[dataset].get("train_size") for summary in raw_by_seed.values() if dataset in summary), None),
            "test_size": next((summary[dataset].get("test_size") for summary in raw_by_seed.values() if dataset in summary), None),
            "methods": {},
            "selected_ranks": [
                summary[dataset].get("rank_selection", {}).get("selected_rank")
                for summary in raw_by_seed.values()
                if dataset in summary
            ],
        }
        for method in method_names:
            values = [
                summary[dataset].get("methods", {}).get(method, {}).get("test_accuracy")
                for summary in raw_by_seed.values()
                if dataset in summary
            ]
            out[dataset]["methods"][method] = {"test_accuracy": mean_std(values)}
    return out


def aggregate_near_tie(raw_by_seed):
    datasets = iter_dataset_names(raw_by_seed)
    metric_keys = (
        "near_tie_log_loss",
        "other_log_loss",
        "near_tie_accuracy_no_ties",
        "other_accuracy_no_ties",
    )
    out = {}
    for dataset in datasets:
        method_names = sorted({
            method
            for summary in raw_by_seed.values()
            if dataset in summary
            for method in summary[dataset].get("methods", {})
        })
        out[dataset] = {
            "near_tie_test_size": mean_std([
                summary[dataset].get("near_tie_test_size")
                for summary in raw_by_seed.values()
                if dataset in summary
            ]),
            "other_test_size": mean_std([
                summary[dataset].get("other_test_size")
                for summary in raw_by_seed.values()
                if dataset in summary
            ]),
            "methods": {},
            "selected_ranks": [
                summary[dataset].get("rank_selection", {}).get("selected_rank")
                for summary in raw_by_seed.values()
                if dataset in summary
            ],
        }
        for method in method_names:
            method_out = {
                "test_accuracy": mean_std([
                    summary[dataset].get("methods", {}).get(method, {}).get("test_accuracy")
                    for summary in raw_by_seed.values()
                    if dataset in summary
                ])
            }
            for key in metric_keys:
                method_out[key] = mean_std([
                    (summary[dataset].get("methods", {}).get(method, {}).get("slice_metrics") or {}).get(key)
                    for summary in raw_by_seed.values()
                    if dataset in summary
                ])
            out[dataset]["methods"][method] = method_out
    return out


def aggregate_bootstrap(raw_by_seed):
    datasets = iter_dataset_names(raw_by_seed)
    out = {}
    for dataset in datasets:
        method_names = sorted({
            method
            for summary in raw_by_seed.values()
            if dataset in summary
            for method in summary[dataset].get("methods", {})
        })
        top_keys = sorted({
            top_key
            for summary in raw_by_seed.values()
            if dataset in summary
            for method in summary[dataset].get("methods", {}).values()
            for top_key in (method.get("top_k_stability") or {})
        })
        out[dataset] = {
            "bootstrap_samples": next((summary[dataset].get("bootstrap_samples") for summary in raw_by_seed.values() if dataset in summary), None),
            "selected_ranks": [
                summary[dataset].get("rank_selection", {}).get("selected_rank")
                for summary in raw_by_seed.values()
                if dataset in summary
            ],
            "methods": {},
        }
        for method in method_names:
            method_out = {
                "successful_samples": mean_std([
                    summary[dataset].get("methods", {}).get(method, {}).get("successful_samples")
                    for summary in raw_by_seed.values()
                    if dataset in summary
                ]),
                "failed_samples": mean_std([
                    summary[dataset].get("methods", {}).get(method, {}).get("failed_samples")
                    for summary in raw_by_seed.values()
                    if dataset in summary
                ]),
                "top_k": {},
            }
            for top_key in top_keys:
                for metric in ("exact_match_rate_vs_baseline_top_k", "mean_jaccard_vs_baseline"):
                    values = [
                        ((summary[dataset].get("methods", {}).get(method, {}).get("top_k_stability") or {}).get(top_key) or {}).get(metric)
                        for summary in raw_by_seed.values()
                        if dataset in summary
                    ]
                    method_out["top_k"].setdefault(top_key, {})[metric] = mean_std(values)
            out[dataset]["methods"][method] = method_out
    return out


def aggregate_experiment(experiment, raw_by_seed):
    if experiment == "stability":
        return aggregate_stability(raw_by_seed)
    if experiment == "near_tie_slice":
        return aggregate_near_tie(raw_by_seed)
    if experiment == "bootstrap":
        return aggregate_bootstrap(raw_by_seed)
    raise ValueError(f"unknown experiment '{experiment}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=("stability", "near_tie_slice", "bootstrap", "all"),
        default="all",
    )
    parser.add_argument("--dataset", choices=sorted(REAL_DATA_FILES.keys()))
    parser.add_argument(
        "--seeds",
        help=(
            "comma-separated seed list; by default stability/near_tie_slice use "
            "20 seeds and bootstrap uses the first 5 seeds"
        ),
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-workers", type=int, default=None)
    parser.add_argument("--near-tie-min-pair-records", type=int, default=20)
    parser.add_argument("--near-tie-max-pairs", type=int, default=20)
    parser.add_argument("--reuse", action="store_true", help="reuse existing per-seed raw JSON when present")
    args = parser.parse_args()

    experiments = ["stability", "near_tie_slice", "bootstrap"] if args.experiment == "all" else [args.experiment]
    output_dir = ensure_seed_sweep_dir()

    for experiment in experiments:
        seeds = parse_seeds(args.seeds) if args.seeds is not None else default_seeds_for_experiment(experiment)
        raw_by_seed = {}
        for seed in seeds:
            existing = load_existing_raw(output_dir, experiment, seed) if args.reuse else None
            if existing is None:
                start_time = time.perf_counter()
                print(f"[seed-sweep] experiment={experiment} seed={seed} start", flush=True)
                summary = run_one_experiment(experiment, seed, args)
                summary["_seed_sweep_metadata"] = {
                    "experiment": experiment,
                    "seed": int(seed),
                    "elapsed_seconds": time.perf_counter() - start_time,
                }
                save_raw(output_dir, experiment, seed, summary)
            else:
                print(f"[seed-sweep] experiment={experiment} seed={seed} reusing raw JSON", flush=True)
                summary = existing
            raw_by_seed[int(seed)] = summary

        aggregate = {
            "experiment": experiment,
            "seeds": [int(seed) for seed in seeds],
            "dataset": args.dataset,
            "summary": aggregate_experiment(experiment, raw_by_seed),
        }
        output_path = os.path.join(output_dir, f"{experiment}_seed_sweep_summary.json")
        write_json(output_path, aggregate)
        print(f"[seed-sweep] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
