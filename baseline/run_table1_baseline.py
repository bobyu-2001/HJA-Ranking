import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks import fit_standard_btl, fit_zhou_github


DATA_PATH = PROJECT_ROOT / "data" / "judge_results_10k_mtbench.json"
OUTPUT_DIR = CURRENT_DIR / "results"

TABLE1_MODELS = [
    "llama-13b",
    "alpaca-13b",
    "vicuna-13b-v1.2",
    "gpt-3.5-turbo",
    "gpt-4",
]

DISPLAY_NAMES = {
    "llama-13b": "LLaMA-13B",
    "alpaca-13b": "Alpaca-13B",
    "vicuna-13b-v1.2": "Vicuna-13B (all)",
    "gpt-3.5-turbo": "GPT-3.5",
    "gpt-4": "GPT-4",
}

PAPER_COLUMNS = {
    "llama-13b": {
        "tokens": "1T",
        "mmlu": 47.0,
        "truthfulqa": 0.26,
        "mt_bench_score": 2.61,
        "paper_s_u": -1.27,
        "paper_s_w": -1.12,
    },
    "alpaca-13b": {
        "tokens": "4.4M",
        "mmlu": 48.1,
        "truthfulqa": 0.30,
        "mt_bench_score": 4.53,
        "paper_s_u": -0.62,
        "paper_s_w": -0.54,
    },
    "vicuna-13b-v1.2": {
        "tokens": "370M",
        "mmlu": 52.1,
        "truthfulqa": 0.35,
        "mt_bench_score": 6.39,
        "paper_s_u": -0.29,
        "paper_s_w": -0.25,
    },
    "gpt-3.5-turbo": {
        "tokens": "-",
        "mmlu": 70.0,
        "truthfulqa": "-",
        "mt_bench_score": 7.94,
        "paper_s_u": 0.51,
        "paper_s_w": 0.43,
    },
    "gpt-4": {
        "tokens": "-",
        "mmlu": 86.4,
        "truthfulqa": "-",
        "mt_bench_score": 8.99,
        "paper_s_u": 0.83,
        "paper_s_w": 0.73,
    },
}



def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def build_table1_dataset(records):
    filtered_records = []
    label_counts = {"a": 0, "b": 0, "c": 0, "unknown": 0, "other": 0}

    judge_names = sorted({record["judge_model"] for record in records if "judge_model" in record})
    item_names = list(TABLE1_MODELS)
    item_set = set(item_names)
    judge_to_idx = {name: idx for idx, name in enumerate(judge_names)}
    item_to_idx = {name: idx for idx, name in enumerate(item_names)}

    for idx, record in enumerate(records):
        model_a = record.get("model_a")
        model_b = record.get("model_b")
        if model_a not in item_set or model_b not in item_set:
            continue

        raw_label = record.get("judge_preferred_model")
        label = str(raw_label).lower() if raw_label is not None else "unknown"
        if label not in {"a", "b", "c", "unknown"}:
            label_counts["other"] += 1
            continue
        label_counts[label] += 1
        if label == "unknown":
            continue
        if model_a == model_b:
            continue

        i_a = item_to_idx[model_a]
        i_b = item_to_idx[model_b]
        i, j = sorted((i_a, i_b))
        if label == "a":
            y = 1.0 if i_a == i else 0.0
        elif label == "b":
            y = 1.0 if i_b == i else 0.0
        else:
            y = 0.5

        filtered_records.append(
            {
                "record_index": idx,
                "question_id": record.get("question_id"),
                "judge_model": record.get("judge_model"),
                "model_a": model_a,
                "model_b": model_b,
                "judge_confidence": record.get("judge_confidence"),
                "preferred_label": label,
                "k": judge_to_idx[record["judge_model"]],
                "i": i,
                "j": j,
                "y": y,
            }
        )

    return {
        "records": filtered_records,
        "item_names": item_names,
        "judge_names": judge_names,
        "item_to_idx": item_to_idx,
        "judge_to_idx": judge_to_idx,
        "summary": {
            "source_path": str(DATA_PATH),
            "num_items": len(item_names),
            "num_judges": len(judge_names),
            "usable_records": len(filtered_records),
            "label_counts": label_counts,
            "tie_count": label_counts["c"],
            "unknown_count": label_counts["unknown"],
        },
    }



def processed_to_aggregated(processed_records, num_items, num_judges):
    n_ijk = np.zeros((num_judges, num_items, num_items), dtype=float)
    y_ijk = np.zeros((num_judges, num_items, num_items), dtype=float)
    for entry in processed_records:
        k = entry["k"]
        i = entry["i"]
        j = entry["j"]
        y = float(entry["y"])
        n_ijk[k, i, j] += 1.0
        y_ijk[k, i, j] += y
    return n_ijk, y_ijk



def fit_all_methods(num_items, num_judges, n_ijk, y_ijk):
    results = {}
    for method_name, fit_fn in (
        ("standard_btl", lambda: fit_standard_btl(num_items, num_judges, n_ijk, y_ijk)),
        ("weighted_mle", lambda: fit_zhou_github(num_items, num_judges, n_ijk, y_ijk)),
    ):
        start = time.perf_counter()
        try:
            fit = fit_fn()
            results[method_name] = {
                "fit": fit,
                "error": None,
                "elapsed_seconds": time.perf_counter() - start,
            }
        except Exception as exc:
            results[method_name] = {
                "fit": None,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - start,
            }
    return results



def round_or_none(value, ndigits=2):
    if value is None:
        return None
    return round(float(value), ndigits)


def ci_bound(fit, key, idx, bound):
    if fit is None:
        return None
    uq = fit.get("uq") or {}
    if uq.get("error") is not None:
        return None
    intervals = uq.get(key) or []
    if idx >= len(intervals):
        return None
    return intervals[idx].get(bound)



def build_rows(fit_results, item_names):
    rows = []
    standard_fit = fit_results["standard_btl"]["fit"]
    weighted_fit = fit_results["weighted_mle"]["fit"]

    for idx, model_name in enumerate(item_names):
        paper = PAPER_COLUMNS[model_name]
        row = {
            "model_key": model_name,
            "model": DISPLAY_NAMES[model_name],
            "tokens": paper["tokens"],
            "mmlu": paper["mmlu"],
            "truthfulqa": paper["truthfulqa"],
            "mt_bench_score": paper["mt_bench_score"],
            "paper_s_u": paper["paper_s_u"],
            "paper_s_w": paper["paper_s_w"],
            "reproduced_s_u": None if standard_fit is None else round_or_none(standard_fit["mu"][idx]),
            "reproduced_s_u_CI_lower": round_or_none(ci_bound(standard_fit, "s_ci", idx, "lower")),
            "reproduced_s_u_CI_upper": round_or_none(ci_bound(standard_fit, "s_ci", idx, "upper")),
            "reproduced_s_w": None if weighted_fit is None else round_or_none(weighted_fit["mu"][idx]),
            "reproduced_s_w_CI_lower": round_or_none(ci_bound(weighted_fit, "s_ci", idx, "lower")),
            "reproduced_s_w_CI_upper": round_or_none(ci_bound(weighted_fit, "s_ci", idx, "upper")),
        }
        if row["reproduced_s_u"] is not None:
            row["delta_s_u"] = round_or_none(row["reproduced_s_u"] - row["paper_s_u"])
        else:
            row["delta_s_u"] = None
        if row["reproduced_s_w"] is not None:
            row["delta_s_w"] = round_or_none(row["reproduced_s_w"] - row["paper_s_w"])
        else:
            row["delta_s_w"] = None
        rows.append(row)
    return rows



def write_csv(rows, output_path):
    fieldnames = [
        "model_key",
        "model",
        "tokens",
        "mmlu",
        "truthfulqa",
        "mt_bench_score",
        "paper_s_u",
        "paper_s_w",
        "reproduced_s_u",
        "reproduced_s_u_CI_lower",
        "reproduced_s_u_CI_upper",
        "reproduced_s_w",
        "reproduced_s_w_CI_lower",
        "reproduced_s_w_CI_upper",
        "delta_s_u",
        "delta_s_w",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def run(output_dir):
    records = load_records(DATA_PATH)
    dataset = build_table1_dataset(records)
    num_items = len(dataset["item_names"])
    num_judges = len(dataset["judge_names"])
    n_ijk, y_ijk = processed_to_aggregated(dataset["records"], num_items, num_judges)
    fit_results = fit_all_methods(num_items, num_judges, n_ijk, y_ijk)
    rows = build_rows(fit_results, dataset["item_names"])

    result = {
        "paper": "A Judge-Aware Ranking Framework for Evaluating Large Language Models without Ground Truth",
        "table": "Table 1",
        "dataset": "MT-Bench",
        "data_path": str(DATA_PATH),
        "summary": dataset["summary"],
        "fit_results": {
            method_name: {
                "error": fit_result["error"],
                "elapsed_seconds": fit_result["elapsed_seconds"],
                "fit_info": None if fit_result["fit"] is None else fit_result["fit"]["fit_info"],
                "mu": None if fit_result["fit"] is None else [float(x) for x in fit_result["fit"]["mu"]],
                "gamma": None if fit_result["fit"] is None else [float(x) for x in fit_result["fit"]["gamma"]],
                "uq": None if fit_result["fit"] is None else fit_result["fit"].get("uq"),
            }
            for method_name, fit_result in fit_results.items()
        },
        "rows": rows,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "table1_mtbench_reproduction.json")
    csv_path = os.path.join(output_dir, "table1_mtbench_reproduction.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_csv(rows, csv_path)

    print(f"wrote {json_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)

    if fit_results["standard_btl"]["error"] is not None:
        print(f"standard_btl failed: {fit_results['standard_btl']['error']}", flush=True)
    if fit_results["weighted_mle"]["error"] is not None:
        print(f"weighted_mle failed: {fit_results['weighted_mle']['error']}", flush=True)

    return result



def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 1 on MT-Bench.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
