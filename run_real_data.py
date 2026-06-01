import argparse
import json
import multiprocessing as mp
import os
import statistics
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.generate_data import (
    bootstrap_processed_records,
    build_real_dataset_from_records,
    compute_rank_shift,
    compute_top_k_stability,
    load_real_dataset,
    make_noisy_judge_records,
    processed_records_to_aggregated,
    rank_items_from_mu,
    split_real_dataset,
    summarize_rank_distribution,
)
from src.benchmarks import fit_proposed, fit_standard_btl, fit_zhou_github


METHOD_LABELS = {
    "proposed": "Proposed",
    "zhou_github": "Zhou github",
    "standard_btl": "Standard BTL",
}
BOOTSTRAP_METHOD_LABELS = {
    "proposed": "Proposed",
    "proposed_full_rank": "Proposed (full rank)",
    "zhou_github": "Zhou github",
    "standard_btl": "Standard BTL",
}

BASE_DIR = os.path.dirname(__file__)
REAL_DATA_FILES = {
    "chatbot_arena": os.path.join(BASE_DIR, "data", "judge_results_10k_chatbot_arena.json"),
    "mtbench": os.path.join(BASE_DIR, "data", "judge_results_10k_mtbench.json"),
    "ultrafeedback": os.path.join(BASE_DIR, "data", "judge_results_10k_ultrafeedback.json"),
    "in_house": os.path.join(BASE_DIR, "data", "in_house_data.json"),
}

NOISY_JUDGE_COUNTS = list(range(1, 11))
STRUCTURED_BIAS_VARIANTS = ("anti_consensus", "family_bias", "cluster_bias")
STRUCTURED_BIAS_UNITS = ("all_records", "question_pair", "matched_judge")
DEFAULT_SEED_SWEEP_SEEDS = [
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
DEFAULT_BOOTSTRAP_SWEEP_SEEDS = DEFAULT_SEED_SWEEP_SEEDS[:5]



def select_real_data_files(dataset_name=None):
    if dataset_name is None:
        return REAL_DATA_FILES
    if dataset_name not in REAL_DATA_FILES:
        raise ValueError(f"unknown dataset '{dataset_name}', expected one of {sorted(REAL_DATA_FILES)}")
    return {dataset_name: REAL_DATA_FILES[dataset_name]}



def select_noisy_judge_counts(max_noisy_step=None):
    if max_noisy_step is None:
        return NOISY_JUDGE_COUNTS
    if max_noisy_step < 1:
        raise ValueError(f"max_noisy_step must be >= 1, got {max_noisy_step}")
    return [step for step in NOISY_JUDGE_COUNTS if step <= max_noisy_step]


def ensure_results_dir():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def ensure_seed_sweep_dir():
    results_dir = os.path.join(ensure_results_dir(), "seed_sweeps")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def ensure_noisy_dataset_dir():
    results_dir = os.path.join(ensure_results_dir(), "noisy_datasets")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir



def ensure_noisy_summary_dir():
    results_dir = os.path.join(ensure_results_dir(), "noisy_summaries")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir



def ensure_proposed_heatmap_dir():
    results_dir = os.path.join(ensure_results_dir(), "proposed_uvt_heatmaps")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir



def ensure_bootstrap_rank_plot_dir():
    results_dir = os.path.join(ensure_results_dir(), "bootstrap_rank_distribution")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir



def ensure_bootstrap_topk_curve_dir():
    results_dir = os.path.join(ensure_results_dir(), "bootstrap_topk_curves")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir



def ensure_bootstrap_summary_archive_dir():
    results_dir = os.path.join(ensure_results_dir(), "bootstrap_summaries")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def ensure_structured_bias_dataset_dir():
    results_dir = os.path.join(ensure_results_dir(), "structured_bias_datasets")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def ensure_stability_excluding_dir():
    results_dir = os.path.join(ensure_results_dir(), "stability_excluding_zai_org")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def ensure_stability_excluding_heatmap_dir():
    results_dir = os.path.join(ensure_stability_excluding_dir(), "proposed_uvt_heatmaps")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir



def infer_model_rank(N, K):
    return 0 if min(K - 1, N - 2) <= 0 else 1



def infer_full_model_rank(N, K):
    return max(0, min(K - 1, N - 2))



def default_cv_candidate_ranks(N, K):
    return list(range(infer_full_model_rank(N, K) + 1))



def compute_validation_nll(fit, validation_records):
    if not validation_records:
        return None
    score = np.asarray(fit["S"], dtype=float)
    loss = 0.0
    for entry in validation_records:
        diff = score[entry["k"], entry["i"]] - score[entry["k"], entry["j"]]
        y = float(entry["y"])
        loss += y * np.logaddexp(0.0, -diff) + (1.0 - y) * np.logaddexp(0.0, diff)
    return float(loss)



def make_k_fold_indices(num_records, n_folds=5, random_seed=42):
    if num_records < 2:
        raise ValueError("cross validation requires at least 2 records")
    actual_folds = min(int(n_folds), int(num_records))
    if actual_folds < 2:
        raise ValueError(f"cross validation requires at least 2 folds, got {actual_folds}")
    rng = np.random.default_rng(random_seed)
    indices = np.arange(num_records)
    rng.shuffle(indices)
    return [fold.astype(int).tolist() for fold in np.array_split(indices, actual_folds)]



def select_rank_by_cross_validation(
    N,
    K,
    processed_records,
    n_folds=5,
    random_seed=42,
    candidate_ranks=None,
    max_steps=40,
    tol=5e-5,
    tau=30.0, # or 5 for UltraFeedback (except in low noise settings) and 300 for in-house near-tie
):
    candidate_ranks = default_cv_candidate_ranks(N, K) if candidate_ranks is None else [int(r) for r in candidate_ranks]
    r_max = infer_full_model_rank(N, K)
    invalid_ranks = [r for r in candidate_ranks if r < 0 or r > r_max]
    if invalid_ranks:
        raise ValueError(f"candidate ranks {invalid_ranks} exceed permissible range 0..{r_max}")
    if not candidate_ranks:
        raise ValueError("candidate_ranks must be non-empty")

    folds = make_k_fold_indices(len(processed_records), n_folds=n_folds, random_seed=random_seed)
    all_indices = set(range(len(processed_records)))
    rank_results = {}

    print(f"[rank-cv] start: folds={len(folds)}, candidates={candidate_ranks}, N={N}, K={K}", flush=True)
    for r_model in candidate_ranks:
        fold_results = []
        total_nll = 0.0
        total_records = 0
        failed = False
        for fold_idx, validation_indices in enumerate(folds):
            validation_set = set(validation_indices)
            train_records = [processed_records[idx] for idx in sorted(all_indices - validation_set)]
            validation_records = [processed_records[idx] for idx in validation_indices]
            n_ijk, y_ijk = processed_records_to_aggregated(train_records, N, K)
            start_time = time.perf_counter()
            try:
                fit = fit_proposed(N, K, r_model, n_ijk, y_ijk, max_steps=max_steps, tol=tol, tau=tau)
                validation_nll = compute_validation_nll(fit, validation_records)
                elapsed = time.perf_counter() - start_time
                total_nll += float(validation_nll)
                total_records += len(validation_records)
                fold_results.append({
                    "fold": int(fold_idx),
                    "validation_size": int(len(validation_records)),
                    "validation_nll": float(validation_nll),
                    "validation_nll_per_record": float(validation_nll / len(validation_records)),
                    "elapsed_seconds": float(elapsed),
                    "error": None,
                })
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                failed = True
                fold_results.append({
                    "fold": int(fold_idx),
                    "validation_size": int(len(validation_records)),
                    "validation_nll": None,
                    "validation_nll_per_record": None,
                    "elapsed_seconds": float(elapsed),
                    "error": str(exc),
                })
                print(f"[rank-cv] r={r_model} fold={fold_idx + 1}/{len(folds)} failed: {exc}", flush=True)
                break

        mean_nll = None if failed or total_records == 0 else float(total_nll / total_records)
        rank_results[str(r_model)] = {
            "rank": int(r_model),
            "mean_validation_nll_per_record": mean_nll,
            "failed": bool(failed),
            "folds": fold_results,
        }
        if mean_nll is not None:
            print(f"[rank-cv] r={r_model} mean nll/record={mean_nll:.6f}", flush=True)

    valid_results = [
        entry for entry in rank_results.values()
        if not entry["failed"] and entry["mean_validation_nll_per_record"] is not None
    ]
    if not valid_results:
        raise RuntimeError("rank cross validation failed for every candidate rank")
    best = min(valid_results, key=lambda entry: (entry["mean_validation_nll_per_record"], entry["rank"]))
    return int(best["rank"]), {
        "method": "k-fold validation_nll",
        "folds": int(len(folds)),
        "random_seed": int(random_seed),
        "candidate_ranks": [int(r) for r in candidate_ranks],
        "selected_rank": int(best["rank"]),
        "rank_results": rank_results,
    }



def infer_bootstrap_workers(requested_workers=None, cpu_fraction=0.75):
    if requested_workers is not None:
        if int(requested_workers) < 1:
            raise ValueError(f"bootstrap_workers must be >= 1, got {requested_workers}")
        return int(requested_workers)
    cpu_total = os.cpu_count() or 1
    return max(1, int(np.floor(cpu_total * float(cpu_fraction))))



def configure_single_threaded_numeric_libraries():
    # Each bootstrap worker is process-parallel. Keeping BLAS/OpenMP at one
    # thread per worker prevents oversubscription and makes worker count track
    # CPU usage more directly.
    for env_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[env_name] = "1"



def fit_all_methods_safe(
    N,
    K,
    r_model,
    n_ijk,
    y_ijk,
    tau=30.0,
    skipped_methods=None,
    max_iter=500,
    proposed_max_steps=40,
    proposed_tol=5e-5,
    proposed_inner_maxiter=500,
):
    out = {}
    skipped_methods = skipped_methods or {}
    for method_name, fit_fn in (
        (
            "proposed",
            lambda: fit_proposed(
                N,
                K,
                r_model,
                n_ijk,
                y_ijk,
                max_steps=proposed_max_steps,
                tol=proposed_tol,
                tau=tau,
                inner_maxiter=proposed_inner_maxiter,
            ),
        ),
        ("zhou_github", lambda: fit_zhou_github(N, K, n_ijk, y_ijk, max_iter=max_iter)),
        ("standard_btl", lambda: fit_standard_btl(N, K, n_ijk, y_ijk)),
    ):
        skip_reason = skipped_methods.get(method_name)
        if skip_reason is not None:
            model_rank = int(r_model) if method_name == "proposed" else None
            out[method_name] = {
                "fit": None,
                "error": skip_reason,
                "elapsed_seconds": 0.0,
                "skipped": True,
                "model_rank": model_rank,
            }
            print(f"[{method_name}] skipped: {skip_reason}", flush=True)
            continue
        start_time = time.perf_counter()
        print(f"[{method_name}] start: N={N}, K={K}, r={r_model}", flush=True)
        try:
            fit = fit_fn()
            elapsed = time.perf_counter() - start_time
            model_rank = int(r_model) if method_name == "proposed" else None
            out[method_name] = {
                "fit": fit,
                "error": None,
                "elapsed_seconds": elapsed,
                "skipped": False,
                "model_rank": model_rank,
            }
            print(f"[{method_name}] done in {elapsed:.2f}s", flush=True)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            model_rank = int(r_model) if method_name == "proposed" else None
            out[method_name] = {
                "fit": None,
                "error": str(exc),
                "elapsed_seconds": elapsed,
                "skipped": False,
                "model_rank": model_rank,
            }
            print(f"[{method_name}] failed after {elapsed:.2f}s: {exc}", flush=True)
    return out


def fit_bootstrap_methods_safe(N, K, proposed_rank, full_rank, n_ijk, y_ijk, tau=30.0):
    out = {}
    for method_name, fit_fn, model_rank in (
        ("proposed", lambda: fit_proposed(N, K, proposed_rank, n_ijk, y_ijk, max_steps=40, tol=5e-5, tau=tau), proposed_rank),
        ("proposed_full_rank", lambda: fit_proposed(N, K, full_rank, n_ijk, y_ijk, max_steps=40, tol=5e-5, tau=tau), full_rank),
        ("zhou_github", lambda: fit_zhou_github(N, K, n_ijk, y_ijk), None),
        ("standard_btl", lambda: fit_standard_btl(N, K, n_ijk, y_ijk), None),
    ):
        start_time = time.perf_counter()
        print(f"[{method_name}] start: N={N}, K={K}, r={model_rank}", flush=True)
        try:
            fit = fit_fn()
            elapsed = time.perf_counter() - start_time
            out[method_name] = {
                "fit": fit,
                "error": None,
                "elapsed_seconds": elapsed,
                "skipped": False,
                "model_rank": None if model_rank is None else int(model_rank),
            }
            print(f"[{method_name}] done in {elapsed:.2f}s", flush=True)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            out[method_name] = {
                "fit": None,
                "error": str(exc),
                "elapsed_seconds": elapsed,
                "skipped": False,
                "model_rank": None if model_rank is None else int(model_rank),
            }
            print(f"[{method_name}] failed after {elapsed:.2f}s: {exc}", flush=True)
    return out


def fit_all_methods_on_sample(sample_idx, N, K, proposed_rank, full_rank, n_ijk, y_ijk, item_names):
    serialized_results = {}
    for method_name in BOOTSTRAP_METHOD_LABELS:
        start_time = time.perf_counter()
        print(f"[{method_name}] sample {sample_idx} start", flush=True)
        model_rank = None
        try:
            if method_name == "proposed":
                model_rank = proposed_rank
                fit = fit_proposed(N, K, proposed_rank, n_ijk, y_ijk, max_steps=40, tol=5e-5, tau=30.0)
            elif method_name == "proposed_full_rank":
                model_rank = full_rank
                fit = fit_proposed(N, K, full_rank, n_ijk, y_ijk, max_steps=40, tol=5e-5, tau=30.0)
            elif method_name == "zhou_github":
                fit = fit_zhou_github(N, K, n_ijk, y_ijk)
            elif method_name == "standard_btl":
                fit = fit_standard_btl(N, K, n_ijk, y_ijk)
            else:
                raise ValueError(f"unknown bootstrap method '{method_name}'")
            elapsed = time.perf_counter() - start_time
            result = {
                "fit": fit,
                "error": None,
                "elapsed_seconds": elapsed,
                "skipped": False,
                "model_rank": None if model_rank is None else int(model_rank),
            }
            print(f"[{method_name}] sample {sample_idx} done in {elapsed:.2f}s", flush=True)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            result = {
                "fit": None,
                "error": str(exc),
                "elapsed_seconds": elapsed,
                "skipped": False,
                "model_rank": None if model_rank is None else int(model_rank),
            }
            print(f"[{method_name}] sample {sample_idx} failed after {elapsed:.2f}s: {exc}", flush=True)
        serialized_results[method_name] = serialize_method_result(result, [], item_names)
    return sample_idx, serialized_results


def compute_test_accuracy(fit, test_records):
    correct = 0
    total = 0
    for entry in test_records:
        pred = int(fit["S"][entry["k"], entry["i"]] > fit["S"][entry["k"], entry["j"]])
        correct += int(pred == entry["y"])
        total += 1
    return float(correct / total) if total > 0 else None


def compute_record_log_loss(fit, records):
    score = np.asarray(fit["S"], dtype=float)
    loss = 0.0
    total = 0
    for entry in records:
        diff = score[entry["k"], entry["i"]] - score[entry["k"], entry["j"]]
        y = float(entry["y"])
        loss += y * np.logaddexp(0.0, -diff) + (1.0 - y) * np.logaddexp(0.0, diff)
        total += 1
    return float(loss / total) if total > 0 else None


def compute_record_accuracy(fit, records, skip_ties=True):
    score = np.asarray(fit["S"], dtype=float)
    correct = 0
    total = 0
    for entry in records:
        y = float(entry["y"])
        if skip_ties and y == 0.5:
            continue
        pred = int(score[entry["k"], entry["i"]] > score[entry["k"], entry["j"]])
        correct += int(pred == int(y))
        total += 1
    return float(correct / total) if total > 0 else None



def serialize_method_result(fit_result, test_records, item_names):
    if fit_result["fit"] is None:
        return {
            "error": fit_result["error"],
            "elapsed_seconds": fit_result.get("elapsed_seconds"),
            "model_rank": fit_result.get("model_rank"),
            "skipped": bool(fit_result.get("skipped", False)),
        }
    fit = fit_result["fit"]
    ranking = rank_items_from_mu(fit["mu"], item_names)
    return {
        "test_accuracy": compute_test_accuracy(fit, test_records),
        "ranking": ranking,
        "mu": [float(x) for x in fit["mu"]],
        "gamma": [float(x) for x in fit["gamma"]],
        "U": np.asarray(fit["U"], dtype=float).tolist(),
        "V": np.asarray(fit["V"], dtype=float).tolist(),
        "uq": fit.get("uq"),
        "fit_info": fit["fit_info"],
        "elapsed_seconds": fit_result.get("elapsed_seconds"),
        "model_rank": fit_result.get("model_rank"),
        "skipped": bool(fit_result.get("skipped", False)),
        "error": None,
    }



def plot_proposed_uvt_heatmap(dataset_name, judge_names, item_names, fit, output_dir=None, sort_rows=True, sort_cols=True):
    U = np.asarray(fit["U"], dtype=float)
    V = np.asarray(fit["V"], dtype=float)
    heatmap_dir = output_dir if output_dir is not None else ensure_proposed_heatmap_dir()
    output_path = os.path.join(heatmap_dir, f"{dataset_name}.png")

    if U.ndim != 2 or V.ndim != 2 or U.shape[1] == 0 or V.shape[1] == 0:
        heterogeneity = np.zeros((len(judge_names), len(item_names)), dtype=float)
    else:
        heterogeneity = U @ V.T

    # Sort rows / columns by their own mean heterogeneity (warm → top-left)
    item_order = list(range(len(item_names)))
    judge_order = list(range(len(judge_names)))
    item_labels = list(item_names)
    judge_labels = list(judge_names)

    if sort_cols and heterogeneity.size > 0:
        col_means = heterogeneity.mean(axis=0)
        item_order = sorted(range(len(item_names)), key=lambda idx: float(col_means[idx]), reverse=True)
        item_labels = [item_names[idx] for idx in item_order]
        heterogeneity = heterogeneity[:, item_order]

    if sort_rows and heterogeneity.size > 0:
        row_means = heterogeneity.mean(axis=1)
        judge_order = sorted(range(len(judge_names)), key=lambda idx: float(row_means[idx]), reverse=True)
        judge_labels = [judge_names[idx] for idx in judge_order]
        heterogeneity = heterogeneity[judge_order, :]

    absmax = float(np.max(np.abs(heterogeneity))) if heterogeneity.size else 0.0
    if absmax <= 0.0:
        absmax = 1.0

    plt.figure(figsize=(max(8, 0.35 * len(item_names)), max(4, 0.35 * len(judge_names))))
    plt.imshow(heterogeneity, aspect="auto", cmap="coolwarm", vmin=-absmax, vmax=absmax)
    plt.colorbar(label="U @ V.T")
    plt.xticks(np.arange(len(item_labels)), item_labels, rotation=90, fontsize=7)
    plt.yticks(np.arange(len(judge_labels)), judge_labels, fontsize=8)
    plt.xlabel("Items")
    plt.ylabel("Judges")
    plt.title(f"{dataset_name} Proposed heterogeneity heatmap (U @ V.T)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path



def plot_real_data_accuracy(stability_summary, output_path=None, title_suffix=None):
    dataset_names = list(stability_summary.keys())
    method_names = list(METHOD_LABELS.keys())
    x = np.arange(len(dataset_names))
    width = 0.25

    plt.figure(figsize=(12, 5))
    for offset, method_name in enumerate(method_names):
        values = []
        for dataset_name in dataset_names:
            method_result = stability_summary[dataset_name]["methods"].get(method_name, {})
            values.append(method_result.get("test_accuracy") if method_result.get("error") is None else np.nan)
        plt.bar(x + (offset - 1) * width, values, width=width, label=METHOD_LABELS[method_name])
    plt.xticks(x, dataset_names, rotation=15)
    plt.ylabel("Test accuracy")
    title = "Real-data stability experiment"
    if title_suffix is not None:
        title = f"{title} {title_suffix}"
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    save_path = output_path if output_path is not None else os.path.join(ensure_results_dir(), "stability_accuracy.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


def normalize_family_name(name):
    normalized = str(name).strip().lower()
    if "/" in normalized:
        normalized = normalized.split("/", 1)[0]
    elif "-" in normalized:
        normalized = normalized.split("-", 1)[0]
    elif "_" in normalized:
        normalized = normalized.split("_", 1)[0]
    return normalized.replace(".", "_")


def group_names_by_family(names):
    groups = {}
    for name in names:
        family = normalize_family_name(name)
        groups.setdefault(family, []).append(name)
    return dict(sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])))


def clone_record_with_judge(record, judge_name, preferred_label, confidence=1.0):
    cloned = dict(record)
    cloned["judge_model"] = judge_name
    cloned["judge_preferred_model"] = preferred_label
    cloned["judge_confidence"] = confidence
    return cloned


def label_for_preferred_item(record, preferred_item_name):
    if record.get("model_a") == preferred_item_name:
        return "a"
    if record.get("model_b") == preferred_item_name:
        return "b"
    return "c"


def select_structured_bias_source_records(base_records, injection_unit, random_seed=42):
    if injection_unit == "all_records":
        return list(base_records)
    valid_records = [
        record for record in base_records
        if record.get("model_a") is not None
        and record.get("model_b") is not None
        and record.get("model_a") != record.get("model_b")
    ]
    if injection_unit == "matched_judge":
        counts = {}
        for record in valid_records:
            judge_name = record.get("judge_model")
            counts[judge_name] = counts.get(judge_name, 0) + 1
        target_count = int(round(float(np.median(list(counts.values()))))) if counts else 0
        if target_count <= 0:
            return []
        rng = np.random.default_rng(random_seed)
        sample_size = min(target_count, len(valid_records))
        indices = rng.choice(len(valid_records), size=sample_size, replace=False)
        return [valid_records[int(idx)] for idx in indices]
    if injection_unit != "question_pair":
        raise ValueError(f"unknown structured-bias injection unit '{injection_unit}'")

    selected = {}
    for record in valid_records:
        model_a = record.get("model_a")
        model_b = record.get("model_b")
        if model_a is None or model_b is None or model_a == model_b:
            continue
        pair_key = tuple(sorted((model_a, model_b)))
        question_key = json.dumps(record.get("question_id"), sort_keys=True)
        key = (question_key, pair_key)
        if key not in selected:
            selected[key] = record
    return list(selected.values())


def build_structured_bias_records(base_records, item_names, item_scores, variant, random_seed=42, injection_unit="all_records"):
    rng = np.random.default_rng(random_seed)
    source_records = select_structured_bias_source_records(base_records, injection_unit, random_seed=random_seed)
    item_scores = {name: float(score) for name, score in item_scores.items()}
    item_families = {name: normalize_family_name(name) for name in item_names}
    family_counts = {}
    for family in item_families.values():
        family_counts[family] = family_counts.get(family, 0) + 1
    target_family = max(family_counts, key=lambda family: (family_counts[family], family)) if family_counts else None
    sorted_items = sorted(item_names, key=lambda name: item_scores.get(name, 0.0), reverse=True)
    cluster_size = max(1, len(sorted_items) // 3)
    favored_cluster = set(sorted_items[-cluster_size:])

    biased_records = []
    for record in source_records:
        model_a = record.get("model_a")
        model_b = record.get("model_b")
        if model_a not in item_scores or model_b not in item_scores or model_a == model_b:
            continue

        if variant == "anti_consensus":
            preferred_item = model_a if item_scores[model_a] < item_scores[model_b] else model_b
        elif variant == "family_bias":
            a_matches = item_families.get(model_a) == target_family
            b_matches = item_families.get(model_b) == target_family
            if a_matches != b_matches:
                preferred_item = model_a if a_matches else model_b
            else:
                preferred_item = model_a if item_scores[model_a] > item_scores[model_b] else model_b
        elif variant == "cluster_bias":
            a_matches = model_a in favored_cluster
            b_matches = model_b in favored_cluster
            if a_matches != b_matches:
                preferred_item = model_a if a_matches else model_b
            else:
                preferred_item = model_a if int(rng.binomial(1, 0.5)) == 1 else model_b
        else:
            raise ValueError(f"unknown structured bias variant '{variant}'")

        biased_records.append(
            clone_record_with_judge(
                record,
                f"structured_{variant}_judge",
                label_for_preferred_item(record, preferred_item),
                confidence=1.0,
            )
        )

    return biased_records, {
        "variant": variant,
        "injection_unit": injection_unit,
        "source_records": len(source_records),
        "target_family": target_family,
        "favored_cluster": sorted(favored_cluster),
    }


def summarize_rank_shift(reference_serialized, current_serialized):
    if reference_serialized.get("error") is not None or current_serialized.get("error") is not None:
        return None
    try:
        return compute_rank_shift(reference_serialized["ranking"], current_serialized["ranking"])
    except Exception as exc:
        return {"error": str(exc)}


def fit_methods_for_records(
    records,
    proposed_rank,
    max_iter=2000,
    proposed_max_steps=120,
    tau=40.0,
    proposed_inner_maxiter=500,
):
    dataset = build_real_dataset_from_records(records)
    N = len(dataset["item_names"])
    K = len(dataset["judge_names"])
    rank_for_fit = min(int(proposed_rank), infer_full_model_rank(N, K))
    n_ijk, y_ijk = processed_records_to_aggregated(dataset["processed"], N, K)
    fit_results = fit_all_methods_safe(
        N,
        K,
        rank_for_fit,
        n_ijk,
        y_ijk,
        max_iter=max_iter,
        proposed_max_steps=proposed_max_steps,
        tau=tau,
        proposed_inner_maxiter=proposed_inner_maxiter,
    )
    method_summary = {
        method_name: serialize_method_result(fit_result, [], dataset["item_names"])
        for method_name, fit_result in fit_results.items()
    }
    return dataset, rank_for_fit, method_summary



def build_noisy_summary_path(dataset_name):
    if dataset_name is None:
        raise ValueError("dataset_name is required for noisy summary path")
    return os.path.join(
        ensure_noisy_summary_dir(),
        f"noisy_judge_summary_{dataset_name}.json",
    )



def write_noisy_summary(noisy_summary, dataset_name):
    summary_path = build_noisy_summary_path(dataset_name)
    dataset_only_summary = {dataset_name: noisy_summary[dataset_name]} if dataset_name in noisy_summary else noisy_summary
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(dataset_only_summary, f, indent=2)
    return {"summary_path": summary_path}



def plot_noisy_rank_shift(noisy_summary, max_step=None, output_path=None):
    dataset_names = list(noisy_summary.keys())
    if not dataset_names:
        return None

    method_names = list(METHOD_LABELS.keys())
    method_plot_styles = {
        "proposed": {"marker": "o", "linestyle": "-", "zorder": 4},
        "zhou_github": {"marker": "s", "linestyle": "-", "zorder": 3},
        "standard_btl": {"marker": "^", "linestyle": "--", "zorder": 2, "alpha": 0.8},
    }
    n_panels = len(dataset_names)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, dataset_name in zip(axes_flat, dataset_names):
        dataset_summary = noisy_summary[dataset_name]
        for method_name in method_names:
            steps = []
            values = []
            for step in dataset_summary["steps"]:
                if max_step is not None and step["step"] > max_step:
                    continue
                method_result = step["methods"].get(method_name, {})
                shift = method_result.get("rank_shift")
                if shift is None or method_result.get("error") is not None:
                    continue
                item_rank_shifts = shift.get("rank_shift", {})
                exact_match_rate = (
                    sum(1 for item_shift in item_rank_shifts.values() if item_shift == 0)
                    / len(item_rank_shifts)
                    if item_rank_shifts
                    else 1.0
                )
                steps.append(step["step"])
                values.append(exact_match_rate)
            if steps:
                ax.plot(
                    steps,
                    values,
                    label=METHOD_LABELS[method_name],
                    **method_plot_styles.get(method_name, {}),
                )
        ax.set_xlabel("# noisy judges")
        ax.set_ylabel("Exact rank-match rate vs base")
        title_suffix = "" if max_step is None else f" (through step {max_step})"
        ax.set_title(f"{dataset_name}{title_suffix}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.tight_layout()
    target_path = output_path or os.path.join(ensure_results_dir(), "noisy_rank_shift.png")
    fig.savefig(target_path, dpi=150)
    plt.close(fig)
    return target_path



def run_real_data_stability_experiment(test_ratio=0.2, random_seed=42, dataset_name=None):
    output_path = os.path.join(ensure_results_dir(), "stability_summary.json")
    merged_summary = {}
    if dataset_name is not None and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            merged_summary = json.load(f)

    for current_dataset_name, dataset_path in select_real_data_files(dataset_name).items():
        print(f"[dataset={current_dataset_name}] loading stability dataset", flush=True)
        dataset_start_time = time.perf_counter()
        dataset = load_real_dataset(dataset_path)
        train_records, test_records = split_real_dataset(dataset["processed"], test_ratio=test_ratio, random_seed=random_seed)
        N = len(dataset["item_names"])
        K = len(dataset["judge_names"])
        print(f"[dataset={current_dataset_name}] selecting proposed rank by 5-fold CV on training split", flush=True)
        r_model, rank_selection = select_rank_by_cross_validation(
            N,
            K,
            train_records,
            n_folds=5,
            random_seed=random_seed,
        )
        n_ijk, y_ijk = processed_records_to_aggregated(train_records, N, K)
        fit_results = fit_all_methods_safe(N, K, r_model, n_ijk, y_ijk, max_iter=2000)
        method_summary = {
            method_name: serialize_method_result(fit_result, test_records, dataset["item_names"])
            for method_name, fit_result in fit_results.items()
        }
        proposed_heatmap_path = None
        proposed_fit_result = fit_results.get("proposed")
        if proposed_fit_result is not None and proposed_fit_result["fit"] is not None:
            proposed_heatmap_path = plot_proposed_uvt_heatmap(
                current_dataset_name,
                dataset["judge_names"],
                dataset["item_names"],
                proposed_fit_result["fit"],
            )
        merged_summary[current_dataset_name] = {
            "dataset_path": dataset_path,
            "num_items": N,
            "num_judges": K,
            "train_size": len(train_records),
            "test_size": len(test_records),
            "rank_selection": rank_selection,
            "elapsed_seconds": time.perf_counter() - dataset_start_time,
            "proposed_uvt_heatmap_path": proposed_heatmap_path,
            **dataset["summary"],
            "methods": method_summary,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_summary, f, indent=2)
        plot_real_data_accuracy(merged_summary)
        print(f"[dataset={current_dataset_name}] stability done in {merged_summary[current_dataset_name]['elapsed_seconds']:.2f}s", flush=True)

    return merged_summary



def run_stability_excluding_judge(test_ratio=0.2, random_seed=42, dataset_name=None, exclude_judges=None):
    """Run stability experiment with specified judges excluded from the data.

    Parameters
    ----------
    exclude_judges : list of str, optional
        Judge names to exclude. Default: ['zai-org/GLM-4.5-Air-FP8']
    """
    if exclude_judges is None:
        exclude_judges = ["zai-org/GLM-4.5-Air-FP8"]
    exclude_set = set(exclude_judges)

    output_dir = ensure_stability_excluding_dir()
    output_path = os.path.join(output_dir, "stability_summary.json")
    merged_summary = {}
    if dataset_name is not None and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            merged_summary = json.load(f)

    for current_dataset_name, dataset_path in select_real_data_files(dataset_name).items():
        print(f"[dataset={current_dataset_name}] loading dataset for stability (excluding judges: {exclude_judges})", flush=True)
        dataset_start_time = time.perf_counter()
        base_dataset = load_real_dataset(dataset_path)

        # Filter out records from excluded judges
        retained_records = [
            record for record in base_dataset["records"]
            if record.get("judge_model") not in exclude_set
        ]
        removed_count = len(base_dataset["records"]) - len(retained_records)
        print(f"[dataset={current_dataset_name}] removed {removed_count} records from excluded judges (out of {len(base_dataset['records'])})", flush=True)

        if not retained_records:
            print(f"[dataset={current_dataset_name}] no records left after filtering, skipping", flush=True)
            continue

        # Rebuild dataset from filtered records
        filtered_dataset = build_real_dataset_from_records(retained_records)
        N = len(filtered_dataset["item_names"])
        K = len(filtered_dataset["judge_names"])
        print(f"[dataset={current_dataset_name}] after filtering: N={N}, K={K}, retained_judges={filtered_dataset['judge_names']}", flush=True)

        # Train/test split
        train_records, test_records = split_real_dataset(
            filtered_dataset["processed"],
            test_ratio=test_ratio,
            random_seed=random_seed,
        )

        print(f"[dataset={current_dataset_name}] selecting proposed rank by 5-fold CV on training split", flush=True)
        r_model, rank_selection = select_rank_by_cross_validation(
            N,
            K,
            train_records,
            n_folds=5,
            random_seed=random_seed,
        )

        n_ijk, y_ijk = processed_records_to_aggregated(train_records, N, K)
        fit_results = fit_all_methods_safe(N, K, r_model, n_ijk, y_ijk, max_iter=100)
        method_summary = {
            method_name: serialize_method_result(fit_result, test_records, filtered_dataset["item_names"])
            for method_name, fit_result in fit_results.items()
        }

        # Generate heatmap
        proposed_heatmap_path = None
        proposed_fit_result = fit_results.get("proposed")
        if proposed_fit_result is not None and proposed_fit_result["fit"] is not None:
            proposed_heatmap_path = plot_proposed_uvt_heatmap(
                current_dataset_name,
                filtered_dataset["judge_names"],
                filtered_dataset["item_names"],
                proposed_fit_result["fit"],
                output_dir=ensure_stability_excluding_heatmap_dir(),
            )

        merged_summary[current_dataset_name] = {
            "dataset_path": dataset_path,
            "excluded_judges": exclude_judges,
            "original_num_judges": len(base_dataset["judge_names"]),
            "retained_num_judges": K,
            "retained_judge_names": filtered_dataset["judge_names"],
            "num_items": N,
            "train_size": len(train_records),
            "test_size": len(test_records),
            "rank_selection": rank_selection,
            "elapsed_seconds": time.perf_counter() - dataset_start_time,
            "proposed_uvt_heatmap_path": proposed_heatmap_path,
            **filtered_dataset["summary"],
            "methods": method_summary,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_summary, f, indent=2)

        plot_real_data_accuracy(
            merged_summary,
            output_path=os.path.join(output_dir, "stability_accuracy.png"),
            title_suffix="(excluding zai-org/GLM-4.5-Air-FP8)",
        )
        print(f"[dataset={current_dataset_name}] stability (excluding judges) done in {merged_summary[current_dataset_name]['elapsed_seconds']:.2f}s", flush=True)

    return merged_summary



def build_bootstrap_summary_paths(dataset_name=None, bootstrap_samples=None):
    canonical_path = os.path.join(ensure_results_dir(), "bootstrap_summary.json")
    archive_path = None
    if dataset_name is not None and bootstrap_samples is not None:
        archive_path = os.path.join(
            ensure_bootstrap_summary_archive_dir(),
            f"bootstrap_summary_{dataset_name}_samples_{int(bootstrap_samples)}.json",
        )
    return canonical_path, archive_path



def write_bootstrap_summary(bootstrap_summary, dataset_name=None, bootstrap_samples=None):
    canonical_path, archive_path = build_bootstrap_summary_paths(dataset_name=dataset_name, bootstrap_samples=bootstrap_samples)
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(bootstrap_summary, f, indent=2)
    if archive_path is not None:
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(bootstrap_summary, f, indent=2)
    return {"canonical_path": canonical_path, "archive_path": archive_path}



def plot_bootstrap_top_k_curve(dataset_name, methods_summary, max_top_k, bootstrap_samples):
    output_path = os.path.join(
        ensure_bootstrap_topk_curve_dir(),
        f"{dataset_name}_top_{int(max_top_k)}_samples_{int(bootstrap_samples)}_exact_match_curve.png",
    )

    plt.figure(figsize=(8, 5))
    for method_name in BOOTSTRAP_METHOD_LABELS:
        method_result = methods_summary.get(method_name, {})
        curve = method_result.get("top_k_stability_curve") or []
        if not curve:
            continue
        x = [entry["top_k"] for entry in curve]
        y = [entry["exact_match_rate_vs_baseline_top_k"] for entry in curve]
        plt.plot(x, y, marker="o", label=BOOTSTRAP_METHOD_LABELS[method_name])

    plt.xlabel("top_k")
    plt.ylabel("Exact-match rate vs baseline top-k")
    plt.title(f"{dataset_name} bootstrap top-k stability")
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path



def plot_bootstrap_top_k_stability_summary(bootstrap_summary, top_k, bootstrap_samples):
    dataset_names = [
        dataset_name
        for dataset_name, dataset_summary in bootstrap_summary.items()
        if int(dataset_summary.get("bootstrap_samples", 0)) == int(bootstrap_samples)
    ]
    if not dataset_names:
        return None

    top_k_key = f"top_{int(top_k)}"
    output_path = os.path.join(
        ensure_results_dir(),
        f"bootstrap_topk_stability_top_{int(top_k)}_samples_{int(bootstrap_samples)}.png",
    )
    x = np.arange(len(dataset_names))
    method_names = list(BOOTSTRAP_METHOD_LABELS.keys())
    width = 0.8 / len(method_names)

    plt.figure(figsize=(12, 5))
    for offset, method_name in enumerate(method_names):
        values = []
        for dataset_name in dataset_names:
            method_result = bootstrap_summary[dataset_name]["methods"].get(method_name, {})
            summary = method_result.get("top_k_summaries", {}).get(top_k_key, {})
            stability = summary.get("top_k_stability")
            values.append(stability.get("exact_match_rate_vs_baseline_top_k") if stability is not None else np.nan)
        plt.bar(
            x + (offset - (len(method_names) - 1) / 2.0) * width,
            values,
            width=width,
            label=BOOTSTRAP_METHOD_LABELS[method_name],
        )

    plt.xticks(x, dataset_names, rotation=15)
    plt.ylabel("Exact-match rate vs baseline top-k")
    plt.title(f"Bootstrap top-k stability (top-{int(top_k)}, samples={int(bootstrap_samples)})")
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path



def plot_bootstrap_rank_distribution(dataset_name, method_name, rank_distribution, top_k, bootstrap_samples):
    if not rank_distribution:
        return None
    item_names = list(rank_distribution.keys())
    median_ranks = [rank_distribution[item_name]["median_rank"] for item_name in item_names]
    low_err = [max(0.0, rank_distribution[item_name]["median_rank"] - rank_distribution[item_name]["rank_p05"]) for item_name in item_names]
    high_err = [max(0.0, rank_distribution[item_name]["rank_p95"] - rank_distribution[item_name]["median_rank"]) for item_name in item_names]
    output_path = os.path.join(
        ensure_bootstrap_rank_plot_dir(),
        f"{dataset_name}_{method_name}_top_{int(top_k)}_samples_{int(bootstrap_samples)}_rank_distribution.png",
    )

    plt.figure(figsize=(max(8, 0.45 * len(item_names)), 5))
    x = np.arange(len(item_names))
    plt.errorbar(x, median_ranks, yerr=[low_err, high_err], fmt="o", capsize=4)
    plt.xticks(x, item_names, rotation=90, fontsize=8)
    plt.ylabel("Bootstrap rank")
    plt.xlabel("Items")
    plt.title(f"{dataset_name} {BOOTSTRAP_METHOD_LABELS[method_name]} rank distribution (top-{int(top_k)})")
    plt.gca().invert_yaxis()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path



def build_bootstrap_top_k_summary(dataset_name, method_name, baseline_serialized, successful_rankings, item_names, top_k, bootstrap_samples):
    actual_top_k = max(1, min(int(top_k), len(item_names)))
    rank_distribution = summarize_rank_distribution(successful_rankings, item_names, actual_top_k)
    top_k_stability = compute_top_k_stability(
        successful_rankings,
        baseline_serialized["ranking"]["ranking"],
        actual_top_k,
        exact_match=True,
    )
    return {
        "top_k": actual_top_k,
        "rank_distribution": rank_distribution,
        "top_k_stability": top_k_stability,
        "rank_distribution_plot": plot_bootstrap_rank_distribution(
            dataset_name,
            method_name,
            rank_distribution,
            actual_top_k,
            bootstrap_samples,
        ),
    }



def normalize_top_k_targets(num_items):
    requested_values = [3, 5]
    deduped = []
    for value in requested_values:
        actual_value = max(1, min(int(value), int(num_items)))
        if actual_value not in deduped:
            deduped.append(actual_value)
    return deduped



def run_real_data_bootstrap_experiment(
    dataset_name=None,
    bootstrap_samples=100,
    bootstrap_seed=42,
    bootstrap_workers=None,
):
    bootstrap_summary = {}
    canonical_output_path, archive_output_path = build_bootstrap_summary_paths(dataset_name=dataset_name, bootstrap_samples=bootstrap_samples)
    preferred_input_path = archive_output_path if archive_output_path is not None and os.path.exists(archive_output_path) else canonical_output_path
    if dataset_name is not None and os.path.exists(preferred_input_path):
        with open(preferred_input_path, "r", encoding="utf-8") as f:
            bootstrap_summary = json.load(f)
        bootstrap_summary = {
            name: summary
            for name, summary in bootstrap_summary.items()
            if int(summary.get("bootstrap_samples", -1)) == int(bootstrap_samples)
        }
        if dataset_name is not None:
            bootstrap_summary = {
                name: summary
                for name, summary in bootstrap_summary.items()
                if name in select_real_data_files(dataset_name)
            }
        if not bootstrap_summary:
            bootstrap_summary = {}

    for current_dataset_name, dataset_path in select_real_data_files(dataset_name).items():
        print(f"[dataset={current_dataset_name}] loading bootstrap dataset", flush=True)
        dataset_start_time = time.perf_counter()
        dataset = load_real_dataset(dataset_path)
        N = len(dataset["item_names"])
        K = len(dataset["judge_names"])
        print(f"[dataset={current_dataset_name}] selecting proposed rank by 5-fold CV on full sample", flush=True)
        r_model, rank_selection = select_rank_by_cross_validation(
            N,
            K,
            dataset["processed"],
            n_folds=5,
            random_seed=bootstrap_seed,
        )
        full_rank = infer_full_model_rank(N, K)
        target_top_k_values = normalize_top_k_targets(N)

        print(f"[dataset={current_dataset_name}] fitting bootstrap baseline methods", flush=True)
        baseline_n_ijk, baseline_y_ijk = processed_records_to_aggregated(dataset["processed"], N, K)
        baseline_fit_results = fit_bootstrap_methods_safe(N, K, r_model, full_rank, baseline_n_ijk, baseline_y_ijk)

        baseline_method_summary = {
            method_name: serialize_method_result(fit_result, [], dataset["item_names"])
            for method_name, fit_result in baseline_fit_results.items()
        }

        bootstrap_rankings = {method_name: [] for method_name in BOOTSTRAP_METHOD_LABELS}
        bootstrap_errors = {method_name: [] for method_name in BOOTSTRAP_METHOD_LABELS}
        bootstrap_elapsed = {method_name: 0.0 for method_name in BOOTSTRAP_METHOD_LABELS}

        sample_data = []
        
        print(f"[dataset={current_dataset_name}] preparing {bootstrap_samples} bootstrap samples", flush=True)
        for sample_idx in range(bootstrap_samples):
            sampled_records = bootstrap_processed_records(dataset["processed"], random_seed=bootstrap_seed + sample_idx)
            n_ijk, y_ijk = processed_records_to_aggregated(sampled_records, N, K)
            sample_data.append((sample_idx, n_ijk, y_ijk))

        print(f"[dataset={current_dataset_name}] running {bootstrap_samples} bootstrap samples in parallel", flush=True)
        cpu_total = os.cpu_count() or 1
        selected_workers = infer_bootstrap_workers(bootstrap_workers)
        max_workers = min(selected_workers, bootstrap_samples)
        configure_single_threaded_numeric_libraries()
        if bootstrap_workers is None:
            print(
                f"[dataset={current_dataset_name}] bootstrap worker count = {max_workers} "
                f"(cpu_total={cpu_total}, target_fraction=0.75)",
                flush=True,
            )
        else:
            print(
                f"[dataset={current_dataset_name}] bootstrap worker count = {max_workers} "
                f"(cpu_total={cpu_total}, requested={int(bootstrap_workers)})",
                flush=True,
            )
        if max_workers == 1:
            sample_results = [
                fit_all_methods_on_sample(
                    sample_idx,
                    N,
                    K,
                    r_model,
                    full_rank,
                    n_ijk,
                    y_ijk,
                    dataset["item_names"],
                )
                for sample_idx, n_ijk, y_ijk in sample_data
            ]
            for sample_idx, serialized_results in sample_results:
                for method_name, serialized in serialized_results.items():
                    bootstrap_elapsed[method_name] += float(serialized.get("elapsed_seconds") or 0.0)
                    if serialized["error"] is None:
                        bootstrap_rankings[method_name].append(serialized["ranking"])
                    else:
                        bootstrap_errors[method_name].append(
                            {
                                "sample_index": sample_idx,
                                "error": serialized["error"],
                                "skipped": bool(serialized.get("skipped", False)),
                            }
                        )
        else:
            mp_context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
                future_to_task = {}
                for sample_idx, n_ijk, y_ijk in sample_data:
                    future = executor.submit(
                        fit_all_methods_on_sample,
                        sample_idx,
                        N,
                        K,
                        r_model,
                        full_rank,
                        n_ijk,
                        y_ijk,
                        dataset["item_names"],
                    )
                    future_to_task[future] = sample_idx

                for future in as_completed(future_to_task):
                    sample_idx, serialized_results = future.result()
                    for method_name, serialized in serialized_results.items():
                        bootstrap_elapsed[method_name] += float(serialized.get("elapsed_seconds") or 0.0)
                        if serialized["error"] is None:
                            bootstrap_rankings[method_name].append(serialized["ranking"])
                        else:
                            bootstrap_errors[method_name].append(
                                {
                                    "sample_index": sample_idx,
                                    "error": serialized["error"],
                                    "skipped": bool(serialized.get("skipped", False)),
                                }
                            )

        method_summary = {}
        for method_name in BOOTSTRAP_METHOD_LABELS:
            baseline_serialized = baseline_method_summary[method_name]
            successful_rankings = bootstrap_rankings[method_name]
            method_summary[method_name] = {
                "baseline": baseline_serialized,
                "bootstrap_samples": int(bootstrap_samples),
                "successful_samples": len(successful_rankings),
                "failed_samples": len(bootstrap_errors[method_name]),
                "elapsed_seconds": bootstrap_elapsed[method_name],
                "errors": bootstrap_errors[method_name],
                "bootstrap_rankings": successful_rankings,
            }
            if baseline_serialized["error"] is None and successful_rankings:
                top_k_summaries = {}
                for target_top_k in target_top_k_values:
                    top_k_key = f"top_{int(target_top_k)}"
                    top_k_summaries[top_k_key] = build_bootstrap_top_k_summary(
                        current_dataset_name,
                        method_name,
                        baseline_serialized,
                        successful_rankings,
                        dataset["item_names"],
                        target_top_k,
                        bootstrap_samples,
                    )
                method_summary[method_name]["top_k_summaries"] = top_k_summaries
                method_summary[method_name]["rank_distribution"] = {
                    top_k_key: summary["rank_distribution"]
                    for top_k_key, summary in top_k_summaries.items()
                }
                method_summary[method_name]["top_k_stability"] = {
                    top_k_key: summary["top_k_stability"]
                    for top_k_key, summary in top_k_summaries.items()
                }
                method_summary[method_name]["rank_distribution_plot"] = {
                    top_k_key: summary["rank_distribution_plot"]
                    for top_k_key, summary in top_k_summaries.items()
                }
            else:
                method_summary[method_name]["top_k_summaries"] = {}
                method_summary[method_name]["rank_distribution"] = {}
                method_summary[method_name]["top_k_stability"] = None
                method_summary[method_name]["rank_distribution_plot"] = None

        bootstrap_summary[current_dataset_name] = {
            "dataset_path": dataset_path,
            "num_items": N,
            "num_judges": K,
            "usable_records": len(dataset["processed"]),
            "bootstrap_samples": int(bootstrap_samples),
            "reported_top_k_values": [int(value) for value in target_top_k_values],
            "rank_selection": rank_selection,
            "proposed_rank": int(r_model),
            "proposed_full_rank": int(full_rank),
            "elapsed_seconds": time.perf_counter() - dataset_start_time,
            **dataset["summary"],
            "methods": method_summary,
        }
        written_paths = write_bootstrap_summary(
            bootstrap_summary,
            dataset_name=current_dataset_name,
            bootstrap_samples=bootstrap_samples,
        )
        bootstrap_summary[current_dataset_name]["bootstrap_summary_path"] = written_paths["canonical_path"]
        bootstrap_summary[current_dataset_name]["bootstrap_summary_archive_path"] = written_paths["archive_path"]
        write_bootstrap_summary(
            bootstrap_summary,
            dataset_name=current_dataset_name,
            bootstrap_samples=bootstrap_samples,
        )
        print(
            f"[dataset={current_dataset_name}] bootstrap done in {bootstrap_summary[current_dataset_name]['elapsed_seconds']:.2f}s",
            flush=True,
        )

    summary_plot_paths = {}
    for requested_top_k in normalize_top_k_targets(max(summary["num_items"] for summary in bootstrap_summary.values())):
        plot_path = plot_bootstrap_top_k_stability_summary(bootstrap_summary, requested_top_k, bootstrap_samples)
        if plot_path is not None:
            summary_plot_paths[f"top_{int(requested_top_k)}"] = plot_path
    if summary_plot_paths:
        for dataset_name_key, dataset_summary in bootstrap_summary.items():
            if int(dataset_summary.get("bootstrap_samples", 0)) == int(bootstrap_samples):
                dataset_summary["bootstrap_topk_stability_plots"] = summary_plot_paths
                write_bootstrap_summary(
                    bootstrap_summary,
                    dataset_name=dataset_name_key,
                    bootstrap_samples=bootstrap_samples,
                )

    return bootstrap_summary


def run_real_data_noisy_judge_experiment(random_seed=42, dataset_name=None, max_noisy_step=None, save_artifacts=True):
    noisy_summary = {}
    noisy_dataset_dir = ensure_noisy_dataset_dir() if save_artifacts else None
    noisy_judge_counts = select_noisy_judge_counts(max_noisy_step)
    for dataset_name, dataset_path in select_real_data_files(dataset_name).items():
        print(f"[dataset={dataset_name}] loading base dataset", flush=True)
        dataset_start_time = time.perf_counter()
        base_dataset = load_real_dataset(dataset_path)
        N = len(base_dataset["item_names"])
        base_K = len(base_dataset["judge_names"])
        print(f"[dataset={dataset_name}] selecting proposed rank by 5-fold CV on base sample", flush=True)
        r_model, rank_selection = select_rank_by_cross_validation(
            N,
            base_K,
            base_dataset["processed"],
            n_folds=5,
            random_seed=random_seed,
        )
        base_n_ijk, base_y_ijk = processed_records_to_aggregated(base_dataset["processed"], N, base_K)
        print(f"[dataset={dataset_name}] fitting base methods", flush=True)
        base_fit_results = fit_all_methods_safe(N, base_K, r_model, base_n_ijk, base_y_ijk)
        base_method_summary = {
            method_name: serialize_method_result(fit_result, [], base_dataset["item_names"])
            for method_name, fit_result in base_fit_results.items()
        }

        dataset_steps = []
        noisy_summary[dataset_name] = {
            "dataset_path": dataset_path,
            "base_summary": base_dataset["summary"],
            "base_judge_names": list(base_dataset["judge_names"]),
            "base_item_names": list(base_dataset["item_names"]),
            "rank_selection": rank_selection,
            "proposed_rank": int(r_model),
            "base_methods": base_method_summary,
            "steps": dataset_steps,
        }
        if save_artifacts:
            write_noisy_summary(noisy_summary, dataset_name=dataset_name)

        augmented_records = list(base_dataset["records"])
        skipped_methods = {}
        for step in noisy_judge_counts:
            print(f"[dataset={dataset_name}] step {step}/{len(noisy_judge_counts)}: generating noisy judges", flush=True)
            step_start_time = time.perf_counter()
            augmented_records.extend(
                make_noisy_judge_records(
                    base_dataset["records"],
                    f"noisy_judge_{step}",
                    random_seed=random_seed + step,
                )
            )
            save_path = None
            if save_artifacts:
                save_path = os.path.join(noisy_dataset_dir, f"{dataset_name}_plus_{step}_noisy_judges.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(augmented_records, f, indent=2)

            augmented_dataset = build_real_dataset_from_records(augmented_records)
            K_aug = len(augmented_dataset["judge_names"])
            print(
                f"[dataset={dataset_name}] step {step}: fitting methods with K={K_aug}, proposed rank fixed at r={r_model}",
                flush=True,
            )
            n_ijk, y_ijk = processed_records_to_aggregated(augmented_dataset["processed"], N, K_aug)
            fit_results = fit_all_methods_safe(N, K_aug, r_model, n_ijk, y_ijk, skipped_methods=skipped_methods)

            method_summary = {}
            for method_name, fit_result in fit_results.items():
                serialized = serialize_method_result(fit_result, [], augmented_dataset["item_names"])
                if (
                    method_name == "proposed"
                    and not serialized.get("skipped", False)
                    and serialized["error"] is not None
                    and "failed to converge" in serialized["error"].lower()
                ):
                    skipped_methods[method_name] = "skipped after earlier noisy-step non-convergence"
                    print(f"[{method_name}] will be skipped for later noisy steps", flush=True)
                if serialized["error"] is None and base_method_summary[method_name]["error"] is None:
                    serialized["rank_shift"] = compute_rank_shift(base_method_summary[method_name]["ranking"], serialized["ranking"])
                    serialized["significant_change"] = bool(serialized["rank_shift"]["spearman"] < 0.95 or serialized["rank_shift"]["top_1_changed"])
                else:
                    serialized["rank_shift"] = None
                    serialized["significant_change"] = None
                method_summary[method_name] = serialized

            step_elapsed = time.perf_counter() - step_start_time
            dataset_steps.append(
                {
                    "step": step,
                    "saved_dataset_path": save_path,
                    "num_judges": K_aug,
                    "judge_names": list(augmented_dataset["judge_names"]),
                    "item_names": list(augmented_dataset["item_names"]),
                    "elapsed_seconds": step_elapsed,
                    "methods": method_summary,
                }
            )
            if save_artifacts:
                write_noisy_summary(noisy_summary, dataset_name=dataset_name)
            print(f"[dataset={dataset_name}] step {step} done in {step_elapsed:.2f}s", flush=True)

        if save_artifacts:
            plot_noisy_rank_shift(noisy_summary)
        dataset_elapsed = time.perf_counter() - dataset_start_time
        print(f"[dataset={dataset_name}] dataset done in {dataset_elapsed:.2f}s", flush=True)

    if save_artifacts:
        plot_noisy_rank_shift(noisy_summary)
    return noisy_summary


def structured_bias_summary_path(injection_unit):
    if injection_unit == "all_records":
        return os.path.join(ensure_results_dir(), "structured_bias_summary.json")
    return os.path.join(ensure_results_dir(), f"structured_bias_summary_{injection_unit}.json")


def run_structured_bias_injection_experiment(
    random_seed=42,
    dataset_name=None,
    injection_unit="question_pair",
    variants=None,
    output_path=None,
    refit_cv=False,
):
    if injection_unit not in STRUCTURED_BIAS_UNITS:
        raise ValueError(f"unknown structured-bias injection unit '{injection_unit}'")
    variants = STRUCTURED_BIAS_VARIANTS if variants is None else tuple(variants)
    output_path = structured_bias_summary_path(injection_unit) if output_path is None else output_path
    structured_dataset_dir = ensure_structured_bias_dataset_dir()
    summary = {}
    if dataset_name is not None and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    for current_dataset_name, dataset_path in select_real_data_files(dataset_name).items():
        print(f"[dataset={current_dataset_name}] loading structured-bias dataset", flush=True)
        dataset_start_time = time.perf_counter()
        base_dataset = load_real_dataset(dataset_path)
        N = len(base_dataset["item_names"])
        K = len(base_dataset["judge_names"])

        print(f"[dataset={current_dataset_name}] selecting proposed rank by 5-fold CV on base sample", flush=True)
        r_model, rank_selection = select_rank_by_cross_validation(
            N,
            K,
            base_dataset["processed"],
            n_folds=5,
            random_seed=random_seed,
        )
        base_n_ijk, base_y_ijk = processed_records_to_aggregated(base_dataset["processed"], N, K)
        print(f"[dataset={current_dataset_name}] fitting base methods", flush=True)
        base_fit_results = fit_all_methods_safe(N, K, r_model, base_n_ijk, base_y_ijk, max_iter=2000)
        base_methods = {
            method_name: serialize_method_result(fit_result, [], base_dataset["item_names"])
            for method_name, fit_result in base_fit_results.items()
        }

        if base_methods["proposed"]["error"] is None:
            item_scores = base_methods["proposed"]["ranking"]["mu"]
        else:
            item_scores = base_methods["standard_btl"]["ranking"]["mu"]

        variant_summaries = []
        for variant_idx, variant in enumerate(variants):
            print(f"[dataset={current_dataset_name}] structured bias variant={variant} unit={injection_unit}", flush=True)
            variant_start_time = time.perf_counter()
            biased_records, bias_metadata = build_structured_bias_records(
                base_dataset["records"],
                base_dataset["item_names"],
                item_scores,
                variant,
                random_seed=random_seed + variant_idx,
                injection_unit=injection_unit,
            )
            augmented_records = list(base_dataset["records"]) + biased_records
            save_path = os.path.join(structured_dataset_dir, f"{current_dataset_name}_{variant}_{injection_unit}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(augmented_records, f, indent=2)

            augmented_dataset = build_real_dataset_from_records(augmented_records)
            rank_for_refit = r_model
            refit_rank_selection = None
            if refit_cv:
                print(
                    f"[dataset={current_dataset_name}] structured bias variant={variant}: selecting proposed rank by CV after injection",
                    flush=True,
                )
                rank_for_refit, refit_rank_selection = select_rank_by_cross_validation(
                    len(augmented_dataset["item_names"]),
                    len(augmented_dataset["judge_names"]),
                    augmented_dataset["processed"],
                    n_folds=5,
                    random_seed=random_seed,
                )

            augmented_dataset, rank_for_fit, method_summary = fit_methods_for_records(
                augmented_records,
                rank_for_refit,
                max_iter=2000,
                proposed_max_steps=300,
                proposed_inner_maxiter=2000,
            )
            for method_name, serialized in method_summary.items():
                serialized["rank_shift"] = summarize_rank_shift(base_methods[method_name], serialized)
                serialized["significant_change"] = (
                    None if serialized["rank_shift"] is None or serialized["rank_shift"].get("error") is not None
                    else bool(serialized["rank_shift"]["spearman"] < 0.95 or serialized["rank_shift"]["top_1_changed"])
                )

            variant_summaries.append(
                {
                    "variant": variant,
                    "metadata": bias_metadata,
                    "saved_dataset_path": save_path,
                    "added_records": len(biased_records),
                    "num_judges": len(augmented_dataset["judge_names"]),
                    "proposed_rank_used": int(rank_for_fit),
                    "refit_rank_selection": refit_rank_selection,
                    "elapsed_seconds": time.perf_counter() - variant_start_time,
                    "methods": method_summary,
                }
            )

        summary[current_dataset_name] = {
            "dataset_path": dataset_path,
            "base_summary": base_dataset["summary"],
            "base_judge_names": list(base_dataset["judge_names"]),
            "base_item_names": list(base_dataset["item_names"]),
            "rank_selection": rank_selection,
            "proposed_rank": int(r_model),
            "injection_unit": injection_unit,
            "refit_cv": bool(refit_cv),
            "base_methods": base_methods,
            "variants": variant_summaries,
            "elapsed_seconds": time.perf_counter() - dataset_start_time,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[dataset={current_dataset_name}] structured bias done", flush=True)

    return summary


def run_leave_one_family_out_experiment(random_seed=42, dataset_name=None, min_family_judges=1):
    output_path = os.path.join(ensure_results_dir(), "leave_one_family_out_summary.json")
    summary = {}

    for current_dataset_name, dataset_path in select_real_data_files(dataset_name).items():
        print(f"[dataset={current_dataset_name}] loading family-ablation dataset", flush=True)
        dataset_start_time = time.perf_counter()
        base_dataset = load_real_dataset(dataset_path)
        N = len(base_dataset["item_names"])
        K = len(base_dataset["judge_names"])
        judge_families = group_names_by_family(base_dataset["judge_names"])

        print(f"[dataset={current_dataset_name}] selecting proposed rank by 5-fold CV on base sample", flush=True)
        r_model, rank_selection = select_rank_by_cross_validation(
            N,
            K,
            base_dataset["processed"],
            n_folds=5,
            random_seed=random_seed,
        )
        base_n_ijk, base_y_ijk = processed_records_to_aggregated(base_dataset["processed"], N, K)
        print(f"[dataset={current_dataset_name}] fitting base methods", flush=True)
        base_fit_results = fit_all_methods_safe(N, K, r_model, base_n_ijk, base_y_ijk, max_iter=2000)
        base_methods = {
            method_name: serialize_method_result(fit_result, [], base_dataset["item_names"])
            for method_name, fit_result in base_fit_results.items()
        }

        family_summaries = []
        for family_name, family_judges in judge_families.items():
            if len(family_judges) < min_family_judges:
                continue
            print(
                f"[dataset={current_dataset_name}] leave out family={family_name} judges={len(family_judges)}",
                flush=True,
            )
            ablation_start_time = time.perf_counter()
            family_judge_set = set(family_judges)
            retained_records = [
                record for record in base_dataset["records"]
                if record.get("judge_model") not in family_judge_set
            ]
            if not retained_records:
                continue

            ablated_dataset, rank_for_fit, method_summary = fit_methods_for_records(
                retained_records,
                r_model,
                max_iter=2000,
            )
            for method_name, serialized in method_summary.items():
                serialized["rank_shift"] = summarize_rank_shift(base_methods[method_name], serialized)
                serialized["significant_change"] = (
                    None if serialized["rank_shift"] is None or serialized["rank_shift"].get("error") is not None
                    else bool(serialized["rank_shift"]["spearman"] < 0.95 or serialized["rank_shift"]["top_1_changed"])
                )

            family_summaries.append(
                {
                    "family": family_name,
                    "removed_judges": list(family_judges),
                    "removed_judge_count": len(family_judges),
                    "retained_judge_count": len(ablated_dataset["judge_names"]),
                    "retained_record_count": len(ablated_dataset["processed"]),
                    "proposed_rank_used": int(rank_for_fit),
                    "elapsed_seconds": time.perf_counter() - ablation_start_time,
                    "methods": method_summary,
                }
            )

        summary[current_dataset_name] = {
            "dataset_path": dataset_path,
            "base_summary": base_dataset["summary"],
            "judge_families": judge_families,
            "rank_selection": rank_selection,
            "proposed_rank": int(r_model),
            "base_methods": base_methods,
            "families": family_summaries,
            "elapsed_seconds": time.perf_counter() - dataset_start_time,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[dataset={current_dataset_name}] family ablation done", flush=True)

    return summary


def select_near_tie_pairs(train_records, min_pair_records=20, max_pairs=20):
    pair_stats = {}
    for entry in train_records:
        key = (entry["i"], entry["j"])
        stats = pair_stats.setdefault(key, {"n": 0, "y_sum": 0.0, "judges": set()})
        stats["n"] += 1
        stats["y_sum"] += float(entry["y"])
        stats["judges"].add(entry["k"])

    candidates = []
    for (i, j), stats in pair_stats.items():
        if stats["n"] < min_pair_records:
            continue
        win_rate = stats["y_sum"] / stats["n"]
        candidates.append(
            {
                "i": int(i),
                "j": int(j),
                "train_count": int(stats["n"]),
                "train_win_rate_for_i": float(win_rate),
                "distance_to_half": float(abs(win_rate - 0.5)),
                "judge_count": int(len(stats["judges"])),
            }
        )
    candidates.sort(key=lambda entry: (entry["distance_to_half"], -entry["train_count"]))
    return candidates[:max_pairs]


def evaluate_fit_on_record_slices(fit, near_tie_records, other_records):
    return {
        "near_tie_count": len(near_tie_records),
        "other_count": len(other_records),
        "near_tie_log_loss": compute_record_log_loss(fit, near_tie_records),
        "other_log_loss": compute_record_log_loss(fit, other_records),
        "near_tie_accuracy_no_ties": compute_record_accuracy(fit, near_tie_records, skip_ties=True),
        "other_accuracy_no_ties": compute_record_accuracy(fit, other_records, skip_ties=True),
    }


def run_real_near_tie_slice_experiment(
    random_seed=42,
    dataset_name=None,
    test_ratio=0.2,
    min_pair_records=20,
    max_pairs=20,
):
    output_path = os.path.join(ensure_results_dir(), "near_tie_slice_summary.json")
    summary = {}

    for current_dataset_name, dataset_path in select_real_data_files(dataset_name).items():
        print(f"[dataset={current_dataset_name}] loading near-tie dataset", flush=True)
        dataset_start_time = time.perf_counter()
        dataset = load_real_dataset(dataset_path)
        train_records, test_records = split_real_dataset(dataset["processed"], test_ratio=test_ratio, random_seed=random_seed)
        N = len(dataset["item_names"])
        K = len(dataset["judge_names"])
        near_tie_pairs = select_near_tie_pairs(
            train_records,
            min_pair_records=min_pair_records,
            max_pairs=max_pairs,
        )
        near_tie_pair_set = {(entry["i"], entry["j"]) for entry in near_tie_pairs}
        near_tie_test_records = [
            entry for entry in test_records
            if (entry["i"], entry["j"]) in near_tie_pair_set
        ]
        other_test_records = [
            entry for entry in test_records
            if (entry["i"], entry["j"]) not in near_tie_pair_set
        ]

        print(f"[dataset={current_dataset_name}] selecting proposed rank by 5-fold CV on training split", flush=True)
        r_model, rank_selection = select_rank_by_cross_validation(
            N,
            K,
            train_records,
            n_folds=5,
            random_seed=random_seed,
        )
        n_ijk, y_ijk = processed_records_to_aggregated(train_records, N, K)
        fit_results = fit_all_methods_safe(N, K, r_model, n_ijk, y_ijk, max_iter=2000)

        method_summary = {}
        for method_name, fit_result in fit_results.items():
            serialized = serialize_method_result(fit_result, test_records, dataset["item_names"])
            if fit_result["fit"] is not None:
                serialized["slice_metrics"] = evaluate_fit_on_record_slices(
                    fit_result["fit"],
                    near_tie_test_records,
                    other_test_records,
                )
            else:
                serialized["slice_metrics"] = None
            method_summary[method_name] = serialized

        summary[current_dataset_name] = {
            "dataset_path": dataset_path,
            "num_items": N,
            "num_judges": K,
            "train_size": len(train_records),
            "test_size": len(test_records),
            "near_tie_test_size": len(near_tie_test_records),
            "other_test_size": len(other_test_records),
            "min_pair_records": int(min_pair_records),
            "max_pairs": int(max_pairs),
            "near_tie_pairs": [
                {
                    **entry,
                    "item_i": dataset["item_names"][entry["i"]],
                    "item_j": dataset["item_names"][entry["j"]],
                }
                for entry in near_tie_pairs
            ],
            "rank_selection": rank_selection,
            "proposed_rank": int(r_model),
            "elapsed_seconds": time.perf_counter() - dataset_start_time,
            **dataset["summary"],
            "methods": method_summary,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[dataset={current_dataset_name}] near-tie slice done", flush=True)

    return summary


def parse_seed_list(seed_text, default_seeds=None):
    if seed_text is None:
        return list(DEFAULT_SEED_SWEEP_SEEDS if default_seeds is None else default_seeds)
    seeds = [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


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


def load_existing_seed_raw(output_dir, experiment, seed):
    path = os.path.join(output_dir, f"{experiment}_seed_{int(seed)}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_seed_raw(output_dir, experiment, seed, summary):
    path = os.path.join(output_dir, f"{experiment}_seed_{int(seed)}.json")
    write_json(path, summary)
    return path


def iter_seed_sweep_dataset_names(raw_by_seed):
    names = set()
    for summary in raw_by_seed.values():
        for dataset, dataset_summary in summary.items():
            if dataset.startswith("_") or not isinstance(dataset_summary, dict):
                continue
            if "methods" in dataset_summary or "base_methods" in dataset_summary:
                names.add(dataset)
    return sorted(names)


def aggregate_seed_sweep_stability(raw_by_seed):
    out = {}
    for dataset in iter_seed_sweep_dataset_names(raw_by_seed):
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


def aggregate_seed_sweep_near_tie(raw_by_seed):
    metric_keys = (
        "near_tie_log_loss",
        "other_log_loss",
        "near_tie_accuracy_no_ties",
        "other_accuracy_no_ties",
    )
    out = {}
    for dataset in iter_seed_sweep_dataset_names(raw_by_seed):
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


def exact_rank_match_rate(rank_shift):
    item_rank_shifts = (rank_shift or {}).get("rank_shift", {})
    if not item_rank_shifts:
        return None
    return sum(1 for item_shift in item_rank_shifts.values() if item_shift == 0) / len(item_rank_shifts)


def aggregate_seed_sweep_robustness(raw_by_seed):
    out = {}
    for dataset in iter_seed_sweep_dataset_names(raw_by_seed):
        method_names = sorted({
            method
            for summary in raw_by_seed.values()
            if dataset in summary
            for step in summary[dataset].get("steps", [])
            for method in step.get("methods", {})
        })
        step_values = sorted({
            int(step.get("step"))
            for summary in raw_by_seed.values()
            if dataset in summary
            for step in summary[dataset].get("steps", [])
            if step.get("step") is not None
        })
        out[dataset] = {
            "base_num_judges": next((summary[dataset].get("base_summary", {}).get("num_judges") for summary in raw_by_seed.values() if dataset in summary), None),
            "steps": step_values,
            "methods": {},
            "selected_ranks": [
                summary[dataset].get("rank_selection", {}).get("selected_rank")
                for summary in raw_by_seed.values()
                if dataset in summary
            ],
        }
        for method in method_names:
            method_out = {}
            for step_value in step_values:
                exact_values = []
                spearman_values = []
                top_1_values = []
                for summary in raw_by_seed.values():
                    dataset_summary = summary.get(dataset, {})
                    step_summary = next(
                        (step for step in dataset_summary.get("steps", []) if int(step.get("step")) == step_value),
                        None,
                    )
                    if step_summary is None:
                        continue
                    method_result = step_summary.get("methods", {}).get(method, {})
                    rank_shift = method_result.get("rank_shift")
                    if rank_shift is None or method_result.get("error") is not None:
                        continue
                    exact_values.append(exact_rank_match_rate(rank_shift))
                    spearman_values.append(rank_shift.get("spearman"))
                    top_1_values.append(float(bool(rank_shift.get("top_1_changed"))))
                method_out[str(step_value)] = {
                    "exact_rank_match_rate": mean_std(exact_values),
                    "spearman": mean_std(spearman_values),
                    "top_1_changed_rate": mean_std(top_1_values),
                }
            out[dataset]["methods"][method] = method_out
    return out


def aggregate_seed_sweep_bootstrap(raw_by_seed):
    out = {}
    for dataset in iter_seed_sweep_dataset_names(raw_by_seed):
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


def aggregate_seed_sweep_experiment(experiment, raw_by_seed):
    if experiment == "stability":
        return aggregate_seed_sweep_stability(raw_by_seed)
    if experiment == "robustness":
        return aggregate_seed_sweep_robustness(raw_by_seed)
    if experiment == "near_tie_slice":
        return aggregate_seed_sweep_near_tie(raw_by_seed)
    if experiment == "bootstrap":
        return aggregate_seed_sweep_bootstrap(raw_by_seed)
    raise ValueError(f"seed sweep is not supported for experiment '{experiment}'")


def run_one_seed_sweep_experiment(experiment, seed, args):
    if experiment == "stability":
        return run_real_data_stability_experiment(
            dataset_name=args.dataset,
            test_ratio=args.test_ratio,
            random_seed=seed,
        )
    if experiment == "robustness":
        return run_real_data_noisy_judge_experiment(
            dataset_name=args.dataset,
            max_noisy_step=args.max_noisy_step,
            random_seed=seed,
            save_artifacts=False,
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
    raise ValueError(f"seed sweep is not supported for experiment '{experiment}'")


def run_real_data_seed_sweep(experiment, args, seeds=None):
    output_dir = ensure_seed_sweep_dir()
    raw_by_seed = {}
    seeds = list(DEFAULT_SEED_SWEEP_SEEDS if seeds is None else seeds)
    for seed in seeds:
        existing = load_existing_seed_raw(output_dir, experiment, seed) if args.reuse else None
        if existing is None:
            start_time = time.perf_counter()
            print(f"[seed-sweep] experiment={experiment} seed={seed} start", flush=True)
            summary = run_one_seed_sweep_experiment(experiment, seed, args)
            summary["_seed_sweep_metadata"] = {
                "experiment": experiment,
                "seed": int(seed),
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            save_seed_raw(output_dir, experiment, seed, summary)
        else:
            print(f"[seed-sweep] experiment={experiment} seed={seed} reusing raw JSON", flush=True)
            summary = existing
        raw_by_seed[int(seed)] = summary

    aggregate = {
        "experiment": experiment,
        "seeds": [int(seed) for seed in seeds],
        "dataset": args.dataset,
        "summary": aggregate_seed_sweep_experiment(experiment, raw_by_seed),
    }
    output_path = os.path.join(output_dir, f"{experiment}_seed_sweep_summary.json")
    write_json(output_path, aggregate)
    print(f"[seed-sweep] wrote {output_path}", flush=True)
    return aggregate


def run_real_data_comparison_seed_sweeps(args):
    seeds = parse_seed_list(args.seeds)
    experiments = ("stability", "robustness", "near_tie_slice")
    return {
        experiment: run_real_data_seed_sweep(experiment, args, seeds=seeds)
        for experiment in experiments
    }



def run_real_data_benchmarks(
    dataset_name=None,
    max_noisy_step=None,
    bootstrap_samples=0,
    bootstrap_seed=42,
    bootstrap_workers=None,
    random_seed=42,
):
    stability_summary = run_real_data_stability_experiment(dataset_name=dataset_name, random_seed=random_seed)
    noisy_summary = run_real_data_noisy_judge_experiment(
        dataset_name=dataset_name,
        max_noisy_step=max_noisy_step,
        random_seed=random_seed,
    )
    bootstrap_summary = None
    if bootstrap_samples > 0:
        bootstrap_summary = run_real_data_bootstrap_experiment(
            dataset_name=dataset_name,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            bootstrap_workers=bootstrap_workers,
        )
    return {"stability": stability_summary, "noisy_judge": noisy_summary, "bootstrap": bootstrap_summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=(
            "stability",
            "robustness",
            "bootstrap",
            "structured_bias",
            "anti_consensus_matched",
            "leave_one_family_out",
            "near_tie_slice",
            "stability_excluding_zai_org",
            "all",
        ),
        default="stability",
    )
    parser.add_argument("--dataset", choices=sorted(REAL_DATA_FILES.keys()))
    parser.add_argument("--max-noisy-step", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-workers", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seeds", help="comma-separated seed list for repeated real-data comparisons; defaults to 20 seeds")
    parser.add_argument("--reuse", action="store_true", help="reuse existing per-seed raw JSON in results/seed_sweeps when present")
    parser.add_argument("--single-run", action="store_true", help="run only --random-seed for non-bootstrap comparison experiments")
    parser.add_argument("--min-family-judges", type=int, default=1)
    parser.add_argument("--near-tie-min-pair-records", type=int, default=20)
    parser.add_argument("--near-tie-max-pairs", type=int, default=20)
    parser.add_argument("--structured-bias-unit", choices=STRUCTURED_BIAS_UNITS, default="question_pair")
    parser.add_argument("--structured-bias-refit-cv", action="store_true")
    args = parser.parse_args()

    repeated_comparison_experiments = {"stability", "robustness", "near_tie_slice"}
    if args.experiment in repeated_comparison_experiments and not args.single_run:
        run_real_data_seed_sweep(args.experiment, args, seeds=parse_seed_list(args.seeds))
    elif args.experiment == "stability":
        run_real_data_stability_experiment(
            dataset_name=args.dataset,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed,
        )
    elif args.experiment == "robustness":
        run_real_data_noisy_judge_experiment(
            dataset_name=args.dataset,
            max_noisy_step=args.max_noisy_step,
            random_seed=args.random_seed,
        )
    elif args.experiment == "bootstrap":
        if args.bootstrap_samples <= 0:
            raise ValueError("--bootstrap-samples must be >= 1 when --experiment bootstrap")
        run_real_data_bootstrap_experiment(
            dataset_name=args.dataset,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_workers=args.bootstrap_workers,
        )
    elif args.experiment == "structured_bias":
        run_structured_bias_injection_experiment(
            dataset_name=args.dataset,
            injection_unit=args.structured_bias_unit,
            refit_cv=args.structured_bias_refit_cv,
            random_seed=args.random_seed,
        )
    elif args.experiment == "anti_consensus_matched":
        run_structured_bias_injection_experiment(
            dataset_name=args.dataset,
            injection_unit="matched_judge",
            variants=("anti_consensus",),
            output_path=os.path.join(
                ensure_results_dir(),
                "anti_consensus_matched_cv_summary.json" if args.structured_bias_refit_cv else "anti_consensus_matched_summary.json",
            ),
            refit_cv=args.structured_bias_refit_cv,
            random_seed=args.random_seed,
        )
    elif args.experiment == "leave_one_family_out":
        run_leave_one_family_out_experiment(
            dataset_name=args.dataset,
            min_family_judges=args.min_family_judges,
            random_seed=args.random_seed,
        )
    elif args.experiment == "stability_excluding_zai_org":
        run_stability_excluding_judge(
            dataset_name=args.dataset,
            random_seed=args.random_seed,
        )
    elif args.experiment == "near_tie_slice":
        run_real_near_tie_slice_experiment(
            dataset_name=args.dataset,
            test_ratio=args.test_ratio,
            min_pair_records=args.near_tie_min_pair_records,
            max_pairs=args.near_tie_max_pairs,
            random_seed=args.random_seed,
        )
    else:
        if args.single_run:
            run_real_data_benchmarks(
                dataset_name=args.dataset,
                max_noisy_step=args.max_noisy_step,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_workers=args.bootstrap_workers,
                random_seed=args.random_seed,
            )
        else:
            run_real_data_comparison_seed_sweeps(args)
            if args.bootstrap_samples > 0:
                run_real_data_bootstrap_experiment(
                    dataset_name=args.dataset,
                    bootstrap_samples=args.bootstrap_samples,
                    bootstrap_seed=args.bootstrap_seed,
                    bootstrap_workers=args.bootstrap_workers,
                )


if __name__ == "__main__":
    main()
