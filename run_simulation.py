import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from src.generate_simulation_data import (
    build_near_tie_parameters,
    comparisons_to_aggregated,
    compute_score_matrix,
    generate_balanced_comparisons,
    generate_true_parameters,
    generate_unbalanced_comparisons,
    identify_near_tie_pairs,
)
from src.benchmarks import fit_proposed, fit_standard_btl, fit_zhou_github
from src.evaluate import compute_near_tie_metrics, compute_parameter_errors, compute_ranking_metrics
from src.models import select_rank_by_bic, uncertainty_quantification


METHOD_LABELS = {
    "proposed": "Proposed",
    "zhou_github": "Zhou github",
    "standard_btl": "Standard BTL",
}

METHOD_COLORS = {
    "proposed": "tab:blue",
    "zhou_github": "tab:orange",
    "standard_btl": "tab:green",
}

PRESENTATION_METRICS = ("mse", "spearman", "ndcg", "score_entry_coverage", "sign_accuracy")



def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value



def ensure_results_dir():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir



def fit_all_methods(N, K, r_model, n_ijk, y_ijk, tau=30.0):
    return {
        "proposed": fit_proposed(N, K, r_model, n_ijk, y_ijk, max_steps=2000, tol=5e-5, tau=tau),
        "zhou_github": fit_zhou_github(N, K, n_ijk, y_ijk),
        "standard_btl": fit_standard_btl(N, K, n_ijk, y_ijk),
    }


def fit_all_methods_safe_simulation(N, K, r_model, n_ijk, y_ijk, tau=30.0):
    out = {}
    for method_name, fit_fn in (
        ("proposed", lambda: fit_proposed(N, K, r_model, n_ijk, y_ijk, max_steps=2000, tol=5e-5, tau=tau)),
        ("zhou_github", lambda: fit_zhou_github(N, K, n_ijk, y_ijk)),
        ("standard_btl", lambda: fit_standard_btl(N, K, n_ijk, y_ijk)),
    ):
        try:
            out[method_name] = {"fit": fit_fn(), "error": None}
        except Exception as exc:
            out[method_name] = {"fit": None, "error": str(exc)}
    return out



def evaluate_methods(mu_true, gamma_true, U_true, V_true, fit_results, near_tie_pairs=None):
    S_true = compute_score_matrix(mu_true, gamma_true, U_true, V_true)
    out = {}

    for name, fit in fit_results.items():
        metrics = {}
        metrics.update(compute_parameter_errors(mu_true, fit["mu"], gamma_true, fit["gamma"], U_true, fit["U"], V_true, fit["V"]))
        metrics.update(compute_ranking_metrics(mu_true, fit["mu"], S_true=S_true, S_est=fit["S"]))
        metrics["n_iter"] = fit["fit_info"]["n_iter"]
        metrics["converged"] = fit["fit_info"]["converged"]
        metrics["nll"] = fit["fit_info"].get("nll")

        if near_tie_pairs is not None:
            metrics.update(compute_near_tie_metrics(S_true, fit["S"], near_tie_pairs))

        out[name] = metrics

    return out



def _plot_metric_grid(x_values, results_by_method, metric_keys, titles, x_label, save_path):
    plt.figure(figsize=(6 * len(metric_keys), 5))
    for idx, metric_key in enumerate(metric_keys, start=1):
        plt.subplot(1, len(metric_keys), idx)
        for method_name, metrics in results_by_method.items():
            plt.plot(x_values, metrics[metric_key], marker="o", label=METHOD_LABELS[method_name])
        plt.xlabel(x_label)
        plt.ylabel(metric_key)
        plt.title(titles[idx - 1])
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()



def run_convergence_experiment(N, K, r_true, T_list, n_repeats):
    means = []
    proposed_gamma_by_T = {}

    for T in T_list:
        iters = []
        gamma_runs = []
        for rep in range(n_repeats):
            seed = 1000 * rep + 79

            mu_true, gamma_true, U_true, V_true = generate_true_parameters(N, K, r_true, random_seed=seed)
            S_true = compute_score_matrix(mu_true, gamma_true, U_true, V_true)
            comparisons = generate_balanced_comparisons(S_true, T, random_seed=seed + 1)
            n_ijk, y_ijk = comparisons_to_aggregated(comparisons, N, K)

            fit = fit_proposed(N, K, r_true, n_ijk, y_ijk, max_steps=2000, tol=5e-5, tau=30.0)
            iters.append(fit["fit_info"]["n_iter"])
            gamma_runs.append({"rep": rep, "seed": seed, "gamma": fit["gamma"]})

        means.append(float(np.mean(iters)))
        proposed_gamma_by_T[str(T)] = gamma_runs

    save_path = os.path.join(ensure_results_dir(), "convergence_vs_T.png")
    plt.figure(figsize=(7, 5))
    plt.plot(T_list, means, marker="o")
    plt.xlabel("Sample Size (T)")
    plt.ylabel("Convergence iterations")
    plt.title("Proposed model convergence vs T")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return {
        "mean_n_iter_by_T": means,
        "proposed_gamma_by_T": proposed_gamma_by_T,
    }



def run_bic_experiment(N, K, r_true, T_list, n_repeats):
    probabilities = []
    proposed_gamma_by_T = {}

    for T in T_list:
        hits = []
        gamma_runs = []
        for rep in range(n_repeats):
            seed = 2000 * rep + 79

            mu_true, gamma_true, U_true, V_true = generate_true_parameters(N, K, r_true, random_seed=seed)
            S_true = compute_score_matrix(mu_true, gamma_true, U_true, V_true)
            comparisons = generate_balanced_comparisons(S_true, T, random_seed=seed + 1)
            n_ijk, y_ijk = comparisons_to_aggregated(comparisons, N, K)

            best_rank, _ = select_rank_by_bic(N, K, n_ijk, y_ijk, candidate_ranks=[0, r_true], max_steps=2000, tol=5e-5)
            hits.append(int(best_rank == r_true))

            fit = fit_proposed(N, K, r_true, n_ijk, y_ijk, max_steps=2000, tol=5e-5, tau=30.0)
            gamma_runs.append({"rep": rep, "seed": seed, "best_rank": int(best_rank), "gamma": fit["gamma"]})

        probabilities.append(float(np.mean(hits)))
        proposed_gamma_by_T[str(T)] = gamma_runs

    save_path = os.path.join(ensure_results_dir(), "bic_rank_recovery.png")
    plt.figure(figsize=(7, 5))
    plt.plot(T_list, probabilities, marker="o")
    plt.xlabel("Sample Size (T)")
    plt.ylabel("P(BIC selects true rank)")
    plt.title("BIC rank recovery")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return {
        "probability_by_T": probabilities,
        "proposed_gamma_by_T": proposed_gamma_by_T,
    }



def run_sample_size_design_experiments(N, K, r_true, T_list, n_repeats):
    design_modes = {
        "balanced": lambda S, T, seed: generate_balanced_comparisons(S, T, random_seed=seed),
        "unbalanced_uniform": lambda S, T, seed: generate_unbalanced_comparisons(S, T, mode="uniform", random_seed=seed),
        "unbalanced_biased": lambda S, T, seed: generate_unbalanced_comparisons(S, T, mode="biased", random_seed=seed),
    }

    metrics_by_design = {}
    proposed_gamma_by_design_and_T = {}
    for design_name, sampler in design_modes.items():
        metric_store = {m: {"score_error": [], "spearman": []} for m in METHOD_LABELS}
        gamma_store = {}

        for T in T_list:
            per_method = {m: {"score_error": [], "spearman": []} for m in METHOD_LABELS}
            gamma_runs = []
            for rep in range(n_repeats):
                seed = 3000 * rep + 79

                mu_true, gamma_true, U_true, V_true = generate_true_parameters(N, K, r_true, random_seed=seed)
                S_true = compute_score_matrix(mu_true, gamma_true, U_true, V_true)
                comparisons = sampler(S_true, T, seed + 1)
                n_ijk, y_ijk = comparisons_to_aggregated(comparisons, N, K)

                fit_results = fit_all_methods(N, K, r_true, n_ijk, y_ijk)
                metrics = evaluate_methods(mu_true, gamma_true, U_true, V_true, fit_results)
                for method_name in METHOD_LABELS:
                    per_method[method_name]["score_error"].append(metrics[method_name]["score_error"])
                    per_method[method_name]["spearman"].append(metrics[method_name]["spearman"])
                gamma_runs.append({"rep": rep, "seed": seed, "gamma": fit_results["proposed"]["gamma"]})

            for method_name in METHOD_LABELS:
                metric_store[method_name]["score_error"].append(float(np.mean(per_method[method_name]["score_error"])))
                metric_store[method_name]["spearman"].append(float(np.mean(per_method[method_name]["spearman"])))
            gamma_store[str(T)] = gamma_runs

        save_path = os.path.join(ensure_results_dir(), f"sample_size_{design_name}.png")
        _plot_metric_grid(T_list, metric_store, ["score_error", "spearman"], [f"{design_name}: score error", f"{design_name}: ranking"], "Sample Size (T)", save_path)
        metrics_by_design[design_name] = metric_store
        proposed_gamma_by_design_and_T[design_name] = gamma_store

    return {
        "metrics_by_design": metrics_by_design,
        "proposed_gamma_by_design_and_T": proposed_gamma_by_design_and_T,
    }



def run_heterogeneity_experiment(N, K, r_true, T_fixed, scale_list, n_repeats):
    metric_store = {m: {"score_error": [], "spearman": []} for m in METHOD_LABELS}
    proposed_gamma_by_scale = {}

    for scale in scale_list:
        per_method = {m: {"score_error": [], "spearman": []} for m in METHOD_LABELS}
        gamma_runs = []
        for rep in range(n_repeats):
            seed = 4000 * rep + 79

            mu_true, gamma_true, U_true, V_true = generate_true_parameters(N, K, r_true, random_seed=seed, heterogeneity_scale=scale)
            S_true = compute_score_matrix(mu_true, gamma_true, U_true, V_true)
            comparisons = generate_balanced_comparisons(S_true, T_fixed, random_seed=seed + 1)
            n_ijk, y_ijk = comparisons_to_aggregated(comparisons, N, K)

            fit_results = fit_all_methods(N, K, r_true, n_ijk, y_ijk)
            metrics = evaluate_methods(mu_true, gamma_true, U_true, V_true, fit_results)
            for method_name in METHOD_LABELS:
                per_method[method_name]["score_error"].append(metrics[method_name]["score_error"])
                per_method[method_name]["spearman"].append(metrics[method_name]["spearman"])
            gamma_runs.append({"rep": rep, "seed": seed, "gamma": fit_results["proposed"]["gamma"]})

        for method_name in METHOD_LABELS:
            metric_store[method_name]["score_error"].append(float(np.mean(per_method[method_name]["score_error"])))
            metric_store[method_name]["spearman"].append(float(np.mean(per_method[method_name]["spearman"])))
        proposed_gamma_by_scale[str(scale)] = gamma_runs

    save_path = os.path.join(ensure_results_dir(), "heterogeneity_levels.png")
    _plot_metric_grid(scale_list, metric_store, ["score_error", "spearman"], ["Heterogeneity: score error", "Heterogeneity: ranking"], "Heterogeneity scale", save_path)
    return {
        "metrics_by_scale": metric_store,
        "proposed_gamma_by_scale": proposed_gamma_by_scale,
    }



def run_near_tie_experiment(N, K, r_true, T_fixed, n_repeats):
    metric_store = {m: {"near_tie_sign_accuracy": [], "near_tie_probability_error": [], "near_tie_reordering_risk": []} for m in METHOD_LABELS}
    proposed_gamma_by_repeat = []

    for rep in range(n_repeats):
        seed = 5000 * rep + 79

        mu_true, gamma_true, U_true, V_true = generate_true_parameters(N, K, r_true, random_seed=seed)
        mu_true, gamma_true, U_true, V_true = build_near_tie_parameters(mu_true, gamma_true, U_true, V_true, pair=(0, 1), target_logit_diff=0.05)
        S_true = compute_score_matrix(mu_true, gamma_true, U_true, V_true)
        near_tie_pairs = [entry["pair"] for entry in identify_near_tie_pairs(S_true, top_m=3)]

        comparisons = generate_balanced_comparisons(S_true, T_fixed, random_seed=seed + 1)
        n_ijk, y_ijk = comparisons_to_aggregated(comparisons, N, K)

        fit_results = fit_all_methods(N, K, r_true, n_ijk, y_ijk, tau=30.0)
        metrics = evaluate_methods(mu_true, gamma_true, U_true, V_true, fit_results, near_tie_pairs=near_tie_pairs)
        for method_name in METHOD_LABELS:
            for key in metric_store[method_name]:
                metric_store[method_name][key].append(metrics[method_name][key])
        proposed_gamma_by_repeat.append({"rep": rep, "seed": seed, "gamma": fit_results["proposed"]["gamma"]})

    averaged = {m: {k: float(np.mean(v)) for k, v in vals.items()} for m, vals in metric_store.items()}

    save_path = os.path.join(ensure_results_dir(), "near_tie.png")
    plt.figure(figsize=(14, 4))
    keys = ["near_tie_sign_accuracy", "near_tie_probability_error", "near_tie_reordering_risk"]
    titles = ["Near-tie sign accuracy", "Near-tie probability error", "Near-tie reordering risk"]
    method_names = list(METHOD_LABELS.keys())
    for idx, key in enumerate(keys, start=1):
        plt.subplot(1, 3, idx)
        values = [averaged[m][key] for m in method_names]
        plt.bar([METHOD_LABELS[m] for m in method_names], values)
        plt.title(titles[idx - 1])
        plt.xticks(rotation=15)
        plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return {
        "averaged_metrics": averaged,
        "proposed_gamma_by_repeat": proposed_gamma_by_repeat,
    }



def run_uq_coverage_experiment(N, K, r_true, T_list, n_repeats, alpha=0.05):
    coverage_by_T = []
    mean_width_by_T = []
    mean_se_by_T = []
    failures_by_T = {}

    targets = [
        {"type": "score_diff", "k": k, "i": i, "j": j}
        for k in range(K)
        for i in range(N)
        for j in range(i + 1, N)
    ]

    for T in T_list:
        covered = []
        widths = []
        ses = []
        failures = []

        for rep in range(n_repeats):
            seed = 6000 * rep + 79
            mu_true, gamma_true, U_true, V_true = generate_true_parameters(N, K, r_true, random_seed=seed)
            S_true = compute_score_matrix(mu_true, gamma_true, U_true, V_true)
            comparisons = generate_balanced_comparisons(S_true, T, random_seed=seed + 1)
            n_ijk, y_ijk = comparisons_to_aggregated(comparisons, N, K)

            try:
                fit = fit_proposed(N, K, r_true, n_ijk, y_ijk, max_steps=2000, tol=5e-5, tau=30.0)
                uq = uncertainty_quantification(
                    fit["gamma"],
                    fit["mu"],
                    fit["U"],
                    fit["V"],
                    n_ijk,
                    targets,
                    alpha=alpha,
                )
            except Exception as exc:
                failures.append({"rep": rep, "seed": seed, "error": str(exc)})
                continue

            for target, interval in zip(targets, uq["intervals"]):
                k = target["k"]
                i = target["i"]
                j = target["j"]
                truth = float(S_true[k, i] - S_true[k, j])
                covered.append(float(interval["lower"] <= truth <= interval["upper"]))
                widths.append(float(interval["upper"] - interval["lower"]))
                ses.append(float(interval["se"]))

        coverage_by_T.append(float(np.mean(covered)) if covered else np.nan)
        mean_width_by_T.append(float(np.mean(widths)) if widths else np.nan)
        mean_se_by_T.append(float(np.mean(ses)) if ses else np.nan)
        failures_by_T[str(T)] = failures

    save_path = os.path.join(ensure_results_dir(), "uq_coverage.png")
    plt.figure(figsize=(7, 5))
    plt.plot(T_list, coverage_by_T, marker="o", label="Empirical coverage")
    plt.axhline(1.0 - alpha, color="black", linestyle="--", linewidth=1.0, label=f"Nominal {1.0 - alpha:.0%}")
    plt.xlabel("Sample Size (T)")
    plt.ylabel("Coverage rate")
    plt.title("UQ coverage for judge-specific score contrasts")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    result = {
        "T_list": [int(T) for T in T_list],
        "n_repeats": int(n_repeats),
        "coverage_by_T": coverage_by_T,
        "mean_width_by_T": mean_width_by_T,
        "mean_se_by_T": mean_se_by_T,
        "failures_by_T": failures_by_T,
        "alpha": float(alpha),
        "nominal_coverage": float(1.0 - alpha),
        "target": "judge_specific_score_diff",
    }
    summary_path = os.path.join(ensure_results_dir(), "uq_coverage_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(result), f, indent=2)
    return result


def compute_ndcg_at_n(mu_true, mu_est):
    mu_true = np.asarray(mu_true, dtype=float)
    mu_est = np.asarray(mu_est, dtype=float)
    N = mu_true.size
    true_order = np.argsort(-mu_true)
    true_rank = np.empty(N, dtype=int)
    true_rank[true_order] = np.arange(1, N + 1)
    relevance = np.power(2.0, N - true_rank) - 1.0

    pred_order = np.argsort(-mu_est)
    discounts = 1.0 / np.log2(np.arange(2, N + 2, dtype=float))
    dcg = float(np.sum(relevance[pred_order] * discounts))
    ideal_dcg = float(np.sum(relevance[true_order] * discounts))
    return dcg / ideal_dcg if ideal_dcg > 0 else 1.0


def compute_score_mse(S_true, S_est):
    S_true = np.asarray(S_true, dtype=float)
    S_est = np.asarray(S_est, dtype=float)
    return float(np.mean((S_est - S_true) ** 2))


def compute_score_sign_accuracy(S_true, S_est):
    S_true = np.asarray(S_true, dtype=float)
    S_est = np.asarray(S_est, dtype=float)
    K, N = S_true.shape
    total = 0
    correct = 0
    for k in range(K):
        for i in range(N):
            for j in range(i + 1, N):
                correct += int(np.sign(S_true[k, i] - S_true[k, j]) == np.sign(S_est[k, i] - S_est[k, j]))
                total += 1
    return float(correct / total) if total > 0 else np.nan


def score_entry_targets(K, N):
    return [
        {"type": "score_entry", "k": k, "i": i}
        for k in range(K)
        for i in range(N)
    ]


def proposed_score_entry_ci_matrix(fit, n_ijk, alpha=0.05):
    K, N = fit["S"].shape
    targets = score_entry_targets(K, N)
    uq = uncertainty_quantification(
        fit["gamma"],
        fit["mu"],
        fit["U"],
        fit["V"],
        n_ijk,
        targets,
        alpha=alpha,
    )
    ci_matrix = []
    cursor = 0
    for _k in range(K):
        row = []
        for _i in range(N):
            interval = uq["intervals"][cursor]
            row.append(interval)
            cursor += 1
        ci_matrix.append(row)
    return ci_matrix


def get_score_entry_ci_matrix(method_name, fit, n_ijk, alpha=0.05):
    if method_name == "proposed":
        return proposed_score_entry_ci_matrix(fit, n_ijk, alpha=alpha)
    uq = fit.get("uq") or {}
    if uq.get("error") is not None:
        raise RuntimeError(uq["error"])
    ci_matrix = uq.get("S_ci")
    if ci_matrix is None:
        raise RuntimeError(f"{method_name} fit did not return S_ci")
    return ci_matrix


def compute_score_entry_coverage(S_true, ci_matrix):
    S_true = np.asarray(S_true, dtype=float)
    K, N = S_true.shape
    covered = []
    if len(ci_matrix) != K:
        raise ValueError(f"S_ci row count {len(ci_matrix)} does not match K={K}")
    for k in range(K):
        if len(ci_matrix[k]) != N:
            raise ValueError(f"S_ci column count {len(ci_matrix[k])} does not match N={N}")
        for i in range(N):
            interval = ci_matrix[k][i]
            covered.append(float(interval["lower"] <= S_true[k, i] <= interval["upper"]))
    return float(np.mean(covered)) if covered else np.nan


def compute_presentation_metrics(mu_true, S_true, method_name, fit, n_ijk, alpha=0.05):
    ranking_metrics = compute_ranking_metrics(mu_true, fit["mu"])
    ci_matrix = get_score_entry_ci_matrix(method_name, fit, n_ijk, alpha=alpha)
    return {
        "mse": compute_score_mse(S_true, fit["S"]),
        "spearman": float(ranking_metrics["spearman"]),
        "ndcg": compute_ndcg_at_n(mu_true, fit["mu"]),
        "score_entry_coverage": compute_score_entry_coverage(S_true, ci_matrix),
        "sign_accuracy": compute_score_sign_accuracy(S_true, fit["S"]),
    }


def make_presentation_parameters(N, K, r_true, random_seed, dgp, heterogeneity_scale=1.0):
    mu_true, gamma_true, U_true, V_true = generate_true_parameters(
        N,
        K,
        r_true,
        random_seed=random_seed,
        heterogeneity_scale=heterogeneity_scale,
    )
    if dgp == "zhou":
        U_true = np.zeros((K, 0), dtype=float)
        V_true = np.zeros((N, 0), dtype=float)
    elif dgp != "ours":
        raise ValueError(f"unknown presentation DGP: {dgp}")
    return mu_true, gamma_true, U_true, V_true


def summarize_metric_records(records):
    summary = {}
    for method_name in METHOD_LABELS:
        summary[method_name] = {}
        for metric_name in PRESENTATION_METRICS:
            values = np.asarray(
                [
                    record["metrics"][method_name][metric_name]
                    for record in records
                    if method_name in record["metrics"]
                    and metric_name in record["metrics"][method_name]
                    and np.isfinite(record["metrics"][method_name][metric_name])
                ],
                dtype=float,
            )
            if values.size == 0:
                summary[method_name][metric_name] = {
                    "n_success": 0,
                    "mean": None,
                    "std": None,
                    "se": None,
                    "mc_ci95": None,
                }
                continue
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            se = float(std / np.sqrt(values.size)) if values.size > 0 else None
            summary[method_name][metric_name] = {
                "n_success": int(values.size),
                "mean": float(np.mean(values)),
                "std": std,
                "se": se,
                "mc_ci95": float(1.96 * se),
            }
    return summary


def run_presentation_condition(
    N,
    K,
    r_true,
    T,
    n_repeats,
    dgp,
    heterogeneity_scale=1.0,
    seed_offset=7000,
    alpha=0.05,
    tau=30.0,
):
    records = []
    failures = []
    for rep in range(n_repeats):
        seed = int(seed_offset + 1000 * rep)
        mu_true, gamma_true, U_true, V_true = make_presentation_parameters(
            N,
            K,
            r_true,
            random_seed=seed,
            dgp=dgp,
            heterogeneity_scale=heterogeneity_scale,
        )
        S_true = compute_score_matrix(mu_true, gamma_true, U_true, V_true)
        comparisons = generate_balanced_comparisons(S_true, T, random_seed=seed + 1)
        n_ijk, y_ijk = comparisons_to_aggregated(comparisons, N, K)
        fit_results = fit_all_methods_safe_simulation(N, K, r_true, n_ijk, y_ijk, tau=tau)

        record = {
            "rep": int(rep),
            "seed": seed,
            "T": int(T),
            "heterogeneity_scale": float(heterogeneity_scale),
            "u_operator_norm": float(np.linalg.norm(U_true, ord=2)) if U_true.size else 0.0,
            "metrics": {},
        }
        for method_name, result in fit_results.items():
            if result["fit"] is None:
                failures.append({"rep": int(rep), "seed": seed, "method": method_name, "error": result["error"]})
                continue
            try:
                record["metrics"][method_name] = compute_presentation_metrics(
                    mu_true,
                    S_true,
                    method_name,
                    result["fit"],
                    n_ijk,
                    alpha=alpha,
                )
            except Exception as exc:
                failures.append({"rep": int(rep), "seed": seed, "method": method_name, "error": str(exc)})
        records.append(record)

    return {
        "T": int(T),
        "heterogeneity_scale": float(heterogeneity_scale),
        "x_u_operator_norm": float(np.mean([record["u_operator_norm"] for record in records])) if records else 0.0,
        "raw": records,
        "summary": summarize_metric_records(records),
        "failures": failures,
    }


def presentation_series_from_conditions(conditions, x_key):
    series = {method_name: {metric: {"mean": [], "err": [], "n_success": []} for metric in PRESENTATION_METRICS} for method_name in METHOD_LABELS}
    x_values = []
    for condition in conditions:
        x_values.append(condition[x_key])
        for method_name in METHOD_LABELS:
            for metric_name in PRESENTATION_METRICS:
                metric_summary = condition["summary"][method_name][metric_name]
                series[method_name][metric_name]["mean"].append(np.nan if metric_summary["mean"] is None else metric_summary["mean"])
                series[method_name][metric_name]["err"].append(np.nan if metric_summary["mc_ci95"] is None else metric_summary["mc_ci95"])
                series[method_name][metric_name]["n_success"].append(metric_summary["n_success"])
    return np.asarray(x_values, dtype=float), series


def plot_metric_panel(ax, x_values, series, metric_name, x_label, y_label, title, ylim=None):
    for method_name in METHOD_LABELS:
        ax.errorbar(
            x_values,
            series[method_name][metric_name]["mean"],
            yerr=series[method_name][metric_name]["err"],
            marker="o",
            linewidth=1.6,
            capsize=3,
            label=METHOD_LABELS[method_name],
            color=METHOD_COLORS[method_name],
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)


def plot_ranking_panel(ax, x_values, series, x_label, title):
    right_ax = ax.twinx()
    for method_name in METHOD_LABELS:
        color = METHOD_COLORS[method_name]
        ax.errorbar(
            x_values,
            series[method_name]["spearman"]["mean"],
            yerr=series[method_name]["spearman"]["err"],
            marker="o",
            linewidth=1.6,
            capsize=3,
            color=color,
            linestyle="-",
            label=f"{METHOD_LABELS[method_name]} Spearman",
        )
        right_ax.errorbar(
            x_values,
            series[method_name]["ndcg"]["mean"],
            yerr=series[method_name]["ndcg"]["err"],
            marker="s",
            linewidth=1.4,
            capsize=3,
            color=color,
            linestyle="--",
            label=f"{METHOD_LABELS[method_name]} NDCG",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Spearman")
    right_ax.set_ylabel("NDCG@N")
    ax.set_title(title)
    ax.set_ylim(0.6, 1)
    right_ax.set_ylim(0.9, 1)
    ax.grid(True, alpha=0.3)
    return right_ax


def plot_presentation_grid(summary, save_path, dgp):
    if dgp == "ours":
        fig, axes = plt.subplots(2, 4, figsize=(22, 9), squeeze=False)
        row_specs = [
            ("ours_sample_size", "T", "Sample size (T)", "Ours DGP: sample size"),
            ("ours_heterogeneity", "x_u_operator_norm", r"Heterogeneity $||U||_2$", "Ours DGP: heterogeneity"),
        ]
    elif dgp == "zhou":
        fig, axes = plt.subplots(1, 4, figsize=(22, 4.8), squeeze=False)
        row_specs = [("zhou_sample_size", "T", "Sample size (T)", "Zhou DGP")]
    else:
        raise ValueError(f"unknown plot DGP: {dgp}")

    legend_handles = None
    legend_labels = None
    for row_idx, (section_key, x_key, x_label, row_title) in enumerate(row_specs):
        x_values, series = presentation_series_from_conditions(summary[section_key], x_key)
        plot_metric_panel(axes[row_idx, 0], x_values, series, "mse", x_label, "MSE", f"{row_title}: MSE")
        ranking_right_ax = plot_ranking_panel(axes[row_idx, 1], x_values, series, x_label, f"{row_title}: ranking")
        plot_metric_panel(
            axes[row_idx, 2],
            x_values,
            series,
            "score_entry_coverage",
            x_label,
            "Coverage",
            f"{row_title}: S entry coverage",
            ylim=(-0.05, 1.05),
        )
        axes[row_idx, 2].axhline(0.95, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        plot_metric_panel(
            axes[row_idx, 3],
            x_values,
            series,
            "sign_accuracy",
            x_label,
            "Accuracy",
            f"{row_title}: sign accuracy",
            ylim=(-0.05, 1.05),
        )
        if legend_handles is None:
            handles_left, labels_left = axes[row_idx, 1].get_legend_handles_labels()
            handles_right, labels_right = ranking_right_ax.get_legend_handles_labels()
            legend_handles = handles_left + handles_right
            legend_labels = labels_left + labels_right

    axes[0, 0].legend(loc="best", fontsize=8)
    if legend_handles is not None:
        axes[0, 1].legend(legend_handles, legend_labels, loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_presentation_simulation_from_summary(summary_path=None):
    results_dir = ensure_results_dir()
    if summary_path is None:
        summary_path = os.path.join(results_dir, "presentation_simulation_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"presentation summary not found at {summary_path}; run "
            "`python experiments/run_simulation.py --experiment presentation` first"
        )

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    ours_path = os.path.join(results_dir, "presentation_ours_dgp_grid.png")
    zhou_path = os.path.join(results_dir, "presentation_zhou_dgp_grid.png")
    plot_presentation_grid(summary, ours_path, "ours")
    plot_presentation_grid(summary, zhou_path, "zhou")

    summary["figures"] = {
        "ours_dgp_grid": ours_path,
        "zhou_dgp_grid": zhou_path,
    }
    summary["summary_path"] = summary_path
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    print(f"[presentation plot-only] read {summary_path}", flush=True)
    print(f"[presentation plot-only] wrote {ours_path}", flush=True)
    print(f"[presentation plot-only] wrote {zhou_path}", flush=True)
    return {
        "summary_path": summary_path,
        "ours_dgp_grid": ours_path,
        "zhou_dgp_grid": zhou_path,
    }


def run_presentation_simulation(
    N=8,
    K=4,
    r_true=1,
    n_repeats=50,
    T_list=None,
    scale_list=None,
    T_fixed=800,
    alpha=0.05,
):
    T_list = [400, 800, 1200, 1600, 2000, 2500, 3000] if T_list is None else [int(T) for T in T_list]
    scale_list = [0.0, 0.5, 1.0, 2.0] if scale_list is None else [float(scale) for scale in scale_list]

    results = {
        "settings": {
            "N": int(N),
            "K": int(K),
            "r_true": int(r_true),
            "n_repeats": int(n_repeats),
            "T_list": T_list,
            "scale_list": scale_list,
            "T_fixed": int(T_fixed),
            "alpha": float(alpha),
            "error_bar": "mean +/- 1.96 * standard_error_across_repeats",
        },
        "ours_sample_size": [],
        "ours_heterogeneity": [],
        "zhou_sample_size": [],
    }

    for idx, T in enumerate(T_list):
        print(f"[presentation] ours DGP sample-size T={T}", flush=True)
        results["ours_sample_size"].append(
            run_presentation_condition(N, K, r_true, T, n_repeats, "ours", 1.0, seed_offset=710000 + 10000 * idx, alpha=alpha)
        )
    for idx, scale in enumerate(scale_list):
        print(f"[presentation] ours DGP heterogeneity scale={scale}", flush=True)
        results["ours_heterogeneity"].append(
            run_presentation_condition(N, K, r_true, T_fixed, n_repeats, "ours", scale, seed_offset=720000 + 10000 * idx, alpha=alpha)
        )
    for idx, T in enumerate(T_list):
        print(f"[presentation] Zhou DGP sample-size T={T}", flush=True)
        results["zhou_sample_size"].append(
            run_presentation_condition(N, K, r_true, T, n_repeats, "zhou", 0.0, seed_offset=730000 + 10000 * idx, alpha=alpha)
        )

    results_dir = ensure_results_dir()
    ours_path = os.path.join(results_dir, "presentation_ours_dgp_grid.png")
    zhou_path = os.path.join(results_dir, "presentation_zhou_dgp_grid.png")
    plot_presentation_grid(results, ours_path, "ours")
    plot_presentation_grid(results, zhou_path, "zhou")

    results["figures"] = {
        "ours_dgp_grid": ours_path,
        "zhou_dgp_grid": zhou_path,
    }
    summary_path = os.path.join(results_dir, "presentation_simulation_summary.json")
    results["summary_path"] = summary_path
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)
    print(f"[presentation] wrote {summary_path}", flush=True)
    print(f"[presentation] wrote {ours_path}", flush=True)
    print(f"[presentation] wrote {zhou_path}", flush=True)
    return results



def run_final_parameter_export(N, K, r_true, T_fixed, seed=79):
    mu_true, gamma_true, U_true, V_true = generate_true_parameters(N, K, r_true, random_seed=seed)
    S_true = compute_score_matrix(mu_true, gamma_true, U_true, V_true)
    comparisons = generate_balanced_comparisons(S_true, T_fixed, random_seed=seed + 1)
    n_ijk, y_ijk = comparisons_to_aggregated(comparisons, N, K)
    fit = fit_proposed(N, K, r_true, n_ijk, y_ijk, max_steps=2000, tol=5e-5, tau=30.0)
    return {
        "seed": int(seed),
        "design": "balanced",
        "T": int(T_fixed),
        "gamma": fit["gamma"].tolist(),
        "mu": fit["mu"].tolist(),
        "U": fit["U"].tolist(),
        "V": fit["V"].tolist(),
        "fit_info": fit["fit_info"],
    }



def run_benchmark(experiments_to_run=
                  [
        "convergence_vs_T",
        "bic_rank_recovery",
        "sample_size_designs",
        "heterogeneity_levels",
        "near_tie",
        "uq_coverage",
        "final_parameter_export",
    ]):
    N = 8
    K = 4
    r_true = 1
    n_repeats = 3
    T_list = [400, 800, 1200, 1600, 2000, 2500, 3000]
    scale_list = [0.0, 0.5, 1.0, 2.0]
    T_fixed = 800

    for exp in experiments_to_run:
        print(f"Running experiment: {exp}...")
        if exp == "convergence_vs_T":
            convergence_vs_T = run_convergence_experiment(N, K, r_true, T_list, n_repeats)
        elif exp == "bic_rank_recovery":
            bic_rank_recovery = run_bic_experiment(N, K, r_true, [100,200,300,400,500], n_repeats)
        elif exp == "sample_size_designs":
            sample_size_designs = run_sample_size_design_experiments(N, K, r_true, T_list, n_repeats)
        elif exp == "heterogeneity_levels":
            heterogeneity_levels = run_heterogeneity_experiment(N, K, r_true, T_fixed, scale_list, n_repeats)
        elif exp == "near_tie":
            near_tie = run_near_tie_experiment(N, K, r_true, T_fixed, n_repeats)
        elif exp == "uq_coverage":
            uq_coverage = run_uq_coverage_experiment(N, K, r_true, T_list, n_repeats)
            print(json.dumps(to_jsonable(uq_coverage), indent=2))
        elif exp == "final_parameter_export":
            final_parameters = run_final_parameter_export(N, K, r_true, T_fixed, seed=79)

    if experiments_to_run==[
        "convergence_vs_T",
        "bic_rank_recovery",
        "sample_size_designs",
        "heterogeneity_levels",
        "near_tie",
        "uq_coverage",
        "final_parameter_export",
    ]: 
        summary = {
            "convergence_vs_T": convergence_vs_T,
            "bic_rank_recovery": bic_rank_recovery,
            "sample_size_designs": sample_size_designs,
            "heterogeneity_levels": heterogeneity_levels,
            "near_tie": near_tie,
            "uq_coverage": uq_coverage,
            "final_parameters": final_parameters,
            "settings": {
                "N": N,
                "K": K,
                "r_true": r_true,
                "n_repeats": n_repeats,
                "T_list": T_list,
                "scale_list": scale_list,
                "T_fixed": T_fixed,
            },
        }
        summary_path = os.path.join(ensure_results_dir(), "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(summary), f, indent=2)

        print("Simulation complete. Results saved in simulation_0420/results/")
        print("Final fitted parameters (reference run):")
        print(json.dumps(final_parameters, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run synthetic simulation experiments.")
    parser.add_argument(
        "--experiment",
        choices=["uq_coverage", "presentation"],
        default="uq_coverage",
        help="Experiment workflow to run.",
    )
    parser.add_argument("--presentation-smoke", action="store_true", help="Run a small presentation-figure smoke test.")
    parser.add_argument("--plot-only", action="store_true", help="Only redraw presentation PNGs from the saved summary JSON.")
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Summary JSON to use with --plot-only; defaults to experiments/results/presentation_simulation_summary.json.",
    )
    parser.add_argument("--n-repeats", type=int, default=None, help="Override repeat count for presentation figures.")
    parser.add_argument("--output-alpha", type=float, default=0.05, help="Interval alpha level for coverage metrics.")
    args = parser.parse_args()

    if args.plot_only:
        plot_presentation_simulation_from_summary(args.summary_path)
        return

    if args.experiment == "presentation":
        if args.presentation_smoke:
            run_presentation_simulation(
                n_repeats=2 if args.n_repeats is None else args.n_repeats,
                T_list=[400, 800],
                scale_list=[0.0, 1.0],
                T_fixed=800,
                alpha=args.output_alpha,
            )
        else:
            run_presentation_simulation(
                n_repeats=50 if args.n_repeats is None else args.n_repeats,
                alpha=args.output_alpha,
            )
    else:
        run_benchmark(["uq_coverage"])


if __name__ == "__main__":
    main()
