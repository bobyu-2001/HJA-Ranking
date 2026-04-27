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


if __name__ == "__main__":
    run_benchmark(["uq_coverage"])
