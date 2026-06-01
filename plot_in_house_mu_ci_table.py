"""Compute 95% bootstrap confidence intervals for Proposed mu on in_house data.

Output: formatted table with columns model | mu | mu-CI | mu+CI
"""

import os
import sys
import json
import time
import numpy as np

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

from src.generate_data import (
    bootstrap_processed_records,
    load_real_dataset,
    processed_records_to_aggregated,
)
from src.benchmarks import fit_proposed
from run_real_data import (
    REAL_DATA_FILES,
    select_rank_by_cross_validation,
)


N_BOOTSTRAP = 200
RANDOM_SEED = 42


def fit_proposed_from_records(processed_records, N, K, r_model):
    """Fit Proposed on aggregated records, returning mu or None on failure."""
    n_ijk, y_ijk = processed_records_to_aggregated(processed_records, N, K)
    try:
        fit = fit_proposed(N, K, r_model, n_ijk, y_ijk, max_steps=40, tol=5e-5, tau=30.0)
        return np.asarray(fit["mu"], dtype=float)
    except Exception:
        return None


def main():
    dataset_path = REAL_DATA_FILES["in_house"]
    dataset_name = "in_house"

    # --- Load data ---
    print(f"[{dataset_name}] loading dataset", flush=True)
    dataset = load_real_dataset(dataset_path)
    processed = dataset["processed"]
    item_names = dataset["item_names"]
    N = len(item_names)
    K = len(dataset["judge_names"])
    print(f"[{dataset_name}] N={N}, K={K}, records={len(processed)}", flush=True)

    # --- CV select rank ---
    print(f"[{dataset_name}] selecting proposed rank by 5-fold CV", flush=True)
    r_model, rank_selection = select_rank_by_cross_validation(
        N, K, processed, n_folds=5, random_seed=RANDOM_SEED,
    )
    print(f"[{dataset_name}] selected rank={r_model}", flush=True)

    # --- Fit baseline ---
    print(f"[{dataset_name}] fitting baseline Proposed", flush=True)
    t0 = time.perf_counter()
    baseline_mu = fit_proposed_from_records(processed, N, K, r_model)
    if baseline_mu is None:
        raise RuntimeError("baseline Proposed fit failed")
    print(f"[{dataset_name}] baseline fit done in {time.perf_counter() - t0:.1f}s", flush=True)

    # --- Bootstrap ---
    print(f"[{dataset_name}] running {N_BOOTSTRAP} bootstrap samples", flush=True)
    bootstrap_mu_list = []
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_seeds = rng.integers(0, 2**31 - 1, size=N_BOOTSTRAP)

    t0 = time.perf_counter()
    for b_idx in range(N_BOOTSTRAP):
        seed = int(bootstrap_seeds[b_idx])
        bootstrap_records = bootstrap_processed_records(processed, random_seed=seed)
        mu_b = fit_proposed_from_records(bootstrap_records, N, K, r_model)
        if mu_b is not None:
            bootstrap_mu_list.append(mu_b)
        else:
            print(f"  bootstrap {b_idx + 1}/{N_BOOTSTRAP} failed, skipping", flush=True)
        if (b_idx + 1) % 50 == 0:
            print(f"  bootstrap {b_idx + 1}/{N_BOOTSTRAP} done ({time.perf_counter() - t0:.1f}s)", flush=True)

    elapsed = time.perf_counter() - t0
    successful = len(bootstrap_mu_list)
    print(f"[{dataset_name}] bootstrap done: {successful}/{N_BOOTSTRAP} successful in {elapsed:.1f}s", flush=True)

    if successful < 50:
        raise RuntimeError(f"too few successful bootstrap samples: {successful}")

    bootstrap_mu = np.stack(bootstrap_mu_list, axis=0)  # shape: (successful, N)

    # --- Compute percentile CIs ---
    alpha = 0.05
    lower_pct = 100.0 * alpha / 2.0      # 2.5
    upper_pct = 100.0 * (1.0 - alpha / 2.0)  # 97.5

    rows = []
    for idx in range(N):
        name = item_names[idx]
        mu_val = float(baseline_mu[idx])
        mu_lower = float(np.percentile(bootstrap_mu[:, idx], lower_pct))
        mu_upper = float(np.percentile(bootstrap_mu[:, idx], upper_pct))
        rows.append((name, mu_val, mu_lower, mu_upper))

    # Sort by mu descending
    rows.sort(key=lambda r: r[1], reverse=True)

    # --- Print table ---
    col_widths = [45, 12, 12, 12]
    header = (
        f"{'model':<{col_widths[0]}}"
        f"{'mu':>{col_widths[1]}}"
        f"{'mu-CI':>{col_widths[2]}}"
        f"{'mu+CI':>{col_widths[3]}}"
    )
    sep = "-" * sum(col_widths)

    print()
    print(f"  In-House Mu 95% CI (bootstrap percentile, n={successful}, rank={r_model})")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")
    for name, mu_val, mu_lower, mu_upper in rows:
        print(
            f"  {name:<{col_widths[0]}}"
            f"{mu_val:>{col_widths[1]}.4f}"
            f"{mu_lower:>{col_widths[2]}.4f}"
            f"{mu_upper:>{col_widths[3]}.4f}"
        )
    print(f"  {sep}")

    # --- Save CSV ---
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "in_house_mu_ci_table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,mu,mu_ci_lower,mu_ci_upper\n")
        for name, mu_val, mu_lower, mu_upper in rows:
            f.write(f"{name},{mu_val:.6f},{mu_lower:.6f},{mu_upper:.6f}\n")
    print(f"\n  Saved to {csv_path}")

    # --- Save JSON summary ---
    json_path = os.path.join(results_dir, "in_house_mu_ci_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_name,
            "method": "proposed",
            "ci_method": "bootstrap_percentile",
            "alpha": alpha,
            "n_bootstrap": N_BOOTSTRAP,
            "n_successful": successful,
            "selected_rank": r_model,
            "items": [{"model": name, "mu": mu_val, "mu_ci_lower": mu_lower, "mu_ci_upper": mu_upper}
                      for name, mu_val, mu_lower, mu_upper in rows],
        }, f, indent=2)
    print(f"  Saved to {json_path}")


if __name__ == "__main__":
    main()
