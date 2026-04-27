import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr


EPS = 1e-12



def _canonicalize_columns(matrix):
    out = np.array(matrix, dtype=float, copy=True)
    for col in range(out.shape[1]):
        nz = np.flatnonzero(np.abs(out[:, col]) > EPS)
        if nz.size == 0:
            continue
        if out[nz[0], col] < 0:
            out[:, col] *= -1.0
    return out



def align_mu(mu_true, mu_est):
    mu_true = np.asarray(mu_true, dtype=float)
    mu_est = np.asarray(mu_est, dtype=float)
    mu_true_c = mu_true - np.mean(mu_true)
    mu_est_c = mu_est - np.mean(mu_est)
    denom = float(mu_est_c @ mu_est_c)
    if denom <= EPS:
        raise ValueError("estimated mu has near-zero norm and cannot be aligned")
    scale = float((mu_true_c @ mu_est_c) / denom)
    return mu_true_c, scale * mu_est_c, scale



def align_factors_with_reanchor(U, V):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    if U.shape[1] != V.shape[1]:
        raise ValueError("U and V must have the same number of columns")
    if U.shape[1] == 0:
        return U, V
    U_out = _canonicalize_columns(U)
    V_out = np.array(V, dtype=float, copy=True)
    for col in range(U.shape[1]):
        if not np.allclose(U_out[:, col], U[:, col]):
            V_out[:, col] *= -1.0
    return U_out, V_out



def compute_parameter_errors(mu_true, mu_est, gamma_true, gamma_est, U_true, U_est, V_true, V_est):
    # 1. Align the consensus score and judge scales.
    mu_true_aligned, mu_est_aligned, _ = align_mu(mu_true, mu_est)

    gamma_true = np.asarray(gamma_true, dtype=float)
    gamma_est = np.asarray(gamma_est, dtype=float)
    gamma_true_aligned = gamma_true / np.mean(gamma_true)
    gamma_est_aligned = gamma_est / np.mean(gamma_est)

    # 2. Align the heterogeneity factors using the same column convention.
    U_true_aligned, V_true_aligned = align_factors_with_reanchor(U_true, V_true)
    U_est_aligned, V_est_aligned = align_factors_with_reanchor(U_est, V_est)

    uv_true = U_true_aligned @ V_true_aligned.T
    uv_est = U_est_aligned @ V_est_aligned.T

    # 3. Report parameter-level and score-level errors.
    mu_error = float(np.linalg.norm(mu_true_aligned - mu_est_aligned))
    gamma_error = float(np.linalg.norm(gamma_true_aligned - gamma_est_aligned))
    uv_error = float(np.linalg.norm(uv_true - uv_est, ord="fro"))

    score_true = np.outer(gamma_true, mu_true) + U_true @ V_true.T
    score_est = np.outer(gamma_est, mu_est) + U_est @ V_est.T
    score_error = float(np.linalg.norm(score_true - score_est, ord="fro"))

    return {
        "mu_error": mu_error,
        "gamma_error": gamma_error,
        "uv_error": uv_error,
        "score_error": score_error,
    }



def compute_ranking_metrics(mu_true, mu_est, S_true=None, S_est=None):
    mu_true = np.asarray(mu_true, dtype=float)
    mu_est = np.asarray(mu_est, dtype=float)
    corr, pvalue = spearmanr(mu_true, mu_est)

    total_pairs = 0
    pairwise_correct = 0
    for i in range(mu_true.size):
        for j in range(i + 1, mu_true.size):
            total_pairs += 1
            sign_true = np.sign(mu_true[i] - mu_true[j])
            sign_est = np.sign(mu_est[i] - mu_est[j])
            pairwise_correct += int(sign_true == sign_est)

    metrics = {
        "spearman": float(corr),
        "spearman_pvalue": float(pvalue),
        "pairwise_accuracy": float(pairwise_correct / total_pairs),
    }

    if S_true is not None and S_est is not None:
        S_true = np.asarray(S_true, dtype=float)
        S_est = np.asarray(S_est, dtype=float)
        correct = 0
        total = 0
        K, N = S_true.shape
        for k in range(K):
            for i in range(N):
                for j in range(i + 1, N):
                    total += 1
                    correct += int(np.sign(S_true[k, i] - S_true[k, j]) == np.sign(S_est[k, i] - S_est[k, j]))
        metrics["score_sign_accuracy"] = float(correct / total)

    return metrics



def compute_near_tie_metrics(S_true, S_est, pairs):
    S_true = np.asarray(S_true, dtype=float)
    S_est = np.asarray(S_est, dtype=float)
    if S_true.shape != S_est.shape:
        raise ValueError("S_true and S_est must have the same shape")

    prob_true = expit(S_true)
    prob_est = expit(S_est)
    del prob_true, prob_est

    sign_correct = []
    distance_errors = []
    reorder_risks = []

    for i, j in pairs:
        true_diff = S_true[:, i] - S_true[:, j]
        est_diff = S_est[:, i] - S_est[:, j]
        true_prob = expit(true_diff)
        est_prob = expit(est_diff)
        sign_correct.append(np.mean(np.sign(true_diff) == np.sign(est_diff)))
        distance_errors.append(np.mean(np.abs(est_prob - true_prob)))
        reorder_risks.append(np.mean(np.abs(est_prob - 0.5) > np.abs(true_prob - 0.5)))

    return {
        "near_tie_sign_accuracy": float(np.mean(sign_correct)),
        "near_tie_probability_error": float(np.mean(distance_errors)),
        "near_tie_reordering_risk": float(np.mean(reorder_risks)),
    }
