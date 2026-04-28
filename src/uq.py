import numpy as np
from scipy.stats import norm

from .models import sigmoid


NEGATIVE_VARIANCE_TOL = 1e-10


def orthonormal_zero_sum_basis(size):
    if int(size) < 2:
        raise ValueError(f"zero-sum basis requires size >= 2, got {size}")
    basis = np.zeros((int(size), int(size) - 1), dtype=float)
    basis[: int(size) - 1, : int(size) - 1] = np.eye(int(size) - 1)
    basis[int(size) - 1, :] = -1.0
    q, _ = np.linalg.qr(basis)
    return q[:, : int(size) - 1]


def observed_pair_counts_by_judge(n_ijk):
    n_ijk = np.asarray(n_ijk, dtype=float)
    if n_ijk.ndim != 3 or n_ijk.shape[1] != n_ijk.shape[2]:
        raise ValueError("n_ijk must have shape (K, N, N)")
    counts = {}
    K, N, _ = n_ijk.shape
    for k in range(K):
        for i in range(N):
            for j in range(i + 1, N):
                n = float(n_ijk[k, i, j])
                if n > 0:
                    counts[(i, j, k)] = n
    return counts


def observed_pair_counts_pooled(n_ijk):
    n_ijk = np.asarray(n_ijk, dtype=float)
    if n_ijk.ndim != 3 or n_ijk.shape[1] != n_ijk.shape[2]:
        raise ValueError("n_ijk must have shape (K, N, N)")
    counts = {}
    _, N, _ = n_ijk.shape
    for i in range(N):
        for j in range(i + 1, N):
            n = float(np.sum(n_ijk[:, i, j]))
            if n > 0:
                counts[(i, j)] = n
    return counts


def zhou_information_matrix(s_hat, gamma_hat, pair_counts):
    s_hat = np.asarray(s_hat, dtype=float)
    gamma_hat = np.asarray(gamma_hat, dtype=float)
    N = s_hat.shape[0]
    K = gamma_hat.shape[0]
    if N < 2 or K < 1:
        raise ValueError("Zhou UQ requires at least two items and one judge")

    A_s = orthonormal_zero_sum_basis(N)
    A_alpha = orthonormal_zero_sum_basis(K) if K > 1 else np.zeros((K, 0), dtype=float)
    total_count = float(sum(pair_counts.values()))
    if total_count <= 0:
        raise ValueError("total comparison count must be positive")

    dim = (N - 1) + max(K - 1, 0)
    information = np.zeros((dim, dim), dtype=float)

    for (i, j, k), n in pair_counts.items():
        n = float(n)
        if n <= 0:
            continue
        z = gamma_hat[k] * (s_hat[i] - s_hat[j])
        weight = (n / total_count) * sigmoid(z) * (1.0 - sigmoid(z))
        if weight == 0.0:
            continue
        score_grad = gamma_hat[k] * (A_s[i, :] - A_s[j, :])
        alpha_grad = gamma_hat[k] * (s_hat[i] - s_hat[j]) * A_alpha[k, :]
        grad = np.concatenate([score_grad, alpha_grad])
        information += weight * np.outer(grad, grad)

    return 0.5 * (information + information.T), A_s, A_alpha


def pooled_btl_information_matrix(s_hat, pair_counts):
    s_hat = np.asarray(s_hat, dtype=float)
    N = s_hat.shape[0]
    if N < 2:
        raise ValueError("pooled BTL UQ requires at least two items")

    A_s = orthonormal_zero_sum_basis(N)
    total_count = float(sum(pair_counts.values()))
    if total_count <= 0:
        raise ValueError("total comparison count must be positive")

    information = np.zeros((N - 1, N - 1), dtype=float)
    for (i, j), n in pair_counts.items():
        n = float(n)
        if n <= 0:
            continue
        z = s_hat[i] - s_hat[j]
        weight = (n / total_count) * sigmoid(z) * (1.0 - sigmoid(z))
        if weight == 0.0:
            continue
        grad = A_s[i, :] - A_s[j, :]
        information += weight * np.outer(grad, grad)

    return 0.5 * (information + information.T), A_s


def covariance_from_information(information, ridge=0.0):
    information = np.asarray(information, dtype=float)
    if ridge > 0:
        information = information + float(ridge) * np.eye(information.shape[0])
    return np.linalg.inv(information)


def zhou_parameter_covariance(s_hat, gamma_hat, n_ijk, ridge=0.0):
    pair_counts = observed_pair_counts_by_judge(n_ijk)
    information, A_s, A_alpha = zhou_information_matrix(s_hat, gamma_hat, pair_counts)
    covariance_reduced = covariance_from_information(information, ridge=ridge)

    N = len(s_hat)
    K = len(gamma_hat)
    jacobian = np.block(
        [
            [A_s, np.zeros((N, max(K - 1, 0)))],
            [np.zeros((K, N - 1)), A_alpha],
        ]
    )
    covariance_s_alpha = jacobian @ covariance_reduced @ jacobian.T

    covariance_ss = covariance_s_alpha[:N, :N]
    covariance_s_alpha_cross = covariance_s_alpha[:N, N:]
    covariance_alpha_s_cross = covariance_s_alpha[N:, :N]
    covariance_alpha_alpha = covariance_s_alpha[N:, N:]
    gamma_diag = np.diag(np.asarray(gamma_hat, dtype=float))
    covariance_s_gamma = np.block(
        [
            [covariance_ss, covariance_s_alpha_cross @ gamma_diag],
            [gamma_diag @ covariance_alpha_s_cross, gamma_diag @ covariance_alpha_alpha @ gamma_diag],
        ]
    )
    return {
        "covariance_reduced": covariance_reduced,
        "covariance_s_alpha": covariance_s_alpha,
        "covariance_s_gamma": covariance_s_gamma,
        "total_count": float(sum(pair_counts.values())),
    }


def pooled_btl_score_covariance(s_hat, n_ijk, ridge=0.0):
    pair_counts = observed_pair_counts_pooled(n_ijk)
    information, A_s = pooled_btl_information_matrix(s_hat, pair_counts)
    covariance_reduced = covariance_from_information(information, ridge=ridge)
    return {
        "covariance_s": A_s @ covariance_reduced @ A_s.T,
        "covariance_reduced": covariance_reduced,
        "total_count": float(sum(pair_counts.values())),
    }


def normal_ci(value, variance, alpha_level=0.05):
    variance = float(variance)
    if variance < 0:
        if variance > -NEGATIVE_VARIANCE_TOL:
            variance = 0.0
        else:
            raise RuntimeError(f"negative variance estimate: {variance}")
    se = float(np.sqrt(variance))
    z_value = float(norm.ppf(1.0 - float(alpha_level) / 2.0))
    value = float(value)
    return {
        "estimate": value,
        "standard_error": se,
        "lower": value - z_value * se,
        "upper": value + z_value * se,
    }


def ci_for_s_or_gamma(s_hat, gamma_hat, covariance_s_gamma, total_count, alpha_level=0.05, which="s", idx=0):
    s_hat = np.asarray(s_hat, dtype=float)
    gamma_hat = np.asarray(gamma_hat, dtype=float)
    theta_hat = np.concatenate([s_hat, gamma_hat])
    if which == "s":
        theta_idx = int(idx)
    elif which == "gamma":
        theta_idx = s_hat.shape[0] + int(idx)
    else:
        raise ValueError("which must be 's' or 'gamma'")
    variance = float(covariance_s_gamma[theta_idx, theta_idx]) / float(total_count)
    return normal_ci(theta_hat[theta_idx], variance, alpha_level=alpha_level)


def ci_for_s(s_hat, covariance_s, total_count, alpha_level=0.05, idx=0):
    s_hat = np.asarray(s_hat, dtype=float)
    idx = int(idx)
    variance = float(covariance_s[idx, idx]) / float(total_count)
    return normal_ci(s_hat[idx], variance, alpha_level=alpha_level)


def ci_for_score_matrix_element(s_hat, gamma_hat, covariance_s_gamma, total_count, judge_idx, item_idx, alpha_level=0.05):
    s_hat = np.asarray(s_hat, dtype=float)
    gamma_hat = np.asarray(gamma_hat, dtype=float)
    N = s_hat.shape[0]
    K = gamma_hat.shape[0]
    judge_idx = int(judge_idx)
    item_idx = int(item_idx)
    if not 0 <= item_idx < N:
        raise IndexError(f"item_idx={item_idx} out of bounds for N={N}")
    if not 0 <= judge_idx < K:
        raise IndexError(f"judge_idx={judge_idx} out of bounds for K={K}")

    grad = np.zeros(N + K, dtype=float)
    grad[item_idx] = gamma_hat[judge_idx]
    grad[N + judge_idx] = s_hat[item_idx]
    variance = float(grad @ covariance_s_gamma @ grad) / float(total_count)
    estimate = float(gamma_hat[judge_idx] * s_hat[item_idx])
    return normal_ci(estimate, variance, alpha_level=alpha_level)


def ci_for_pooled_score_matrix_element(s_hat, covariance_s, total_count, judge_idx, item_idx, alpha_level=0.05):
    del judge_idx
    return ci_for_s(s_hat, covariance_s, total_count, alpha_level=alpha_level, idx=item_idx)


def zhou_uq_summary(s_hat, gamma_hat, n_ijk, alpha_level=0.05, ridge=0.0):
    covariance = zhou_parameter_covariance(s_hat, gamma_hat, n_ijk, ridge=ridge)
    covariance_s_gamma = covariance["covariance_s_gamma"]
    total_count = covariance["total_count"]
    N = len(s_hat)
    K = len(gamma_hat)
    return {
        "method": "zhou_asymptotic_delta",
        "alpha_level": float(alpha_level),
        "total_count": total_count,
        "s_ci": [
            ci_for_s_or_gamma(s_hat, gamma_hat, covariance_s_gamma, total_count, alpha_level, "s", idx)
            for idx in range(N)
        ],
        "gamma_ci": [
            ci_for_s_or_gamma(s_hat, gamma_hat, covariance_s_gamma, total_count, alpha_level, "gamma", idx)
            for idx in range(K)
        ],
        "S_ci": [
            [
                ci_for_score_matrix_element(
                    s_hat,
                    gamma_hat,
                    covariance_s_gamma,
                    total_count,
                    judge_idx=k,
                    item_idx=i,
                    alpha_level=alpha_level,
                )
                for i in range(N)
            ]
            for k in range(K)
        ],
    }


def pooled_btl_uq_summary(s_hat, n_ijk, num_judges, alpha_level=0.05, ridge=0.0):
    covariance = pooled_btl_score_covariance(s_hat, n_ijk, ridge=ridge)
    covariance_s = covariance["covariance_s"]
    total_count = covariance["total_count"]
    N = len(s_hat)
    K = int(num_judges)
    s_ci = [ci_for_s(s_hat, covariance_s, total_count, alpha_level=alpha_level, idx=idx) for idx in range(N)]
    return {
        "method": "pooled_btl_asymptotic_delta",
        "alpha_level": float(alpha_level),
        "total_count": total_count,
        "s_ci": s_ci,
        "S_ci": [
            [
                ci_for_pooled_score_matrix_element(
                    s_hat,
                    covariance_s,
                    total_count,
                    judge_idx=k,
                    item_idx=i,
                    alpha_level=alpha_level,
                )
                for i in range(N)
            ]
            for k in range(K)
        ],
    }
