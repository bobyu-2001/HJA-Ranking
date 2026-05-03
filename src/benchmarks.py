import numpy as np
from scipy.optimize import minimize

from .models import estimate_parameters, make_centering_basis, sigmoid
from .uq import normal_ci, pooled_btl_uq_summary, zhou_uq_summary


EPS = 1e-10



def compute_score_matrix(mu, gamma, U, V):
    mu = np.asarray(mu, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)

    if mu.ndim != 1 or gamma.ndim != 1:
        raise ValueError("mu and gamma must be vectors")
    if U.ndim != 2 or V.ndim != 2:
        raise ValueError("U and V must be matrices")
    if U.shape[0] != gamma.shape[0]:
        raise ValueError("U row count must equal len(gamma)")
    if V.shape[0] != mu.shape[0]:
        raise ValueError("V row count must equal len(mu)")
    if U.shape[1] != V.shape[1]:
        raise ValueError("U and V must have the same rank")

    return np.outer(gamma, mu) + U @ V.T



def fit_proposed(N, K, r_model, n_ijk, y_ijk, max_steps=120, tol=1e-5, tau=10.0, inner_maxiter=500):
    mu, gamma, U, V, fit_info = estimate_parameters(
        N,
        K,
        r_model,
        n_ijk,
        y_ijk,
        max_steps=max_steps,
        tol=tol,
        tau=tau,
        inner_maxiter=inner_maxiter,
    )
    S = compute_score_matrix(mu, gamma, U, V)
    return {
        "mu": mu,
        "gamma": gamma,
        "U": U,
        "V": V,
        "S": S,
        "fit_info": fit_info,
    }


def fit_proposed_no_reanchor(N, K, r_model, n_ijk, y_ijk, max_steps=120, tol=1e-5, tau=10.0, inner_maxiter=500):
    mu, gamma, U, V, fit_info = estimate_parameters(
        N,
        K,
        r_model,
        n_ijk,
        y_ijk,
        max_steps=max_steps,
        tol=tol,
        tau=tau,
        inner_maxiter=inner_maxiter,
        reanchor_steps=False,
    )
    S = compute_score_matrix(mu, gamma, U, V)
    return {
        "mu": mu,
        "gamma": gamma,
        "U": U,
        "V": V,
        "S": S,
        "fit_info": fit_info,
    }



def fit_standard_btl(N, K, n_ijk, y_ijk, maxiter=800):
    def objective(mu_free):
        mu = mu_free - np.mean(mu_free)
        loss = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                n = np.sum(n_ijk[:, i, j])
                if n <= 0:
                    continue
                y = np.sum(y_ijk[:, i, j])
                diff = mu[i] - mu[j]
                loss += y * np.logaddexp(0.0, -diff) + (n - y) * np.logaddexp(0.0, diff)
        return float(loss)

    result = minimize(objective, x0=np.zeros(N, dtype=float), method="L-BFGS-B", options={"maxiter": maxiter})
    if not result.success:
        raise RuntimeError(f"standard BTL fit failed: {result.message}")
    mu = result.x - np.mean(result.x)
    gamma = np.ones(K, dtype=float)
    U = np.zeros((K, 0), dtype=float)
    V = np.zeros((N, 0), dtype=float)
    S = np.outer(gamma, mu)
    try:
        uq = pooled_btl_uq_summary(mu, n_ijk, K)
    except Exception as exc:
        uq = {"method": "pooled_btl_asymptotic_delta", "error": str(exc)}
    return {
        "mu": mu,
        "gamma": gamma,
        "U": U,
        "V": V,
        "S": S,
        "uq": uq,
        "fit_info": {"n_iter": int(result.nit), "converged": bool(result.success), "nll": float(result.fun)},
    }


def _direct_score_nll_and_grad(S, n_ijk, y_ijk):
    tri_i, tri_j = np.triu_indices(S.shape[1], k=1)
    n_obs = n_ijk[:, tri_i, tri_j]
    y_obs = y_ijk[:, tri_i, tri_j]
    mask = n_obs > 0
    if not np.any(mask):
        return 0.0, np.zeros_like(S)

    diff = S[:, tri_i] - S[:, tri_j]
    prob = sigmoid(diff)
    loss = y_obs * np.logaddexp(0.0, -diff) + (n_obs - y_obs) * np.logaddexp(0.0, diff)
    residual = (n_obs * prob - y_obs) * mask

    grad = np.zeros_like(S)
    for k in range(S.shape[0]):
        np.add.at(grad[k], tri_i, residual[k])
        np.add.at(grad[k], tri_j, -residual[k])
    return float(np.sum(loss[mask])), grad


def _direct_score_uq_summary(S_hat, n_ijk, alpha_level=0.05, rcond=1e-8):
    S_hat = np.asarray(S_hat, dtype=float)
    n_ijk = np.asarray(n_ijk, dtype=float)
    K, N = S_hat.shape
    basis = make_centering_basis(N)
    S_ci = []
    total_count = 0.0
    tri_i, tri_j = np.triu_indices(N, 1)

    for k in range(K):
        info = np.zeros((N - 1, N - 1), dtype=float)
        judge_total = float(np.sum(n_ijk[k, tri_i, tri_j]))
        total_count += judge_total
        if judge_total <= 0:
            raise ValueError(f"direct score UQ requires judge {k} to have observed comparisons")

        for i in range(N):
            for j in range(i + 1, N):
                n = float(n_ijk[k, i, j])
                if n <= 0:
                    continue
                diff = S_hat[k, i] - S_hat[k, j]
                p = sigmoid(diff)
                contrast_grad = basis[i, :] - basis[j, :]
                info += (n / judge_total) * p * (1.0 - p) * np.outer(contrast_grad, contrast_grad)

        cov_reduced = np.linalg.pinv(0.5 * (info + info.T), rcond=rcond)
        row_ci = []
        for i in range(N):
            entry_grad = basis[i, :]
            variance = float(entry_grad @ cov_reduced @ entry_grad) / judge_total
            row_ci.append(normal_ci(S_hat[k, i], variance, alpha_level=alpha_level))
        S_ci.append(row_ci)

    return {
        "method": "direct_score_rowwise_btl_delta",
        "alpha_level": float(alpha_level),
        "total_count": float(total_count),
        "S_ci": S_ci,
    }


def fit_direct_score_svd(N, K, r_model, n_ijk, y_ijk, maxiter=500, tol=1e-6):
    basis = make_centering_basis(N)
    z0 = np.zeros((K, N - 1), dtype=float)

    def objective(params):
        Z = params.reshape(K, N - 1)
        S = Z @ basis.T
        loss, grad_S = _direct_score_nll_and_grad(S, n_ijk, y_ijk)
        grad_Z = grad_S @ basis
        return loss, grad_Z.ravel()

    result = minimize(
        objective,
        x0=z0.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "gtol": tol, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(f"direct score MLE fit failed: {result.message}")

    S_mle = result.x.reshape(K, N - 1) @ basis.T
    mu = np.mean(S_mle, axis=0)
    mu = mu - np.mean(mu)
    mu_norm_sq = float(mu @ mu)
    if mu_norm_sq <= EPS:
        raise RuntimeError("direct score MLE produced a near-zero consensus score")

    gamma = (S_mle @ mu) / mu_norm_sq
    residual = S_mle - np.outer(gamma, mu)

    rank = int(min(r_model, min(residual.shape)))
    if rank > 0:
        P, singular_values, Qt = np.linalg.svd(residual, full_matrices=False)
        sigma = singular_values[:rank]
        U = P[:, :rank] @ np.diag(np.sqrt(sigma))
        V = Qt[:rank, :].T @ np.diag(np.sqrt(sigma))
    else:
        U = np.zeros((K, 0), dtype=float)
        V = np.zeros((N, 0), dtype=float)

    S = compute_score_matrix(mu, gamma, U, V)
    try:
        uq = _direct_score_uq_summary(S, n_ijk)
    except Exception as exc:
        uq = {"method": "direct_score_rowwise_btl_delta", "error": str(exc)}

    return {
        "mu": mu,
        "gamma": gamma,
        "U": U,
        "V": V,
        "S": S,
        "S_mle": S_mle,
        "uq": uq,
        "fit_info": {
            "n_iter": int(result.nit),
            "converged": bool(result.success),
            "nll": float(result.fun),
            "direct_score_nll": float(result.fun),
            "rank": int(rank),
        },
    }


def fit_unstructured_btl_svd(N, K, r_model, n_ijk, y_ijk, maxiter=500, tol=1e-6):
    basis = make_centering_basis(N)
    z0 = np.zeros((K, N - 1), dtype=float)

    def objective(params):
        Z = params.reshape(K, N - 1)
        S = Z @ basis.T
        loss, grad_S = _direct_score_nll_and_grad(S, n_ijk, y_ijk)
        grad_Z = grad_S @ basis
        return loss, grad_Z.ravel()

    result = minimize(
        objective,
        x0=z0.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "gtol": tol, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(f"unstructured BTL score fit failed: {result.message}")

    S_mle = result.x.reshape(K, N - 1) @ basis.T
    rank = int(min(r_model + 1, min(S_mle.shape)))
    P, singular_values, Qt = np.linalg.svd(S_mle, full_matrices=False)
    S = (P[:, :rank] * singular_values[:rank]) @ Qt[:rank, :]
    mu = np.mean(S, axis=0)
    mu = mu - np.mean(mu)

    gamma = np.ones(K, dtype=float)
    residual = S - np.outer(gamma, mu)
    if residual.size:
        P_res, singular_values_res, Qt_res = np.linalg.svd(residual, full_matrices=False)
        residual_rank = int(np.sum(singular_values_res > EPS))
        if residual_rank > 0:
            sigma = singular_values_res[:residual_rank]
            U = P_res[:, :residual_rank] @ np.diag(np.sqrt(sigma))
            V = Qt_res[:residual_rank, :].T @ np.diag(np.sqrt(sigma))
        else:
            U = np.zeros((K, 0), dtype=float)
            V = np.zeros((N, 0), dtype=float)
    else:
        U = np.zeros((K, 0), dtype=float)
        V = np.zeros((N, 0), dtype=float)

    return {
        "mu": mu,
        "gamma": gamma,
        "U": U,
        "V": V,
        "S": S,
        "S_mle": S_mle,
        "uq": {
            "method": "unstructured_btl_rank_truncated_svd",
            "error": "Coverage is omitted because the post-MLE truncated SVD is a nonsmooth projection with no simple delta-method interval formula here.",
        },
        "fit_info": {
            "n_iter": int(result.nit),
            "converged": bool(result.success),
            "nll": float(result.fun),
            "direct_score_nll": float(result.fun),
            "rank": int(rank),
        },
    }



def _zhou_loglik(mu, alpha, pairs):
    gamma = np.exp(alpha)
    ll = 0.0
    for k, i, j, n, ybar in pairs:
        diff = gamma[k] * (mu[i] - mu[j])
        p = sigmoid(diff)
        ll += n * (ybar * np.log(p + EPS) + (1.0 - ybar) * np.log(1.0 - p + EPS))
    return float(ll)



def _zhou_mle_adam(
    N,
    K,
    pairs,
    lr_s=1e-2,
    lr_a=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-6,
    max_iter=500,
    tol=1e-5,
):
    mu = np.zeros(N, dtype=float)
    mu -= np.mean(mu)
    alpha = np.zeros(K, dtype=float)
    alpha -= np.mean(alpha)

    m_s = np.zeros_like(mu)
    v_s = np.zeros_like(mu)
    m_a = np.zeros_like(alpha)
    v_a = np.zeros_like(alpha)

    converged = False
    diff_norm = np.inf
    grad_norm = np.inf
    t = 0

    for t in range(1, max_iter + 1):
        g_s = np.zeros_like(mu)
        g_a = np.zeros_like(alpha)

        for k, i, j, n, ybar in pairs:
            gamma_k = np.exp(alpha[k])
            diff = gamma_k * (mu[i] - mu[j])
            p = sigmoid(diff)
            residual = ybar - p

            g_s[i] += n * gamma_k * residual
            g_s[j] -= n * gamma_k * residual
            g_a[k] += n * gamma_k * residual * (mu[i] - mu[j])

        grad_norm = max(np.linalg.norm(g_s), np.linalg.norm(g_a))

        m_s = beta1 * m_s + (1.0 - beta1) * g_s
        v_s = beta2 * v_s + (1.0 - beta2) * (g_s ** 2)
        m_s_hat = m_s / (1.0 - beta1 ** t)
        v_s_hat = v_s / (1.0 - beta2 ** t)
        mu_new = mu + lr_s * m_s_hat / (np.sqrt(v_s_hat) + eps)

        m_a = beta1 * m_a + (1.0 - beta1) * g_a
        v_a = beta2 * v_a + (1.0 - beta2) * (g_a ** 2)
        m_a_hat = m_a / (1.0 - beta1 ** t)
        v_a_hat = v_a / (1.0 - beta2 ** t)
        alpha_new = alpha + lr_a * m_a_hat / (np.sqrt(v_a_hat) + eps)

        mu_new -= np.mean(mu_new)
        alpha_new -= np.mean(alpha_new)

        diff_norm = max(np.linalg.norm(mu_new - mu), np.linalg.norm(alpha_new - alpha))

        mu, alpha = mu_new, alpha_new

        if diff_norm < tol:
            converged = True
            break

    gamma = np.exp(alpha)
    nll = -_zhou_loglik(mu, alpha, pairs)
    return mu, gamma, {
        "n_iter": int(t),
        "converged": bool(converged),
        "nll": float(nll),
        "diff_norm": float(diff_norm),
        "grad_norm": float(grad_norm),
    }



def fit_zhou_github(N, K, n_ijk, y_ijk, max_iter=500, tol=1e-5):
    pairs = []
    for k in range(K):
        for i in range(N):
            for j in range(i + 1, N):
                n = float(n_ijk[k, i, j])
                if n <= 0:
                    continue
                y = float(y_ijk[k, i, j])
                pairs.append((k, i, j, n, y / n))
    if not pairs:
        raise ValueError("Zhou benchmark requires at least one observed comparison")

    mu, gamma, fit_info = _zhou_mle_adam(N, K, pairs, max_iter=max_iter, tol=tol)
    U = np.zeros((K, 0), dtype=float)
    V = np.zeros((N, 0), dtype=float)
    S = np.outer(gamma, mu)
    try:
        uq = zhou_uq_summary(mu, gamma, n_ijk)
    except Exception as exc:
        uq = {"method": "zhou_asymptotic_delta", "error": str(exc)}
    return {
        "mu": mu,
        "gamma": gamma,
        "U": U,
        "V": V,
        "S": S,
        "uq": uq,
        "fit_info": fit_info,
    }
