import numpy as np
from scipy.optimize import minimize

from .models import estimate_parameters, sigmoid
from .uq import pooled_btl_uq_summary, zhou_uq_summary


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
