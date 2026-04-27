import numpy as np
from scipy.optimize import minimize

from .models import (
    aggregate_judge_pairs,
    centered_btl_loss_and_grad_from_pairs,
    estimate_parameters,
    make_centering_basis,
    sigmoid,
)


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



def fit_proposed(N, K, r_model, n_ijk, y_ijk, max_steps=120, tol=1e-5, tau=10.0):
    mu, gamma, U, V, fit_info = estimate_parameters(N, K, r_model, n_ijk, y_ijk, max_steps=max_steps, tol=tol, tau=tau)
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
    pairs = aggregate_judge_pairs(n_ijk, y_ijk)

    def objective(mu_free):
        mu = mu_free - np.mean(mu_free)
        loss, grad = centered_btl_loss_and_grad_from_pairs(N, pairs, mu)
        grad = grad - np.mean(grad)
        return loss, grad

    result = minimize(
        objective,
        x0=np.zeros(N, dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "gtol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"standard BTL fit failed: {result.message}")
    mu = result.x - np.mean(result.x)
    gamma = np.ones(K, dtype=float)
    U = np.zeros((K, 0), dtype=float)
    V = np.zeros((N, 0), dtype=float)
    S = np.outer(gamma, mu)
    return {
        "mu": mu,
        "gamma": gamma,
        "U": U,
        "V": V,
        "S": S,
        "fit_info": {"n_iter": int(result.nit), "converged": bool(result.success), "nll": float(result.fun)},
    }



def fit_zhou_github(N, K, n_ijk, y_ijk, maxiter=5000, tol=1e-6):
    mu_basis = make_centering_basis(N)
    alpha_basis = make_centering_basis(K)
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

    def objective(params):
        mu = mu_basis @ params[:N - 1]
        alpha = alpha_basis @ params[N - 1:]
        gamma = np.exp(alpha)
        loss = 0.0
        grad_mu = np.zeros(N, dtype=float)
        grad_alpha = np.zeros(K, dtype=float)
        for k, i, j, n, ybar in pairs:
            diff = gamma[k] * (mu[i] - mu[j])
            p = sigmoid(diff)
            y = n * ybar
            residual = n * p - y
            loss -= n * (ybar * np.log(p + EPS) + (1.0 - ybar) * np.log(1.0 - p + EPS))
            grad_mu[i] += residual * gamma[k]
            grad_mu[j] -= residual * gamma[k]
            grad_alpha[k] += residual * diff
        grad = np.concatenate([mu_basis.T @ grad_mu, alpha_basis.T @ grad_alpha])
        return float(loss), grad

    x0 = np.zeros(N + K - 2, dtype=float)
    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "gtol": tol, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(f"Zhou github benchmark fit failed: {result.message}")

    mu = mu_basis @ result.x[:N - 1]
    alpha = alpha_basis @ result.x[N - 1:]
    gamma = np.exp(alpha)
    U = np.zeros((K, 0), dtype=float)
    V = np.zeros((N, 0), dtype=float)
    S = np.outer(gamma, mu)
    return {
        "mu": mu,
        "gamma": gamma,
        "U": U,
        "V": V,
        "S": S,
        "fit_info": {"n_iter": int(result.nit), "converged": bool(result.success), "nll": float(result.fun)},
    }
