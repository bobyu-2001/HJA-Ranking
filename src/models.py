import time
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


EPS = 1e-10



def validate_rank(N, K, r):
    r_max = min(K - 1, N - 2)
    if r < 0:
        raise ValueError(f"rank r must be nonnegative, got {r}")
    if r > r_max:
        raise ValueError(f"rank r={r} exceeds identifiable maximum {r_max} for N={N}, K={K}")
    return r_max



def sigmoid(x):
    return expit(x)



def project_zero_sum(vec):
    arr = np.asarray(vec, dtype=float)
    return arr - np.mean(arr)



def canonicalize_columns(U, V):
    U_out = np.array(U, dtype=float, copy=True)
    V_out = np.array(V, dtype=float, copy=True)
    for col in range(U_out.shape[1]):
        nz = np.flatnonzero(np.abs(U_out[:, col]) > EPS)
        if nz.size == 0:
            raise ValueError(f"column {col} is numerically zero and cannot be canonicalized")
        if U_out[nz[0], col] < 0:
            U_out[:, col] *= -1.0
            V_out[:, col] *= -1.0
    return U_out, V_out



def negative_log_likelihood(mu, gamma, U, V, n_ijk, y_ijk):
    # Negative log-likelihood L_n(theta) for S = gamma * mu^T + U V^T.
    # This is the objective used throughout Algorithm 1 and also inside
    # Initialize / rank-0 fitting.
    score = np.outer(gamma, mu) + U @ V.T
    tri_i, tri_j = np.triu_indices(score.shape[1], k=1)
    n_obs = n_ijk[:, tri_i, tri_j]
    mask = n_obs > 0
    if not np.any(mask):
        return 0.0

    y_obs = y_ijk[:, tri_i, tri_j]
    diff = score[:, tri_i] - score[:, tri_j]
    loss = y_obs * np.logaddexp(0.0, -diff) + (n_obs - y_obs) * np.logaddexp(0.0, diff)
    return float(np.sum(loss[mask]))



def negative_log_likelihood_and_grad(mu, gamma, U, V, n_ijk, y_ijk):
    score = np.outer(gamma, mu) + U @ V.T
    tri_i, tri_j = np.triu_indices(score.shape[1], k=1)
    n_obs = n_ijk[:, tri_i, tri_j]
    mask = n_obs > 0
    if not np.any(mask):
        zero_gamma = np.zeros_like(gamma)
        zero_mu = np.zeros_like(mu)
        zero_U = np.zeros_like(U)
        zero_V = np.zeros_like(V)
        return 0.0, zero_mu, zero_gamma, zero_U, zero_V

    y_obs = y_ijk[:, tri_i, tri_j]
    diff = score[:, tri_i] - score[:, tri_j]
    prob = expit(diff)
    residual = (n_obs * prob - y_obs) * mask

    loss = y_obs * np.logaddexp(0.0, -diff) + (n_obs - y_obs) * np.logaddexp(0.0, diff)
    loss_value = float(np.sum(loss[mask]))

    grad_score = np.zeros_like(score)
    for judge_index in range(score.shape[0]):
        np.add.at(grad_score[judge_index], tri_i, residual[judge_index])
        np.add.at(grad_score[judge_index], tri_j, -residual[judge_index])

    grad_gamma = grad_score @ mu
    grad_mu = grad_score.T @ gamma
    grad_U = grad_score @ V if U.size else np.zeros_like(U)
    grad_V = grad_score.T @ U if V.size else np.zeros_like(V)
    return loss_value, grad_mu, grad_gamma, grad_U, grad_V



def aggregate_judge_pairs(n_ijk, y_ijk, judge_index=None):
    K, N, _ = n_ijk.shape
    pairs = []
    if judge_index is None:
        for i in range(N):
            for j in range(i + 1, N):
                n = float(np.sum(n_ijk[:, i, j]))
                if n <= 0:
                    continue
                y = float(np.sum(y_ijk[:, i, j]))
                pairs.append((i, j, n, y))
    else:
        for i in range(N):
            for j in range(i + 1, N):
                n = float(n_ijk[judge_index, i, j])
                if n <= 0:
                    continue
                y = float(y_ijk[judge_index, i, j])
                pairs.append((i, j, n, y))
    if not pairs:
        raise ValueError("no observed pairs available for BTL fit")
    return pairs



def centered_btl_loss_and_grad_from_pairs(N, pairs, s):
    loss = 0.0
    grad = np.zeros(N, dtype=float)
    for i, j, n, y in pairs:
        diff = s[i] - s[j]
        loss += y * np.logaddexp(0.0, -diff) + (n - y) * np.logaddexp(0.0, diff)
        residual = n * expit(diff) - y
        grad[i] += residual
        grad[j] -= residual
    return float(loss), grad



def fit_centered_btl_from_pairs(N, pairs, initial=None, maxiter=500):
    # Centered BTL subroutine used by Initialize (Appendix B.1):
    # once on pooled comparisons to get mu^(0), and once per judge to get the
    # rows of \tilde S^(0).
    if initial is None:
        x0 = np.zeros(N, dtype=float)
    else:
        x0 = project_zero_sum(initial)

    def objective(s_free):
        s = project_zero_sum(s_free)
        loss, grad = centered_btl_loss_and_grad_from_pairs(N, pairs, s)
        return loss, project_zero_sum(grad)

    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "gtol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"centered BTL fit failed: {result.message}")
    return project_zero_sum(result.x)



def reanchor(gamma, mu, U, V, delta_mu=1e-8, delta_sigma=1e-8, sum_tol=1e-8, check_sum=True):
    # Appendix B.2 / ReAnchor: restore the identified representation after the
    # two block updates without changing the implied score matrix.
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)

    mu_plus = np.array(mu, dtype=float, copy=True)
    mu_norm_sq = float(mu_plus @ mu_plus)
    if mu_norm_sq < delta_mu:
        raise ValueError("ReAnchor failed: mu norm is too small")

    if V.shape[1] == 0:
        if check_sum and abs(np.sum(gamma) - gamma.size) > sum_tol:
            raise ValueError("ReAnchor failed: gamma sum constraint violated")
        return gamma, mu_plus, np.zeros_like(U), np.zeros_like(V)

    a = (V.T @ mu_plus) / mu_norm_sq
    V_bar = V - np.outer(mu_plus, a)
    gamma_cand = gamma + U @ a

    if check_sum and abs(np.sum(gamma_cand) - gamma_cand.size) > sum_tol:
        raise ValueError("ReAnchor failed: gamma candidate sum constraint violated")

    H_bar = U @ V_bar.T
    P, singular_values, Qt = np.linalg.svd(H_bar, full_matrices=False)
    rank = V.shape[1]
    if singular_values.shape[0] < rank or singular_values[rank - 1] < delta_sigma:
        raise ValueError("ReAnchor failed: heterogeneity term is rank deficient")

    sigma = singular_values[:rank]
    U_plus = P[:, :rank] @ np.diag(sigma / np.sqrt(mu_plus.size))
    V_plus = Qt[:rank, :].T * np.sqrt(mu_plus.size)
    U_plus, V_plus = canonicalize_columns(U_plus, V_plus)
    return gamma_cand, mu_plus, U_plus, V_plus



def initialize_parameters(N, K, r, n_ijk, y_ijk):
    # Appendix B.1 / Initialize: pooled centered BTL -> judgewise centered BTL
    # -> residual centering -> truncated SVD -> ReAnchor.
    validate_rank(N, K, r)

    mu0 = fit_centered_btl_from_pairs(N, aggregate_judge_pairs(n_ijk, y_ijk))
    mu0 = project_zero_sum(mu0)
    gamma0 = np.ones(K, dtype=float)

    judge_scores = np.zeros((K, N), dtype=float)
    for k in range(K):
        judge_scores[k, :] = fit_centered_btl_from_pairs(N, aggregate_judge_pairs(n_ijk, y_ijk, judge_index=k), initial=mu0)

    if r == 0:
        U0 = np.zeros((K, 0), dtype=float)
        V0 = np.zeros((N, 0), dtype=float)
        gamma0, mu0, U0, V0 = reanchor(gamma0, mu0, U0, V0, check_sum=False)
        gamma0 = gamma0 / np.sum(gamma0) * K
        return gamma0, mu0, U0, V0

    residual = judge_scores - np.outer(gamma0, mu0)
    residual_centered = residual - np.mean(residual, axis=0, keepdims=True)
    P, singular_values, Qt = np.linalg.svd(residual_centered, full_matrices=False)
    if singular_values.shape[0] < r or singular_values[r - 1] <= EPS:
        raise ValueError("initialization failed: residual SVD is rank deficient")
    sigma_root = np.sqrt(singular_values[:r])
    U0 = P[:, :r] @ np.diag(sigma_root)
    V0 = Qt[:r, :].T @ np.diag(sigma_root)
    gamma0, mu0, U0, V0 = reanchor(gamma0, mu0, U0, V0, check_sum=False)
    gamma0 = gamma0 / np.sum(gamma0) * K
    return gamma0, mu0, U0, V0



def _pack_params(gamma, mu, U, V):
    return np.concatenate([gamma, mu, U.ravel(), V.ravel()])



def _param_slices(K, N, r):
    gamma_end = K
    mu_end = gamma_end + N
    U_end = mu_end + K * r
    V_end = U_end + N * r
    return {
        "gamma": slice(0, gamma_end),
        "mu": slice(gamma_end, mu_end),
        "U": slice(mu_end, U_end),
        "V": slice(U_end, V_end),
        "size": V_end,
    }



@lru_cache(maxsize=None)
def make_centering_basis(dim):
    if dim <= 1:
        return np.zeros((dim, 0), dtype=float)
    raw = np.eye(dim, dim - 1, dtype=float)
    raw[-1, :] = -1.0
    basis, _ = np.linalg.qr(raw, mode="reduced")
    return basis



def reduce_judge_block(gamma, U):
    basis = make_centering_basis(gamma.size)
    gamma_reduced = basis.T @ (np.asarray(gamma, dtype=float) - 1.0)
    U_reduced = basis.T @ np.asarray(U, dtype=float)
    return gamma_reduced, U_reduced



def expand_judge_block(gamma_reduced, U_reduced, K, r):
    basis = make_centering_basis(K)
    gamma = np.ones(K, dtype=float) + basis @ gamma_reduced
    U = basis @ U_reduced.reshape(K - 1, r)
    return gamma, U



def reduce_item_block(mu, V):
    basis = make_centering_basis(mu.size)
    mu_reduced = basis.T @ np.asarray(mu, dtype=float)
    V_reduced = basis.T @ np.asarray(V, dtype=float)
    return mu_reduced, V_reduced



def expand_item_block(mu_reduced, V_reduced, N, r):
    basis = make_centering_basis(N)
    mu = basis @ mu_reduced
    V = basis @ V_reduced.reshape(N - 1, r)
    return mu, V


def alternating_mle(
    N,
    K,
    r,
    n_ijk,
    y_ijk,
    max_steps=120,
    tol=1e-5,
    tau=10.0,
    inner_maxiter=500,
    reanchor_steps=True,
):
    # Algorithm 1 / proximal anchored alternating MLE:
    # initialize -> judge-side update -> item-side update -> ReAnchor ->
    # stop when the relative NLL change is small.
    gamma, mu, U, V = initialize_parameters(N, K, r, n_ijk, y_ijk)
    history = []

    judge_basis = make_centering_basis(K)
    item_basis = make_centering_basis(N)

    for step in range(max_steps):
        step_index = step + 1
        step_start = time.perf_counter()
        current = _pack_params(gamma, mu, U, V)
        current_gamma_reduced, current_U_reduced = reduce_judge_block(gamma, U)
        current_judge_block = np.concatenate([current_gamma_reduced, current_U_reduced.ravel()])
        current_mu_reduced, current_V_reduced = reduce_item_block(mu, V)
        current_item_block = np.concatenate([current_mu_reduced, current_V_reduced.ravel()])
        #print(f"[alternating_mle] step {step_index}/{max_steps} start", flush=True)

        def objective_judge(block):
            gamma_new = np.ones(K, dtype=float) + judge_basis @ block[:K - 1]
            U_new = judge_basis @ block[K - 1:].reshape(K - 1, r)
            loss, _, grad_gamma, grad_U, _ = negative_log_likelihood_and_grad(mu, gamma_new, U_new, V, n_ijk, y_ijk)
            objective_value = loss + 0.5 * tau * np.sum((block - current_judge_block) ** 2)
            grad_reduced = np.empty_like(block)
            grad_reduced[:K - 1] = judge_basis.T @ grad_gamma + tau * (block[:K - 1] - current_gamma_reduced)
            grad_reduced[K - 1:] = (
                (judge_basis.T @ grad_U).ravel() + tau * (block[K - 1:] - current_U_reduced.ravel())
            )
            return objective_value, grad_reduced

        judge_start = time.perf_counter()
        result_j = minimize(
            objective_judge,
            x0=current_judge_block,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": inner_maxiter, "gtol": 1e-5, "maxls": 50},
        )
        judge_elapsed = time.perf_counter() - judge_start
        #print(
        #    f"[alternating_mle] step {step_index} judge done in {judge_elapsed:.2f}s success={result_j.success} nit={getattr(result_j, 'nit', 'NA')} message={result_j.message}",
        #    flush=True,
        #)
        if not result_j.success:
            raise RuntimeError(f"judge-side update failed: {result_j.message} (nit={getattr(result_j, 'nit', 'NA')})")
        gamma_tilde = np.ones(K, dtype=float) + judge_basis @ result_j.x[:K - 1]
        U_tilde = judge_basis @ result_j.x[K - 1:].reshape(K - 1, r)

        def objective_item(block):
            mu_new = item_basis @ block[:N - 1]
            V_new = item_basis @ block[N - 1:].reshape(N - 1, r)
            loss, grad_mu, _, _, grad_V = negative_log_likelihood_and_grad(
                mu_new,
                gamma_tilde,
                U_tilde,
                V_new,
                n_ijk,
                y_ijk,
            )
            objective_value = loss + 0.5 * tau * np.sum((block - current_item_block) ** 2)
            grad_reduced = np.empty_like(block)
            grad_reduced[:N - 1] = item_basis.T @ grad_mu + tau * (block[:N - 1] - current_mu_reduced)
            grad_reduced[N - 1:] = (item_basis.T @ grad_V).ravel() + tau * (block[N - 1:] - current_V_reduced.ravel())
            return objective_value, grad_reduced

        item_start = time.perf_counter()
        result_i = minimize(
            objective_item,
            x0=current_item_block,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": inner_maxiter, "gtol": 1e-5, "maxls": 50},
        )
        item_elapsed = time.perf_counter() - item_start
        #print(
        #    f"[alternating_mle] step {step_index} item done in {item_elapsed:.2f}s success={result_i.success} nit={getattr(result_i, 'nit', 'NA')} message={result_i.message}",
        #    flush=True,
        #)
        if not result_i.success:
            raise RuntimeError(f"item-side update failed: {result_i.message} (nit={getattr(result_i, 'nit', 'NA')})")
        mu_tilde = item_basis @ result_i.x[:N - 1]
        V_tilde = item_basis @ result_i.x[N - 1:].reshape(N - 1, r)

        if reanchor_steps:
            gamma_new, mu_new, U_new, V_new = reanchor(gamma_tilde, mu_tilde, U_tilde, V_tilde)
        else:
            gamma_new, mu_new, U_new, V_new = gamma_tilde, mu_tilde, U_tilde, V_tilde

        nll = negative_log_likelihood(mu_new, gamma_new, U_new, V_new, n_ijk, y_ijk)
        prev_nll = negative_log_likelihood(mu, gamma, U, V, n_ijk, y_ijk)
        diff = np.linalg.norm(_pack_params(gamma_new, mu_new, U_new, V_new) - current)
        rel_nll = abs(nll - prev_nll) / (1.0 + abs(prev_nll))
        step_elapsed = time.perf_counter() - step_start
        history.append({"iteration": step_index, "nll": float(nll), "diff": float(diff), "rel_nll": float(rel_nll), "tau": float(tau)})
        '''
        print(
            f"[alternating_mle] step {step_index} summary: total={step_elapsed:.2f}s nll={nll:.6f} rel_nll={rel_nll:.6e} diff={diff:.6f}",
            flush=True,
        )
        '''
        gamma, mu, U, V = gamma_new, mu_new, U_new, V_new
        if rel_nll < tol:
            return mu, gamma, U, V, {
                "n_iter": step_index,
                "converged": True,
                "history": history,
                "nll": float(nll),
                "reanchor_steps": bool(reanchor_steps),
            }

    raise RuntimeError("alternating MLE failed to converge within max_steps")



def fit_rank0_model(N, K, n_ijk, y_ijk, maxiter=2000):
    # Special case r = 0: fit S = gamma * mu^T under the same sum / zero-sum
    # constraints, without a heterogeneity term U V^T.
    item_basis = make_centering_basis(N)
    judge_basis = make_centering_basis(K)
    mu0 = fit_centered_btl_from_pairs(N, aggregate_judge_pairs(n_ijk, y_ijk))
    x0 = np.concatenate([item_basis.T @ mu0, np.zeros(K - 1, dtype=float)])

    def objective(params):
        mu = item_basis @ params[:N - 1]
        gamma = np.ones(K, dtype=float) + judge_basis @ params[N - 1:]
        loss, grad_mu, grad_gamma, _, _ = negative_log_likelihood_and_grad(
            mu,
            gamma,
            np.zeros((K, 0), dtype=float),
            np.zeros((N, 0), dtype=float),
            n_ijk,
            y_ijk,
        )
        grad = np.concatenate([item_basis.T @ grad_mu, judge_basis.T @ grad_gamma])
        return loss, grad

    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "gtol": 1e-5, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(f"rank-0 fit failed: {result.message}")

    mu = item_basis @ result.x[:N - 1]
    gamma = np.ones(K, dtype=float) + judge_basis @ result.x[N - 1:]
    U = np.zeros((K, 0), dtype=float)
    V = np.zeros((N, 0), dtype=float)
    gamma, mu, U, V = reanchor(gamma, mu, U, V)
    fit_info = {"n_iter": int(result.nit), "converged": bool(result.success), "nll": float(result.fun)}
    return mu, gamma, U, V, fit_info



def estimate_parameters(
    N,
    K,
    r,
    n_ijk,
    y_ijk,
    max_steps=120,
    tol=1e-5,
    tau=10.0,
    inner_maxiter=500,
    reanchor_steps=True,
):
    # Public estimator entry: dispatch to the rank-0 fit or Algorithm 1 depending
    # on whether the requested latent rank is zero.
    validate_rank(N, K, r)
    if r == 0:
        return fit_rank0_model(N, K, n_ijk, y_ijk)
    return alternating_mle(
        N,
        K,
        r,
        n_ijk,
        y_ijk,
        max_steps=max_steps,
        tol=tol,
        tau=tau,
        inner_maxiter=inner_maxiter,
        reanchor_steps=reanchor_steps,
    )



def score_contrast_gradient(gamma, mu, U, V, k, i, j):
    # Gradient of eta_kij = S_ki - S_kj in the full local coordinates
    # (gamma, mu, vec(U), vec(V)). The information matrix may be singular because
    # these coordinates include constrained directions; downstream UQ uses a
    # Moore-Penrose inverse, which is invariant for estimable score contrasts.
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    K = gamma.size
    N = mu.size
    r = U.shape[1]
    if not (0 <= k < K and 0 <= i < N and 0 <= j < N and i != j):
        raise ValueError(f"invalid score contrast indices {(k, i, j)} for K={K}, N={N}")
    slices = _param_slices(K, N, r)
    grad = np.zeros(slices["size"], dtype=float)
    grad[slices["gamma"].start + k] = mu[i] - mu[j]
    grad[slices["mu"].start + i] = gamma[k]
    grad[slices["mu"].start + j] = -gamma[k]
    if r > 0:
        U_offset = slices["U"].start + k * r
        grad[U_offset : U_offset + r] = V[i, :] - V[j, :]
        V_i_offset = slices["V"].start + i * r
        V_j_offset = slices["V"].start + j * r
        grad[V_i_offset : V_i_offset + r] = U[k, :]
        grad[V_j_offset : V_j_offset + r] = -U[k, :]
    return grad



def score_entry_gradient(gamma, mu, U, V, k, i):
    # Gradient of S_ki = gamma_k * mu_i + U_k^T V_i in the full local
    # coordinates used by the plug-in information matrix.
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    K = gamma.size
    N = mu.size
    r = U.shape[1]
    if not (0 <= k < K and 0 <= i < N):
        raise ValueError(f"invalid score entry indices {(k, i)} for K={K}, N={N}")
    slices = _param_slices(K, N, r)
    grad = np.zeros(slices["size"], dtype=float)
    grad[slices["gamma"].start + k] = mu[i]
    grad[slices["mu"].start + i] = gamma[k]
    if r > 0:
        U_offset = slices["U"].start + k * r
        V_i_offset = slices["V"].start + i * r
        grad[U_offset : U_offset + r] = V[i, :]
        grad[V_i_offset : V_i_offset + r] = U[k, :]
    return grad



def consensus_contrast_gradient(gamma, mu, U, V, i, j):
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    U = np.asarray(U, dtype=float)
    K = gamma.size
    N = mu.size
    r = U.shape[1]
    if not (0 <= i < N and 0 <= j < N and i != j):
        raise ValueError(f"invalid consensus contrast indices {(i, j)} for N={N}")
    slices = _param_slices(K, N, r)
    grad = np.zeros(slices["size"], dtype=float)
    grad[slices["mu"].start + i] = 1.0
    grad[slices["mu"].start + j] = -1.0
    return grad



def compute_information_matrix(gamma, mu, U, V, n_ijk):
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    n_ijk = np.asarray(n_ijk, dtype=float)
    K = gamma.size
    N = mu.size
    r = U.shape[1]
    if n_ijk.shape != (K, N, N):
        raise ValueError(f"n_ijk must have shape {(K, N, N)}, got {n_ijk.shape}")

    slices = _param_slices(K, N, r)
    info = np.zeros((slices["size"], slices["size"]), dtype=float)
    total_n = float(np.sum(n_ijk[:, np.triu_indices(N, k=1)[0], np.triu_indices(N, k=1)[1]]))
    if total_n <= 0:
        raise ValueError("information matrix requires at least one observed comparison")

    score = np.outer(gamma, mu) + U @ V.T
    for k in range(K):
        for i in range(N):
            for j in range(i + 1, N):
                cell_n = float(n_ijk[k, i, j])
                if cell_n <= 0:
                    continue
                p = expit(score[k, i] - score[k, j])
                grad = score_contrast_gradient(gamma, mu, U, V, k, i, j)
                info += (cell_n / total_n) * p * (1.0 - p) * np.outer(grad, grad)
    return info



def _evaluate_uq_target(target, gamma, mu, U, V):
    target_type = target.get("type", "score_diff")
    if target_type in {"score_diff", "judge_score_diff"}:
        k = int(target["k"])
        i = int(target["i"])
        j = int(target["j"])
        value = gamma[k] * (mu[i] - mu[j])
        if U.shape[1] > 0:
            value += U[k, :] @ (V[i, :] - V[j, :])
        grad = score_contrast_gradient(gamma, mu, U, V, k, i, j)
        label = target.get("label", f"S[{k},{i}]-S[{k},{j}]")
    elif target_type in {"score_entry", "judge_score_entry"}:
        k = int(target["k"])
        i = int(target["i"])
        value = gamma[k] * mu[i]
        if U.shape[1] > 0:
            value += U[k, :] @ V[i, :]
        grad = score_entry_gradient(gamma, mu, U, V, k, i)
        label = target.get("label", f"S[{k},{i}]")
    elif target_type == "consensus_diff":
        i = int(target["i"])
        j = int(target["j"])
        value = mu[i] - mu[j]
        grad = consensus_contrast_gradient(gamma, mu, U, V, i, j)
        label = target.get("label", f"mu[{i}]-mu[{j}]")
    elif target_type == "consensus_score":
        i = int(target["i"])
        N = mu.size
        if not (0 <= i < N):
            raise ValueError(f"invalid consensus score index {i} for N={N}")
        value = mu[i]
        slices = _param_slices(gamma.size, N, U.shape[1])
        grad = np.zeros(slices["size"], dtype=float)
        grad[slices["mu"].start + i] = 1.0
        label = target.get("label", f"mu[{i}]")
    elif target_type == "gamma":
        k = int(target["k"])
        K = gamma.size
        if not (0 <= k < K):
            raise ValueError(f"invalid gamma index {k} for K={K}")
        value = gamma[k]
        slices = _param_slices(K, mu.size, U.shape[1])
        grad = np.zeros(slices["size"], dtype=float)
        grad[slices["gamma"].start + k] = 1.0
        label = target.get("label", f"gamma[{k}]")
    else:
        raise ValueError(f"unknown UQ target type: {target_type}")
    return label, float(value), grad



def uncertainty_quantification(gamma, mu, U, V, n_ijk, targets, alpha=0.05, rcond=1e-8):
    # Algorithm B.4 / UncertaintyQuantification.
    # Returns Wald intervals for smooth scalar targets, using the plug-in
    # information matrix from Algorithm B.3 and delta-method standard errors.
    info = compute_information_matrix(gamma, mu, U, V, n_ijk)
    total_n = float(np.sum(n_ijk[:, np.triu_indices(mu.size, k=1)[0], np.triu_indices(mu.size, k=1)[1]]))
    covariance = np.linalg.pinv(info, rcond=rcond) / total_n
    z_value = float(norm.ppf(1.0 - alpha / 2.0))

    intervals = []
    for target in targets:
        label, estimate, grad = _evaluate_uq_target(target, gamma, mu, U, V)
        variance = float(grad @ covariance @ grad)
        se = float(np.sqrt(max(variance, 0.0)))
        intervals.append(
            {
                "label": label,
                "type": target.get("type", "score_diff"),
                "estimate": estimate,
                "se": se,
                "alpha": float(alpha),
                "lower": estimate - z_value * se,
                "upper": estimate + z_value * se,
            }
        )
    return {
        "intervals": intervals,
        "information_matrix": info,
        "covariance": covariance,
        "total_n": total_n,
        "z_value": z_value,
    }



def select_rank_by_bic(N, K, n_ijk, y_ijk, candidate_ranks=None, max_steps=800, tol=1e-5):
    # Fit each candidate rank and compare BIC(r) = 2 L_n(theta_hat_r) + d_r log n,
    # with d_r = r (K + N - r).
    r_max = min(K - 1, N - 2)
    if candidate_ranks is None:
        candidate_ranks = list(range(r_max + 1))
    results = {}
    n_total = float(np.sum(n_ijk[:, np.triu_indices(N, 1)[0], np.triu_indices(N, 1)[1]]))
    if n_total <= 0:
        raise ValueError("BIC selection requires positive total comparisons")

    for r in candidate_ranks:
        mu, gamma, U, V, fit_info = estimate_parameters(N, K, r, n_ijk, y_ijk, max_steps=max_steps, tol=tol, tau=10.0)
        d_r = r * (K + N - r-3)  # degrees of freedom: r (K + N - r) minus 3 for the constraints
        bic = 2.0 * fit_info["nll"] + d_r * np.log(n_total)
        results[r] = {
            "bic": float(bic),
            "mu": mu,
            "gamma": gamma,
            "U": U,
            "V": V,
            "fit_info": fit_info,
        }
    best_rank = min(results, key=lambda r: results[r]["bic"])
    return best_rank, results
