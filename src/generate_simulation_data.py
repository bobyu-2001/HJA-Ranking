import math
import numpy as np


EPS = 1e-12


def validate_rank(N, K, r):
    # Identifiability restricts the latent rank to r <= min(K - 1, N - 2):
    # the judge-side factor U must satisfy 1_K^T U = 0, so its columns live in a
    # (K - 1)-dimensional subspace; the item-side factor V must satisfy both
    # 1_N^T V = 0 and (after re-anchoring) (mu^+)^T V = 0, so its columns live in
    # at most an (N - 2)-dimensional subspace. This is exactly the admissible rank
    # used by the paper's parameter space Theta_r / identifiable chart.
    r_max = min(K - 1, N - 2)
    if r < 0:
        raise ValueError(f"rank r must be nonnegative, got {r}")
    if r > r_max:
        raise ValueError(f"rank r={r} exceeds identifiable maximum {r_max} for N={N}, K={K}")
    return r_max



def zero_sum_basis(dim):
    if dim < 2:
        raise ValueError(f"dim must be at least 2, got {dim}")
    basis = np.zeros((dim, dim - 1), dtype=float)
    basis[: dim - 1, : dim - 1] = np.eye(dim - 1)
    basis[dim - 1, :] = -1.0
    q, _ = np.linalg.qr(basis)
    return q[:, : dim - 1]



def _canonicalize_columns(matrix):
    # The factorization U V^T is invariant under column-wise sign flips, so without
    # a sign convention the same heterogeneity term has multiple equivalent
    # representations. We canonicalize each column by requiring the first nonzero
    # entry to be positive, matching the paper's ReAnchor step: "Flip the
    # corresponding columns ... so that the first nonzero entry of each column of
    # U^+ is positive."
    out = np.array(matrix, dtype=float, copy=True)
    if out.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    for col in range(out.shape[1]):
        nz = np.flatnonzero(np.abs(out[:, col]) > EPS)
        if nz.size == 0:
            raise ValueError(f"column {col} is numerically zero and cannot be canonicalized")
        if out[nz[0], col] < 0:
            out[:, col] *= -1.0
    return out



def generate_true_parameters(N, K, r, random_seed=42, heterogeneity_scale=1.0):
    '''
    Parameters:
    - N: number of items
    - K: number of judges
    - r: rank
    - heterogeneity_scale: scale of the UV^T term

    Returns:
    - mu_true: consensus item score vector, (N,), sum(mu) = 0
    - gamma_true: judge-specific discrimination parameters, (K,), sum(gamma) = K, gamma >= 0
    - U_true: judge loadings on latent disagreement axes, (K, r)
    - V_true: item positions on those axes, (N, r)
    '''
    validate_rank(N, K, r)
    if heterogeneity_scale < 0:
        raise ValueError(f"heterogeneity_scale must be nonnegative, got {heterogeneity_scale}")

    rng = np.random.default_rng(random_seed)

    ## 1. generate mu_true, sum(mu) = 0 (condition 2)
    mu_true = rng.normal(size=N)
    mu_true = mu_true - np.mean(mu_true)

    ## 2. generate gamma_true
    # np.ones(K).T @ gamma_true should be K (condition 3)
    # use a more heterogeneous simplex draw so judge scales are more dispersed
    gamma_true = K * rng.dirichlet(np.ones(K))

    if r == 0:
        U_true = np.zeros((K, 0), dtype=float)
        V_true = np.zeros((N, 0), dtype=float)
        return mu_true, gamma_true, U_true, V_true

    ## 3. generate V_true
    # V_true.T @ np.ones(N) should be 0 (condition 2)
    # mu_true.T @ V_true should be 0 (condition 5)
    # the singular strengths are fixed in descending order to preserve identifiability
    raw_v = rng.normal(size=(N, r))
    raw_v = raw_v - np.mean(raw_v, axis=0, keepdims=True)
    raw_v = raw_v - np.outer(mu_true, (mu_true @ raw_v) / max(mu_true @ mu_true, EPS))
    q_v, _ = np.linalg.qr(raw_v)
    strengths = np.linspace(r, 1, r, dtype=float)
    V_true = q_v[:, :r] @ np.diag(np.sqrt(N) * strengths)
    V_true = V_true - np.mean(V_true, axis=0, keepdims=True)
    V_true = V_true - np.outer(mu_true, (mu_true @ V_true) / max(mu_true @ mu_true, EPS))

    ## 4. generate U_true
    # np.ones(K).T @ U_true should be 0 (condition 3)
    # the column strengths match V_true and remain strictly descending (condition 4)
    raw_u = rng.normal(size=(K, r))
    raw_u = raw_u - np.mean(raw_u, axis=0, keepdims=True)
    q_u, _ = np.linalg.qr(raw_u)
    U_true = q_u[:, :r] @ np.diag(np.sqrt(K) * strengths)

    # canonicalize signs and then scale the heterogeneity part
    U_true = _canonicalize_columns(U_true) * math.sqrt(heterogeneity_scale)
    V_true = _canonicalize_columns(V_true) * math.sqrt(heterogeneity_scale)
    mu_true = mu_true - np.mean(mu_true)

    return mu_true, gamma_true, U_true, V_true



def compute_score_matrix(mu, gamma, U, V):
    '''
    S = gamma * mu.T + U @ V.T

    Shape:
    - gamma: (K,)
    - mu: (N,)
    - U: (K, r)
    - V: (N, r)

    Returns:
    - S: (K, N)
    '''
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

    consensus_score = np.outer(gamma, mu)
    heterogeneity_score = U @ V.T
    score_matrix = consensus_score + heterogeneity_score
    return score_matrix



def build_near_tie_parameters(mu_true, gamma_true, U_true, V_true, pair=(0, 1), target_logit_diff=0.05):
    '''
    Modify a true parameter tuple so that one selected item pair becomes a near-tie.

    The goal is to make the target pair's judge-specific logit difference
    S_{k,i} - S_{k,j} small, so the corresponding comparison probability is close
    to 1/2. This is useful for the near-tie benchmark, where we want a controlled
    pair whose ordering is intentionally hard to recover.

    Parameters:
    - pair=(i, j): the item pair whose preference gap should be shrunk toward zero.
    - target_logit_diff: the desired consensus-scale logit gap for that pair. A
      value near 0 means a harder near-tie; exactly 0 would correspond to a 50-50
      comparison in the consensus part.
    '''

    mu = np.array(mu_true, dtype=float, copy=True)
    gamma = np.array(gamma_true, dtype=float, copy=True)
    U = np.array(U_true, dtype=float, copy=True)
    V = np.array(V_true, dtype=float, copy=True)

    i, j = pair
    if not (0 <= i < mu.size and 0 <= j < mu.size and i != j):
        raise ValueError(f"invalid near-tie pair {pair} for N={mu.size}")

    # 1. Make the heterogeneity part agree on the target pair.
    midpoint = 0.5 * (V[i, :] + V[j, :]) if V.shape[1] > 0 else np.zeros(0)
    if V.shape[1] > 0:
        # Setting V[i, :] and V[j, :] to the same midpoint makes their
        # heterogeneity contributions identical, so U_k^T(V_i - V_j) = 0 for every
        # judge k on the target pair. This isolates the near-tie adjustment to the
        # consensus term gamma_k (mu_i - mu_j).
        V[i, :] = midpoint
        V[j, :] = midpoint
        # Re-center V so it continues to satisfy the item-side zero-sum constraint
        # 1_N^T V = 0 used throughout the paper.
        V = V - np.mean(V, axis=0, keepdims=True)
        mu_norm_sq = max(mu @ mu, EPS)
        V = V - np.outer(mu, (mu @ V) / mu_norm_sq)

    # 2. Move the consensus scores so the target pair has logit gap close to target_logit_diff.
    avg_gamma = np.mean(gamma)
    # Since the score difference contributed by the consensus term is
    # gamma_k (mu_i - mu_j), we back out the required gap in mu by dividing the
    # target logit difference by a representative judge scale (avg_gamma). Here mean(gamma)=1
    # because sum(gamma)=K, so this mostly serves as a transparent normalization.
    target_gap = target_logit_diff / max(avg_gamma, EPS)
    # Keep the pair's midpoint fixed while shrinking only the difference; this
    # makes the modification as local as possible instead of shifting both items
    # in the same direction.
    pair_mean = 0.5 * (mu[i] + mu[j])
    mu[i] = pair_mean + 0.5 * target_gap
    mu[j] = pair_mean - 0.5 * target_gap
    mu = mu - np.mean(mu)

    # 3. Re-enforce orthogonality after changing mu.
    # The paper's identified parameter space requires mu^T V = 0. After editing mu,
    # the old V need not remain orthogonal to the new mu, so we project V back off
    # the mu direction to restore that constraint.
    if V.shape[1] > 0:
        mu_norm_sq = max(mu @ mu, EPS)
        V = V - np.outer(mu, (mu @ V) / mu_norm_sq)

    return mu, gamma, U, V


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))



def generate_balanced_comparisons(S, total_comparisons, random_seed=42):
    '''
    Near-balanced design over judge-pair cells.
    Every (k, i, j) cell gets either floor(T / n_cells) or ceil(T / n_cells) comparisons, 
    so arbitrary total sample sizes are allowed while the design stays as even as possible.
    '''
    S = np.asarray(S, dtype=float)
    K, N = S.shape
    n_pairs = N * (N - 1) // 2
    n_cells = K * n_pairs
    if total_comparisons <= 0:
        raise ValueError(f"total_comparisons must be positive, got {total_comparisons}")

    base_per_cell = total_comparisons // n_cells
    remainder = total_comparisons % n_cells
    rng = np.random.default_rng(random_seed)
    comparisons = []
    extra_cell_indices = set(rng.choice(n_cells, size=remainder, replace=False).tolist()) if remainder > 0 else set()

    cell_idx = 0
    for k in range(K):
        for i in range(N):
            for j in range(i + 1, N):
                prob = _logistic(S[k, i] - S[k, j])
                cell_count = base_per_cell + int(cell_idx in extra_cell_indices)
                if cell_count > 0:
                    draws = rng.binomial(1, prob, size=cell_count)
                    comparisons.extend((k, i, j, int(y)) for y in draws)
                cell_idx += 1

    return comparisons



def generate_unbalanced_comparisons(S, total_comparisons, mode="uniform", random_seed=42):
    '''
    Unbalanced design with either uniform or biased judge/pair sampling.
    '''
    S = np.asarray(S, dtype=float)
    K, N = S.shape
    if total_comparisons <= 0:
        raise ValueError(f"total_comparisons must be positive, got {total_comparisons}")
    if mode not in {"uniform", "biased"}:
        raise ValueError(f"unknown unbalanced mode: {mode}")

    rng = np.random.default_rng(random_seed)
    pair_list = [(i, j) for i in range(N) for j in range(i + 1, N)]
    pair_count = len(pair_list)

    if mode == "uniform":
        judge_weights = np.full(K, 1.0 / K)
        pair_weights = np.full(pair_count, 1.0 / pair_count)
    else:
        judge_weights = np.linspace(1.0, 2.5, K, dtype=float)
        judge_weights = judge_weights / np.sum(judge_weights)
        pair_weights = np.array([1.0 + 2.0 / (j - i) for i, j in pair_list], dtype=float)
        pair_weights = pair_weights / np.sum(pair_weights)

    comparisons = []
    for _ in range(total_comparisons):
        k = int(rng.choice(K, p=judge_weights))
        pair_idx = int(rng.choice(pair_count, p=pair_weights))
        i, j = pair_list[pair_idx]
        prob = _logistic(S[k, i] - S[k, j])
        y = int(rng.binomial(1, prob))
        comparisons.append((k, i, j, y))

    return comparisons



def comparisons_to_aggregated(comparisons, N, K):
    '''
    Returns:
    - n_ijk: (K, N, N), the number of times (i, j) is compared by judge k
    - y_ijk: (K, N, N), the number of times judge k says item i beats item j

    Note:
    - unlike the older symmetric aggregation in simulation_0411, this version keeps
      only the upper-triangular i < j entries because the downstream likelihood uses
      that convention directly.
    '''
    n_ijk = np.zeros((K, N, N), dtype=float)
    y_ijk = np.zeros((K, N, N), dtype=float)

    for k, i, j, y in comparisons:
        if i >= j:
            raise ValueError(f"comparison indices must satisfy i < j, got {(i, j)}")
        n_ijk[k, i, j] += 1.0
        y_ijk[k, i, j] += float(y)

    return n_ijk, y_ijk



def identify_near_tie_pairs(S, top_m=5):
    S = np.asarray(S, dtype=float)
    K, N = S.shape
    avg_scores = np.mean(S, axis=0)
    pairs = []

    for i in range(N):
        for j in range(i + 1, N):
            avg_diff = avg_scores[i] - avg_scores[j]
            pair_probs = _logistic(S[:, i] - S[:, j])
            pairs.append(
                {
                    "pair": (i, j),
                    "avg_diff": float(avg_diff),
                    "avg_probability": float(np.mean(pair_probs)),
                    "distance_to_half": float(abs(np.mean(pair_probs) - 0.5)),
                }
            )

    pairs.sort(key=lambda x: x["distance_to_half"])
    return pairs[:top_m]
