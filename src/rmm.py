"""Core utilities for Riemannian Metric Matching (RMM).

This module is deliberately verbose and heavily commented so each operation is easy
to understand when reproducing paper-style geometric matching experiments.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, logm


def random_spd(dim: int, rng: np.random.Generator, eps: float = 1e-2) -> np.ndarray:
    """Create a random Symmetric Positive Definite (SPD) matrix.

    We sample a dense matrix A and form A A^T + eps*I, which is guaranteed SPD.
    """
    a = rng.normal(size=(dim, dim))
    return a @ a.T + eps * np.eye(dim)


def affine_invariant_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """Affine-invariant Riemannian distance between two SPD covariance matrices.

    d(C1, C2) = || log( C1^{-1/2} C2 C1^{-1/2} ) ||_F

    This is a standard metric on SPD manifolds and captures geometry better than
    naive Euclidean matrix differences.
    """
    # Eigen-decompose C1 to build C1^{-1/2} stably.
    w, v = np.linalg.eigh(c1)
    c1_inv_sqrt = v @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ v.T

    # Transport C2 into C1's tangent-normalized frame.
    inner = c1_inv_sqrt @ c2 @ c1_inv_sqrt

    # Matrix logarithm maps SPD manifold point to tangent space.
    log_inner = logm(inner)

    # Frobenius norm of the log gives geodesic distance.
    return float(np.linalg.norm(log_inner, ord="fro"))


def generate_synthetic_pair(
    n_dists: int,
    dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic source/target distribution families.

    Returns:
        means_src, covs_src, means_tgt, covs_tgt, true_map

    Construction:
        - Draw source means/covariances.
        - Draw a hidden linear map M (ground-truth).
        - Transform source family with M to produce target family (+ small noise).
    """
    means_src = rng.normal(size=(n_dists, dim))
    covs_src = np.stack([random_spd(dim, rng) for _ in range(n_dists)], axis=0)

    # Hidden map used to synthesize targets.
    true_map = expm(0.15 * rng.normal(size=(dim, dim)))

    means_tgt = means_src @ true_map.T + 0.02 * rng.normal(size=(n_dists, dim))

    covs_tgt = []
    for c in covs_src:
        transformed = true_map @ c @ true_map.T
        transformed += 0.02 * np.eye(dim)  # slight regularization/noise
        covs_tgt.append(transformed)
    covs_tgt = np.stack(covs_tgt, axis=0)

    return means_src, covs_src, means_tgt, covs_tgt, true_map


def objective_and_grad(
    m: np.ndarray,
    means_src: np.ndarray,
    covs_src: np.ndarray,
    means_tgt: np.ndarray,
    covs_tgt: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.25,
) -> tuple[float, np.ndarray]:
    """Compute loss and a practical gradient approximation.

    Loss = alpha * mean Euclidean mismatch + beta * mean SPD Riemannian mismatch

    For clarity and stability in a small reproduction script, we use:
    - exact gradient for mean term
    - finite-difference gradient for covariance Riemannian term
    """
    n, dim = means_src.shape

    # ----- Mean matching term (closed-form gradient) -----
    pred_means = means_src @ m.T
    mean_res = pred_means - means_tgt
    mean_loss = np.mean(np.sum(mean_res**2, axis=1))
    grad_mean = (2.0 / n) * (mean_res.T @ means_src)

    # ----- Covariance matching term (distance average) -----
    cov_loss = 0.0
    for i in range(n):
        pred_c = m @ covs_src[i] @ m.T + 1e-6 * np.eye(dim)
        cov_loss += affine_invariant_distance(pred_c, covs_tgt[i])
    cov_loss /= n

    # Finite-difference gradient for covariance term.
    # Expensive but simple and explicit for educational reproducibility.
    fd_eps = 1e-4
    grad_cov = np.zeros_like(m)
    for r in range(dim):
        for c in range(dim):
            delta = np.zeros_like(m)
            delta[r, c] = fd_eps
            plus = m + delta
            minus = m - delta

            plus_loss = 0.0
            minus_loss = 0.0
            for i in range(n):
                pc = plus @ covs_src[i] @ plus.T + 1e-6 * np.eye(dim)
                mc = minus @ covs_src[i] @ minus.T + 1e-6 * np.eye(dim)
                plus_loss += affine_invariant_distance(pc, covs_tgt[i])
                minus_loss += affine_invariant_distance(mc, covs_tgt[i])

            grad_cov[r, c] = (plus_loss - minus_loss) / (2.0 * fd_eps * n)

    total_loss = alpha * mean_loss + beta * cov_loss
    grad = alpha * grad_mean + beta * grad_cov
    return float(total_loss), grad


def fit_metric_map(
    means_src: np.ndarray,
    covs_src: np.ndarray,
    means_tgt: np.ndarray,
    covs_tgt: np.ndarray,
    iters: int = 200,
    lr: float = 0.03,
    alpha: float = 1.0,
    beta: float = 0.25,
) -> tuple[np.ndarray, list[float]]:
    """Fit map matrix M via gradient descent.

    Starts near identity and iteratively reduces the geometric mismatch objective.
    """
    dim = means_src.shape[1]
    m = np.eye(dim)
    history: list[float] = []

    for _ in range(iters):
        loss, grad = objective_and_grad(m, means_src, covs_src, means_tgt, covs_tgt, alpha, beta)
        m -= lr * grad
        history.append(loss)

    return m, history
