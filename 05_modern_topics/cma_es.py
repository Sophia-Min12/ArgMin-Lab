"""CMA-ES: covariance matrix adaptation evolution strategy.

Sample a population from N(m, σ²C), then move the mean m toward the best
samples, adapt the step size σ, and *learn the shape* C of the landscape.
On a narrow curved valley like Rosenbrock, C stretches along the valley —
something no isotropic mutation (GA, PSO) can do.

Follows Hansen (2016), "The CMA Evolution Strategy: A Tutorial".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class CMAESResult:
    best_x: np.ndarray
    best_f: float
    history: list[float] = field(default_factory=list)
    evaluations: int = 0
    final_sigma: float = 0.0
    final_cov: np.ndarray | None = None


def cma_es(
    func: Callable[[np.ndarray], float],
    bounds: tuple[float, float],
    dim: int,
    sigma0: float | None = None,
    pop_size: int | None = None,
    max_evals: int = 12000,
    tol_sigma: float = 1e-12,
    seed: int | None = None,
) -> CMAESResult:
    """Minimize `func` with (μ/μ_w, λ)-CMA-ES.

    Args:
        sigma0: initial step size; defaults to 30% of the box width.
        pop_size: λ; defaults to 4 + ⌊3 ln n⌋.
        tol_sigma: stop once σ collapses below this.

    Samples outside the box are clipped (repaired) before evaluation, and the
    repaired point is what the update learns from.
    """
    rng = np.random.default_rng(seed)
    low, high = bounds
    n = dim

    # --- strategy parameters (Hansen's defaults) ---
    lam = pop_size or 4 + int(3 * np.log(n))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mu_eff = 1.0 / np.sum(weights**2)

    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))  # E‖N(0, I)‖

    # --- state ---
    mean = rng.uniform(low, high, size=n)
    sigma = sigma0 if sigma0 is not None else 0.3 * (high - low)
    cov = np.eye(n)
    p_sigma = np.zeros(n)  # evolution path for σ
    p_c = np.zeros(n)  # evolution path for C

    best_x, best_f = mean.copy(), func(mean)
    history = [best_f]
    evals, generation = 1, 0

    while evals + lam <= max_evals and sigma > tol_sigma:
        generation += 1
        # C = B D² Bᵀ  →  y = B D z  ~  N(0, C)
        eigvals, basis = np.linalg.eigh(cov)
        d = np.sqrt(np.maximum(eigvals, 1e-20))
        z = rng.standard_normal((lam, n))
        x = np.clip(mean + sigma * (z * d) @ basis.T, low, high)
        y = (x - mean) / sigma
        f = np.array([func(xi) for xi in x])
        evals += lam

        order = np.argsort(f)
        if f[order[0]] < best_f:
            best_x, best_f = x[order[0]].copy(), float(f[order[0]])
        history.append(best_f)

        # 1. move the mean toward the weighted best μ samples
        y_sel = y[order[:mu]]
        y_w = weights @ y_sel
        mean = mean + sigma * y_w

        # 2. step-size control via the conjugate evolution path
        inv_sqrt_cov = basis @ np.diag(1 / d) @ basis.T
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
            c_sigma * (2 - c_sigma) * mu_eff
        ) * (inv_sqrt_cov @ y_w)
        h_sigma = float(
            np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * generation))
            < (1.4 + 2 / (n + 1)) * chi_n
        )

        # 3. covariance: rank-one (path) + rank-μ (selected steps) updates
        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
        rank_one = np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * cov
        rank_mu = (y_sel * weights[:, None]).T @ y_sel
        cov = (1 - c_1 - c_mu) * cov + c_1 * rank_one + c_mu * rank_mu
        cov = (cov + cov.T) / 2  # keep it numerically symmetric

        sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))

    return CMAESResult(best_x, best_f, history, evals, sigma, cov)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1] / "01_foundations"))
    from objective_functions import BENCHMARKS

    for name in ("sphere", "rosenbrock", "rastrigin"):
        bench = BENCHMARKS[name]
        res = cma_es(bench.func, bench.bounds, dim=5, seed=0)
        print(f"{name:<10}: best_f={res.best_f:.2e} ({res.evaluations} evals)")
