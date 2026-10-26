"""Bayesian optimization: a Gaussian-process surrogate + expected improvement.

When one evaluation of F is expensive (training a model, running a RAG
pipeline), we cannot afford thousands of samples. Instead we fit a cheap
probabilistic model of F to the points seen so far and ask it where the next
evaluation is most promising.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

_erf = np.frompyfunc(math.erf, 1, 1)


def normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)).astype(float))


def rbf_kernel(a: np.ndarray, b: np.ndarray, length_scale: float) -> np.ndarray:
    """Squared-exponential kernel k(a, b) = exp(−‖a − b‖² / 2ℓ²)."""
    sq_dist = np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None, :] - 2.0 * a @ b.T
    return np.exp(-0.5 * np.maximum(sq_dist, 0.0) / length_scale**2)


class GaussianProcess:
    """Zero-mean GP regression on standardized targets.

    The length scale ℓ is chosen by maximizing the log marginal likelihood
    over a small grid — the model picks how smooth it believes F is.
    """

    def __init__(
        self,
        length_scales: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2),
        noise: float = 1e-6,
    ) -> None:
        self.length_scales = length_scales
        self.noise = noise

    def fit(self, x: np.ndarray, y: np.ndarray) -> GaussianProcess:
        self.x = x
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) or 1.0
        y_std = (y - self.y_mean) / self.y_std

        best_lml = -np.inf
        for ell in self.length_scales:
            k = rbf_kernel(x, x, ell) + self.noise * np.eye(len(x))
            try:
                chol = np.linalg.cholesky(k)
            except np.linalg.LinAlgError:
                continue
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_std))
            # log p(y | X, ℓ) = −½ yᵀK⁻¹y − ½ log|K| − n/2 log 2π
            lml = -0.5 * y_std @ alpha - np.sum(np.log(np.diag(chol)))
            if lml > best_lml:
                best_lml = lml
                self.length_scale, self.chol, self.alpha = ell, chol, alpha
        return self

    def predict(self, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation, in the original units of F."""
        k_star = rbf_kernel(x_new, self.x, self.length_scale)
        mean = k_star @ self.alpha
        v = np.linalg.solve(self.chol, k_star.T)
        var = np.maximum(1.0 - np.sum(v**2, axis=0), 1e-12)
        return self.y_mean + self.y_std * mean, self.y_std * np.sqrt(var)


def expected_improvement(
    mean: np.ndarray, std: np.ndarray, best_f: float, xi: float = 0.01
) -> np.ndarray:
    """EI(x) = E[max(best_f − F(x) − ξ, 0)] under the GP posterior.

    High where the mean is low (exploitation) *or* the uncertainty is high
    (exploration) — the trade-off is built into one closed-form number.
    """
    improvement = best_f - mean - xi
    z = improvement / std
    return improvement * normal_cdf(z) + std * normal_pdf(z)


@dataclass
class BOResult:
    best_x: np.ndarray
    best_f: float
    history: list[float] = field(default_factory=list)
    evaluations: int = 0


def bayesian_optimization(
    func: Callable[[np.ndarray], float],
    bounds: tuple[float, float],
    dim: int,
    n_initial: int = 5,
    iterations: int = 30,
    n_candidates: int = 2000,
    xi: float = 0.01,
    seed: int | None = None,
) -> BOResult:
    """Minimize an expensive `func` in only `n_initial + iterations` calls.

    Args:
        n_initial: random points used to seed the surrogate.
        n_candidates: random points on which EI is scored each round; half
            are uniform over the box, half are perturbations of the incumbent.
        xi: exploration bonus inside EI.
    """
    rng = np.random.default_rng(seed)
    low, high = bounds
    width = high - low

    # The GP works in the unit cube so one length-scale grid fits every problem.
    u = rng.random((n_initial, dim))
    y = np.array([func(low + width * ui) for ui in u])
    history = [float(np.min(y[: i + 1])) for i in range(n_initial)]

    for _ in range(iterations):
        gp = GaussianProcess().fit(u, y)
        incumbent = u[np.argmin(y)]
        n_local = n_candidates // 2
        candidates = np.vstack(
            [
                rng.random((n_candidates - n_local, dim)),
                np.clip(incumbent + rng.normal(0.0, 0.05, (n_local, dim)), 0.0, 1.0),
            ]
        )
        mean, std = gp.predict(candidates)
        ei = expected_improvement(mean, std, float(np.min(y)), xi)
        u_next = candidates[np.argmax(ei)]

        u = np.vstack([u, u_next])
        y = np.append(y, func(low + width * u_next))
        history.append(float(np.min(y)))

    best = int(np.argmin(y))
    return BOResult(low + width * u[best], float(y[best]), history, len(y))


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1] / "01_foundations"))
    from grid_random_search import random_search
    from objective_functions import BENCHMARKS

    budget = 40
    for name in ("sphere", "rosenbrock"):
        bench = BENCHMARKS[name]
        bo = bayesian_optimization(bench.func, bench.bounds, dim=2, iterations=budget - 5, seed=0)
        rs = random_search(bench.func, bench.bounds, dim=2, n_samples=budget, seed=0)
        print(f"{name:<10}: BO best_f={bo.best_f:.4f}   random search best_f={rs.best_f:.4f}   ({budget} evals each)")
