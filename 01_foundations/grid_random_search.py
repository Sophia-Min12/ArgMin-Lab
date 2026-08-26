"""Grid search and random search — the two baselines every optimizer must beat."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np


@dataclass
class SearchResult:
    """Best configuration found and its objective value."""

    best_x: np.ndarray
    best_f: float
    evaluations: int


def grid_search(
    func: Callable[[np.ndarray], float],
    bounds: tuple[float, float],
    dim: int,
    points_per_dim: int = 21,
) -> SearchResult:
    """Exhaustively evaluate a regular grid over the search box.

    Cost grows as points_per_dim ** dim — the curse of dimensionality.
    """
    low, high = bounds
    axis = np.linspace(low, high, points_per_dim)
    best_x, best_f, count = None, np.inf, 0
    for candidate in product(axis, repeat=dim):
        x = np.array(candidate)
        f = func(x)
        count += 1
        if f < best_f:
            best_x, best_f = x, f
    return SearchResult(best_x, best_f, count)


def random_search(
    func: Callable[[np.ndarray], float],
    bounds: tuple[float, float],
    dim: int,
    n_samples: int = 1000,
    seed: int | None = None,
) -> SearchResult:
    """Evaluate uniformly random points; cost is independent of dimension."""
    rng = np.random.default_rng(seed)
    low, high = bounds
    best_x, best_f = None, np.inf
    for _ in range(n_samples):
        x = rng.uniform(low, high, size=dim)
        f = func(x)
        if f < best_f:
            best_x, best_f = x, f
    return SearchResult(best_x, best_f, n_samples)


if __name__ == "__main__":
    from objective_functions import BENCHMARKS

    bench = BENCHMARKS["rastrigin"]
    g = grid_search(bench.func, bench.bounds, dim=2, points_per_dim=41)
    r = random_search(bench.func, bench.bounds, dim=2, n_samples=g.evaluations, seed=0)
    print(f"grid   : best_f={g.best_f:.4f} with {g.evaluations} evals")
    print(f"random : best_f={r.best_f:.4f} with {r.evaluations} evals")
