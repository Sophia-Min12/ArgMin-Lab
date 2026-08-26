"""Continuous hill climbing with Gaussian neighbors and random restarts.

(Here we descend — minimizing — but the classic name stays.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class HCResult:
    best_x: np.ndarray
    best_f: float
    restarts_used: int
    evaluations: int


def hill_climb_once(
    func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: tuple[float, float],
    step_size: float,
    max_stall: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int]:
    """Greedy walk: accept a Gaussian neighbor only if it improves.

    Stops after `max_stall` consecutive failures, taking that as having
    reached a local optimum.
    """
    low, high = bounds
    x, f = x0.copy(), func(x0)
    evals, stall = 1, 0
    while stall < max_stall:
        candidate = np.clip(x + rng.normal(0.0, step_size, size=x.size), low, high)
        f_cand = func(candidate)
        evals += 1
        if f_cand < f:
            x, f, stall = candidate, f_cand, 0
        else:
            stall += 1
    return x, f, evals


def hill_climbing(
    func: Callable[[np.ndarray], float],
    bounds: tuple[float, float],
    dim: int,
    n_restarts: int = 10,
    step_size: float = 0.1,
    max_stall: int = 100,
    seed: int | None = None,
) -> HCResult:
    """Random-restart hill climbing: run greedy walks from random starts.

    Restarts are the escape mechanism — walk from many different starting
    points and keep the best result.
    """
    rng = np.random.default_rng(seed)
    low, high = bounds
    best_x, best_f, total_evals = None, np.inf, 0
    for _ in range(n_restarts):
        x0 = rng.uniform(low, high, size=dim)
        x, f, evals = hill_climb_once(func, x0, bounds, step_size, max_stall, rng)
        total_evals += evals
        if f < best_f:
            best_x, best_f = x, f
    return HCResult(best_x, best_f, n_restarts, total_evals)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1] / "01_foundations"))
    from objective_functions import BENCHMARKS

    for name in ("sphere", "rastrigin"):
        bench = BENCHMARKS[name]
        res = hill_climbing(bench.func, bench.bounds, dim=2, n_restarts=20, seed=0)
        print(f"{name:<10}: best_f={res.best_f:.4f} "
              f"({res.evaluations} evals, {res.restarts_used} restarts)")
