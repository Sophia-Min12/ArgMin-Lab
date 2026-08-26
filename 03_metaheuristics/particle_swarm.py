"""Particle swarm optimization (global-best topology).

v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class PSOResult:
    best_x: np.ndarray
    best_f: float
    history: list[float] = field(default_factory=list)
    iterations: int = 0


def particle_swarm(
    func: Callable[[np.ndarray], float],
    bounds: tuple[float, float],
    dim: int,
    n_particles: int = 40,
    iterations: int = 300,
    inertia: float = 0.72,
    cognitive: float = 1.49,
    social: float = 1.49,
    seed: int | None = None,
) -> PSOResult:
    """Minimize `func` with a standard global-best PSO.

    Args:
        inertia: how much old velocity carries over (w).
        cognitive: pull toward each particle's own best (c1).
        social: pull toward the swarm's best (c2).
        (0.72 / 1.49 constriction values follow Clerc & Kennedy 2002.)
    """
    rng = np.random.default_rng(seed)
    low, high = bounds
    v_max = 0.2 * (high - low)

    x = rng.uniform(low, high, size=(n_particles, dim))
    v = rng.uniform(-v_max, v_max, size=(n_particles, dim))
    f = np.array([func(p) for p in x])

    pbest_x, pbest_f = x.copy(), f.copy()
    g_idx = int(np.argmin(f))
    gbest_x, gbest_f = x[g_idx].copy(), float(f[g_idx])
    history = [gbest_f]

    for it in range(1, iterations + 1):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        v = (
            inertia * v
            + cognitive * r1 * (pbest_x - x)
            + social * r2 * (gbest_x - x)
        )
        v = np.clip(v, -v_max, v_max)
        x = np.clip(x + v, low, high)
        f = np.array([func(p) for p in x])

        improved = f < pbest_f
        pbest_x[improved], pbest_f[improved] = x[improved], f[improved]
        g_idx = int(np.argmin(pbest_f))
        if pbest_f[g_idx] < gbest_f:
            gbest_x, gbest_f = pbest_x[g_idx].copy(), float(pbest_f[g_idx])
        history.append(gbest_f)

    return PSOResult(gbest_x, gbest_f, history, iterations)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1] / "01_foundations"))
    from objective_functions import BENCHMARKS

    for name in ("rastrigin", "ackley"):
        bench = BENCHMARKS[name]
        res = particle_swarm(bench.func, bench.bounds, dim=5, seed=0)
        print(f"{name:<10}: best_f={res.best_f:.4f} after {res.iterations} iterations")
