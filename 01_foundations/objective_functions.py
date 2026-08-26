"""Classic benchmark objective functions for optimization.

Every function is written to be minimized: X* = arg min F(X').
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def sphere(x: np.ndarray) -> float:
    """Sphere: F(x) = Σ xᵢ².  Convex, unimodal. Global minimum 0 at origin."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock: narrow curved valley where gradient methods struggle.

    Global minimum 0 at (1, ..., 1).
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin: highly multimodal — a grid of local optima. Minimum 0 at origin.

    Local search gets trapped; metaheuristics shine here.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    return float(10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Ackley: nearly flat outer region, deep hole at origin — a test of the
    exploration/exploitation balance. Minimum 0 at origin.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    s1 = np.sum(x**2) / n
    s2 = np.sum(np.cos(c * x)) / n
    return float(-a * np.exp(-b * np.sqrt(s1)) - np.exp(s2) + a + np.e)


@dataclass(frozen=True)
class Benchmark:
    """An objective together with its search bounds and known optimum."""

    name: str
    func: Callable[[np.ndarray], float]
    bounds: tuple[float, float]  # same (low, high) for every dimension
    optimum_value: float
    optimum_point: Callable[[int], np.ndarray] = field(
        default=lambda dim: np.zeros(dim)
    )


BENCHMARKS: dict[str, Benchmark] = {
    "sphere": Benchmark("sphere", sphere, (-5.12, 5.12), 0.0),
    "rosenbrock": Benchmark(
        "rosenbrock", rosenbrock, (-2.048, 2.048), 0.0, lambda dim: np.ones(dim)
    ),
    "rastrigin": Benchmark("rastrigin", rastrigin, (-5.12, 5.12), 0.0),
    "ackley": Benchmark("ackley", ackley, (-32.768, 32.768), 0.0),
}


if __name__ == "__main__":
    dim = 2
    print(f"{'function':<12} {'F(0)':>12} {'F(optimum)':>12}")
    for bench in BENCHMARKS.values():
        at_zero = bench.func(np.zeros(dim))
        at_opt = bench.func(bench.optimum_point(dim))
        print(f"{bench.name:<12} {at_zero:>12.4f} {at_opt:>12.4f}")
