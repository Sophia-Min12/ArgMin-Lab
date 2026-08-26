"""Traveling Salesman Problem solved with simulated annealing + 2-opt moves.

Here 𝒳 is the set of tours (permutations) — a *combinatorial* search space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def tour_length(cities: np.ndarray, tour: np.ndarray) -> float:
    """Total closed-loop length of `tour` (including the return leg)."""
    ordered = cities[tour]
    diffs = ordered - np.roll(ordered, -1, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


def two_opt_neighbor(tour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reverse a random segment — the classic 2-opt move that untangles
    crossing edges.
    """
    n = len(tour)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    neighbor = tour.copy()
    neighbor[i : j + 1] = neighbor[i : j + 1][::-1]
    return neighbor


@dataclass
class TSPResult:
    best_tour: np.ndarray
    best_length: float
    initial_length: float
    evaluations: int


def solve_tsp(
    cities: np.ndarray,
    t_initial: float = 10.0,
    t_final: float = 1e-3,
    cooling: float = 0.995,
    steps_per_temp: int = 50,
    seed: int | None = None,
) -> TSPResult:
    """Anneal over tour space with Metropolis acceptance.

    Same algorithmic skeleton as the continuous SA in Level 2 — only the
    neighbor definition changes.
    """
    rng = np.random.default_rng(seed)
    n = len(cities)

    tour = rng.permutation(n)
    length = tour_length(cities, tour)
    initial_length = length
    best_tour, best_length = tour.copy(), length
    evals = 1

    temperature = t_initial
    while temperature > t_final:
        for _ in range(steps_per_temp):
            candidate = two_opt_neighbor(tour, rng)
            cand_length = tour_length(cities, candidate)
            evals += 1
            delta = cand_length - length
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                tour, length = candidate, cand_length
                if length < best_length:
                    best_tour, best_length = tour.copy(), length
        temperature *= cooling

    return TSPResult(best_tour, best_length, initial_length, evals)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    cities = rng.uniform(0.0, 100.0, size=(30, 2))

    result = solve_tsp(cities, seed=0)
    improvement = 100.0 * (1.0 - result.best_length / result.initial_length)
    print(f"cities            : {len(cities)}")
    print(f"random tour length: {result.initial_length:.1f}")
    print(f"annealed length   : {result.best_length:.1f}  ({improvement:.1f}% shorter)")
    print(f"evaluations       : {result.evaluations}")
