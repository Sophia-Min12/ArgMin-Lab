"""Run every implemented optimizer on the classic benchmarks and print a table.

구현된 모든 최적화 기법을 고전 벤치마크 함수 위에서 실행해 표로 비교한다.
Roughly equal evaluation budgets are used for a fair comparison.

Usage: python benchmarks/compare_optimizers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("01_foundations", "02_local_search", "03_metaheuristics"):
    sys.path.append(str(ROOT / sub))

from objective_functions import BENCHMARKS  # noqa: E402
from grid_random_search import random_search  # noqa: E402
from gradient_descent import gradient_descent  # noqa: E402
from hill_climbing import hill_climbing  # noqa: E402
from simulated_annealing import simulated_annealing  # noqa: E402
from genetic_algorithm import genetic_algorithm  # noqa: E402
from particle_swarm import particle_swarm  # noqa: E402

DIM = 5
SEED = 0


def run_all() -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for name, bench in BENCHMARKS.items():
        f, bounds = bench.func, bench.bounds
        rng = np.random.default_rng(SEED)
        x0 = rng.uniform(*bounds, size=DIM)

        results[name] = {
            "random search": random_search(f, bounds, DIM, n_samples=12000, seed=SEED).best_f,
            "gradient descent": gradient_descent(f, x0, learning_rate=1e-4, momentum=0.9, max_iter=5000).best_f,
            "hill climbing": hill_climbing(f, bounds, DIM, n_restarts=12, seed=SEED).best_f,
            "simulated annealing": simulated_annealing(f, bounds, DIM, seed=SEED).best_f,
            "genetic algorithm": genetic_algorithm(f, bounds, DIM, seed=SEED).best_f,
            "particle swarm": particle_swarm(f, bounds, DIM, seed=SEED).best_f,
        }
    return results


def print_table(results: dict[str, dict[str, float]]) -> None:
    optimizers = list(next(iter(results.values())).keys())
    col = 22
    header = f"{'optimizer':<{col}}" + "".join(f"{fn:>14}" for fn in results)
    print(header)
    print("-" * len(header))
    for opt in optimizers:
        row = f"{opt:<{col}}"
        for fn in results:
            value = results[fn][opt]
            best = min(results[fn].values())
            mark = "*" if np.isclose(value, best) else " "
            row += f"{value:>13.4f}{mark}"
        print(row)
    print("\n* = best on that objective (dim=%d, seed=%d). 목적함수별 최고 성능." % (DIM, SEED))


if __name__ == "__main__":
    print_table(run_all())
