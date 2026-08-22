"""Real-coded genetic algorithm: tournament selection, BLX-α crossover,
Gaussian mutation, and elitism.

실수 인코딩 유전 알고리즘 — 토너먼트 선택, BLX-α 교차, 가우시안 변이,
엘리트 보존.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GAResult:
    best_x: np.ndarray
    best_f: float
    history: list[float] = field(default_factory=list)
    generations: int = 0


def tournament_select(
    pop: np.ndarray, fitness: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Pick the best of k random individuals. k개 중 최고를 고르는 토너먼트."""
    idx = rng.integers(0, len(pop), size=k)
    return pop[idx[np.argmin(fitness[idx])]].copy()


def blx_crossover(
    p1: np.ndarray, p2: np.ndarray, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """BLX-α: child sampled uniformly from the parents' interval, widened by α.

    부모 구간을 α만큼 확장한 범위에서 자식을 균일 표집 — 탐색성이 있는 교차.
    """
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    span = hi - lo
    return rng.uniform(lo - alpha * span, hi + alpha * span)


def genetic_algorithm(
    func: Callable[[np.ndarray], float],
    bounds: tuple[float, float],
    dim: int,
    pop_size: int = 60,
    generations: int = 200,
    tournament_k: int = 3,
    crossover_alpha: float = 0.3,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    elite: int = 2,
    seed: int | None = None,
) -> GAResult:
    """Minimize `func` with a generational GA.

    Args:
        elite: top-n individuals copied unchanged into the next generation,
            guaranteeing the best fitness never worsens.
            (상위 n개는 그대로 다음 세대로 — 최고 적합도의 퇴보를 방지.)
        mutation_scale: Gaussian std as a fraction of the box width.
            (변이 표준편차 — 탐색 범위 폭에 대한 비율.)
    """
    rng = np.random.default_rng(seed)
    low, high = bounds
    width = high - low

    pop = rng.uniform(low, high, size=(pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])
    history: list[float] = [float(fitness.min())]

    for gen in range(1, generations + 1):
        order = np.argsort(fitness)
        next_pop = [pop[i].copy() for i in order[:elite]]  # elitism

        while len(next_pop) < pop_size:
            p1 = tournament_select(pop, fitness, tournament_k, rng)
            p2 = tournament_select(pop, fitness, tournament_k, rng)
            child = blx_crossover(p1, p2, crossover_alpha, rng)
            # Gaussian mutation, gene-wise. 유전자 단위 가우시안 변이.
            mask = rng.random(dim) < mutation_rate
            child[mask] += rng.normal(0.0, mutation_scale * width, size=mask.sum())
            next_pop.append(np.clip(child, low, high))

        pop = np.array(next_pop)
        fitness = np.array([func(ind) for ind in pop])
        history.append(float(fitness.min()))

    best_idx = int(np.argmin(fitness))
    return GAResult(pop[best_idx], float(fitness[best_idx]), history, generations)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1] / "01_foundations"))
    from objective_functions import BENCHMARKS

    for name in ("rastrigin", "rosenbrock"):
        bench = BENCHMARKS[name]
        res = genetic_algorithm(bench.func, bench.bounds, dim=5, seed=0)
        print(f"{name:<10}: best_f={res.best_f:.4f} after {res.generations} generations")
