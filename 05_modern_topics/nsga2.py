"""NSGA-II: multi-objective optimization with Pareto fronts.

With two objectives F = (f₁, f₂) there is usually no single arg min — making
f₁ smaller makes f₂ larger. The answer becomes a *set*: the Pareto front of
solutions no other solution beats on every objective at once.

Follows Deb et al. (2002), "A Fast and Elitist Multiobjective Genetic
Algorithm: NSGA-II".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def zdt1(x: np.ndarray) -> np.ndarray:
    """ZDT1 test problem on [0, 1]ⁿ. True Pareto front: f₂ = 1 − √f₁ (g = 1)."""
    x = np.asarray(x, dtype=float)
    f1 = x[0]
    g = 1.0 + 9.0 * np.mean(x[1:])
    return np.array([f1, g * (1.0 - np.sqrt(f1 / g))])


def dominates_matrix(objs: np.ndarray) -> np.ndarray:
    """D[i, j] = True iff solution i Pareto-dominates solution j:
    no worse on every objective and strictly better on at least one.
    """
    a, b = objs[:, None, :], objs[None, :, :]
    return np.all(a <= b, axis=2) & np.any(a < b, axis=2)


def non_dominated_sort(objs: np.ndarray) -> list[np.ndarray]:
    """Peel the population into fronts F₀, F₁, … (F₀ = non-dominated)."""
    dom = dominates_matrix(objs)
    dominated_by = dom.sum(axis=0)  # how many solutions dominate each one
    fronts = []
    remaining = np.ones(len(objs), dtype=bool)
    while remaining.any():
        front = np.flatnonzero(remaining & (dominated_by == 0))
        fronts.append(front)
        remaining[front] = False
        dominated_by -= dom[front].sum(axis=0)
    return fronts


def crowding_distance(objs: np.ndarray) -> np.ndarray:
    """How much empty objective space surrounds each point of one front.

    Boundary points get ∞ so the extremes of the front are always kept.
    """
    n, m = objs.shape
    dist = np.zeros(n)
    for k in range(m):
        order = np.argsort(objs[:, k])
        span = objs[order[-1], k] - objs[order[0], k]
        dist[order[0]] = dist[order[-1]] = np.inf
        if span > 0 and n > 2:
            dist[order[1:-1]] += (objs[order[2:], k] - objs[order[:-2], k]) / span
    return dist


def rank_and_crowding(objs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rank = np.empty(len(objs), dtype=int)
    crowd = np.empty(len(objs))
    for r, front in enumerate(non_dominated_sort(objs)):
        rank[front] = r
        crowd[front] = crowding_distance(objs[front])
    return rank, crowd


def sbx_crossover(
    p1: np.ndarray, p2: np.ndarray, eta: float, low: float, high: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulated binary crossover: children spread around the parents like
    one-point crossover on binary strings would; larger η = closer to parents.
    As in Deb's reference code, each gene is crossed with probability ½ and
    otherwise inherited unchanged (β = 1).
    """
    u = rng.random(p1.shape)
    beta = np.where(
        u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    )
    beta = np.where(rng.random(p1.shape) < 0.5, beta, 1.0)
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, low, high), np.clip(c2, low, high)


def polynomial_mutation(
    x: np.ndarray, eta: float, rate: float, low: float, high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    u = rng.random(x.shape)
    delta = np.where(
        u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta + 1))
    )
    mask = rng.random(x.shape) < rate
    return np.clip(x + mask * delta * (high - low), low, high)


@dataclass
class NSGA2Result:
    pareto_x: np.ndarray  # decision vectors on the final first front
    pareto_f: np.ndarray  # their objective vectors
    population_f: np.ndarray
    generations: int
    evaluations: int


def nsga2(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: tuple[float, float],
    dim: int,
    pop_size: int = 100,
    generations: int = 250,
    eta_crossover: float = 10.0,
    eta_mutation: float = 10.0,
    crossover_rate: float = 0.9,
    mutation_rate: float | None = None,
    seed: int | None = None,
) -> NSGA2Result:
    """Approximate the Pareto front of a vector-valued `func`.

    Selection is a binary tournament on (rank ↑, crowding distance ↓);
    survival keeps the best `pop_size` of parents ∪ children — elitism.

    Args:
        eta_crossover / eta_mutation: distribution indices; larger = children
            closer to parents. Deb et al. use 20 / 20 with *bounded* operators;
            with the simpler clip-to-box operators here, 10 / 10 converged
            markedly faster on ZDT1 (mean gap to the true front ≈ 0.01 vs ≈ 0.2
            after 250 generations).
    """
    rng = np.random.default_rng(seed)
    low, high = bounds
    mutation_rate = mutation_rate if mutation_rate is not None else 1.0 / dim

    pop = rng.uniform(low, high, size=(pop_size, dim))
    objs = np.array([func(ind) for ind in pop])
    evals = pop_size
    rank, crowd = rank_and_crowding(objs)

    def tournament() -> np.ndarray:
        i, j = rng.integers(0, pop_size, size=2)
        better = rank[i] < rank[j] or (rank[i] == rank[j] and crowd[i] > crowd[j])
        return pop[i if better else j]

    for _ in range(generations):
        children = []
        while len(children) < pop_size:
            p1, p2 = tournament(), tournament()
            if rng.random() < crossover_rate:
                c1, c2 = sbx_crossover(p1, p2, eta_crossover, low, high, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            children.append(polynomial_mutation(c1, eta_mutation, mutation_rate, low, high, rng))
            children.append(polynomial_mutation(c2, eta_mutation, mutation_rate, low, high, rng))
        children = np.array(children[:pop_size])
        child_objs = np.array([func(c) for c in children])
        evals += pop_size

        # (μ + λ) survival: fill by front, break the last front by crowding.
        merged = np.vstack([pop, children])
        merged_objs = np.vstack([objs, child_objs])
        survivors: list[int] = []
        for front in non_dominated_sort(merged_objs):
            if len(survivors) + len(front) <= pop_size:
                survivors.extend(front)
            else:
                crowd_front = crowding_distance(merged_objs[front])
                need = pop_size - len(survivors)
                survivors.extend(front[np.argsort(-crowd_front)[:need]])
                break
        pop, objs = merged[survivors], merged_objs[survivors]
        rank, crowd = rank_and_crowding(objs)

    first = rank == 0
    return NSGA2Result(pop[first], objs[first], objs, generations, evals)


def hypervolume_2d(front: np.ndarray, reference: tuple[float, float]) -> float:
    """Area dominated by a 2-objective front and bounded by `reference`.
    Bigger is better: it rewards both convergence and spread.
    """
    pts = front[np.all(front < reference, axis=1)]
    pts = pts[np.argsort(pts[:, 0])]
    area, prev_f2 = 0.0, reference[1]
    for f1, f2 in pts:
        if f2 < prev_f2:
            area += (reference[0] - f1) * (prev_f2 - f2)
            prev_f2 = f2
    return area


if __name__ == "__main__":
    res = nsga2(zdt1, (0.0, 1.0), dim=30, seed=0)
    gap = np.abs(res.pareto_f[:, 1] - (1 - np.sqrt(res.pareto_f[:, 0])))
    true_hv = 0.1 + 2 / 3 + 0.11  # area under 1 − √f₁ w.r.t. reference (1.1, 1.1)
    print(f"ZDT1 (30-D): {len(res.pareto_f)} Pareto points after {res.evaluations} evals")
    print(f"mean distance to true front f2 = 1 - sqrt(f1): {gap.mean():.4f}")
    print(f"hypervolume: {hypervolume_2d(res.pareto_f, (1.1, 1.1)):.4f} (true front: {true_hv:.4f})")
