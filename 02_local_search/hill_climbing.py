"""Continuous hill climbing with Gaussian neighbors and random restarts.

가우시안 이웃 제안과 랜덤 재시작을 사용하는 연속 공간 언덕 오르기.
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

    개선될 때만 이웃해를 받아들이는 탐욕적 걷기. `max_stall`번 연속
    실패하면 지역 최적점에 도달했다고 보고 멈춘다.
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

    랜덤 재시작 언덕 오르기 — 서로 다른 시작점에서 여러 번 걸어
    가장 좋은 결과를 취한다. 재시작이 지역 최적점 탈출 수단이다.
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
