"""Simulated annealing with Metropolis acceptance and geometric cooling.

메트로폴리스 수용 기준과 기하 냉각을 사용하는 담금질 기법.
Accept a worse move with probability exp(−ΔF / T).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class SAResult:
    best_x: np.ndarray
    best_f: float
    history: list[float] = field(default_factory=list)
    evaluations: int = 0


def simulated_annealing(
    func: Callable[[np.ndarray], float],
    bounds: tuple[float, float],
    dim: int,
    t_initial: float = 10.0,
    t_final: float = 1e-4,
    cooling: float = 0.995,
    steps_per_temp: int = 20,
    step_size: float = 0.5,
    seed: int | None = None,
) -> SAResult:
    """Minimize `func` over a box via simulated annealing.

    Args:
        t_initial / t_final: temperature range. High T ≈ random walk,
            low T ≈ greedy hill climbing. (높은 온도는 무작위 걷기,
            낮은 온도는 탐욕적 언덕 오르기에 가깝다.)
        cooling: geometric factor, T ← cooling · T each stage.
            (기하 냉각 계수 — 1에 가까울수록 천천히 식는다.)
        step_size: neighbor proposal scale, shrinks with temperature.
            (이웃 제안 폭 — 온도에 비례해 줄어든다.)
    """
    rng = np.random.default_rng(seed)
    low, high = bounds

    x = rng.uniform(low, high, size=dim)
    f = func(x)
    best_x, best_f = x.copy(), f
    history = [f]
    evals = 1

    temperature = t_initial
    while temperature > t_final:
        scale = step_size * max(temperature / t_initial, 0.01)
        for _ in range(steps_per_temp):
            candidate = np.clip(x + rng.normal(0.0, scale, size=dim), low, high)
            f_cand = func(candidate)
            evals += 1
            delta = f_cand - f
            # Metropolis criterion: always accept improvement, sometimes accept worse.
            # 메트로폴리스 기준 — 개선은 항상, 악화는 확률적으로 수용.
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                x, f = candidate, f_cand
                if f < best_f:
                    best_x, best_f = x.copy(), f
        history.append(best_f)
        temperature *= cooling

    return SAResult(best_x, best_f, history, evals)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1] / "01_foundations"))
    from objective_functions import BENCHMARKS

    for name in ("rastrigin", "ackley"):
        bench = BENCHMARKS[name]
        res = simulated_annealing(bench.func, bench.bounds, dim=2, seed=0)
        print(f"{name:<10}: best_f={res.best_f:.4f} ({res.evaluations} evals)")
