"""Level 3 tests: population methods handle multimodal landscapes.

Level 3 검증 — 개체군 기반 기법의 다봉 함수 성능.
"""

from objective_functions import BENCHMARKS
from genetic_algorithm import genetic_algorithm
from particle_swarm import particle_swarm


def test_ga_on_rastrigin():
    bench = BENCHMARKS["rastrigin"]
    result = genetic_algorithm(bench.func, bench.bounds, dim=5, seed=0)
    assert result.best_f < 5.0


def test_ga_elitism_makes_history_monotone():
    """With elitism the best fitness can never get worse between generations."""
    bench = BENCHMARKS["rosenbrock"]
    result = genetic_algorithm(bench.func, bench.bounds, dim=5, seed=0)
    assert all(b <= a for a, b in zip(result.history, result.history[1:]))


def test_pso_on_ackley():
    bench = BENCHMARKS["ackley"]
    result = particle_swarm(bench.func, bench.bounds, dim=5, seed=0)
    assert result.best_f < 1.0


def test_pso_stays_in_bounds():
    bench = BENCHMARKS["sphere"]
    result = particle_swarm(bench.func, bench.bounds, dim=3, seed=0)
    low, high = bench.bounds
    assert (result.best_x >= low).all() and (result.best_x <= high).all()
