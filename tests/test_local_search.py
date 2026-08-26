"""Level 2 tests: local search escapes traps well enough to near the optimum."""

from objective_functions import BENCHMARKS
from hill_climbing import hill_climbing
from simulated_annealing import simulated_annealing


def test_hill_climbing_on_sphere():
    bench = BENCHMARKS["sphere"]
    result = hill_climbing(bench.func, bench.bounds, dim=2, n_restarts=10, seed=0)
    assert result.best_f < 0.1


def test_hill_climbing_restarts_help_on_rastrigin():
    """More restarts must never hurt the best result (same seed stream aside)."""
    bench = BENCHMARKS["rastrigin"]
    few = hill_climbing(bench.func, bench.bounds, dim=2, n_restarts=1, seed=0)
    many = hill_climbing(bench.func, bench.bounds, dim=2, n_restarts=30, seed=0)
    assert many.best_f <= few.best_f


def test_simulated_annealing_on_rastrigin():
    """SA should land near the global basin on a multimodal surface."""
    bench = BENCHMARKS["rastrigin"]
    result = simulated_annealing(bench.func, bench.bounds, dim=2, seed=0)
    assert result.best_f < 2.0


def test_simulated_annealing_history_is_monotone():
    """`history` tracks the best-so-far, so it can never increase."""
    bench = BENCHMARKS["ackley"]
    result = simulated_annealing(bench.func, bench.bounds, dim=2, seed=1)
    assert all(b <= a for a, b in zip(result.history, result.history[1:]))
