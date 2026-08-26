"""Level 1 tests: objectives have correct optima; baselines find them."""

import numpy as np
import pytest

from objective_functions import BENCHMARKS
from grid_random_search import grid_search, random_search
from gradient_descent import gradient_descent, numerical_gradient


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_known_optima(name):
    """Each benchmark evaluates to its documented optimum at the optimum point."""
    bench = BENCHMARKS[name]
    for dim in (2, 5):
        value = bench.func(bench.optimum_point(dim))
        assert value == pytest.approx(bench.optimum_value, abs=1e-9)


def test_numerical_gradient_matches_analytic():
    """For sphere, ∇F(x) = 2x — the numerical gradient must agree."""
    x = np.array([1.5, -2.0, 0.5])
    grad = numerical_gradient(BENCHMARKS["sphere"].func, x)
    np.testing.assert_allclose(grad, 2.0 * x, rtol=1e-4)


def test_grid_search_finds_sphere_minimum():
    bench = BENCHMARKS["sphere"]
    result = grid_search(bench.func, bench.bounds, dim=2, points_per_dim=21)
    assert result.best_f < 0.5
    assert result.evaluations == 21**2


def test_random_search_beats_nothing():
    bench = BENCHMARKS["sphere"]
    result = random_search(bench.func, bench.bounds, dim=2, n_samples=2000, seed=0)
    assert result.best_f < 0.5


def test_gradient_descent_converges_on_sphere():
    bench = BENCHMARKS["sphere"]
    result = gradient_descent(bench.func, np.array([3.0, -4.0]), learning_rate=0.1)
    assert result.converged
    assert result.best_f < 1e-6
    # History must be monotonically non-increasing for convex sphere + small lr.
    assert all(b <= a + 1e-12 for a, b in zip(result.history, result.history[1:]))
