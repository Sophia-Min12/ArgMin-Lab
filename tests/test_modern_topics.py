"""Level 5 tests: sample-efficient, shape-learning, and multi-objective search."""

import numpy as np

from objective_functions import BENCHMARKS
from grid_random_search import random_search
from bayesian_optimization import GaussianProcess, bayesian_optimization
from cma_es import cma_es
from nsga2 import dominates_matrix, hypervolume_2d, non_dominated_sort, nsga2, zdt1


def test_gp_interpolates_training_points():
    """With (almost) no noise the posterior passes through the data, with ~0 doubt."""
    rng = np.random.default_rng(0)
    x = rng.random((8, 2))
    y = np.sin(3 * x[:, 0]) + x[:, 1] ** 2
    mean, std = GaussianProcess().fit(x, y).predict(x)
    assert np.allclose(mean, y, atol=1e-3)
    assert (std < 1e-2).all()


def test_bo_beats_random_search_on_small_budget():
    bench = BENCHMARKS["rosenbrock"]
    budget = 40
    bo = bayesian_optimization(bench.func, bench.bounds, dim=2, iterations=budget - 5, seed=0)
    rs = random_search(bench.func, bench.bounds, dim=2, n_samples=budget, seed=0)
    assert bo.evaluations == budget
    assert bo.best_f < 0.1
    assert bo.best_f < rs.best_f


def test_cma_es_solves_rosenbrock_valley():
    bench = BENCHMARKS["rosenbrock"]
    result = cma_es(bench.func, bench.bounds, dim=5, seed=0)
    assert result.best_f < 1e-6
    assert np.allclose(result.best_x, np.ones(5), atol=1e-3)


def test_cma_es_covariance_stays_positive_definite():
    bench = BENCHMARKS["rosenbrock"]
    result = cma_es(bench.func, bench.bounds, dim=5, max_evals=2000, seed=0)
    assert np.allclose(result.final_cov, result.final_cov.T)
    assert (np.linalg.eigvalsh(result.final_cov) > 0).all()


def test_non_dominated_sort_on_hand_made_points():
    objs = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [3.0, 3.0], [4.0, 4.0]])
    fronts = non_dominated_sort(objs)
    assert sorted(fronts[0].tolist()) == [0, 1, 2]
    assert fronts[1].tolist() == [3]
    assert fronts[2].tolist() == [4]


def test_nsga2_approaches_zdt1_front():
    result = nsga2(zdt1, (0.0, 1.0), dim=30, seed=0)
    f1, f2 = result.pareto_f[:, 0], result.pareto_f[:, 1]
    assert np.mean(np.abs(f2 - (1 - np.sqrt(f1)))) < 0.05
    # spread: the front should cover most of f₁ ∈ [0, 1]
    assert f1.min() < 0.05 and f1.max() > 0.95
    # true-front hypervolume w.r.t. (1.1, 1.1) is ≈ 0.8767
    assert hypervolume_2d(result.pareto_f, (1.1, 1.1)) > 0.84
    # nothing on the returned front dominates anything else on it
    assert not dominates_matrix(result.pareto_f).any()
