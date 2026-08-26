"""Level 6 tests: TSP annealing improves clearly over a random tour."""

import numpy as np

from tsp_simulated_annealing import solve_tsp, tour_length


def test_tsp_improves_over_random_tour():
    rng = np.random.default_rng(42)
    cities = rng.uniform(0.0, 100.0, size=(25, 2))
    result = solve_tsp(cities, seed=0)
    assert result.best_length < 0.7 * result.initial_length


def test_tsp_result_is_valid_permutation():
    rng = np.random.default_rng(1)
    cities = rng.uniform(0.0, 100.0, size=(15, 2))
    result = solve_tsp(cities, seed=0)
    assert sorted(result.best_tour.tolist()) == list(range(15))
    assert result.best_length == tour_length(cities, result.best_tour)
