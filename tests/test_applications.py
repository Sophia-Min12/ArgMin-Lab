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


# --- exam timetabling ---------------------------------------------------------

from exam_scheduling_ga import ExamInstance, evaluate, make_instance, schedule_exams  # noqa: E402


def test_timetable_cost_counted_by_hand():
    # student 0 sits exams 0, 1; student 1 sits exams 1, 2; one seat per slot
    enrollment = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    cost = evaluate(ExamInstance(enrollment, n_slots=3, capacity=1), np.array([0, 0, 1]))
    assert cost.clashes == 1  # student 0 has exams 0 and 1 together in slot 0
    assert cost.overflow == 2  # slot 0 needs 3 seats
    assert cost.back_to_back == 1  # student 1: slot 0 then slot 1


def test_ga_finds_clash_free_timetable():
    instance = make_instance(seed=0)
    result = schedule_exams(instance, seed=0)
    random_tt = np.random.default_rng(0).integers(0, instance.n_slots, instance.enrollment.shape[1])
    assert result.best_cost.hard == 0
    assert result.best_cost.total < evaluate(instance, random_tt).total


# --- RAG hyperparameter tuning ------------------------------------------------

from bayesian_optimization import bayesian_optimization  # noqa: E402
from grid_random_search import random_search  # noqa: E402
from rag_hyperparameter_tuning import (  # noqa: E402
    CORPUS,
    QUESTIONS,
    RAGConfig,
    chunk_words,
    contains_span,
    evaluate_config,
    exhaustive_grid,
    rag_objective,
    tokenize,
)


def test_every_answer_span_appears_exactly_once_in_corpus():
    text = tokenize(" ".join(CORPUS.values()))
    for _, answer in QUESTIONS:
        span = tokenize(answer)
        hits = sum(text[i : i + len(span)] == span for i in range(len(text)))
        assert hits == 1, answer


def test_chunking_overlap_and_span_check():
    words = [str(i) for i in range(10)]
    chunks = chunk_words(words, chunk_size=4, overlap=0.5)
    assert chunks[0] == ["0", "1", "2", "3"] and chunks[1] == ["2", "3", "4", "5"]
    assert chunks[-1][-1] == "9"  # nothing is dropped at the end
    assert contains_span(chunks[0], ["1", "2"]) and not contains_span(chunks[0], ["3", "4"])


def test_tiny_chunks_and_k1_lose_answers_but_cost_little():
    tiny = evaluate_config(RAGConfig(chunk_size=10, overlap=0.0, top_k=1))
    huge = evaluate_config(RAGConfig(chunk_size=150, overlap=0.0, top_k=8))
    assert tiny.answerable < 0.5 < huge.answerable
    assert tiny.context_words < huge.context_words


def test_bo_tunes_rag_better_than_random_search_on_average():
    _, grid_f = exhaustive_grid()
    bo = [
        rag_objective(bayesian_optimization(rag_objective, (0.0, 1.0), 3, iterations=25, seed=s).best_x)
        for s in range(5)
    ]
    rs = [random_search(rag_objective, (0.0, 1.0), 3, n_samples=30, seed=s).best_f for s in range(5)]
    assert np.mean(bo) < np.mean(rs)
    assert np.mean(bo) < grid_f + 0.03  # 30 evals land close to the 720-eval optimum
