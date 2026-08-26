# Level 1 · Foundations

> Everything starts from the definition: an **objective function** `F(X)` and the set of all possible **configurations** `𝒳`. Optimization is finding `X* = arg min F(X')`.

## Files

| File | What it teaches |
|------|-----------------|
| [`objective_functions.py`](objective_functions.py) | Classic benchmark objectives (Sphere, Rosenbrock, Rastrigin, Ackley) with known optima |
| [`grid_random_search.py`](grid_random_search.py) | The two most naive baselines every optimizer must beat |
| [`gradient_descent.py`](gradient_descent.py) | Gradient descent with numerical differentiation and momentum |

## Key ideas

- **Why minimize?** Maximizing `F` is minimizing `−F` — one convention covers both.
- **Curse of dimensionality**: grid search cost grows exponentially with dimension; random search does not.
- **Gradient descent** follows `x ← x − η∇F(x)`; step size `η` decides convergence or divergence.

## 📖 Study note — Grid Search vs. Greedy Search

They sound similar but are completely different ideas.

| | **Grid Search** | **Greedy Search** |
|---|---|---|
| **What it is** | Divide the search space into a uniform **grid** and evaluate *every* point — exhaustive, no cleverness. | At every step take whatever looks **best right now** — a *strategy*, not an enumeration. |
| **Example** | Hyperparameter tuning: `lr ∈ {0.1, 0.01, 0.001} × batch ∈ {16, 32, 64}` → try all 9 combos (sklearn's `GridSearchCV`). | Hill climbing (accept a neighbor only if it improves); greedy best-first search in AIMA Ch. 3. |
| **Weakness** | Cost explodes exponentially with dimension — the curse of dimensionality. | Gets trapped in local optima — it never looks past the next step. |
| **In this repo** | [`grid_random_search.py`](grid_random_search.py) (Level 1) | [`../02_local_search/hill_climbing.py`](../02_local_search/hill_climbing.py) (Level 2) is exactly a greedy walk |

So the curriculum order is intentional: Level 1's grid search is the "dumb but certain" exhaustive baseline, and Level 2's hill climbing is the canonical greedy strategy — including its local-optimum weakness, which Levels 2–3 then learn to escape.

## Reference

- AIMA 3rd ed., §4.2 *Local Search in Continuous Spaces*
