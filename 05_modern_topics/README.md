# Level 5 · Modern Topics

> Research-grade optimizers used in today's ML pipelines and papers. Each one relaxes an assumption the earlier levels quietly made: that evaluations are cheap, that every direction looks the same, and that there is only one objective.

## Files

| File | What it teaches |
|------|-----------------|
| [`bayesian_optimization.py`](bayesian_optimization.py) | Gaussian-process surrogate + expected improvement — optimizing when every evaluation is **expensive** |
| [`cma_es.py`](cma_es.py) | CMA-ES — an evolution strategy that **learns the shape** of the landscape through its covariance matrix |
| [`nsga2.py`](nsga2.py) | NSGA-II — **multi-objective** search: non-dominated sorting, crowding distance, hypervolume |

## Key ideas

- **Bayesian optimization** fits a cheap model of `F` (a Gaussian process) to the points seen so far and evaluates `F` only where the model says it is worth it. **Expected improvement** puts exploitation (low predicted mean) and exploration (high uncertainty) into one closed-form score. On a 40-evaluation budget it reaches `F ≈ 0.01` on 2-D Rosenbrock, where random search stops at `0.21`.
- **CMA-ES** samples from `N(m, σ²C)`. Each generation it moves the mean `m` toward the best samples, adapts the step size `σ` from an evolution path, and stretches the covariance `C` along directions that keep paying off. That is why it solves the curved Rosenbrock valley (`0.0000` in the benchmark), where the isotropic GA stalls at `2.27`.
- **No free lunch, in action**: CMA-ES with its default small population scores `9.95` on Rastrigin, where GA and PSO reach `0`. Restart strategies with growing populations (IPOP-CMA-ES) are the standard fix.
- **Multi-objective**: with `F = (f₁, f₂)` there is usually no single arg min, only a **Pareto front** of trade-offs. NSGA-II ranks the population into fronts, keeps the front evenly spread with **crowding distance**, and is elitist (parents ∪ children compete for survival). Its result is measured by **hypervolume**: on 30-D ZDT1 it reaches `0.864` against `0.877` for the true front.

> Run `python benchmarks/compare_optimizers.py` to see CMA-ES next to the Level 1–3 optimizers. Bayesian optimization is deliberately left out of that 12,000-evaluation race: it is built for budgets of tens of evaluations, and fitting the GP costs O(n³) in the number of evaluations.

## Next

- [ ] **Constrained optimization** — penalty & barrier methods.
- [ ] **Gradient-based deep learning optimizers** — SGD → Momentum → Adam, from scratch.

## Reference

- AIMA 3rd ed., §4.1.4 *Genetic Algorithms* (the population idea NSGA-II and CMA-ES build on)
- Shahriari et al. (2016), *Taking the Human Out of the Loop: A Review of Bayesian Optimization*
- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, Ch. 2 & 5
- Hansen (2016), *The CMA Evolution Strategy: A Tutorial*
- Deb, Pratap, Agarwal & Meyarivan (2002), *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*
