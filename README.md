# ArgMin-Lab

$$X^{\star} = \underset{X' \in \mathcal{X}}{\arg\min} F\left(X'\right)$$

**From first principles to research-grade optimization — one repository, written in code, not words.**

> *I see the real world in 0 and 1. That is why I am 경01 Min.*
> This lab is where that worldview becomes executable: every concept I study about **searching and optimizing objective functions** is implemented, tested, and benchmarked here.

---

## 🗺️ Curriculum Roadmap

| Level | Stage | Topics | Status |
|:-----:|-------|--------|:------:|
| 0 | [`00_basics`](00_basics/) | What "arg min" means — min vs. arg min, reading the master equation | ✅ |
| 1 | [`01_foundations`](01_foundations/) | Objective functions, grid/random search, gradient descent | ✅ |
| 2 | [`02_local_search`](02_local_search/) | Hill climbing, simulated annealing | ✅ |
| 3 | [`03_metaheuristics`](03_metaheuristics/) | Genetic algorithm, particle swarm optimization | ✅ |
| 4 | [`04_adversarial_search`](04_adversarial_search/) | Minimax, alpha-beta pruning — **Min VS. Max** | ✅ |
| 5 | [`05_modern_topics`](05_modern_topics/) | Bayesian optimization, CMA-ES, multi-objective (NSGA-II) | 🚧 |
| 6 | [`06_applications`](06_applications/) | TSP, scheduling, LLM/RAG hyperparameter tuning | 🚧 |

✅ implemented & tested &nbsp;·&nbsp; 🚧 in progress &nbsp;·&nbsp; 📝 planned

---

## 🚀 Quick Start

```bash
git clone https://github.com/Sophia-Min12/ArgMin-Lab.git
cd ArgMin-Lab
pip install -r requirements.txt

# Run all tests
pytest

# Compare every implemented optimizer on classic benchmark functions
python benchmarks/compare_optimizers.py
```

---

## 🧭 Design Principles

1. **NumPy only, no black boxes** — every algorithm is implemented from scratch so the mathematics stays visible.
2. **Every algorithm is tested** — `tests/` proves each optimizer actually converges.
3. **Benchmarked side by side** — `benchmarks/` runs them on the same objective functions for honest comparison.
4. **Theory ↔ Reference** — each stage README cites *Artificial Intelligence: A Modern Approach* (Russell & Norvig) chapters it builds on.

---

## 📚 Primary Reference

- Stuart Russell & Peter Norvig, *Artificial Intelligence: A Modern Approach* (3rd ed.)
  — esp. **Ch. 4 Beyond Classical Search** and **Ch. 5 Adversarial Search**

## License

[MIT](LICENSE) © Sophia Min
