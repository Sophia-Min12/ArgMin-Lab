# Level 6 · Applications

> Theory earns its keep here: the same `arg min` applied to problems that look like real work and real research.

## Files

| File | Problem | `𝒳` | `F` | Solver |
|------|---------|-----|-----|--------|
| [`tsp_simulated_annealing.py`](tsp_simulated_annealing.py) | **Traveling Salesman Problem** | tours (permutations) | tour length | simulated annealing + 2-opt (Level 2) |
| [`exam_scheduling_ga.py`](exam_scheduling_ga.py) | **Exam timetabling** | exam → slot assignments | penalties: clashes, seat overflow, back-to-back exams | integer-coded GA (Level 3) |
| [`rag_hyperparameter_tuning.py`](rag_hyperparameter_tuning.py) | **RAG hyperparameter tuning** | chunk size × overlap × top-k | (1 − answerable rate) + cost of context words | Bayesian optimization (Level 5) |

## Key ideas

- **Same skeleton, new `𝒳`.** TSP reuses Level 2's annealing loop and only redefines the neighbor (2-opt). The timetabler reuses Level 3's select → crossover → mutate → elitism loop with operators rewritten for discrete slots.
- **Constraints as penalties.** Hard constraints (no student sits two exams at once, no slot over capacity) are weighted ×1000, so any clash-free timetable beats any timetable with a clash. Among clash-free timetables, the soft term (back-to-back exams) decides. On the demo instance (30 exams, 10 slots, 300 students) the GA goes from 181 clashes and 169 overflowing seats in a random timetable to **0 / 0**. The random timetable shows *fewer* back-to-back exams only because its clashing exams share a slot.
- **When `F` is a pipeline, evaluations are expensive.** Every RAG evaluation re-chunks the corpus, rebuilds the index and answers 22 questions. That is exactly the setting Bayesian optimization was built for:

  | Method | Evaluations | Mean `F` (10 seeds) | Worst `F` |
  |--------|:-----------:|:-------------------:|:---------:|
  | Exhaustive grid | 720 | 0.0292 (optimum) | — |
  | **Bayesian optimization** | 30 | **0.0431** | **0.0499** |
  | Random search | 30 | 0.0515 | 0.0688 |

  The trade-off it finds is real. Tiny chunks cut answers in half, and huge chunks or a large top-k pay for words the generator does not need. The grid optimum is the middle ground: 40-word chunks, 30% overlap, top-3.
- **Honest scope.** No LLM is called: the corpus, questions and TF-IDF retriever are in the file, so every result is reproducible offline. Generator-side knobs (temperature, prompt template) need an LLM judge to score and are listed below.

## Roadmap

- [ ] **Generator-side RAG tuning** — temperature and prompt template, scored by an LLM judge.
- [ ] **Neural architecture / feature selection** with metaheuristics.
- [ ] **Portfolio optimization** — risk vs. return as a multi-objective problem with NSGA-II (Level 5).

## Reference

- AIMA 3rd ed., §4.1 *Local Search* (TSP and 2-opt) and Ch. 6 *Constraint Satisfaction Problems* (timetabling as CSP)
- Burke & Petrovic (2002), *Recent research directions in automated timetabling*
- Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
