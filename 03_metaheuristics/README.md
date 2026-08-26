# Level 3 · Metaheuristics

> One walker gets trapped; a **population** shares what it learns. Evolution and swarms are search strategies borrowed from nature.

## Files

| File | What it teaches |
|------|-----------------|
| [`genetic_algorithm.py`](genetic_algorithm.py) | Selection → crossover → mutation → elitism, real-coded |
| [`particle_swarm.py`](particle_swarm.py) | Velocity update pulled by personal & global bests |

## Key ideas

- **Exploration vs. exploitation** is population-level here: diversity explores, selection pressure exploits.
- **GA**: tournament selection + blend crossover (BLX-α) + Gaussian mutation; elitism guarantees monotone best fitness.
- **PSO**: each particle balances inertia, its own memory, and the swarm's memory — `v ← wv + c₁r₁(p−x) + c₂r₂(g−x)`.

## Reference

- AIMA 3rd ed., §4.1.4 *Genetic Algorithms*
- Kennedy & Eberhart (1995), *Particle Swarm Optimization*
