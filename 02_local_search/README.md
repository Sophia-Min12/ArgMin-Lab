# Level 2 · Local Search

> When gradients are unavailable or the landscape is rugged, we walk: propose a neighbor, keep it if it is better — and sometimes even if it is worse.

## Files

| File | What it teaches |
|------|-----------------|
| [`hill_climbing.py`](hill_climbing.py) | Greedy neighbor-taking + random restarts to escape local optima |
| [`simulated_annealing.py`](simulated_annealing.py) | Metropolis acceptance `exp(−ΔF/T)` with a cooling schedule |

## Key ideas

- **Hill climbing *is* greedy search**: at every step it takes whatever looks best right now — see the grid-vs-greedy study note in [Level 1](../01_foundations/README.md).
- **Local optimum trap**: pure hill climbing stops at the nearest peak of its valley.
- **Random restarts** trade compute for coverage.
- **Annealing**: accept worse moves with probability `exp(−ΔF/T)`; as temperature `T` cools, the walk anneals from exploration into exploitation.

## Reference

- AIMA 3rd ed., §4.1 *Local Search Algorithms and Optimization Problems*
