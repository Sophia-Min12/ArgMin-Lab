# Level 3 · Metaheuristics · 메타휴리스틱

> One walker gets trapped; a **population** shares what it learns. Evolution and swarms are search strategies borrowed from nature.
> <sub>혼자 걷는 탐색자는 갇히지만, **개체군**은 배운 것을 공유한다. 진화와 군집은 자연에서 빌려온 탐색 전략이다.</sub>

## Files

| File | What it teaches |
|------|-----------------|
| [`genetic_algorithm.py`](genetic_algorithm.py) | Selection → crossover → mutation → elitism, real-coded <sub>· 선택→교차→변이→엘리트 보존 (실수 인코딩)</sub> |
| [`particle_swarm.py`](particle_swarm.py) | Velocity update pulled by personal & global bests <sub>· 개인 최적과 전역 최적이 끌어당기는 속도 갱신</sub> |

## Key ideas · 핵심 아이디어

- **Exploration vs. exploitation** is population-level here: diversity explores, selection pressure exploits. <sub>(다양성은 탐험을, 선택압은 활용을 담당)</sub>
- **GA**: tournament selection + blend crossover (BLX-α) + Gaussian mutation; elitism guarantees monotone best fitness. <sub>(엘리트 보존은 최고 적합도의 단조 개선을 보장)</sub>
- **PSO**: each particle balances inertia, its own memory, and the swarm's memory — `v ← wv + c₁r₁(p−x) + c₂r₂(g−x)`. <sub>(관성·개인 기억·군집 기억의 균형)</sub>

## Reference

- AIMA 3rd ed., §4.1.4 *Genetic Algorithms*
- Kennedy & Eberhart (1995), *Particle Swarm Optimization*
