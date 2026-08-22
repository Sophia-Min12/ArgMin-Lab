# Level 2 · Local Search · 지역 탐색

> When gradients are unavailable or the landscape is rugged, we walk: propose a neighbor, keep it if it is better — and sometimes even if it is worse.
> <sub>그레이디언트를 쓸 수 없거나 지형이 험할 때는 걷는다: 이웃해를 제안하고, 더 좋으면 받아들이고 — 때로는 더 나빠도 받아들인다.</sub>

## Files

| File | What it teaches |
|------|-----------------|
| [`hill_climbing.py`](hill_climbing.py) | Greedy neighbor-taking + random restarts to escape local optima <sub>· 탐욕적 이웃 선택과 랜덤 재시작</sub> |
| [`simulated_annealing.py`](simulated_annealing.py) | Metropolis acceptance `exp(−ΔF/T)` with a cooling schedule <sub>· 메트로폴리스 수용 확률과 냉각 스케줄</sub> |

## Key ideas · 핵심 아이디어

- **Local optimum trap**: pure hill climbing stops at the nearest peak of its valley. <sub>(순수 언덕 오르기는 가장 가까운 지역 최적점에 갇힌다)</sub>
- **Random restarts** trade compute for coverage. <sub>(재시작은 계산량을 대가로 탐색 범위를 넓힌다)</sub>
- **Annealing**: accept worse moves with probability `exp(−ΔF/T)`; as temperature `T` cools, the walk anneals from exploration into exploitation. <sub>(온도가 식을수록 탐험에서 활용으로 옮겨간다)</sub>

## Reference

- AIMA 3rd ed., §4.1 *Local Search Algorithms and Optimization Problems*
