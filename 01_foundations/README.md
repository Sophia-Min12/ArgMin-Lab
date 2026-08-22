# Level 1 · Foundations · 기초

> Everything starts from the definition: an **objective function** `F(X)` and the set of all possible **configurations** `𝒳`. Optimization is finding `X* = arg min F(X')`.
> <sub>모든 것은 정의에서 시작한다: **목적함수** `F(X)`와 가능한 모든 **구성(configuration)** 의 집합 `𝒳`. 최적화란 `X* = arg min F(X')`를 찾는 일이다.</sub>

## Files

| File | What it teaches |
|------|-----------------|
| [`objective_functions.py`](objective_functions.py) | Classic benchmark objectives (Sphere, Rosenbrock, Rastrigin, Ackley) with known optima <sub>· 대표 벤치마크 목적함수와 알려진 최적해</sub> |
| [`grid_random_search.py`](grid_random_search.py) | The two most naive baselines every optimizer must beat <sub>· 모든 최적화 기법이 이겨야 할 가장 순진한 두 기준선</sub> |
| [`gradient_descent.py`](gradient_descent.py) | Gradient descent with numerical differentiation and momentum <sub>· 수치 미분과 모멘텀을 이용한 경사하강법</sub> |

## Key ideas · 핵심 아이디어

- **Why minimize?** Maximizing `F` is minimizing `−F` — one convention covers both. <sub>(최대화는 `−F` 최소화와 동일 — 하나의 관례로 둘 다 처리)</sub>
- **Curse of dimensionality**: grid search cost grows exponentially with dimension; random search does not. <sub>(그리드 탐색 비용은 차원에 지수적으로 증가, 랜덤 탐색은 그렇지 않음)</sub>
- **Gradient descent** follows `x ← x − η∇F(x)`; step size `η` decides convergence or divergence. <sub>(학습률이 수렴/발산을 결정)</sub>

## Reference

- AIMA 3rd ed., §4.2 *Local Search in Continuous Spaces*
