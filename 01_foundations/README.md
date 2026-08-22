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

## 📖 Study note · 공부 노트 — Grid Search vs. Greedy Search

They sound similar but are completely different ideas. <sub>발음은 비슷하지만 완전히 다른 개념이다.</sub>

| | **Grid Search** (그리드/격자 탐색) | **Greedy Search** (탐욕 탐색) |
|---|---|---|
| **What it is** | Divide the search space into a uniform **grid** and evaluate *every* point — exhaustive, no cleverness. <sub>탐색 공간을 격자로 나눠 모든 점을 전부 평가하는 전수조사.</sub> | At every step take whatever looks **best right now** — a *strategy*, not an enumeration. <sub>매 순간 당장 가장 좋아 보이는 선택만 하는 전략.</sub> |
| **Example** | Hyperparameter tuning: `lr ∈ {0.1, 0.01, 0.001} × batch ∈ {16, 32, 64}` → try all 9 combos (sklearn's `GridSearchCV`). | Hill climbing (accept a neighbor only if it improves); greedy best-first search in AIMA Ch. 3. |
| **Weakness** | Cost explodes exponentially with dimension — the curse of dimensionality. <sub>차원의 저주로 비용이 지수적으로 폭발.</sub> | Gets trapped in local optima — it never looks past the next step. <sub>다음 한 수만 보므로 지역 최적점에 갇힌다.</sub> |
| **In this repo** | [`grid_random_search.py`](grid_random_search.py) (Level 1) | [`../02_local_search/hill_climbing.py`](../02_local_search/hill_climbing.py) (Level 2) is exactly a greedy walk |

So the curriculum order is intentional: Level 1's grid search is the "dumb but certain" exhaustive baseline, and Level 2's hill climbing is the canonical greedy strategy — including its local-optimum weakness, which Levels 2–3 then learn to escape.
<sub>커리큘럼 순서가 곧 이야기다: Level 1의 그리드 탐색은 "무식하지만 확실한" 기준선, Level 2의 언덕 오르기가 탐욕 전략의 대표 — 그리고 그 약점(지역 최적점)을 Level 2–3이 탈출하는 법을 배운다.</sub>

## Reference

- AIMA 3rd ed., §4.2 *Local Search in Continuous Spaces*
