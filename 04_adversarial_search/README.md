# Level 4 · Adversarial Search · 적대적 탐색 — **Min VS. Max**

> So far the objective sat still. Now it fights back: an opponent **maximizes** exactly what we **minimize**.
> <sub>지금까지 목적함수는 가만히 있었다. 이제는 반격한다 — 상대는 우리가 **최소화**하려는 값을 정확히 **최대화**한다.</sub>

## Files

| File | What it teaches |
|------|-----------------|
| [`minimax.py`](minimax.py) | Minimax with alpha-beta pruning on Tic-Tac-Toe, with a perfect-play demo <sub>· 틱택토 위 미니맥스 + 알파-베타 가지치기, 완전 플레이 데모</sub> |

## Key ideas · 핵심 아이디어

- **Minimax value**: assume both sides play optimally; the game tree's value backs up from the leaves. <sub>(양쪽 모두 최선을 다한다고 가정하고 리프에서 값을 거슬러 올린다)</sub>
- **Alpha-beta pruning** returns the *same* answer while skipping branches that cannot change it — often reducing O(bᵐ) toward O(b^(m/2)). <sub>(답은 그대로, 결과에 영향 없는 가지만 건너뛴다)</sub>
- **Perfect play in Tic-Tac-Toe is a draw** — the test suite proves it by self-play. <sub>(완전 플레이 틱택토는 무승부 — 테스트가 자가 대국으로 증명한다)</sub>

## Why this matters to me · 왜 이 주제인가

Ever since I was young, I enjoyed learning about things that held completely opposite information — **Min VS. Max** is that idea, formalized.

## Reference

- AIMA 3rd ed., **Ch. 5 Adversarial Search** (retitled *Adversarial Search and Games* in the 4th ed.)
