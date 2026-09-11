# Level 4 · Adversarial Search — **Min VS. Max**

> So far the objective sat still. Now it fights back: an opponent **maximizes** exactly what we **minimize**.

## Files

| File | What it teaches |
|------|-----------------|
| [`minimax.py`](minimax.py) | Minimax with alpha-beta pruning on Tic-Tac-Toe, with a perfect-play demo |

## Key ideas

- **Minimax value**: assume both sides play optimally; the game tree's value backs up from the leaves.
- **Alpha-beta pruning** returns the *same* answer while skipping branches that cannot change it — often reducing O(bᵐ) toward O(b^(m/2)).
- **Perfect play in Tic-Tac-Toe is a draw** — the test suite proves it by self-play.

## Reference

- AIMA 3rd ed., **Ch. 5 Adversarial Search** (retitled *Adversarial Search and Games* in the 4th ed.)
