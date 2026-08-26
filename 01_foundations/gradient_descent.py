"""Gradient descent with numerical differentiation and optional momentum.

Update rule: x ← x − η∇F(x)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def numerical_gradient(
    func: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Central-difference gradient: (F(x+ε) − F(x−ε)) / 2ε per coordinate."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        grad[i] = (func(x + step) - func(x - step)) / (2.0 * eps)
    return grad


@dataclass
class GDResult:
    """Optimization trace."""

    best_x: np.ndarray
    best_f: float
    history: list[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False


def gradient_descent(
    func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    learning_rate: float = 0.01,
    momentum: float = 0.0,
    max_iter: int = 5000,
    tol: float = 1e-8,
) -> GDResult:
    """Minimize `func` from `x0`.

    Args:
        learning_rate: step size η. Too large diverges, too small crawls.
        momentum: velocity coefficient in [0, 1). 0 disables momentum;
            higher values damp oscillation in valleys and accelerate progress.
        tol: stop when the gradient norm falls below this.
    """
    x = np.asarray(x0, dtype=float).copy()
    velocity = np.zeros_like(x)
    history: list[float] = [func(x)]

    for it in range(1, max_iter + 1):
        grad = numerical_gradient(func, x)
        if np.linalg.norm(grad) < tol:
            return GDResult(x, func(x), history, it, converged=True)
        velocity = momentum * velocity - learning_rate * grad
        x = x + velocity
        history.append(func(x))

    return GDResult(x, func(x), history, max_iter, converged=False)


if __name__ == "__main__":
    from objective_functions import rosenbrock, sphere

    r1 = gradient_descent(sphere, x0=np.array([3.0, -4.0]), learning_rate=0.1)
    print(f"sphere     : f={r1.best_f:.2e} at {np.round(r1.best_x, 4)} "
          f"({r1.iterations} iters, converged={r1.converged})")

    r2 = gradient_descent(
        rosenbrock, x0=np.array([-1.5, 2.0]), learning_rate=1e-3,
        momentum=0.9, max_iter=20000,
    )
    print(f"rosenbrock : f={r2.best_f:.2e} at {np.round(r2.best_x, 4)} "
          f"({r2.iterations} iters, converged={r2.converged})")
