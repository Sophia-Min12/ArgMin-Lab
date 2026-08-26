# Level 0 · Basics — What does "arg min" mean?

> This repository is named **ArgMin-Lab**. Before any algorithm, understand the two words in the name.

## `arg` = **argument** (the input of a function)

The key point: **`min` and `arg min` return different things.**

Take a simple example:

$$F(x) = (x-3)^2$$

| Notation | Question it asks | Answer |
|---|---|:---:|
| $\min F(x)$ | What is the smallest **value** of the function? (output) | $0$ |
| $\arg\min F(x)$ | Which **input** achieves that minimum? (argument) | $x = 3$ |

So **arg min** literally reads as *"the **arg**ument that **min**imizes"* — the input that makes the function smallest.

## Reading the master equation

$$X^{\star} = \underset{X' \in \mathcal{X}}{\arg\min}\, F\left(X'\right)$$

- We are **not** asking for the smallest value of $F(X')$.
- We are asking: among all possible **configurations** $\mathcal{X}$, which configuration $X^{\star}$ **achieves** that smallest value?

## Where you meet it in practice

- **Machine learning training**
  $\theta^{\star} = \arg\min_{\theta} \mathrm{Loss}(\theta)$ — what we want is not the loss value but the **optimal weights** $\theta^{\star}$.
- **TSP** ([`06_applications`](../06_applications/)) — we want not the shortest distance (min) but the **shortest tour itself** (arg min).
- **Every optimizer in this repo** returns both: `best_f` is the *min*, `best_x` is the *arg min*.

## Why the repo is named ArgMin-Lab

> A lab for finding **not the answer's value, but the configuration that produces the answer.**
