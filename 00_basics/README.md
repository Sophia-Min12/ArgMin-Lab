# Level 0 · Basics · 기초 개념 — What does "arg min" mean?

> This repository is named **ArgMin-Lab**. Before any algorithm, understand the two words in the name.
> <sub>이 저장소의 이름은 **ArgMin-Lab**이다. 어떤 알고리즘보다 먼저, 이름에 담긴 두 단어부터 이해하자.</sub>

## `arg` = **argument** (the input of a function) · 함수의 입력값(인수)

The key point: **`min` and `arg min` return different things.**
<sub>핵심: **`min`과 `arg min`은 반환하는 것이 다르다.**</sub>

Take a simple example · 간단한 예시:

$$F(x) = (x-3)^2$$

| Notation | Question it asks | Answer |
|---|---|:---:|
| $\min F(x)$ | What is the smallest **value** of the function? (output)<br><sub>함수의 **최솟값**이 얼마인가? (출력값)</sub> | $0$ |
| $\arg\min F(x)$ | Which **input** achieves that minimum? (argument)<br><sub>그 최솟값을 만드는 **입력값**이 무엇인가? (인수)</sub> | $x = 3$ |

So **arg min** literally reads as *"the **arg**ument that **min**imizes"* — the input that makes the function smallest.
<sub>즉 **arg min**은 "함수를 최소로 만드는 **입력(인수)**"라는 뜻이다.</sub>

## Reading the master equation · 대표 수식 읽기

$$X^{\star} = \underset{X' \in \mathcal{X}}{\arg\min}\, F\left(X'\right)$$

- We are **not** asking for the smallest value of $F(X')$.
  <sub>$F(X')$의 가장 작은 **값**을 묻는 게 아니다.</sub>
- We are asking: among all possible **configurations** $\mathcal{X}$, which configuration $X^{\star}$ **achieves** that smallest value?
  <sub>모든 **구성(configuration)** $\mathcal{X}$ 중에서 그 최솟값을 **달성하는 구성 $X^{\star}$ 자체**를 찾겠다는 선언이다.</sub>

## Where you meet it in practice · 실전에서 만나는 곳

- **Machine learning training** · 머신러닝 학습
  $\theta^{\star} = \arg\min_{\theta} \operatorname{Loss}(\theta)$ — what we want is not the loss value but the **optimal weights** $\theta^{\star}$.
  <sub>우리가 원하는 건 손실값이 아니라 **최적의 가중치**다.</sub>
- **TSP** ([`06_applications`](../06_applications/)) — we want not the shortest distance (min) but the **shortest tour itself** (arg min).
  <sub>최단 거리(min)가 아니라 **최단 경로 그 자체**(arg min)를 찾는다.</sub>
- **Every optimizer in this repo** returns both: `best_f` is the *min*, `best_x` is the *arg min*.
  <sub>이 저장소의 모든 최적화 기법이 `best_f`(min)와 `best_x`(arg min)를 **둘 다** 반환하는 이유다.</sub>

## Why the repo is named ArgMin-Lab · 저장소 이름의 의미

> A lab for finding **not the answer's value, but the configuration that produces the answer.**
> <sub>정답 **값**이 아니라, 정답을 **만들어내는 구성**을 찾는 실험실.</sub>
