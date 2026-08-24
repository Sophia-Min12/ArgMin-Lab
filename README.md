# ArgMin-Lab

$$X^{\star} = \underset{X' \in \mathcal{X}}{\arg\min} F\left(X'\right)$$

**From first principles to research-grade optimization — one repository, written in code, not words.**
**기초 원리부터 연구 수준의 최적화까지 — 말이 아닌 코드로 쌓아 올리는 저장소.**

> *I see the real world in 0 and 1. That is why I am 경01 Min.*
> This lab is where that worldview becomes executable: every concept I study about **searching and optimizing objective functions** is implemented, tested, and benchmarked here.

---

## 🗺️ Curriculum Roadmap · 커리큘럼 로드맵

| Level | Stage | Topics | Status |
|:-----:|-------|--------|:------:|
| 0 | [`00_basics`](00_basics/) | What "arg min" means — min vs. arg min, reading the master equation<br><sub>arg min의 의미 — min과 arg min의 차이, 대표 수식 읽는 법</sub> | ✅ |
| 1 | [`01_foundations`](01_foundations/) | Objective functions, grid/random search, gradient descent<br><sub>목적함수, 그리드/랜덤 탐색, 경사하강법</sub> | ✅ |
| 2 | [`02_local_search`](02_local_search/) | Hill climbing, simulated annealing<br><sub>언덕 오르기, 담금질 기법</sub> | ✅ |
| 3 | [`03_metaheuristics`](03_metaheuristics/) | Genetic algorithm, particle swarm optimization<br><sub>유전 알고리즘, 입자 군집 최적화</sub> | ✅ |
| 4 | [`04_adversarial_search`](04_adversarial_search/) | Minimax, alpha-beta pruning — **Min VS. Max**<br><sub>미니맥스, 알파-베타 가지치기</sub> | ✅ |
| 5 | [`05_modern_topics`](05_modern_topics/) | Bayesian optimization, CMA-ES, multi-objective (NSGA-II)<br><sub>베이지안 최적화, CMA-ES, 다목적 최적화</sub> | 🚧 |
| 6 | [`06_applications`](06_applications/) | TSP, scheduling, LLM/RAG hyperparameter tuning<br><sub>외판원 문제, 스케줄링, LLM/RAG 하이퍼파라미터 튜닝</sub> | 🚧 |

✅ implemented & tested &nbsp;·&nbsp; 🚧 in progress &nbsp;·&nbsp; 📝 planned

---

## 🚀 Quick Start

```bash
git clone https://github.com/Sophia-Min12/ArgMin-Lab.git
cd ArgMin-Lab
pip install -r requirements.txt

# Run all tests · 전체 테스트 실행
pytest

# Compare every implemented optimizer on classic benchmark functions
# 구현된 모든 최적화 기법을 벤치마크 함수에서 비교
python benchmarks/compare_optimizers.py
```

---

## 🧭 Design Principles · 설계 원칙

1. **NumPy only, no black boxes** — every algorithm is implemented from scratch so the mathematics stays visible.
   <sub>NumPy만 사용, 블랙박스 없음 — 수학이 그대로 보이도록 모든 알고리즘을 밑바닥부터 구현.</sub>
2. **Every algorithm is tested** — `tests/` proves each optimizer actually converges.
   <sub>모든 알고리즘은 테스트로 수렴을 증명.</sub>
3. **Benchmarked side by side** — `benchmarks/` runs them on the same objective functions for honest comparison.
   <sub>동일한 목적함수 위에서 정직하게 비교.</sub>
4. **Theory ↔ Reference** — each stage README cites *Artificial Intelligence: A Modern Approach* (Russell & Norvig) chapters it builds on.
   <sub>각 단계는 AIMA(러셀 & 노빅)의 해당 챕터와 연결.</sub>

---

## 📚 Primary Reference

- Stuart Russell & Peter Norvig, *Artificial Intelligence: A Modern Approach* (3rd ed.)
  — esp. **Ch. 4 Beyond Classical Search** and **Ch. 5 Adversarial Search**

## License

[MIT](LICENSE) © Sophia Min (경01 Min)
