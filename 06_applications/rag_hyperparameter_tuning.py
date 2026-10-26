"""RAG hyperparameter tuning with Bayesian optimization.

The retrieval half of a RAG pipeline has knobs nobody can set by theory:

    𝒳 = { chunk_size ∈ [10, 150] words,  overlap ∈ [0, 0.5],  top_k ∈ {1, …, 8} }

and F is measured, not derived:

    F = (1 − answerable rate) + COST_WEIGHT · (context words per query / 1000)

"Answerable" means at least one retrieved chunk contains the full answer
span, i.e. a generator *could* answer from the prompt. The cost term charges
for every word pasted into the prompt (tokens = latency = money).

No LLM is called: the corpus, questions and a TF-IDF retriever live in this
file, so every evaluation is deterministic and offline. Generator-side knobs
such as temperature need an LLM judge to score and are out of scope here.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "05_modern_topics"))
from bayesian_optimization import bayesian_optimization  # noqa: E402

COST_WEIGHT = 0.25

# Written as short sections for readability, then joined into four long
# chapters below — real RAG documents are long, which is what makes chunking
# a genuine trade-off.
SECTIONS: dict[str, str] = {
    "gradient descent": """
        Gradient descent updates the parameters by stepping against the gradient
        of the objective. The step size, usually called the learning rate,
        controls how far each update moves. If the learning rate is too large
        the iterates overshoot the minimum and the loss diverges; if it is too
        small, training crawls. Momentum keeps a running average of past
        gradients so that the walker builds speed along consistent directions
        and damps oscillations across steep walls. Gradient descent needs a
        differentiable objective; when only function values are available, a
        finite difference approximation replaces the analytic gradient at the
        cost of extra evaluations per step. On convex problems gradient descent
        reaches the global minimum, but on nonconvex landscapes it settles in
        whichever basin the starting point belongs to.""",
    "simulated annealing": """
        Simulated annealing borrows its name from metallurgy, where a metal is
        heated and then cooled slowly so its atoms settle into a low energy
        crystal. The algorithm proposes a random neighbor and always accepts it
        when it improves the objective. A worse neighbor is accepted with
        probability exp of minus delta over temperature, the Metropolis
        criterion. Early on the temperature is high and the search behaves like
        a random walk that can climb out of poor basins. A geometric cooling
        schedule multiplies the temperature by a constant slightly below one
        after each stage. If cooling is too fast the walker freezes in a local
        optimum, a failure known as quenching. Given a logarithmically slow
        schedule annealing provably finds the global optimum, although that
        schedule is far too slow in practice.""",
    "genetic algorithms": """
        A genetic algorithm keeps a population of candidate solutions and
        improves it over generations. Tournament selection picks the fittest of
        a few randomly drawn individuals to become a parent. Crossover
        recombines two parents so that useful building blocks from each can
        meet in one child. Mutation applies small random changes that keep
        diversity in the population and reintroduce lost genes. Elitism copies
        the best individuals unchanged into the next generation, which
        guarantees that the best fitness never gets worse. Too much selection
        pressure makes the population converge prematurely onto one region,
        while too little turns the algorithm into random search.""",
    "particle swarm": """
        Particle swarm optimization was introduced by Kennedy and Eberhart in
        1995, inspired by flocks of birds. Each particle has a position and a
        velocity. The velocity update blends three terms: inertia from the
        previous velocity, a cognitive pull toward the particle's own best
        position, and a social pull toward the best position found by the whole
        swarm. Constriction coefficients of 0.72 for inertia and 1.49 for both
        pulls give stable convergence. Clamping the velocity stops particles
        from flying out of the search box.""",
    "bayesian optimization": """
        Bayesian optimization is designed for objectives that are expensive to
        evaluate, such as training a deep network or running a full retrieval
        pipeline. It fits a Gaussian process surrogate to every evaluation so
        far. The surrogate returns a posterior mean and an uncertainty at any
        untested point. An acquisition function such as expected improvement
        turns the mean and uncertainty into one score that balances
        exploitation and exploration. The next evaluation is placed where the
        acquisition function is maximal. Because the Gaussian process needs a
        cubic time matrix factorization, Bayesian optimization is practical for
        a few hundred evaluations, not for millions.""",
    "cma-es": """
        The covariance matrix adaptation evolution strategy samples candidates
        from a multivariate normal distribution. After ranking the samples it
        moves the mean toward the best ones, adapts a global step size using an
        evolution path, and updates the covariance matrix with rank one and
        rank mu terms. Over time the covariance matrix learns the shape of the
        landscape, stretching along narrow curved valleys such as the
        Rosenbrock function. CMA-ES is invariant to rotations of the search
        space, which isotropic mutations in a genetic algorithm are not. The
        default population size grows only logarithmically with dimension.""",
    "multi-objective": """
        In multi-objective optimization there is rarely one best answer,
        because improving one objective worsens another. A solution dominates
        another when it is no worse on every objective and strictly better on
        at least one. The set of non-dominated solutions forms the Pareto
        front. NSGA-II ranks the population into successive fronts by fast
        non-dominated sorting and uses crowding distance to prefer solutions in
        sparsely populated regions of the front. Hypervolume, the area
        dominated by a front up to a reference point, measures convergence and
        spread at the same time.""",
    "adversarial search": """
        In a two player zero sum game, one player maximizes the same value
        that the other minimizes. Minimax assumes both players play optimally
        and backs values up from the leaves of the game tree. Alpha beta
        pruning returns exactly the same decision as minimax while skipping
        branches that cannot influence it. With perfect move ordering, alpha
        beta reduces the effective branching factor to roughly its square root,
        which lets the search look twice as deep in the same time. Tic tac toe
        played perfectly by both sides always ends in a draw.""",
    "traveling salesman": """
        The traveling salesman problem asks for the shortest closed tour that
        visits every city exactly once. The number of possible tours grows
        factorially with the number of cities, so exhaustive enumeration is
        hopeless beyond a handful of cities. The 2-opt move removes two edges
        and reconnects the tour by reversing the segment between them, which
        untangles crossing edges. Combined with simulated annealing, 2-opt
        neighbors quickly shorten a random tour.""",
    "timetabling": """
        Exam timetabling assigns every exam to a time slot. Hard constraints
        must never be broken: no student can sit two exams in the same slot,
        and the number of students in a slot cannot exceed the seats
        available. Soft constraints express preferences, such as avoiding back
        to back exams for the same student. A common approach turns constraints
        into penalties added to the objective, with hard violations weighted
        far more heavily than soft ones. Timetabling generalizes graph coloring
        and is NP hard.""",
    "retrieval augmented generation": """
        Retrieval augmented generation lets a language model answer from
        documents it was not trained on. Documents are split into chunks, each
        chunk is indexed, and at question time the top k most similar chunks
        are pasted into the prompt. Small chunks are precise but may cut an
        answer in half; large chunks keep context but dilute the similarity
        score and cost more tokens. Overlap between neighboring chunks reduces
        the chance that an answer is split across a boundary. Raising top k
        improves recall but lengthens the prompt, which raises latency and
        cost.""",
    "no free lunch": """
        The no free lunch theorem of Wolpert and Macready states that, averaged
        over all possible objective functions, every search algorithm performs
        equally well. An optimizer can only beat random search on a class of
        problems by exploiting structure that the class shares, such as
        smoothness, separability or a single basin. This is why benchmark
        suites mix unimodal functions like the sphere with multimodal ones like
        Rastrigin and Ackley.""",
}

CHAPTERS: dict[str, list[str]] = {
    "Ch. 1  Single-walker search": ["gradient descent", "simulated annealing", "no free lunch"],
    "Ch. 2  Population search": ["genetic algorithms", "particle swarm", "cma-es"],
    "Ch. 3  Opponents, surrogates and fronts": ["adversarial search", "bayesian optimization", "multi-objective"],
    "Ch. 4  Applications": ["traveling salesman", "timetabling", "retrieval augmented generation"],
}

CORPUS: dict[str, str] = {
    title: " ".join(SECTIONS[name] for name in names) for title, names in CHAPTERS.items()
}

# (question, answer span that must appear inside a retrieved chunk)
QUESTIONS: list[tuple[str, str]] = [
    ("What happens when the step size is set too high?", "overshoot the minimum and the loss diverges"),
    ("How can a gradient be estimated when only function values are known?", "finite difference approximation"),
    ("Why is annealing named after metal work?", "heated and then cooled slowly"),
    ("What is it called when the temperature drops too quickly and the search freezes?", "known as quenching"),
    ("Which mechanism ensures the best fitness never gets worse between generations?", "elitism copies the best individuals unchanged"),
    ("What is the risk of very strong selection pressure?", "converge prematurely onto one region"),
    ("Who introduced particle swarm optimization?", "kennedy and eberhart in 1995"),
    ("What stops particles from leaving the search box?", "clamping the velocity"),
    ("Why is Bayesian optimization limited to a few hundred evaluations?", "cubic time matrix factorization"),
    ("What does the Gaussian process surrogate return at an untested point?", "posterior mean and an uncertainty"),
    ("How does the CMA-ES population size scale with dimension?", "grows only logarithmically with dimension"),
    ("Why does rotating the coordinates not affect CMA-ES?", "invariant to rotations of the search space"),
    ("How does NSGA-II prefer solutions in sparse regions of the front?", "uses crowding distance"),
    ("Which metric measures convergence and spread of a front at the same time?", "hypervolume the area dominated by a front"),
    ("How does a perfectly played noughts and crosses game end?", "always ends in a draw"),
    ("How much deeper can alpha beta search with perfect move ordering?", "look twice as deep"),
    ("What does the 2-opt move do between the two removed edges?", "reversing the segment between them"),
    ("How do metaheuristics usually handle timetabling constraints?", "turns constraints into penalties"),
    ("Which classic problem does exam timetabling generalize?", "generalizes graph coloring"),
    ("Why add overlap between neighboring chunks?", "reduces the chance that an answer is split"),
    ("What is the downside of raising top k?", "lengthens the prompt"),
    ("Which theorem says no optimizer wins on every objective function?", "no free lunch theorem of wolpert and macready"),
]

STOPWORDS = frozenset(
    "a an the and or of to in on at by for from with as is are was be it its "
    "that this which what when why how who does do can into one".split()
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk_words(words: list[str], chunk_size: int, overlap: float) -> list[list[str]]:
    """Fixed-size word windows; consecutive windows share ⌊overlap · size⌋ words."""
    stride = max(1, chunk_size - int(overlap * chunk_size))
    chunks = []
    for start in range(0, len(words), stride):
        chunks.append(words[start : start + chunk_size])
        if start + chunk_size >= len(words):
            break
    return chunks


def contains_span(chunk: list[str], span: list[str]) -> bool:
    n = len(span)
    return any(chunk[i : i + n] == span for i in range(len(chunk) - n + 1))


class TfidfRetriever:
    """Bag-of-words TF-IDF with cosine similarity — the classic sparse retriever."""

    def __init__(self, chunks: list[list[str]]) -> None:
        docs = [[w for w in c if w not in STOPWORDS] for c in chunks]
        self.vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        counts = np.zeros((len(docs), len(self.vocab)))
        for row, doc in enumerate(docs):
            for w in doc:
                counts[row, self.vocab[w]] += 1
        df = (counts > 0).sum(axis=0)
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1.0
        self.matrix = self._normalize(counts * self.idf)

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(norms == 0, 1.0, norms)

    def top_k(self, query: str, k: int) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for w in tokenize(query):
            if w in self.vocab:
                vec[self.vocab[w]] += 1
        scores = self.matrix @ self._normalize(vec * self.idf)
        return np.argsort(-scores, kind="stable")[:k]


@dataclass(frozen=True)
class RAGConfig:
    chunk_size: int
    overlap: float
    top_k: int


@dataclass(frozen=True)
class RAGMetrics:
    answerable: float  # fraction of questions whose answer span was retrieved
    context_words: float  # mean words pasted into the prompt per question

    @property
    def objective(self) -> float:
        return (1.0 - self.answerable) + COST_WEIGHT * self.context_words / 1000.0


@lru_cache(maxsize=None)
def evaluate_config(config: RAGConfig) -> RAGMetrics:
    """Run every question through the pipeline built with `config`."""
    chunks = [
        c
        for text in CORPUS.values()
        for c in chunk_words(tokenize(text), config.chunk_size, config.overlap)
    ]
    retriever = TfidfRetriever(chunks)
    hits, words = 0, 0
    for question, answer in QUESTIONS:
        retrieved = [chunks[i] for i in retriever.top_k(question, config.top_k)]
        span = tokenize(answer)
        hits += any(contains_span(c, span) for c in retrieved)
        words += sum(len(c) for c in retrieved)
    return RAGMetrics(hits / len(QUESTIONS), words / len(QUESTIONS))


def decode(u: np.ndarray) -> RAGConfig:
    """Map a point of the unit cube [0, 1]³ to a concrete configuration."""
    u = np.clip(u, 0.0, 1.0)
    return RAGConfig(
        chunk_size=int(round(10 + 140 * u[0])),
        overlap=round(0.5 * float(u[1]), 2),
        top_k=int(round(1 + 7 * u[2])),
    )


def rag_objective(u: np.ndarray) -> float:
    return evaluate_config(decode(u)).objective


def exhaustive_grid() -> tuple[RAGConfig, float]:
    """Ground truth on a coarse grid: 15 sizes × 6 overlaps × 8 k = 720 runs."""
    configs = [
        RAGConfig(size, overlap, k)
        for size in range(10, 151, 10)
        for overlap in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
        for k in range(1, 9)
    ]
    scores = [evaluate_config(c).objective for c in configs]
    best = int(np.argmin(scores))
    return configs[best], scores[best]


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1] / "01_foundations"))
    from grid_random_search import random_search

    budget, seeds = 30, range(10)
    grid_config, grid_f = exhaustive_grid()
    m = evaluate_config(grid_config)
    print(f"{len(CORPUS)} chapters, {len(QUESTIONS)} questions, COST_WEIGHT={COST_WEIGHT}\n")
    print(
        f"grid optimum (720 evals): F={grid_f:.4f}  answerable={m.answerable:.0%}  "
        f"words={m.context_words:.0f}  size={grid_config.chunk_size}, "
        f"overlap={grid_config.overlap}, k={grid_config.top_k}\n"
    )

    bo = [
        rag_objective(bayesian_optimization(rag_objective, (0.0, 1.0), 3, iterations=budget - 5, seed=s).best_x)
        for s in seeds
    ]
    rs = [random_search(rag_objective, (0.0, 1.0), 3, n_samples=budget, seed=s).best_f for s in seeds]
    print(f"{budget} evals each, {len(seeds)} seeds:   mean F   worst F")
    print(f"  Bayesian optimization   {np.mean(bo):.4f}    {np.max(bo):.4f}")
    print(f"  random search           {np.mean(rs):.4f}    {np.max(rs):.4f}")
