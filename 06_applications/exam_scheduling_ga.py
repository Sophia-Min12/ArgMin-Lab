"""Exam timetabling with a genetic algorithm — constraints as penalties.

𝒳 is every way of assigning E exams to T time slots (Tᴱ configurations).
F adds up what is wrong with a timetable:

    F = HARD_WEIGHT · (student clashes + seat overflow) + back-to-back exams

Hard constraints are weighted so heavily that any clash-free timetable beats
any timetable with a clash; among clash-free timetables, the soft term decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HARD_WEIGHT = 1000.0


@dataclass(frozen=True)
class ExamInstance:
    enrollment: np.ndarray  # (students, exams) bool — who sits which exam
    n_slots: int
    capacity: int  # seats available per slot


@dataclass(frozen=True)
class TimetableCost:
    clashes: int  # a student with 2+ exams in one slot (counted per extra exam)
    overflow: int  # students beyond the seats of a slot
    back_to_back: int  # a student with exams in two consecutive slots

    @property
    def hard(self) -> int:
        return self.clashes + self.overflow

    @property
    def total(self) -> float:
        return HARD_WEIGHT * self.hard + self.back_to_back


def make_instance(
    n_exams: int = 30,
    n_slots: int = 10,
    n_students: int = 300,
    exams_per_student: int = 4,
    seed: int | None = None,
) -> ExamInstance:
    """Random instance that is guaranteed to have a clash-free timetable.

    A hidden timetable is drawn first and every student only enrolls in exams
    from distinct hidden slots; seats are set 10% above that timetable's
    busiest slot. The GA never sees the hidden timetable.
    """
    rng = np.random.default_rng(seed)
    hidden = rng.permutation(np.arange(n_exams) % n_slots)
    enrollment = np.zeros((n_students, n_exams), dtype=bool)
    for s in range(n_students):
        for slot in rng.choice(n_slots, size=exams_per_student, replace=False):
            enrollment[s, rng.choice(np.flatnonzero(hidden == slot))] = True
    busiest = max(enrollment[:, hidden == t].sum() for t in range(n_slots))
    return ExamInstance(enrollment, n_slots, int(np.ceil(1.1 * busiest)))


def evaluate(instance: ExamInstance, timetable: np.ndarray) -> TimetableCost:
    """Score one timetable (array: exam → slot) — fully vectorized."""
    slot_onehot = np.eye(instance.n_slots, dtype=int)[timetable]  # (exams, slots)
    per_student = instance.enrollment.astype(int) @ slot_onehot  # (students, slots)
    clashes = int(np.maximum(per_student - 1, 0).sum())
    overflow = int(np.maximum(per_student.sum(axis=0) - instance.capacity, 0).sum())
    busy = per_student > 0
    back_to_back = int((busy[:, :-1] & busy[:, 1:]).sum())
    return TimetableCost(clashes, overflow, back_to_back)


@dataclass
class ScheduleResult:
    best_timetable: np.ndarray
    best_cost: TimetableCost
    history: list[float] = field(default_factory=list)
    generations: int = 0


def schedule_exams(
    instance: ExamInstance,
    pop_size: int = 80,
    generations: int = 300,
    tournament_k: int = 3,
    mutation_rate: float = 0.03,
    elite: int = 2,
    seed: int | None = None,
) -> ScheduleResult:
    """Integer-coded GA: the same select → crossover → mutate → elitism loop
    as Level 3, with operators rewritten for a discrete 𝒳.

    - uniform crossover: each exam inherits its slot from either parent;
    - random-reset mutation: an exam jumps to a random slot.
    """
    rng = np.random.default_rng(seed)
    n_exams = instance.enrollment.shape[1]

    def fitness(tt: np.ndarray) -> float:
        return evaluate(instance, tt).total

    pop = rng.integers(0, instance.n_slots, size=(pop_size, n_exams))
    scores = np.array([fitness(ind) for ind in pop])
    history = [float(scores.min())]

    for _ in range(generations):
        order = np.argsort(scores)
        next_pop = [pop[i].copy() for i in order[:elite]]
        while len(next_pop) < pop_size:
            parents = []
            for _ in range(2):
                idx = rng.integers(0, pop_size, size=tournament_k)
                parents.append(pop[idx[np.argmin(scores[idx])]])
            child = np.where(rng.random(n_exams) < 0.5, parents[0], parents[1])
            mask = rng.random(n_exams) < mutation_rate
            child[mask] = rng.integers(0, instance.n_slots, size=mask.sum())
            next_pop.append(child)
        pop = np.array(next_pop)
        scores = np.array([fitness(ind) for ind in pop])
        history.append(float(scores.min()))

    best = int(np.argmin(scores))
    return ScheduleResult(pop[best], evaluate(instance, pop[best]), history, generations)


if __name__ == "__main__":
    instance = make_instance(seed=0)
    n_students, n_exams = instance.enrollment.shape
    rng = np.random.default_rng(0)
    random_cost = evaluate(instance, rng.integers(0, instance.n_slots, n_exams))

    result = schedule_exams(instance, seed=0)
    c = result.best_cost
    print(f"{n_exams} exams, {instance.n_slots} slots, {n_students} students, {instance.capacity} seats/slot")
    print(f"random timetable : clashes={random_cost.clashes:>3}  overflow={random_cost.overflow:>3}  back-to-back={random_cost.back_to_back}")
    print(f"GA timetable     : clashes={c.clashes:>3}  overflow={c.overflow:>3}  back-to-back={c.back_to_back}")
