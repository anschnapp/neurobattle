"""Genetic algorithm for evolving robot brains."""

from __future__ import annotations

import numpy as np
from typing import Callable

from brain import Brain


class Population:
    """A population of brains that evolves via genetic algorithm.

    Usage:
        pop = Population(size=20, input_size=4, hidden_size=16, output_size=4)
        for generation in range(100):
            for i, brain in enumerate(pop.brains):
                score = evaluate(brain)  # user-defined
                pop.set_fitness(i, score)
            pop.evolve()
    """

    def __init__(
        self,
        size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        elite_count: int = 3,
        mutation_rate: float = 0.1,
    ):
        self.size = size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.generation = 0

        self.brains: list[Brain] = [
            Brain(input_size, hidden_size, output_size) for _ in range(size)
        ]
        self.fitness: np.ndarray = np.zeros(size, dtype=np.float32)

    def set_fitness(self, index: int, score: float):
        self.fitness[index] = score

    def get_best(self) -> Brain:
        """Return a copy of the highest-fitness brain."""
        best_idx = int(np.argmax(self.fitness))
        return self.brains[best_idx].copy()

    def get_stats(self) -> dict:
        return {
            "generation": self.generation,
            "best_fitness": float(np.max(self.fitness)),
            "avg_fitness": float(np.mean(self.fitness)),
            "worst_fitness": float(np.min(self.fitness)),
        }

    def evolve(self):
        """Create the next generation: keep elites, crossover+mutate the rest."""
        ranked = np.argsort(self.fitness)[::-1]  # best first

        # Keep elites unchanged
        new_brains: list[Brain] = []
        for i in range(self.elite_count):
            new_brains.append(self.brains[ranked[i]].copy())

        # Fill remaining slots with crossover + mutation from top performers
        top_half = ranked[: self.size // 2]
        while len(new_brains) < self.size:
            # Pick two different parents from top half
            p1, p2 = np.random.choice(top_half, size=2, replace=False)
            parent1 = self.brains[p1]
            parent2 = self.brains[p2]
            child = self._crossover(parent1, parent2)
            self._mutate(child)
            new_brains.append(child)

        self.brains = new_brains
        self.fitness = np.zeros(self.size, dtype=np.float32)
        self.generation += 1

    def _crossover(self, parent1: Brain, parent2: Brain) -> Brain:
        """Uniform crossover: each weight randomly picked from either parent."""
        flat1 = parent1.get_flat_weights()
        flat2 = parent2.get_flat_weights()
        mask = np.random.rand(len(flat1)) < 0.5
        child_weights = np.where(mask, flat1, flat2)
        child = parent1.copy()
        child.set_flat_weights(child_weights)
        return child

    def _mutate(self, brain: Brain):
        """Apply gaussian noise to a random subset of weights."""
        flat = brain.get_flat_weights()
        # Only mutate ~20% of weights per child — preserves parent behavior
        mask = np.random.rand(len(flat)) < 0.2
        noise = np.random.randn(len(flat)).astype(np.float32) * self.mutation_rate
        flat += noise * mask
        brain.set_flat_weights(flat)
