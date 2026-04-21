"""
evolutionary_algorithm.py - FIXED VERSION 3
--------------------------------------------
Added adaptive mutation:
  - If best fitness does not improve for `stagnation_limit` generations,
    temporarily increase mutation sigma to escape local optima
  - Once improvement resumes, sigma returns to normal
  - This is called a restart-free adaptive EA strategy
"""

import numpy as np
from typing import Callable


class GeneticAlgorithm:
    def __init__(
        self,
        fitness_fn: Callable,
        genome_size: int,
        pop_size: int = 50,
        n_generations: int = 100,
        cx_prob: float = 0.5,
        mut_prob: float = 0.05,
        mut_sigma: float = 0.1,
        tournament_k: int = 3,
        elite_n: int = 8,
        weight_clip: float = 3.0,
        stagnation_limit: int = 15,   # gens without improvement before boost
        mut_sigma_boost: float = 0.4, # boosted sigma when stagnating
        seed: int = None,
    ):
        self.fitness_fn       = fitness_fn
        self.genome_size      = genome_size
        self.pop_size         = pop_size
        self.n_generations    = n_generations
        self.cx_prob          = cx_prob
        self.mut_prob         = mut_prob
        self.mut_sigma_base   = mut_sigma
        self.mut_sigma        = mut_sigma        # current sigma (adaptive)
        self.tournament_k     = tournament_k
        self.elite_n          = elite_n
        self.weight_clip      = weight_clip
        self.stagnation_limit = stagnation_limit
        self.mut_sigma_boost  = mut_sigma_boost
        self.rng              = np.random.default_rng(seed)

        self.best_individual  = None
        self.best_fitness     = -np.inf
        self.history          = []
        self.stagnation_count = 0

    def run(self, verbose=True):
        population = [
            self.rng.uniform(-2.0, 2.0, self.genome_size)
            for _ in range(self.pop_size)
        ]

        for gen in range(self.n_generations):
            fitnesses = [self.fitness_fn(ind) for ind in population]

            best_idx = int(np.argmax(fitnesses))
            gen_best = fitnesses[best_idx]
            gen_mean = float(np.mean(fitnesses))

            # Track improvement
            if gen_best > self.best_fitness + 0.001:
                self.best_fitness    = gen_best
                self.best_individual = population[best_idx].copy()
                self.stagnation_count = 0
                self.mut_sigma = self.mut_sigma_base  # reset to normal
            else:
                self.stagnation_count += 1

            # Adaptive mutation: boost if stagnating
            if self.stagnation_count >= self.stagnation_limit:
                self.mut_sigma = self.mut_sigma_boost
                stag_flag = " [BOOSTING MUTATION]"
            else:
                stag_flag = ""

            self.history.append((gen, self.best_fitness, gen_mean))

            if verbose and gen % 10 == 0:
                print(f"  Gen {gen:4d} | Best: {self.best_fitness:8.3f} | "
                      f"Gen Best: {gen_best:8.3f} | "
                      f"Mean: {gen_mean:8.3f}{stag_flag}")

            # Build next generation
            sorted_idx = np.argsort(fitnesses)[::-1]
            elites = [population[i].copy() for i in sorted_idx[:self.elite_n]]

            # Always keep all-time best
            if self.best_individual is not None:
                elites[0] = self.best_individual.copy()

            offspring = []
            while len(offspring) < self.pop_size - self.elite_n:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                offspring.append(c1)
                if len(offspring) < self.pop_size - self.elite_n:
                    offspring.append(c2)

            population = elites + offspring

        if verbose:
            print(f"\nEvolution complete. Best fitness: {self.best_fitness:.3f}")

        return self.best_individual, self.history

    def _tournament_select(self, population, fitnesses):
        candidates = self.rng.choice(len(population), self.tournament_k, replace=False)
        best = candidates[np.argmax([fitnesses[i] for i in candidates])]
        return population[best].copy()

    def _crossover(self, p1, p2):
        c1, c2 = p1.copy(), p2.copy()
        if self.rng.random() < self.cx_prob:
            mask = self.rng.random(self.genome_size) < 0.5
            c1[mask] = p2[mask]
            c2[mask] = p1[mask]
        return c1, c2

    def _mutate(self, individual):
        mask = self.rng.random(self.genome_size) < self.mut_prob
        individual[mask] += self.rng.normal(0, self.mut_sigma, mask.sum())
        individual = np.clip(individual, -self.weight_clip, self.weight_clip)
        return individual