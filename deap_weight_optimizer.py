"""
deap_weight_optimizer.py
------------------------
DEAP-based inner-loop weight optimizer for a fixed network topology.

Usage:
    best_weights, best_fitness = optimize_weights(
        fitness_fn=fn,   # callable: list[float] -> float
        n_weights=42,    # number of edges in the topology
        pop_size=20,
        n_gen=25,
        seed=None,
    )

The DEAP creator classes are registered once at module level to avoid the
"class already registered" error when this function is called many times.
"""

import random
from deap import base, creator, tools, algorithms

# Register once at import time; guard against re-imports in interactive sessions
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def optimize_weights(fitness_fn, n_weights, pop_size=20, n_gen=25, seed=None):
    """
    Run DEAP eaSimple to find the best weight vector for a fixed topology.

    Parameters
    ----------
    fitness_fn : callable
        Takes a list of floats (weights) and returns a float (higher = better).
    n_weights : int
        Number of weights (= number of edges in the topology).
    pop_size : int
        DEAP population size.
    n_gen : int
        Number of DEAP generations.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    best_weights : list[float]
    best_fitness : float
    """
    if n_weights == 0:
        return [], 0.0

    if seed is not None:
        random.seed(seed)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -2.0, 2.0)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_float, n=n_weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate",   tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", lambda ind: (fitness_fn(list(ind)),))

    pop = toolbox.population(n=pop_size)

    # Evaluate initial population before handing to eaSimple
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    algorithms.eaSimple(pop, toolbox,
                        cxpb=0.5, mutpb=0.3, ngen=n_gen,
                        verbose=False)

    best = tools.selBest(pop, 1)[0]
    return list(best), best.fitness.values[0]
