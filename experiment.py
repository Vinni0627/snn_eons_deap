"""
experiment.py
-------------
Main experiment runner.

Runs 4 conditions (2x2 design):
  - Controller type : SNN  vs  ANN
  - Environment     : Static  vs  Dynamic

FIXES:
  1. Saves .npy after EVERY trial (not just at the end)
     so Ctrl+C never loses your data
  2. Stronger elitism + smaller mutations to prevent fitness collapse
"""

import os
import numpy as np
from environment import GridEnvironment
from snn_controller import SNNController
from ann_controller import ANNController
from evolutionary_algorithm import GeneticAlgorithm

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

CONFIG = {
    # Environment
    "grid_size"         : (20, 20),
    "n_sensors"         : 8,
    "n_obstacles"       : 15,
    "dynamic_interval"  : 5,
    "max_steps"         : 33,

    # Network
    "n_hidden"          : 10,
    "n_actions"         : 5,

    # EA - tuned to prevent fitness collapse
    "pop_size"          : 50,
    "n_generations"     : 100,
    "cx_prob"           : 0.5,
    "mut_prob"          : 0.05,   # FIX: was 0.1 - fewer genes mutated per gen
    "mut_sigma"         : 0.1,   # FIX: was 0.5 - smaller mutation steps
    "tournament_k"      : 3,
    "elite_n"           : 8,     # FIX: was 2 - protect more good solutions

    # Experiment
    "n_trials"          : 5,
    "n_eval_episodes"   : 5,
}

N_INPUTS = CONFIG["n_sensors"] + 2


# ------------------------------------------------------------------
# Fitness function
# ------------------------------------------------------------------

def make_fitness_fn(controller_class, dynamic, seed_offset=0):
    def fitness_fn(weights):
        total_reward = 0.0
        for ep in range(CONFIG["n_eval_episodes"]):
            env = GridEnvironment(
                grid_size=CONFIG["grid_size"],
                n_sensors=CONFIG["n_sensors"],
                n_obstacles=CONFIG["n_obstacles"],
                dynamic=dynamic,
                dynamic_interval=CONFIG["dynamic_interval"],
                max_steps=CONFIG["max_steps"],
                seed=ep + seed_offset,
            )
            if controller_class == SNNController:
                ctrl = SNNController(
                    n_inputs=N_INPUTS,
                    n_hidden=CONFIG["n_hidden"],
                    n_outputs=CONFIG["n_actions"],
                    weights=weights,
                )
            else:
                ctrl = ANNController(
                    n_inputs=N_INPUTS,
                    n_hidden=CONFIG["n_hidden"],
                    n_outputs=CONFIG["n_actions"],
                    weights=weights,
                )

            sensors = env.reset()
            ctrl.reset_state()
            ep_reward = 0.0
            done = False
            while not done:
                action, _ = ctrl.forward(sensors)
                sensors, reward, done, _ = env.step(action)
                ep_reward += reward
            total_reward += ep_reward

        return total_reward / CONFIG["n_eval_episodes"]
    return fitness_fn


# ------------------------------------------------------------------
# Run one condition - saves after EVERY trial
# ------------------------------------------------------------------

def run_condition(label, controller_class, dynamic, n_trials, save_dir):
    print(f"\n{'='*60}")
    print(f"Condition: {label}")
    print(f"{'='*60}")

    os.makedirs(save_dir, exist_ok=True)

    genome_size = (
        CONFIG["n_hidden"] * N_INPUTS +
        CONFIG["n_actions"] * CONFIG["n_hidden"]
    )

    # Load existing results if resuming after Ctrl+C
    fits_path    = os.path.join(save_dir, f"{label}_fitnesses.npy")
    weights_path = os.path.join(save_dir, f"{label}_best_weights.npy")

    if os.path.exists(fits_path):
        all_histories     = list(np.load(fits_path))
        all_best_weights  = list(np.load(weights_path))
        start_trial       = len(all_histories)
        print(f"  Resuming from trial {start_trial + 1} (found existing results)")
    else:
        all_histories    = []
        all_best_weights = []
        start_trial      = 0

    for trial in range(start_trial, n_trials):
        print(f"\n  --- Trial {trial+1}/{n_trials} ---")

        fitness_fn = make_fitness_fn(
            controller_class,
            dynamic,
            seed_offset=trial * 1000,
        )
        ea = GeneticAlgorithm(
            fitness_fn=fitness_fn,
            genome_size=genome_size,
            pop_size=CONFIG["pop_size"],
            n_generations=CONFIG["n_generations"],
            cx_prob=CONFIG["cx_prob"],
            mut_prob=CONFIG["mut_prob"],
            mut_sigma=CONFIG["mut_sigma"],
            tournament_k=CONFIG["tournament_k"],
            elite_n=CONFIG["elite_n"],
            seed=trial,
        )
        best_weights, history = ea.run(verbose=True)

        gen_best_fitnesses = [h[1] for h in history]
        all_histories.append(gen_best_fitnesses)
        all_best_weights.append(best_weights)

        # FIX: Save after EVERY trial so Ctrl+C never loses data
        np.save(fits_path,    np.array(all_histories))
        np.save(weights_path, np.array(all_best_weights))
        print(f"  Saved trial {trial+1} -> {fits_path}")

    print(f"\nCondition {label} complete.")
    return np.array(all_histories)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    SAVE_DIR = "results"

    conditions = [
        ("snn_static",  SNNController, False),
        ("snn_dynamic", SNNController, True),
        ("ann_static",  ANNController, False),
        ("ann_dynamic", ANNController, True),
    ]

    for label, ctrl_class, dynamic in conditions:
        run_condition(
            label=label,
            controller_class=ctrl_class,
            dynamic=dynamic,
            n_trials=CONFIG["n_trials"],
            save_dir=SAVE_DIR,
        )

    print("\n\nAll conditions complete!")
    print("Run plot_results.py to generate figures.")