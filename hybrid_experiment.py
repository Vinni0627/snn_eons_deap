"""
hybrid_experiment.py
--------------------
Bilevel neuroevolution experiment:
  Outer loop  — EONS evolves the SNN topology (hidden nodes + edges)
  Inner loop  — DEAP optimises edge weights for each topology candidate

Run:
    python hybrid_experiment.py

Results are saved to results/hybrid_<condition>_*.npy after every EONS epoch
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Framework paths
# ---------------------------------------------------------------------------
FRAMEWORK_DIR = os.path.join(os.path.dirname(__file__), "framework")
sys.path.insert(0, os.path.join(FRAMEWORK_DIR, "build"))
sys.path.insert(0, os.path.join(FRAMEWORK_DIR, "eons", "build"))

import neuro  
import eons  

from environment import GridEnvironment
from topology_snn import TopologySNN
from deap_weight_optimizer import optimize_weights

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Environment (matches experiment.py)
    "grid_size"        : (20, 20),
    "n_sensors"        : 8,
    "n_obstacles"      : 15,
    "dynamic_interval" : 5,
    "max_steps"        : 33,
    "n_actions"        : 5,

    # EONS outer loop
    "n_eons_epochs"    : 30,
    "eons_pop_size"    : 10,
    "starting_nodes"   : 3,   # rough number of hidden nodes at initialisation
    "starting_edges"   : 6,   # rough number of edges at initialisation
    "max_nodes"        : 30,

    # DEAP inner loop (per topology candidate)
    "deap_pop_size"    : 20,
    "deap_n_gen"       : 25,

    # Evaluation
    "n_eval_episodes"  : 5,

    # Experiment
    "n_trials"         : 15,
}

N_INPUTS  = CONFIG["n_sensors"] + 2   # 10 total (8 directional + 2 goal)
N_OUTPUTS = CONFIG["n_actions"]        # 5


# ---------------------------------------------------------------------------
# Build the EONS template network
# ---------------------------------------------------------------------------

def make_template():
    """10 input nodes (0-9) + 5 output nodes (10-14), no hidden, no edges."""
    net = neuro.Network()
    for i in range(N_INPUTS):
        net.add_node(i)
        net.add_input(i)
    for i in range(N_OUTPUTS):
        net.add_node(N_INPUTS + i)
        net.add_output(N_INPUTS + i)
    return net


# ---------------------------------------------------------------------------
# EONS parameters
# ---------------------------------------------------------------------------

EONS_PARAMS = {
    "population_size"           : CONFIG["eons_pop_size"],
    "max_nodes"                 : CONFIG["max_nodes"],
    "multi_edges"               : 0,
    "starting_nodes"            : CONFIG["starting_nodes"],
    "starting_edges"            : CONFIG["starting_edges"],

    # Topology mutation rates (relative weights — EONS normalises them)
    "mutation_rate"             : 0.9,
    "add_node_rate"             : 0.3,
    "add_edge_rate"             : 0.6,
    "delete_node_rate"          : 0.2,
    "delete_edge_rate"          : 0.4,

    # Disable parameter mutations — weights are handled by DEAP, not EONS
    "node_params_rate"          : 0.0,
    "edge_params_rate"          : 0.0,
    "net_params_rate"           : 0.0,
    "node_mutations"            : {},
    "edge_mutations"            : {},
    "net_mutations"             : {},

    "num_mutations"             : 1,
    "num_best"                  : 1,
    "random_factor"             : 0.05,

    "selection_type"            : "tournament",
    "tournament_best_net_factor": 0.8,
    "tournament_size_factor"    : 0.2,
}


# ---------------------------------------------------------------------------
# Fitness function (DEAP inner loop wraps this)
# ---------------------------------------------------------------------------

def make_fitness_fn(topology_json, dynamic, seed_offset=0):
    """Return a function: weights -> mean episode reward for this topology."""
    def fitness_fn(weights):
        total_reward = 0.0
        for ep in range(CONFIG["n_eval_episodes"]):
            env = GridEnvironment(
                grid_size        = CONFIG["grid_size"],
                n_sensors        = CONFIG["n_sensors"],
                n_obstacles      = CONFIG["n_obstacles"],
                dynamic          = dynamic,
                dynamic_interval = CONFIG["dynamic_interval"],
                max_steps        = CONFIG["max_steps"],
                seed             = ep + seed_offset,
            )
            ctrl = TopologySNN(topology_json, weights)
            sensors  = env.reset()
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


# ---------------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------------

def run_condition(label, dynamic, n_trials, save_dir):
    print(f"\n{'='*60}")
    print(f"Condition: {label}")
    print(f"{'='*60}")
    os.makedirs(save_dir, exist_ok=True)

    fits_path    = os.path.join(save_dir, f"hybrid_{label}_fitnesses.npy")
    weights_path = os.path.join(save_dir, f"hybrid_{label}_best_weights.npy")
    topo_path    = os.path.join(save_dir, f"hybrid_{label}_best_topos.npy")

    # Resume from checkpoint if available
    if os.path.exists(fits_path):
        all_histories    = list(np.load(fits_path, allow_pickle=True))
        all_best_weights = list(np.load(weights_path, allow_pickle=True))
        all_best_topos   = list(np.load(topo_path,    allow_pickle=True))
        start_trial      = len(all_histories)
        print(f"  Resuming from trial {start_trial + 1}")
    else:
        all_histories    = []
        all_best_weights = []
        all_best_topos   = []
        start_trial      = 0

    for trial in range(start_trial, n_trials):
        print(f"\n  --- Trial {trial+1}/{n_trials} ---")
        history, best_weights, best_topo = run_trial(
            dynamic=dynamic,
            seed_offset=trial * 1000,
            trial=trial,
        )
        all_histories.append(history)
        all_best_weights.append(best_weights)
        all_best_topos.append(best_topo)

        np.save(fits_path,    np.array(all_histories,    dtype=object))
        np.save(weights_path, np.array(all_best_weights, dtype=object))
        np.save(topo_path,    np.array(all_best_topos,   dtype=object))
        print(f"  Saved trial {trial+1} → {fits_path}")

    print(f"\nCondition {label} complete.")


def run_trial(dynamic, seed_offset, trial):
    """Run one trial: EONS outer loop + DEAP inner loop."""
    template   = make_template()
    eons_obj   = eons.EONS(EONS_PARAMS)
    eons_obj.set_template_network(template)
    population = eons_obj.generate_population(EONS_PARAMS)

    history       = []   # best fitness per EONS epoch
    overall_best  = -np.inf
    best_weights  = None
    best_topo     = None

    for epoch in range(CONFIG["n_eons_epochs"]):
        fitnesses    = []
        epoch_best   = -np.inf
        epoch_bw     = None
        epoch_btopo  = None

        # Serialise the whole population to pure Python dicts before the loop.
        # get_network() returns a raw Network* that pybind11 takes ownership of
        # and deletes when the variable goes out of scope, corrupting the
        # Population before do_epoch runs. as_json() gives a safe copy.
        pop_json = population.as_json(True)
        topologies = [entry["network"] for entry in pop_json["network_info"]]

        for i, topo in enumerate(topologies):
            n_edges = len(topo["Edges"])

            if n_edges == 0:
                fitnesses.append(0.0)
                continue

            fit_fn = make_fitness_fn(topo, dynamic, seed_offset=seed_offset + i * 100)
            bw, bf = optimize_weights(
                fitness_fn = fit_fn,
                n_weights  = n_edges,
                pop_size   = CONFIG["deap_pop_size"],
                n_gen      = CONFIG["deap_n_gen"],
                seed       = trial * 10000 + epoch * 100 + i,
            )

            fitnesses.append(bf)
            if bf > epoch_best:
                epoch_best  = bf
                epoch_bw    = bw
                epoch_btopo = topo

        history.append(epoch_best)

        if epoch_best > overall_best:
            overall_best = epoch_best
            best_weights = epoch_bw
            best_topo    = epoch_btopo

        print(f"  Epoch {epoch+1:3d}/{CONFIG['n_eons_epochs']} | "
              f"Best: {overall_best:8.3f} | Epoch Best: {epoch_best:8.3f}")

        population = eons_obj.do_epoch(population, fitnesses, EONS_PARAMS)

    return history, best_weights, best_topo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAVE_DIR = "results_extended"

    conditions = [
        ("snn_static",  False),
        ("snn_dynamic", True),
    ]

    for label, dynamic in conditions:
        run_condition(
            label    = label,
            dynamic  = dynamic,
            n_trials = CONFIG["n_trials"],
            save_dir = SAVE_DIR,
        )

    print("\n\nAll conditions complete!")
