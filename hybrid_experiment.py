"""
hybrid_experiment.py
--------------------
Bilevel neuroevolution experiment:
  Outer loop  — EONS evolves the SNN topology (hidden nodes + edges)
  Inner loop  — DEAP optimises edge weights for each topology candidate

Run:
    python hybrid_experiment.py

Results are saved to results/hybrid_<condition>_*.npy after every EONS epoch.

Statistics collected (for the question:
"How effectively can a GA evolve SNN synaptic weights for autonomous navigation?"):

Per EONS epoch (history list of dicts):
  best_fitness      — best individual in the EONS population this epoch
  mean_fitness      — mean fitness across all evaluated topologies
  std_fitness       — std (population diversity signal)
  min_fitness       — worst non-zero individual
  mean_edges        — mean edge count across population
  mean_hidden_nodes — mean hidden node count across population
  best_n_edges      — edge count of the epoch-best topology
  best_n_hidden     — hidden node count of the epoch-best topology

Per trial:
  deap_history      — DEAP per-gen best fitness for the overall best individual
  behavioral_stats  — goal success rate, avg steps, collisions, final dist
                      on 20 held-out evaluation episodes
"""

import os
import sys
import argparse
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


# Command-line Arguments for Experiment Parameters and Files
parser = argparse.ArgumentParser(prog="hybrid_experiment.py", description="Experiment Parameters")
parser.add_argument("--grid", type=int, default=20, help="N by N grid")
parser.add_argument("--obj", type=int, default=15, help="Number of obstacles")
parser.add_argument("--int", type=int, default=5, help="Dynamic Interval")
parser.add_argument("--trials", type=int, default=15, help="Number of trials")
parser.add_argument("--out_dir", type=str, required=True, help="Results Local Directory Path")
args = parser.parse_args()



CONFIG = {
    # Environment (matches experiment.py)
    "grid_size"        : (args.grid, args.grid),
    "n_sensors"        : 8,
    "n_obstacles"      : args.obj,
    "dynamic_interval" : args.int,
    "max_steps"        : 33,
    "n_actions"        : 5,

    # EONS outer loop
    "n_eons_epochs"    : 30,
    "eons_pop_size"    : 10,
    "starting_nodes"   : 3,
    "starting_edges"   : 6,
    "max_nodes"        : 30,

    # DEAP inner loop (per topology candidate)
    "deap_pop_size"    : 20,
    "deap_n_gen"       : 25,

    # Evaluation
    "n_eval_episodes"  : 5,    # episodes used during GA fitness evaluation
    "n_behav_episodes" : 20,   # held-out episodes for behavioral stats

    # Experiment
    "n_trials"         : args.trials,
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
# Behavioral evaluation (held-out episodes, post-training)
# ---------------------------------------------------------------------------

def behavioral_eval(topology_json, weights, dynamic, seed_offset=99999):
    """
    Run held-out episodes and return navigation quality metrics.

    Returns a dict with:
      goal_success_rate   — fraction of episodes where the goal was reached
      avg_reward          — mean cumulative reward per episode
      avg_steps           — mean steps per episode
      avg_steps_to_goal   — mean steps on goal-reaching episodes (None if 0)
      avg_final_dist      — mean Manhattan distance to goal at episode end
      avg_collisions      — mean collisions per episode
      n_episodes          — number of evaluation episodes
    """
    n_ep = CONFIG["n_behav_episodes"]
    results = []

    for ep in range(n_ep):
        env = GridEnvironment(
            grid_size        = CONFIG["grid_size"],
            n_sensors        = CONFIG["n_sensors"],
            n_obstacles      = CONFIG["n_obstacles"],
            dynamic          = dynamic,
            dynamic_interval = CONFIG["dynamic_interval"],
            max_steps        = CONFIG["max_steps"],
            seed             = seed_offset + ep,
        )
        ctrl = TopologySNN(topology_json, weights)
        sensors = env.reset()
        ctrl.reset_state()

        ep_reward = 0.0
        n_collisions = 0
        done = False
        info = {}

        while not done:
            action, _ = ctrl.forward(sensors)
            sensors, reward, done, info = env.step(action)
            ep_reward += reward
            if info.get("collision", False):
                n_collisions += 1

        results.append({
            "reward"      : ep_reward,
            "goal_reached": info.get("goal_reached", False),
            "steps"       : info.get("step", CONFIG["max_steps"]),
            "final_dist"  : info.get("dist_to_goal", float("nan")),
            "collisions"  : n_collisions,
        })

    goal_eps = [r for r in results if r["goal_reached"]]
    return {
        "goal_success_rate": len(goal_eps) / n_ep,
        "avg_reward"       : float(np.mean([r["reward"]     for r in results])),
        "avg_steps"        : float(np.mean([r["steps"]      for r in results])),
        "avg_steps_to_goal": float(np.mean([r["steps"]      for r in goal_eps]))
                             if goal_eps else None,
        "avg_final_dist"   : float(np.mean([r["final_dist"] for r in results])),
        "avg_collisions"   : float(np.mean([r["collisions"] for r in results])),
        "n_episodes"       : n_ep,
    }


# ---------------------------------------------------------------------------
# Topology complexity helpers
# ---------------------------------------------------------------------------

def _topo_hidden_nodes(topo):
    all_ids = {n["id"] for n in topo.get("Nodes", [])}
    fixed   = set(topo.get("Inputs", [])) | set(topo.get("Outputs", []))
    return len(all_ids - fixed)


def _topo_edges(topo):
    return len(topo.get("Edges", []))


# ---------------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------------

def run_condition(label, dynamic, n_trials, save_dir):
    print(f"\n{'='*60}")
    print(f"Condition: {label}")
    print(f"{'='*60}")
    os.makedirs(save_dir, exist_ok=True)

    history_path  = os.path.join(save_dir, f"hybrid_{label}_history.npy")
    weights_path  = os.path.join(save_dir, f"hybrid_{label}_best_weights.npy")
    topo_path     = os.path.join(save_dir, f"hybrid_{label}_best_topos.npy")
    deap_path     = os.path.join(save_dir, f"hybrid_{label}_deap_history.npy")
    behav_path    = os.path.join(save_dir, f"hybrid_{label}_behavioral.npy")

    # Resume from checkpoint if available
    if os.path.exists(history_path):
        all_histories    = list(np.load(history_path,  allow_pickle=True))
        all_best_weights = list(np.load(weights_path,  allow_pickle=True))
        all_best_topos   = list(np.load(topo_path,     allow_pickle=True))
        all_deap_history = list(np.load(deap_path,     allow_pickle=True))
        all_behavioral   = list(np.load(behav_path,    allow_pickle=True))
        start_trial      = len(all_histories)
        print(f"  Resuming from trial {start_trial + 1}")
    else:
        all_histories    = []
        all_best_weights = []
        all_best_topos   = []
        all_deap_history = []
        all_behavioral   = []
        start_trial      = 0

    for trial in range(start_trial, n_trials):
        print(f"\n  --- Trial {trial+1}/{n_trials} ---")
        history, best_weights, best_topo, deap_hist, behav = run_trial(
            dynamic     = dynamic,
            seed_offset = trial * 1000,
            trial       = trial,
        )
        all_histories.append(history)
        all_best_weights.append(best_weights)
        all_best_topos.append(best_topo)
        all_deap_history.append(deap_hist)
        all_behavioral.append(behav)

        np.save(history_path,  np.array(all_histories,    dtype=object))
        np.save(weights_path,  np.array(all_best_weights, dtype=object))
        np.save(topo_path,     np.array(all_best_topos,   dtype=object))
        np.save(deap_path,     np.array(all_deap_history, dtype=object))
        np.save(behav_path,    np.array(all_behavioral,   dtype=object))
        print(f"  Saved trial {trial+1} → {save_dir}/hybrid_{label}_*.npy")

    print(f"\nCondition {label} complete.")


def run_trial(dynamic, seed_offset, trial):
    """Run one trial: EONS outer loop + DEAP inner loop."""
    template   = make_template()
    eons_obj   = eons.EONS(EONS_PARAMS)
    eons_obj.set_template_network(template)
    population = eons_obj.generate_population(EONS_PARAMS)

    history          = []        # list of per-epoch stat dicts
    overall_best     = -np.inf
    best_weights     = None
    best_topo        = None
    best_deap_hist   = None      # DEAP gen history for the overall-best individual

    for epoch in range(CONFIG["n_eons_epochs"]):
        fitnesses   = []
        epoch_best  = -np.inf
        epoch_bw    = None
        epoch_btopo = None
        epoch_dh    = None

        pop_json   = population.as_json(True)
        topologies = [entry["network"] for entry in pop_json["network_info"]]

        # Topology complexity across the population this epoch
        pop_edges   = []
        pop_hidden  = []
        has_edges   = []   # True for topologies that were actually evaluated

        for i, topo in enumerate(topologies):
            n_edges = _topo_edges(topo)
            n_hid   = _topo_hidden_nodes(topo)
            pop_edges.append(n_edges)
            pop_hidden.append(n_hid)

            if n_edges == 0:
                fitnesses.append(0.0)
                has_edges.append(False)
                continue

            has_edges.append(True)
            fit_fn = make_fitness_fn(topo, dynamic, seed_offset=seed_offset + i * 100)
            bw, bf, dh = optimize_weights(
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
                epoch_dh    = dh

        # Fitness and topology stats — only over topologies that had edges
        valid_fits   = [f for f, v in zip(fitnesses,  has_edges) if v]
        valid_edges  = [e for e, v in zip(pop_edges,  has_edges) if v]
        valid_hidden = [h for h, v in zip(pop_hidden, has_edges) if v]

        ep_mean    = float(np.mean(valid_fits))   if valid_fits   else 0.0
        ep_std     = float(np.std(valid_fits))    if valid_fits   else 0.0
        ep_min     = float(np.min(valid_fits))    if valid_fits   else 0.0
        mean_edges  = float(np.mean(valid_edges))  if valid_edges  else 0.0
        mean_hidden = float(np.mean(valid_hidden)) if valid_hidden else 0.0

        best_n_edges  = _topo_edges(epoch_btopo)        if epoch_btopo else 0
        best_n_hidden = _topo_hidden_nodes(epoch_btopo) if epoch_btopo else 0

        history.append({
            "epoch"            : epoch,
            "best_fitness"     : float(epoch_best),
            "mean_fitness"     : ep_mean,
            "std_fitness"      : ep_std,
            "min_fitness"      : ep_min,
            "mean_edges"       : mean_edges,
            "mean_hidden_nodes": mean_hidden,
            "best_n_edges"     : best_n_edges,
            "best_n_hidden"    : best_n_hidden,
        })

        if epoch_best > overall_best:
            overall_best   = epoch_best
            best_weights   = epoch_bw
            best_topo      = epoch_btopo
            best_deap_hist = epoch_dh

        print(f"  Epoch {epoch+1:3d}/{CONFIG['n_eons_epochs']} | "
              f"Best: {overall_best:8.3f} | Epoch Best: {epoch_best:8.3f} | "
              f"Pop Mean: {ep_mean:7.3f} ± {ep_std:.3f} | "
              f"Edges: {mean_edges:.1f}  Hidden: {mean_hidden:.1f}")

        population = eons_obj.do_epoch(population, fitnesses, EONS_PARAMS)

    # --- Behavioral evaluation on the best agent found ---
    print("  Running behavioral evaluation...")
    behav = behavioral_eval(
        topology_json = best_topo,
        weights       = best_weights,
        dynamic       = dynamic,
        seed_offset   = 99999 + trial * 1000,
    ) if best_topo is not None else {}

    print(f"  Behavioral: success={behav.get('goal_success_rate', 0):.1%} | "
          f"avg_steps={behav.get('avg_steps', -1):.1f} | "
          f"avg_collisions={behav.get('avg_collisions', -1):.2f} | "
          f"avg_final_dist={behav.get('avg_final_dist', -1):.2f}")

    return history, best_weights, best_topo, best_deap_hist, behav





def run_cross_eval(save_dir, n_trials):
    print(f"\n{'='*60}")
    print("Running Cross-Environment Fitness Evaluation")
    print(f"{'='*60}")

    static_topos = np.load(os.path.join(save_dir, "hybrid_snn_static_best_topos.npy"), allow_pickle=True)
    static_weights = np.load(os.path.join(save_dir, "hybrid_snn_static_best_weights.npy"), allow_pickle=True)
    
    dynamic_topos = np.load(os.path.join(save_dir, "hybrid_snn_dynamic_best_topos.npy"), allow_pickle=True)
    dynamic_weights = np.load(os.path.join(save_dir, "hybrid_snn_dynamic_best_weights.npy"), allow_pickle=True) 

    cross_results = {"static": [], "dynamic": []}
    for trial in range(n_trials):
        # Run static on dynamic environment
        static = behavioral_eval(static_topos[trial], static_weights[trial], dynamic=True, seed_offset=88888+trial*1000)
        cross_results["static"].append(static)

        # Run dynamic on static
        dynamic = behavioral_eval(dynamic_topos[trial], dynamic_weights[trial], dynamic=False, seed_offset=88888+trial*1000)
        cross_results["dynamic"].append(dynamic)

        print(f"Trial {trial+1}: S->D Success: {static['goal_success_rate']:.1%} | D->S Success: {dynamic['goal_success_rate']:.1%}")

    np.save(os.path.join(save_dir, "hybrid_cross_eval_results.npy"), cross_results)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAVE_DIR = args.out_dir

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

    run_cross_eval(save_dir=SAVE_DIR, n_trials=CONFIG["n_trials"])

    print("\n\nAll conditions complete!")
