"""
plot_hybrid_results.py
----------------------
Visualise results from hybrid_experiment.py.

Figures produced (answering: "How effectively can a GA evolve SNN synaptic
weights to achieve autonomous navigation in a 2D environment?"):

  Fig 1  — GA Convergence: best & mean±std fitness across EONS epochs
  Fig 2  — Population Diversity: mean±std fitness per epoch (static vs dynamic)
  Fig 3  — Topology Complexity Evolution: mean edges & hidden nodes per epoch
  Fig 4  — DEAP Inner-Loop Convergence: GA weight optimisation curve
  Fig 5  — Behavioral Performance: goal success rate, steps, collisions, dist

Run after hybrid_experiment.py has finished:
    python plot_hybrid_results.py
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser(prog="plot_hybrid_results.py", description="Plot Hybrid Results")
parser.add_argument("--res_dir", type=str, default="results", help="path to results directory")
parser.add_argument("--fig_dir", type=str, default="figures", help="path to figures directory")
args = parser.parse_args()


RESULTS_DIR = args.res_dir
FIGURES_DIR = args.fig_dir
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    "snn_static" : "#5480E0",
    "snn_dynamic": "#B03A2E",
}
DISPLAY = {
    "snn_static" : "Static Environment",
    "snn_dynamic": "Dynamic Environment",
}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(label, suffix):
    path = os.path.join(RESULTS_DIR, f"hybrid_{label}_{suffix}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def extract_epoch_stat(histories, key):
    """
    histories : array of shape (n_trials,) where each element is a list of dicts.
    Returns array of shape (n_trials, n_epochs).
    """
    return np.array([[ep[key] for ep in trial] for trial in histories],
                    dtype=float)


data = {}
for label in ["snn_static", "snn_dynamic"]:
    try:
        data[label] = {
            "history" : load(label, "history"),
            "deap"    : load(label, "deap_history"),
            "behav"   : load(label, "behavioral"),
        }
        print(f"Loaded {label}: {len(data[label]['history'])} trial(s)")
    except FileNotFoundError as e:
        print(f"Warning — missing file: {e}")

if not data:
    print("No hybrid results found. Run hybrid_experiment.py first.")
    exit()


# ---------------------------------------------------------------------------
# Fig 1 — GA Convergence (best fitness per EONS epoch, mean ± std over trials)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.suptitle(
    "GA Convergence: Fitness Over EONS Epochs\n"
    "(How well does the GA evolve SNN weights for navigation?)",
    fontsize=13,
)

for ax, (label, d) in zip(axes, data.items()):
    histories = d["history"]
    best = extract_epoch_stat(histories, "best_fitness")
    mean = extract_epoch_stat(histories, "mean_fitness")
    std  = extract_epoch_stat(histories, "std_fitness")

    epochs = np.arange(best.shape[1]) + 1
    color  = COLORS[label]

    # Best fitness
    best_mean = best.mean(axis=0)
    best_std  = best.std(axis=0)
    ax.plot(epochs, best_mean, color=color, linewidth=2, label="Best (mean over trials)")
    ax.fill_between(epochs, best_mean - best_std, best_mean + best_std,
                    alpha=0.2, color=color)

    # Population mean fitness
    pop_mean = mean.mean(axis=0)
    pop_std  = std.mean(axis=0)
    ax.plot(epochs, pop_mean, color=color, linewidth=1.5, linestyle="--",
            label="Pop. mean (mean over trials)")
    ax.fill_between(epochs, pop_mean - pop_std, pop_mean + pop_std,
                    alpha=0.1, color=color)

    ax.set_title(DISPLAY[label], fontsize=11)
    ax.set_xlabel("EONS Epoch", fontsize=11)
    ax.set_ylabel("Fitness (avg reward)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "hybrid_fig1_ga_convergence.pdf"), dpi=300)
fig.savefig(os.path.join(FIGURES_DIR, "hybrid_fig1_ga_convergence.png"), dpi=300)
print("Saved hybrid_fig1_ga_convergence")


# ---------------------------------------------------------------------------
# Fig 2 — Static vs Dynamic: best fitness learning curves on same axes
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title(
    "Learning Curves: Static vs Dynamic Environment\n"
    "(Best fitness per EONS epoch, mean ± std across trials)",
    fontsize=13,
)

for label, d in data.items():
    best   = extract_epoch_stat(d["history"], "best_fitness")
    m, s   = best.mean(axis=0), best.std(axis=0)
    epochs = np.arange(len(m)) + 1
    color  = COLORS[label]
    ax.plot(epochs, m, color=color, linewidth=2, label=DISPLAY[label])
    ax.fill_between(epochs, m - s, m + s, alpha=0.2, color=color)

ax.set_xlabel("EONS Epoch", fontsize=12)
ax.set_ylabel("Best Fitness (avg reward)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "hybrid_fig2_learning_curves.pdf"), dpi=300)
fig.savefig(os.path.join(FIGURES_DIR, "hybrid_fig2_learning_curves.png"), dpi=300)
print("Saved hybrid_fig2_learning_curves")


# ---------------------------------------------------------------------------
# Fig 3 — Topology Complexity Evolution
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Topology Complexity over EONS Epochs", fontsize=13)

for ax_e, ax_h in [axes]:
    break

for label, d in data.items():
    histories = d["history"]
    edges  = extract_epoch_stat(histories, "mean_edges")
    hidden = extract_epoch_stat(histories, "mean_hidden_nodes")
    epochs = np.arange(edges.shape[1]) + 1
    color  = COLORS[label]

    e_mean, e_std = edges.mean(axis=0),  edges.std(axis=0)
    h_mean, h_std = hidden.mean(axis=0), hidden.std(axis=0)

    axes[0].plot(epochs, e_mean, color=color, linewidth=2, label=DISPLAY[label])
    axes[0].fill_between(epochs, e_mean - e_std, e_mean + e_std, alpha=0.2, color=color)

    axes[1].plot(epochs, h_mean, color=color, linewidth=2, label=DISPLAY[label])
    axes[1].fill_between(epochs, h_mean - h_std, h_mean + h_std, alpha=0.2, color=color)

axes[0].set_title("Mean Edges per Population", fontsize=11)
axes[0].set_xlabel("EONS Epoch", fontsize=11)
axes[0].set_ylabel("Mean Edge Count", fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Mean Hidden Nodes per Population", fontsize=11)
axes[1].set_xlabel("EONS Epoch", fontsize=11)
axes[1].set_ylabel("Mean Hidden Node Count", fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "hybrid_fig3_topology_evolution.pdf"), dpi=300)
fig.savefig(os.path.join(FIGURES_DIR, "hybrid_fig3_topology_evolution.png"), dpi=300)
print("Saved hybrid_fig3_topology_evolution")


# ---------------------------------------------------------------------------
# Fig 4 — DEAP Inner-Loop Convergence (weight optimisation curve)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 5), sharey=True)
if len(data) == 1:
    axes = [axes]
fig.suptitle(
    "DEAP Inner-Loop Convergence\n"
    "(GA weight optimisation for the best SNN topology per trial)",
    fontsize=13,
)

for ax, (label, d) in zip(axes, data.items()):
    color = COLORS[label]
    deap_trials = d["deap"]  # object array, each element is a list of floats

    all_curves = []
    for trial_dh in deap_trials:
        if trial_dh is not None and len(trial_dh) > 0:
            all_curves.append(np.array(trial_dh, dtype=float))
            ax.plot(all_curves[-1], color=color, linewidth=1, alpha=0.4)

    if all_curves:
        min_len = min(len(c) for c in all_curves)
        mat = np.array([c[:min_len] for c in all_curves])
        ax.plot(mat.mean(axis=0), color=color, linewidth=2.5,
                label="Mean across trials")

    ax.set_title(DISPLAY[label], fontsize=11)
    ax.set_xlabel("DEAP Generation", fontsize=11)
    ax.set_ylabel("Best Fitness in DEAP Population", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "hybrid_fig4_deap_convergence.pdf"), dpi=300)
fig.savefig(os.path.join(FIGURES_DIR, "hybrid_fig4_deap_convergence.png"), dpi=300)
print("Saved hybrid_fig4_deap_convergence")


# ---------------------------------------------------------------------------
# Fig 5 — Behavioral Performance metrics
# ---------------------------------------------------------------------------

metrics = [
    ("goal_success_rate", "Goal Success Rate",        "Fraction of Episodes"),
    ("avg_steps",         "Avg Steps per Episode",     "Steps"),
    ("avg_collisions",    "Avg Collisions per Episode","Count"),
    ("avg_final_dist",    "Avg Final Distance to Goal","Manhattan Distance"),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle(
    "Behavioral Navigation Performance (20 held-out episodes)\n"
    "Best evolved SNN agent per trial",
    fontsize=13,
)

for ax, (key, title, ylabel) in zip(axes, metrics):
    vals_per_label = {}
    for label, d in data.items():
        behav = d["behav"]
        vals  = []
        for b in behav:
            v = b.item().get(key) if hasattr(b, "item") else b.get(key)
            if v is not None:
                vals.append(float(v))
        vals_per_label[label] = vals

    positions = list(range(len(vals_per_label)))
    bp = ax.boxplot(
        [vals_per_label[l] for l in data.keys()],
        positions=positions,
        patch_artist=True,
        widths=0.4,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 6},
    )
    for patch, label in zip(bp["boxes"], data.keys()):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.7)

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY[l] for l in data.keys()], fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "hybrid_fig5_behavioral.pdf"), dpi=300)
fig.savefig(os.path.join(FIGURES_DIR, "hybrid_fig5_behavioral.png"), dpi=300)
print("Saved hybrid_fig5_behavioral")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("BEHAVIORAL SUMMARY (mean ± std across trials)")
print("=" * 65)
fmt = "{:<22} {:>12} {:>12} {:>12} {:>12}"
print(fmt.format("Condition", "Success %", "Avg Steps", "Avg Collisions", "Avg Dist"))
print("-" * 65)

for label, d in data.items():
    behav = d["behav"]

    def stat(key):
        vals = []
        for b in behav:
            v = b.item().get(key) if hasattr(b, "item") else b.get(key)
            if v is not None:
                vals.append(float(v))
        return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))

    sr_m, sr_s = stat("goal_success_rate")
    st_m, st_s = stat("avg_steps")
    co_m, co_s = stat("avg_collisions")
    di_m, di_s = stat("avg_final_dist")
    print(fmt.format(
        DISPLAY[label],
        f"{sr_m*100:.1f}±{sr_s*100:.1f}%",
        f"{st_m:.1f}±{st_s:.1f}",
        f"{co_m:.2f}±{co_s:.2f}",
        f"{di_m:.2f}±{di_s:.2f}",
    ))

print("\nFigures saved to:", FIGURES_DIR)
