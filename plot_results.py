"""
plot_results.py
---------------
Load saved results and generate publication-quality figures + statistics.

Produces:
  1. Learning curves (fitness over generations, mean ± std across trials)
  2. Box plots of final fitness per condition
  3. Performance drop table (static vs dynamic for each controller)
  4. Mann-Whitney U test results (SNN vs ANN in each environment type)

Run after experiment.py has finished:
    python plot_results.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load results
# ------------------------------------------------------------------

def load(label):
    path = os.path.join(RESULTS_DIR, f"{label}_fitnesses.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results found at {path}. Run experiment.py first.")
    return np.load(path)   # shape (n_trials, n_generations)


labels = ["snn_static", "snn_dynamic", "ann_static", "ann_dynamic"]
colors = {
    "snn_static" : "#E07B54",   # orange-red
    "snn_dynamic": "#B03A2E",   # dark red
    "ann_static" : "#5DADE2",   # blue
    "ann_dynamic": "#1A5276",   # dark blue
}
display_names = {
    "snn_static" : "SNN – Static",
    "snn_dynamic": "SNN – Dynamic",
    "ann_static" : "ANN – Static",
    "ann_dynamic": "ANN – Dynamic",
}

data = {}
for label in labels:
    try:
        data[label] = load(label)
    except FileNotFoundError as e:
        print(f"Warning: {e}")

if not data:
    print("No results found. Please run experiment.py first.")
    exit()

# ------------------------------------------------------------------
# Figure 1: Learning Curves (mean ± std)
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

for label, arr in data.items():
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    gens = np.arange(len(mean))
    ax.plot(gens, mean, label=display_names[label],
            color=colors[label], linewidth=2)
    ax.fill_between(gens, mean - std, mean + std,
                    alpha=0.2, color=colors[label])

ax.set_xlabel("Generation", fontsize=12)
ax.set_ylabel("Best Fitness (avg reward)", fontsize=12)
ax.set_title("Learning Curves: EA-evolved SNN vs ANN Controllers", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig1_learning_curves.pdf"), dpi=300)
fig.savefig(os.path.join(FIGURES_DIR, "fig1_learning_curves.png"), dpi=300)
print("Saved fig1_learning_curves")

# ------------------------------------------------------------------
# Figure 2: Box Plots of Final Fitness
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

final_fitnesses = {label: arr[:, -1] for label, arr in data.items()}
positions = [1, 2, 3, 4]
bp = ax.boxplot(
    [final_fitnesses[l] for l in labels],
    positions=positions,
    patch_artist=True,
    widths=0.5,
)

for patch, label in zip(bp["boxes"], labels):
    patch.set_facecolor(colors[label])
    patch.set_alpha(0.7)

ax.set_xticks(positions)
ax.set_xticklabels([display_names[l] for l in labels], fontsize=10)
ax.set_ylabel("Final Fitness (best of last generation)", fontsize=11)
ax.set_title("Final Performance Distribution per Condition", fontsize=13)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig2_boxplots.pdf"), dpi=300)
fig.savefig(os.path.join(FIGURES_DIR, "fig2_boxplots.png"), dpi=300)
print("Saved fig2_boxplots")

# ------------------------------------------------------------------
# Figure 3: Performance Drop Bar Chart (Static → Dynamic)
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))

for i, ctrl in enumerate(["snn", "ann"]):
    static_mean  = final_fitnesses[f"{ctrl}_static"].mean()
    dynamic_mean = final_fitnesses[f"{ctrl}_dynamic"].mean()
    drop = static_mean - dynamic_mean
    drop_pct = (drop / abs(static_mean)) * 100 if static_mean != 0 else 0
    label = "SNN" if ctrl == "snn" else "ANN"
    bar_color = colors[f"{ctrl}_static"]
    ax.bar(i, drop_pct, color=bar_color, alpha=0.8, width=0.4, label=label)
    ax.text(i, drop_pct + 0.5, f"{drop_pct:.1f}%", ha="center", fontsize=11)

ax.set_xticks([0, 1])
ax.set_xticklabels(["SNN", "ANN"], fontsize=12)
ax.set_ylabel("Performance Drop (%)\nStatic → Dynamic", fontsize=11)
ax.set_title("Robustness to Dynamic Environments", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig3_performance_drop.pdf"), dpi=300)
fig.savefig(os.path.join(FIGURES_DIR, "fig3_performance_drop.png"), dpi=300)
print("Saved fig3_performance_drop")

# ------------------------------------------------------------------
# Statistical Tests: Mann-Whitney U (non-parametric, appropriate for EA)
# ------------------------------------------------------------------

print("\n" + "="*60)
print("STATISTICAL ANALYSIS")
print("="*60)

def mwu(a, b, label_a, label_b):
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    print(f"\n  {label_a} vs {label_b}")
    print(f"    Mean: {a.mean():.3f} vs {b.mean():.3f}")
    print(f"    U={stat:.1f}, p={p:.4f} {sig}")
    return p

if "snn_static" in data and "ann_static" in data:
    mwu(final_fitnesses["snn_static"],
        final_fitnesses["ann_static"],
        "SNN-Static", "ANN-Static")

if "snn_dynamic" in data and "ann_dynamic" in data:
    mwu(final_fitnesses["snn_dynamic"],
        final_fitnesses["ann_dynamic"],
        "SNN-Dynamic", "ANN-Dynamic")

if "snn_static" in data and "snn_dynamic" in data:
    mwu(final_fitnesses["snn_static"],
        final_fitnesses["snn_dynamic"],
        "SNN-Static", "SNN-Dynamic")

if "ann_static" in data and "ann_dynamic" in data:
    mwu(final_fitnesses["ann_static"],
        final_fitnesses["ann_dynamic"],
        "ANN-Static", "ANN-Dynamic")

# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------

print("\n" + "="*60)
print("SUMMARY TABLE")
print(f"{'Condition':<20} {'Mean':>8} {'Std':>8} {'Median':>8}")
print("-"*50)
for label in labels:
    if label in final_fitnesses:
        ff = final_fitnesses[label]
        print(f"{display_names[label]:<20} {ff.mean():>8.3f} {ff.std():>8.3f} {np.median(ff):>8.3f}")

print("\nFigures saved to:", FIGURES_DIR)
