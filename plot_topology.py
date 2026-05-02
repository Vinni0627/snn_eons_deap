"""
plot_topology.py
----------------
Visualize the initial vs. best-evolved SNN topology for both conditions,
plus summary statistics (node/edge counts, fitness table, learning curves).

Run after hybrid_experiment.py:
    python plot_topology.py

Outputs saved to figures/:
  topology_comparison.png / .pdf   — network structure side-by-side
  topology_stats.png / .pdf        — summary statistics panel + Mann-Whitney U
  hybrid_learning_curves.png / .pdf — fitness over EONS epochs (per-trial lines)
  hybrid_topo_complexity.png / .pdf — hidden nodes & edges over EONS epochs
  hybrid_deap_convergence.png / .pdf — inner-loop DEAP fitness curves
  hybrid_trial_stripplot.png / .pdf  — box + individual trial dots

results/hybrid_summary.json is also written with key stats.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
from scipy import stats as scipy_stats

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Labels / colours
# -------------------------------------------------------------------------

INPUT_LABELS  = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "GoalΔr", "GoalΔc"]
OUTPUT_LABELS = ["North", "East", "South", "West", "Stay"]

NODE_COLORS = {
    "input"  : "#5DADE2",
    "hidden" : "#F39C12",
    "output" : "#27AE60",
}

COND_COLORS = {
    "static" : "#E07B54",
    "dynamic": "#1A5276",
}

# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

def load_condition(label):
    def _load(name, required=True):
        path = os.path.join(RESULTS_DIR, f"hybrid_snn_{label}_{name}.npy")
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Missing: {path}")
            return None
        return np.load(path, allow_pickle=True)

    fits    = np.array(_load("fitnesses"),    dtype=float)
    weights = _load("best_weights")
    topos   = _load("best_topos")

    # Optional richer outputs (present only after multiprocessed rerun)
    hidden  = _load("topo_hidden",    required=False)
    edges   = _load("topo_edges",     required=False)
    times   = _load("trial_times",    required=False)
    deap_h  = _load("deap_histories", required=False)

    return {
        "fits":    fits,
        "weights": weights,
        "topos":   topos,
        "hidden":  hidden,
        "edges":   edges,
        "times":   times,
        "deap_h":  deap_h,
    }


def best_trial(fits):
    return int(np.argmax(fits[:, -1]))


# -------------------------------------------------------------------------
# Graph construction helpers
# -------------------------------------------------------------------------

def make_initial_graph(seed=42):
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    input_ids  = list(range(10))
    output_ids = list(range(10, 15))
    hidden_ids = [15, 16, 17]

    for i, nid in enumerate(input_ids):
        G.add_node(nid, role="input",  label=INPUT_LABELS[i],  layer=0)
    for i, nid in enumerate(hidden_ids):
        G.add_node(nid, role="hidden", label=f"H{nid}",        layer=1)
    for i, nid in enumerate(output_ids):
        G.add_node(nid, role="output", label=OUTPUT_LABELS[i], layer=2)

    all_nodes = input_ids + hidden_ids + output_ids
    pool = [(u, v) for u in all_nodes for v in all_nodes if u != v]
    chosen = rng.choice(len(pool), size=6, replace=False)
    for idx in chosen:
        u, v = pool[idx]
        G.add_edge(u, v, weight=float(rng.uniform(-2.0, 2.0)))
    return G


def make_evolved_graph(topo, weights):
    G = nx.DiGraph()
    inp_set = set(topo["Inputs"])
    out_set = set(topo["Outputs"])

    for node in topo["Nodes"]:
        nid = node["id"]
        if nid in inp_set:
            role, lyr = "input",  0
            lbl = INPUT_LABELS[topo["Inputs"].index(nid)]
        elif nid in out_set:
            role, lyr = "output", 2
            lbl = OUTPUT_LABELS[topo["Outputs"].index(nid)]
        else:
            role, lyr = "hidden", 1
            lbl = f"H{nid}"
        G.add_node(nid, role=role, label=lbl, layer=lyr)

    for idx, edge in enumerate(topo["Edges"]):
        w = float(weights[idx]) if idx < len(weights) else 0.0
        G.add_edge(edge["from"], edge["to"], weight=w)
    return G


def layered_positions(G):
    layers = {0: [], 1: [], 2: []}
    for nid, d in G.nodes(data=True):
        layers[d["layer"]].append(nid)
    pos = {}
    for lyr, nodes in layers.items():
        x = {0: 0.0, 1: 1.0, 2: 2.0}[lyr]
        ys = np.linspace(1, -1, len(nodes)) if len(nodes) > 1 else [0.0]
        for node, y in zip(nodes, ys):
            pos[node] = (x, y)
    return pos


def draw_topology(ax, G, title, fitness=None):
    pos = layered_positions(G)
    node_colors = [NODE_COLORS[G.nodes[n]["role"]] for n in G.nodes]
    labels = {n: G.nodes[n]["label"] for n in G.nodes}

    nx.draw_networkx_nodes(ax=ax, G=G, pos=pos,
                           node_color=node_colors, node_size=500, alpha=0.9)
    nx.draw_networkx_labels(ax=ax, G=G, pos=pos, labels=labels,
                            font_size=6, font_color="white", font_weight="bold")

    edges = list(G.edges(data=True))
    if edges:
        w_arr   = np.array([d["weight"] for _, _, d in edges])
        abs_w   = np.abs(w_arr)
        max_w   = abs_w.max() if abs_w.max() > 0 else 1.0
        widths  = 1.0 + 3.0 * (abs_w / max_w)
        ecolors = ["#C0392B" if w < 0 else "#2980B9" for w in w_arr]
        nx.draw_networkx_edges(ax=ax, G=G, pos=pos,
                               edgelist=[(u, v) for u, v, _ in edges],
                               width=widths, edge_color=ecolors,
                               arrows=True, arrowsize=12,
                               connectionstyle="arc3,rad=0.1",
                               node_size=500)

    n_hidden = sum(1 for _, d in G.nodes(data=True) if d["role"] == "hidden")
    info = f"Nodes: {G.number_of_nodes()}  |  Hidden: {n_hidden}  |  Edges: {G.number_of_edges()}"
    if fitness is not None:
        info += f"\nFitness: {fitness:.3f}"
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.text(0.5, -0.04, info, transform=ax.transAxes,
            fontsize=7, ha="center", va="top", style="italic", color="#555555")
    ax.axis("off")


# -------------------------------------------------------------------------
# Figure 1: Network structure comparison
# -------------------------------------------------------------------------

def fig_topology_comparison(data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("SNN Topology: Initial vs. Best Evolved", fontsize=14,
                 fontweight="bold", y=1.01)
    G_init = make_initial_graph()

    for row, cond in enumerate(["static", "dynamic"]):
        d = data[cond]
        bt = best_trial(d["fits"])
        G_best = make_evolved_graph(d["topos"][bt], d["weights"][bt])
        draw_topology(axes[row, 0], G_init,
                      title=f"{'Static' if cond=='static' else 'Dynamic'} — Representative Gen-0 Individual")
        draw_topology(axes[row, 1], G_best,
                      title=f"{'Static' if cond=='static' else 'Dynamic'} — Best Evolved (Trial {bt+1})",
                      fitness=d["fits"][bt, -1])

    legend_patches = [
        mpatches.Patch(color=NODE_COLORS["input"],  label="Input node"),
        mpatches.Patch(color=NODE_COLORS["hidden"], label="Hidden node (evolved)"),
        mpatches.Patch(color=NODE_COLORS["output"], label="Output node"),
        mpatches.Patch(color="#2980B9", label="Excitatory edge (w > 0)"),
        mpatches.Patch(color="#C0392B", label="Inhibitory edge (w < 0)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout()
    _save(fig, "topology_comparison")


# -------------------------------------------------------------------------
# Figure 2: Summary statistics panel + Mann-Whitney U
# -------------------------------------------------------------------------

def fig_summary_stats(data):
    fig = plt.figure(figsize=(13, 5))
    fig.suptitle("Hybrid SNN-EONS-DEAP: Summary Statistics", fontsize=13, fontweight="bold")

    ax_nodes = fig.add_subplot(1, 3, 1)
    ax_edges = fig.add_subplot(1, 3, 2)
    ax_table = fig.add_subplot(1, 3, 3)

    conditions  = ["static", "dynamic"]
    cond_labels = ["Static", "Dynamic"]
    x = np.arange(len(conditions))
    width = 0.3

    node_counts = {"input": [], "hidden": [], "output": []}
    edge_counts = []

    for cond in conditions:
        d = data[cond]
        bt = best_trial(d["fits"])
        G = make_evolved_graph(d["topos"][bt], d["weights"][bt])
        node_counts["input"].append(sum(1 for _, nd in G.nodes(data=True) if nd["role"]=="input"))
        node_counts["hidden"].append(sum(1 for _, nd in G.nodes(data=True) if nd["role"]=="hidden"))
        node_counts["output"].append(sum(1 for _, nd in G.nodes(data=True) if nd["role"]=="output"))
        edge_counts.append(G.number_of_edges())

    bars_in  = ax_nodes.bar(x, node_counts["input"],  width, label="Input",
                            color=NODE_COLORS["input"],  alpha=0.85)
    bars_hid = ax_nodes.bar(x, node_counts["hidden"], width, bottom=node_counts["input"],
                            label="Hidden", color=NODE_COLORS["hidden"], alpha=0.85)
    bars_out = ax_nodes.bar(x, node_counts["output"], width,
                            bottom=[i+h for i,h in zip(node_counts["input"],node_counts["hidden"])],
                            label="Output", color=NODE_COLORS["output"], alpha=0.85)
    ax_nodes.set_xticks(x); ax_nodes.set_xticklabels(cond_labels)
    ax_nodes.set_ylabel("Node count")
    ax_nodes.set_title("Best Evolved Node Breakdown")
    ax_nodes.legend(fontsize=8, loc="upper right")
    ax_nodes.grid(True, axis="y", alpha=0.3)
    for bar, h in zip(bars_hid, node_counts["hidden"]):
        if h > 0:
            ax_nodes.text(bar.get_x() + bar.get_width()/2,
                          bar.get_y() + h/2, str(h),
                          ha="center", va="center", fontsize=9,
                          color="white", fontweight="bold")

    bar_colors = [COND_COLORS[c] for c in conditions]
    bars = ax_edges.bar(x, edge_counts, width, color=bar_colors, alpha=0.85)
    ax_edges.set_xticks(x); ax_edges.set_xticklabels(cond_labels)
    ax_edges.set_ylabel("Edge count")
    ax_edges.set_title("Best Evolved Edge Count")
    ax_edges.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, edge_counts):
        ax_edges.text(bar.get_x() + bar.get_width()/2, v + 0.2, str(v),
                      ha="center", fontsize=10, fontweight="bold")

    # Fitness summary table + Mann-Whitney U
    ax_table.axis("off")
    col_labels = ["Condition", "Mean", "Std", "Median", "Max"]
    rows = []
    ff_static  = data["static"]["fits"][:, -1]
    ff_dynamic = data["dynamic"]["fits"][:, -1]

    for cond, lbl, ff in [("static", "Static", ff_static), ("dynamic", "Dynamic", ff_dynamic)]:
        rows.append([lbl, f"{ff.mean():.3f}", f"{ff.std():.3f}",
                     f"{np.median(ff):.3f}", f"{ff.max():.3f}"])

    # Mann-Whitney U
    mwu_stat, mwu_p = scipy_stats.mannwhitneyu(ff_static, ff_dynamic, alternative="two-sided")
    sig = "***" if mwu_p < 0.001 else ("**" if mwu_p < 0.01 else ("*" if mwu_p < 0.05 else "n.s."))
    rows.append(["MWU Static vs Dynamic", f"U={mwu_stat:.0f}", f"p={mwu_p:.4f}", sig, ""])

    tbl = ax_table.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            cell.set_facecolor(mcolors.to_rgba(COND_COLORS["static"], alpha=0.25))
        elif r == 2:
            cell.set_facecolor(mcolors.to_rgba(COND_COLORS["dynamic"], alpha=0.25))
        else:
            cell.set_facecolor("#F0F0F0")
    ax_table.set_title(f"Final Epoch Fitness ({data['static']['fits'].shape[0]} trials)",
                       fontsize=10, fontweight="bold")

    print(f"\nMann-Whitney U (Static vs Dynamic): U={mwu_stat:.1f}, p={mwu_p:.4f} {sig}")

    plt.tight_layout()
    _save(fig, "topology_stats")


# -------------------------------------------------------------------------
# Figure 3: Learning curves — mean±std + individual trial lines
# -------------------------------------------------------------------------

def fig_learning_curves(data):
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {"static": "-", "dynamic": "--"}

    for cond in ["static", "dynamic"]:
        fits   = data[cond]["fits"]
        mean   = fits.mean(axis=0)
        std    = fits.std(axis=0)
        epochs = np.arange(len(mean))
        lbl    = cond.capitalize()

        # Individual trial lines (thin, transparent)
        for t in range(fits.shape[0]):
            ax.plot(epochs, fits[t], linestyle=styles[cond],
                    color=COND_COLORS[cond], linewidth=0.6, alpha=0.35)

        ax.plot(epochs, mean, linestyle=styles[cond], color=COND_COLORS[cond],
                linewidth=2.5, label=f"{lbl} (mean ± std)")
        ax.fill_between(epochs, mean - std, mean + std,
                        alpha=0.18, color=COND_COLORS[cond])
        ax.axhline(mean[-1], linestyle=":", color=COND_COLORS[cond], linewidth=1, alpha=0.6)
        ax.text(len(mean) - 1, mean[-1] + 0.15, f"{mean[-1]:.2f}",
                color=COND_COLORS[cond], fontsize=8, va="bottom", ha="right")

    ax.set_xlabel("EONS Epoch", fontsize=12)
    ax.set_ylabel("Best Fitness (avg reward)", fontsize=12)
    ax.set_title("Hybrid SNN-EONS-DEAP: Learning Curves", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "hybrid_learning_curves")


# -------------------------------------------------------------------------
# Figure 4: Topology complexity over EONS epochs
# -------------------------------------------------------------------------

def fig_topo_complexity(data):
    has_data = all(data[c]["hidden"] is not None for c in ["static", "dynamic"])
    if not has_data:
        print("Skipping topology complexity plot (topo_hidden.npy not found — run new experiment)")
        return

    fig, (ax_h, ax_e) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Topology Complexity over EONS Epochs", fontsize=13, fontweight="bold")

    for cond in ["static", "dynamic"]:
        d      = data[cond]
        lbl    = cond.capitalize()
        epochs = np.arange(d["fits"].shape[1])

        hidden = np.array(d["hidden"], dtype=float)   # (n_trials, n_epochs)
        edges  = np.array(d["edges"],  dtype=float)

        for ax, arr, ylabel in [(ax_h, hidden, "Hidden node count"),
                                 (ax_e, edges,  "Edge count")]:
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            ax.plot(epochs, mean, color=COND_COLORS[cond], linewidth=2, label=lbl)
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.2, color=COND_COLORS[cond])
            ax.set_xlabel("EONS Epoch", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

    ax_h.set_title("Hidden Nodes (epoch-best individual)")
    ax_e.set_title("Edges (epoch-best individual)")
    plt.tight_layout()
    _save(fig, "hybrid_topo_complexity")


# -------------------------------------------------------------------------
# Figure 5: Inner-loop DEAP convergence for selected EONS epochs
# -------------------------------------------------------------------------

def fig_deap_convergence(data):
    has_data = all(data[c]["deap_h"] is not None for c in ["static", "dynamic"])
    if not has_data:
        print("Skipping DEAP convergence plot (deap_histories.npy not found — run new experiment)")
        return

    n_epochs = data["static"]["fits"].shape[1]
    # Select 3 representative epochs
    probe_epochs = [0, n_epochs // 2, n_epochs - 1]

    fig, axes = plt.subplots(1, len(probe_epochs), figsize=(13, 4), sharey=False)
    fig.suptitle("DEAP Inner-Loop Convergence at Selected EONS Epochs",
                 fontsize=13, fontweight="bold")

    for ax, ep in zip(axes, probe_epochs):
        for cond in ["static", "dynamic"]:
            d   = data[cond]
            bt  = best_trial(d["fits"])
            # deap_h shape: (n_trials, n_eons_epochs, n_deap_gens+1)
            dh  = np.array(d["deap_h"][bt][ep], dtype=float)
            gens = np.arange(len(dh))
            ax.plot(gens, dh, color=COND_COLORS[cond], linewidth=2,
                    label=cond.capitalize())

        ax.set_title(f"EONS Epoch {ep + 1}", fontsize=10)
        ax.set_xlabel("DEAP Generation", fontsize=10)
        ax.set_ylabel("Best Fitness", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "hybrid_deap_convergence")


# -------------------------------------------------------------------------
# Figure 6: Box plot + individual trial strip
# -------------------------------------------------------------------------

def fig_trial_stripplot(data):
    fig, ax = plt.subplots(figsize=(6, 5))

    conditions  = ["static", "dynamic"]
    cond_labels = ["Static", "Dynamic"]
    final_fits  = [data[c]["fits"][:, -1] for c in conditions]
    positions   = [1, 2]

    bp = ax.boxplot(final_fits, positions=positions, patch_artist=True,
                    widths=0.35, zorder=2)
    for patch, cond in zip(bp["boxes"], conditions):
        patch.set_facecolor(mcolors.to_rgba(COND_COLORS[cond], alpha=0.5))

    rng = np.random.default_rng(0)
    for pos, ff, cond in zip(positions, final_fits, conditions):
        jitter = rng.uniform(-0.08, 0.08, size=len(ff))
        ax.scatter(pos + jitter, ff, color=COND_COLORS[cond],
                   s=60, zorder=3, edgecolors="white", linewidths=0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(cond_labels, fontsize=12)
    ax.set_ylabel("Final Fitness (last EONS epoch)", fontsize=11)
    ax.set_title("Per-Trial Final Fitness Distribution", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "hybrid_trial_stripplot")


# -------------------------------------------------------------------------
# JSON summary export
# -------------------------------------------------------------------------

def export_summary(data):
    summary = {}
    conds = list(data.keys())

    ff_by_cond = {c: data[c]["fits"][:, -1] for c in conds}

    for cond in conds:
        ff    = ff_by_cond[cond]
        times = data[cond]["times"]
        entry = {
            "mean"   : float(ff.mean()),
            "std"    : float(ff.std()),
            "median" : float(np.median(ff)),
            "max"    : float(ff.max()),
            "min"    : float(ff.min()),
            "trials" : ff.tolist(),
        }
        if times is not None:
            entry["trial_times_s"] = [float(t) for t in times]

        # MWU against every other condition
        mwu_results = {}
        for other in conds:
            if other == cond:
                continue
            u, p = scipy_stats.mannwhitneyu(ff, ff_by_cond[other], alternative="two-sided")
            mwu_results[f"vs_{other}"] = {"U": float(u), "p": float(p)}
        entry["mwu"] = mwu_results
        summary[cond] = entry

    path = os.path.join(RESULTS_DIR, "hybrid_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {path}")


# -------------------------------------------------------------------------
# Save helper
# -------------------------------------------------------------------------

def _save(fig, name):
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {name}")
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    data = {}
    for cond in ["static", "dynamic"]:
        try:
            data[cond] = load_condition(cond)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if not data:
        print("No hybrid results found. Run hybrid_experiment.py first.")
        exit(1)

    fig_topology_comparison(data)
    fig_summary_stats(data)
    fig_learning_curves(data)
    fig_topo_complexity(data)
    fig_deap_convergence(data)
    fig_trial_stripplot(data)
    export_summary(data)

    print(f"\nAll figures saved to: {FIGURES_DIR}/")
