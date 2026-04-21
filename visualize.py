"""
visualize.py
------------
Visualizes the best evolved agent navigating the grid environment.

Run AFTER experiment.py has saved at least one condition:
    python visualize.py

What it shows:
  - The 20x20 grid with obstacles, goal, and agent
  - Agent moving step by step
  - Sensor rays drawn from the agent
  - Live fitness/reward counter
  - Side-by-side: SNN vs ANN in same environment

Controls:
  - Press SPACE to toggle auto-play
  - Press LEFT/RIGHT to step through frames
  - Press Q to quit
"""

import numpy as np
import os
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.lines as mlines
    from matplotlib.animation import FuncAnimation
    import matplotlib.gridspec as gridspec
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)

from environment import GridEnvironment
from topology_snn import TopologySNN

RESULTS_DIR = "results"
N_SENSORS   = 8
N_INPUTS    = N_SENSORS + 2
N_HIDDEN    = 10
N_ACTIONS   = 5

# Colors
COLOR_FREE     = "#F5F5F5"
COLOR_OBSTACLE = "#2C3E50"
COLOR_GOAL     = "#2ECC71"
COLOR_AGENT    = "#E74C3C"
COLOR_SENSOR   = "#3498DB"
COLOR_PATH     = "#E67E22"
COLOR_BG       = "#FFFFFF"

ACTION_NAMES  = {0: "North", 1: "East", 2: "South", 3: "West", 4: "Stay"}
ACTION_ARROWS = {0: "↑",     1: "→",    2: "↓",     3: "←",    4: "•"}

SENSOR_DIRS = [
    (-1,  0), (-1,  1), ( 0,  1), ( 1,  1),
    ( 1,  0), ( 1, -1), ( 0, -1), (-1, -1),
]


# ------------------------------------------------------------------
# Load best weights
# ------------------------------------------------------------------

def load_weights(condition, trial_idx=None):
    prefix       = f"hybrid_{condition}"
    fits_path    = os.path.join(RESULTS_DIR, f"{prefix}_fitnesses.npy")
    weights_path = os.path.join(RESULTS_DIR, f"{prefix}_best_weights.npy")
    topo_path    = os.path.join(RESULTS_DIR, f"{prefix}_best_topos.npy")

    if not os.path.exists(fits_path):
        return None, None, None, None

    fits    = np.load(fits_path, allow_pickle=True)
    weights = np.load(weights_path, allow_pickle=True)
    topos   = np.load(topo_path, allow_pickle=True)

    # Each trial's history is a 1-D list of per-epoch bests
    final_fits = np.array([h[-1] for h in fits])
    if trial_idx is None:
        trial_idx = int(np.argmax(final_fits))

    print(f"  Loaded {condition}: trial {trial_idx+1}, "
          f"final fitness = {final_fits[trial_idx]:.3f}")
    return weights[trial_idx], topos[trial_idx], fits, trial_idx


# ------------------------------------------------------------------
# Run one episode (returns full trajectory for replay)
# ------------------------------------------------------------------

def run_episode(ctrl, env):
    sensors      = env.reset()
    ctrl.reset_state()
    trajectory   = []
    done         = False
    total_reward = 0.0
    goal_reached = False

    while not done:
        action, spike_counts_raw = ctrl.forward(sensors)
        spike_counts = np.array(spike_counts_raw, dtype=float)
        state = {
            "agent_pos"    : env.agent_pos,
            "goal_pos"     : env.goal_pos,
            "obstacles"    : list(env.obstacle_positions),
            "action"       : action,
            "spike_counts" : spike_counts.copy(),
            "sensors"      : sensors.copy(),
            "reward"       : 0.0,
            "total_reward" : total_reward,
            "step"         : env.step_count,
            "dist_to_goal" : env._manhattan(env.agent_pos, env.goal_pos),
        }
        sensors, reward, done, info = env.step(action)
        state["reward"]       = reward
        total_reward         += reward
        state["total_reward"] = total_reward
        state["dist_to_goal"] = info["dist_to_goal"]

        if info.get("goal_reached", False):
            goal_reached = True

        trajectory.append(state)

    return trajectory, total_reward, goal_reached


# ------------------------------------------------------------------
# Grid renderer
# ------------------------------------------------------------------

def draw_grid(ax, state, rows=20, cols=20, title="", show_sensors=True):
    ax.clear()
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    # Grid lines
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="#CCCCCC", linewidth=0.3)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="#CCCCCC", linewidth=0.3)

    # Obstacles
    for (r, c) in state["obstacles"]:
        ax.add_patch(patches.Rectangle(
            (c - 0.5, r - 0.5), 1, 1,
            linewidth=0, facecolor=COLOR_OBSTACLE, zorder=2
        ))

    # Goal
    gr, gc = state["goal_pos"]
    ax.add_patch(patches.Rectangle(
        (gc - 0.5, gr - 0.5), 1, 1,
        linewidth=1.5, edgecolor="#27AE60",
        facecolor=COLOR_GOAL, zorder=3
    ))
    ax.text(gc, gr, "G", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=4)

    # Sensor rays
    if show_sensors:
        ar, ac   = state["agent_pos"]
        max_dist = max(rows, cols)
        for i, (dr, dc) in enumerate(SENSOR_DIRS[:N_SENSORS]):
            dist  = state["sensors"][i] * max_dist
            end_r = ar + dr * dist
            end_c = ac + dc * dist
            ax.plot([ac, end_c], [ar, end_r],
                    color=COLOR_SENSOR, linewidth=0.8,
                    alpha=0.5, linestyle="--", zorder=3)

    # Agent
    ar, ac = state["agent_pos"]
    dist   = state.get("dist_to_goal", 99)
    color  = "#F39C12" if dist <= 1 else COLOR_AGENT
    ax.add_patch(patches.Circle(
        (ac, ar), 0.38,
        facecolor=color, edgecolor="#922B21",
        linewidth=1.5, zorder=5
    ))
    arrow = ACTION_ARROWS.get(state["action"], "•")
    ax.text(ac, ar, arrow, ha="center", va="center",
            fontsize=9, color="white", fontweight="bold", zorder=6)

    # Distance indicator
    ax.text(0.02, 0.02, f"Dist to goal: {dist}",
            transform=ax.transAxes, fontsize=8,
            color="#E74C3C" if dist <= 3 else "#888888")


# ------------------------------------------------------------------
# Spike bar renderer
# ------------------------------------------------------------------

def draw_spikes(ax, spike_counts, title="Output Spikes"):
    ax.clear()
    actions = ["N", "E", "S", "W", "Stay"]
    colors  = [COLOR_AGENT if i == int(np.argmax(spike_counts))
               else "#AED6F1" for i in range(len(spike_counts))]
    ax.bar(actions, spike_counts, color=colors, edgecolor="white")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Spike count", fontsize=8)
    ax.set_ylim(0, max(spike_counts.max() + 1, 5))
    for bar, val in zip(ax.patches, spike_counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1, str(int(val)),
                    ha="center", va="bottom", fontsize=8)


# ------------------------------------------------------------------
# Main visualization
# ------------------------------------------------------------------

def visualize():
    print("\nLoading results...")

    # Fixed comparison: hybrid SNN static vs hybrid SNN dynamic
    cond_a, cond_b = "snn_static", "snn_dynamic"
    dynamic_a, dynamic_b = False, True
    env_label = "Static vs Dynamic"

    # Trial selection for A
    print(f"\nSelect trial for {cond_a}:")
    print("  0 = best trial (default)")
    t_a = input("Enter trial number (1-3) or 0 for best: ").strip()
    trial_idx_a = int(t_a) - 1 if t_a.isdigit() and int(t_a) > 0 else None
    weights_a, topo_a, fits_a, idx_a = load_weights(cond_a, trial_idx_a)

    # Trial selection for B
    print(f"\nSelect trial for {cond_b}:")
    print("  0 = best trial (default)")
    t_b = input("Enter trial number (1-3) or 0 for best: ").strip()
    trial_idx_b = int(t_b) - 1 if t_b.isdigit() and int(t_b) > 0 else None
    weights_b, topo_b, fits_b, idx_b = load_weights(cond_b, trial_idx_b)

    if weights_a is None or topo_a is None:
        print(f"No results found for {cond_a} — run hybrid_experiment.py first.")
        return
    if weights_b is None or topo_b is None:
        print(f"No results found for {cond_b} — run hybrid_experiment.py first.")
        return

    # Build controllers
    ctrl_a = TopologySNN(topo_a, weights_a)
    ctrl_b = TopologySNN(topo_b, weights_b)

    # Run on best training seed (seed that gives highest reward)
    print("\nFinding best episode across training seeds...")

    def find_best_episode(ctrl, dynamic):
        best_traj, best_reward, best_goal, best_seed = None, -np.inf, False, 0
        for seed in range(5):
            env = GridEnvironment(
                grid_size=(20, 20), n_sensors=N_SENSORS,
                n_obstacles=15, dynamic=dynamic,
                dynamic_interval=5, max_steps=200, seed=seed
            )
            traj, total, goal = run_episode(ctrl, env)
            if goal or total > best_reward:
                best_traj, best_reward = traj, total
                best_goal, best_seed   = goal, seed
        return best_traj, best_reward, best_goal, best_seed

    traj_a, total_a, goal_a, seed_a = find_best_episode(ctrl_a, dynamic_a)
    traj_b, total_b, goal_b, seed_b = find_best_episode(ctrl_b, dynamic_b)

    out_a = "GOAL REACHED" if goal_a else "TIMED OUT"
    out_b = "GOAL REACHED" if goal_b else "TIMED OUT"

    print(f"\n  {cond_a} trial {idx_a+1}: {out_a} "
          f"(seed {seed_a}, fitness {total_a:.3f}, steps {len(traj_a)})")
    print(f"  {cond_b} trial {idx_b+1}: {out_b} "
          f"(seed {seed_b}, fitness {total_b:.3f}, steps {len(traj_b)})")

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(COLOR_BG)

    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[6, 2, 0.3],
        hspace=0.4, wspace=0.3
    )

    ax_snn       = fig.add_subplot(gs[0, 0])
    ax_ann       = fig.add_subplot(gs[0, 1])
    ax_snn_spike = fig.add_subplot(gs[1, 0])
    ax_ann_act   = fig.add_subplot(gs[1, 1])
    ax_status    = fig.add_subplot(gs[2, :])
    ax_status.axis("off")

    max_steps = max(len(traj_a), len(traj_b))
    step_idx  = [0]
    playing   = [True]

    label_a = cond_a.replace("_", " ").upper()
    label_b = cond_b.replace("_", " ").upper()

    def get_state(traj, idx):
        return traj[min(idx, len(traj) - 1)]

    def update(frame=None):
        i  = step_idx[0]
        sa = get_state(traj_a, i)
        sb = get_state(traj_b, i)

        draw_grid(
            ax_snn, sa,
            title=(
                f"{label_a} | Trial {idx_a+1} | {out_a}\n"
                f"Step {sa['step']} | "
                f"Dist: {sa['dist_to_goal']} | "
                f"Reward: {sa['reward']:+.2f} | "
                f"Total: {sa['total_reward']:.2f} | "
                f"{ACTION_NAMES[sa['action']]}"
            ),
            show_sensors=True
        )

        draw_grid(
            ax_ann, sb,
            title=(
                f"{label_b} | Trial {idx_b+1} | {out_b}\n"
                f"Step {sb['step']} | "
                f"Dist: {sb['dist_to_goal']} | "
                f"Reward: {sb['reward']:+.2f} | "
                f"Total: {sb['total_reward']:.2f} | "
                f"{ACTION_NAMES[sb['action']]}"
            ),
            show_sensors=False
        )

        # SNN spike bar
        draw_spikes(ax_snn_spike, sa["spike_counts"],
                    title=f"{label_a} — Output Spike Counts")

        # B spike bar
        draw_spikes(ax_ann_act, sb["spike_counts"],
                    title=f"{label_b} — Output Spike Counts")

        # Status bar
        ax_status.clear()
        ax_status.axis("off")
        play_state = "[PLAYING]" if playing[0] else "[PAUSED - press SPACE]"
        status = (
            f"Frame {i+1}/{max_steps}  |  "
            f"{label_a}: {total_a:.2f} ({out_a})  |  "
            f"{label_b}: {total_b:.2f} ({out_b})  |  "
            f"SPACE=play/pause   LEFT/RIGHT=step   Q=quit   {play_state}"
        )
        ax_status.text(0.5, 0.5, status,
                       ha="center", va="center",
                       fontsize=10, transform=ax_status.transAxes,
                       color="#2C3E50")

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            playing[0] = not playing[0]
        elif event.key in ("right", "enter"):
            if step_idx[0] < max_steps - 1:
                step_idx[0] += 1
            update()
        elif event.key == "left":
            if step_idx[0] > 0:
                step_idx[0] -= 1
            update()
        elif event.key == "q":
            plt.close("all")

    fig.canvas.mpl_connect("key_press_event", on_key)

    def animate(frame):
        if playing[0] and step_idx[0] < max_steps - 1:
            step_idx[0] += 1
        update()

    ani = FuncAnimation(fig, animate, frames=max_steps,
                        interval=150, repeat=False)

    plt.suptitle(
        f"EA-Evolved SNN vs ANN Navigation — {env_label} Environment\n"
        f"{label_a} Trial {idx_a+1}: "
        f"{'GOAL REACHED' if goal_a else 'TIMED OUT'}  |  "
        f"{label_b} Trial {idx_b+1}: "
        f"{'GOAL REACHED' if goal_b else 'TIMED OUT'}",
        fontsize=13, fontweight="bold", y=0.98
    )

    print("\nControls:")
    print("  SPACE      = play / pause")
    print("  LEFT/RIGHT = step backward / forward")
    print("  Q          = quit")

    update()
    plt.show()


if __name__ == "__main__":
    visualize()