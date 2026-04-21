"""
sanity_check.py
---------------
Quick test to verify all components work BEFORE running the full experiment.
Run this first:
    python sanity_check.py

Expected output:
  - Environment resets and steps without errors
  - SNN and ANN produce valid actions (0-4)
  - EA runs 5 generations without crashing
  - All shapes and values are as expected
"""

import numpy as np
from environment import GridEnvironment
from snn_controller import SNNController
from ann_controller import ANNController
from evolutionary_algorithm import GeneticAlgorithm

import sys, os
FRAMEWORK_DIR = os.path.join(os.path.dirname(__file__), "framework")
sys.path.insert(0, os.path.join(FRAMEWORK_DIR, "build"))
sys.path.insert(0, os.path.join(FRAMEWORK_DIR, "eons", "build"))
import neuro, eons 
print('Neuro and Eons correctly imported')

N_SENSORS = 8
N_INPUTS  = N_SENSORS + 2   # sensors + goal direction
N_HIDDEN  = 10
N_ACTIONS = 5

print("=" * 50)
print("SANITY CHECK")
print("=" * 50)

# ---- 1. Environment ----
print("\n[1] Testing Environment...")

for dynamic in [False, True]:
    env = GridEnvironment(
        grid_size=(20, 20),
        n_sensors=N_SENSORS,
        n_obstacles=15,
        dynamic=dynamic,
        max_steps=50,
        seed=42,
    )
    sensors = env.reset()
    assert sensors.shape == (N_INPUTS,), f"Expected {N_INPUTS} sensors, got {sensors.shape}"
    total_reward = 0
    for step in range(10):
        action = np.random.randint(0, N_ACTIONS)
        sensors, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    label = "Dynamic" if dynamic else "Static"
    print(f"  {label}: OK | sensor shape={sensors.shape} | reward after 10 steps={total_reward:.3f}")

# ---- 2. SNN Controller ----
print("\n[2] Testing SNN Controller...")

snn = SNNController(n_inputs=N_INPUTS, n_hidden=N_HIDDEN, n_outputs=N_ACTIONS)
print(f"  Weight vector size: {snn.total_weights}")
print(f"  W_ih shape: {snn.W_ih.shape}")
print(f"  W_ho shape: {snn.W_ho.shape}")

test_sensors = np.random.uniform(0, 1, N_INPUTS)
action, spikes = snn.forward(test_sensors)
assert 0 <= action <= 4, f"Invalid action: {action}"
print(f"  Forward pass OK | action={action} | spike counts={spikes}")

snn.reset_state()
print("  Reset state OK")

# ---- 3. ANN Controller ----
print("\n[3] Testing ANN Controller...")

ann = ANNController(n_inputs=N_INPUTS, n_hidden=N_HIDDEN, n_outputs=N_ACTIONS)
assert ann.total_weights == snn.total_weights, "Weight count mismatch between SNN and ANN!"
action_ann, out_ann = ann.forward(test_sensors)
assert 0 <= action_ann <= 4
print(f"  Forward pass OK | action={action_ann} | output={out_ann.round(3)}")
print(f"  Weight count matches SNN: {ann.total_weights} == {snn.total_weights} (OK)")

# ---- 4. EA (mini run: 2 generations, 5 individuals) ----
print("\n[4] Testing EA (mini run: 5 individuals, 5 generations)...")

def quick_fitness(weights):
    """Simple dummy fitness: just returns sum of weights for speed."""
    env = GridEnvironment(grid_size=(10, 10), n_sensors=N_SENSORS,
                          n_obstacles=5, dynamic=False, max_steps=30, seed=0)
    ctrl = ANNController(n_inputs=N_INPUTS, n_hidden=N_HIDDEN,
                         n_outputs=N_ACTIONS, weights=weights)
    sensors = env.reset()
    total_r = 0.0
    done = False
    while not done:
        action, _ = ctrl.forward(sensors)
        sensors, reward, done, _ = env.step(action)
        total_r += reward
    return total_r

ea = GeneticAlgorithm(
    fitness_fn=quick_fitness,
    genome_size=snn.total_weights,
    pop_size=5,
    n_generations=5,
    seed=0,
)
best_weights, history = ea.run(verbose=True)

assert len(history) == 5
assert best_weights.shape == (snn.total_weights,)
print(f"\n  EA OK | best fitness after 5 gens: {ea.best_fitness:.3f}")

# ---- Summary ----
print("\n" + "="*50)
print("ALL CHECKS PASSED")
print("You can now run: python experiment.py")
print("="*50)
print(f"\nEstimated runtime for full experiment:")
print(f"  {4} conditions × {30} trials × {100} generations × {50} individuals")
print(f"  ≈ several hours. Consider running overnight or reducing n_trials to 10 first.")