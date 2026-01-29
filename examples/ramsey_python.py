import sys
import os
sys.path.append(os.getcwd())
import pybaha
import random
import itertools

# Problem: Ramsey R(3,3,3) @ N=16
# Find a 3-coloring of edges in K16 with no monochromatic K3 (triangle).
# R(3,3,3) = 17, so a solution exists for N=16.

N_NODES = 16
EDGES = list(itertools.combinations(range(N_NODES), 2))
TRIANGLES = list(itertools.combinations(range(N_NODES), 3))

print(f"🧩 Problem: Ramsey R(3,3,3) @ N={N_NODES}")
print(f"  - Edges: {len(EDGES)}")
print(f"  - Triangles: {len(TRIANGLES)}")
print(f"  - Search Space: 3^{len(EDGES)} ≈ 1.7e57")

# Pre-index triangles to edge indices for O(1) checks
TRI_EDGE_IDXS = []
for u, v, w in TRIANGLES:
    e1 = EDGES.index(tuple(sorted((u, v))))
    e2 = EDGES.index(tuple(sorted((v, w))))
    e3 = EDGES.index(tuple(sorted((w, u))))
    TRI_EDGE_IDXS.append((e1, e2, e3))

def energy(state_list):
    # state_list is a list of colors [c0, c1, ..., c119]
    violations = 0
    # The hot loop: pre-indexed integer lookups
    for e1, e2, e3 in TRI_EDGE_IDXS:
        if state_list[e1] == state_list[e2] == state_list[e3]:
            violations += 1
    return float(violations)

def sampler():
    return [random.randint(0, 2) for _ in range(len(EDGES))]

def neighbors(state):
    nbrs = []
    # Pick a random edge and try all other colors
    for _ in range(32):
        edge_idx = random.randrange(len(EDGES))
        current_color = state[edge_idx]
        for new_color in range(3):
            if new_color != current_color:
                nbr = list(state)
                nbr[edge_idx] = new_color
                nbrs.append(nbr)
    return nbrs

# BAHA Setup
opt = pybaha.Optimizer(energy, sampler, neighbors)
config = pybaha.Config()
config.beta_steps = 1500
config.beta_end = 30.0
config.fracture_threshold = 1.15  # More sensitive to shattering
config.samples_per_beta = 50     # Faster estimation
config.max_branches = 8         # More parallel exploration
config.verbose = True
config.schedule_type = pybaha.ScheduleType.GEOMETRIC

print("\nStarting Optimized BAHA (Lightning Mode)...")
result = opt.optimize(config)

print(f"\nFinal Violations: {result.best_energy}")
print(f"Fractures: {result.fractures_detected}, Jumps: {result.branch_jumps}")
print(f"Solve Time: {result.time_ms/1000.0:.2f} seconds")

if result.best_energy == 0:
    print("💎 SUCCESS: MONOCHROMATIC-TRIANGLE-FREE COLORING FOUND!")
else:
    print("🛑 FAILED TO FIND GROUND STATE.")
