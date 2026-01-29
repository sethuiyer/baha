import sys
import os
sys.path.append(os.getcwd())
import pybaha
import random
import itertools
from dataclasses import dataclass

# Problem: Ramsey R(3,3,3) @ N=16
# Find a 3-coloring of edges in K16 with no monochromatic triangles.
N_NODES = 16
EDGES = list(itertools.combinations(range(N_NODES), 2))
TRIANGLES = list(itertools.combinations(range(N_NODES), 3))

# Pre-map each edge to the triangles it belongs to
EDGE_TO_TRI_IDXS = [[] for _ in range(len(EDGES))]
TRI_EDGE_IDXS = []

# Map triangles to edge indices
for u, v, w in TRIANGLES:
    e1 = EDGES.index(tuple(sorted((u, v))))
    e2 = EDGES.index(tuple(sorted((v, w))))
    e3 = EDGES.index(tuple(sorted((w, u))))
    tri_idx = len(TRI_EDGE_IDXS)
    TRI_EDGE_IDXS.append((e1, e2, e3))
    EDGE_TO_TRI_IDXS[e1].append(tri_idx)
    EDGE_TO_TRI_IDXS[e2].append(tri_idx)
    EDGE_TO_TRI_IDXS[e3].append(tri_idx)

@dataclass
class RamseyState:
    colors: list
    energy: float

    # To make state printable in baha-cli or result dumps
    def __repr__(self):
        return "".join(map(str, self.colors))

def compute_initial_energy(colors):
    violations = 0
    for e1, e2, e3 in TRI_EDGE_IDXS:
        if colors[e1] == colors[e2] == colors[e3]:
            violations += 1
    return float(violations)

def energy_fn(state):
    return state.energy

def sampler():
    cols = [random.randint(0, 2) for _ in range(len(EDGES))]
    return RamseyState(cols, compute_initial_energy(cols))

def neighbors(state):
    nbrs = []
    # Try flipping 64 random edges (N=16 is denser)
    for _ in range(64):
        edge_idx = random.randrange(len(EDGES))
        current_color = state.colors[edge_idx]
        for new_color in range(3):
            if new_color != current_color:
                delta = 0
                for tri_idx in EDGE_TO_TRI_IDXS[edge_idx]:
                    e1, e2, e3 = TRI_EDGE_IDXS[tri_idx]
                    if state.colors[e1] == state.colors[e2] == state.colors[e3]:
                        delta -= 1
                    other1 = e1 if e1 != edge_idx else e2
                    other2 = e3 if e3 != edge_idx else e2
                    if state.colors[other1] == new_color and state.colors[other2] == new_color:
                        delta += 1
                
                new_cols = list(state.colors)
                new_cols[edge_idx] = new_color
                nbrs.append(RamseyState(new_cols, float(state.energy + delta)))
    return nbrs

print(f"💎 HUNTING RAMSEY R(3,3,3) @ N={N_NODES} 💎")
print(f"Edges: {len(EDGES)}, Triangles: {len(TRIANGLES)}")
print("60-Second Timeout Engaged.")

# BAHA Setup
opt = pybaha.Optimizer(energy_fn, sampler, neighbors)
config = pybaha.Config()
config.beta_steps = 100000 
config.beta_end = 50.0
config.timeout_ms = 60000.0 # 60 seconds
config.verbose = False
config.max_branches = 8

result = opt.optimize(config)

print(f"\nFinal Violations: {result.best_energy}")
print(f"Timeout Hit: {result.timeout_reached}")
print(f"Time: {result.time_ms/1000.0:.2f}s")
print(f"\nBEST STATE (Edge Coloring):")
print(result.best_state)

print(f"\nFinal Violations: {result.best_energy}")
print(f"Fractures: {result.fractures_detected}, Jumps: {result.branch_jumps}")
print(f"Time: {result.time_ms/1000.0:.2f}s")

if result.best_energy == 0:
    print("🏆 SUCCESS! Ground state reached.")
else:
    print("❌ Near-miss. Energy exists.")
