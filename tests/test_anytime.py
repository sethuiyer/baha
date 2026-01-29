import sys
import os
sys.path.append(os.getcwd())
import pybaha
import random
import itertools
import time

# Problem: Simplified Ramsey R(3,3,3) @ N=16 (known to take >30s)
N_NODES = 16
EDGES = list(itertools.combinations(range(N_NODES), 2))
TRIANGLES = list(itertools.combinations(range(N_NODES), 3))

TRI_EDGE_IDXS = []
for u, v, w in TRIANGLES:
    e1 = EDGES.index(tuple(sorted((u, v))))
    e2 = EDGES.index(tuple(sorted((v, w))))
    e3 = EDGES.index(tuple(sorted((w, u))))
    TRI_EDGE_IDXS.append((e1, e2, e3))

def energy(state_list):
    violations = 0
    for e1, e2, e3 in TRI_EDGE_IDXS:
        if state_list[e1] == state_list[e2] == state_list[e3]:
            violations += 1
    return float(violations)

def sampler():
    return [random.randint(0, 2) for _ in range(len(EDGES))]

def neighbors(state):
    nbrs = []
    for _ in range(10):
        edge_idx = random.randrange(len(EDGES))
        nbr = list(state)
        nbr[edge_idx] = random.randint(0, 2)
        nbrs.append(nbr)
    return nbrs

print("⏱️ TESTING ANYTIME MECHANISM (Timeout = 2.0s) ⏱️")
opt = pybaha.Optimizer(energy, sampler, neighbors)
config = pybaha.Config()
config.timeout_ms = 2000.0  # 2 seconds
config.beta_steps = 1000000 # Way too many steps for 2s
config.verbose = True

start = time.time()
result = opt.optimize(config)
end = time.time()

print(f"\nElapsed (Python-side): {end - start:.4f}s")
print(f"BAHA Reported Time: {result.time_ms/1000.0:.4f}s")
print(f"Timeout Reached: {result.timeout_reached}")
print(f"Best Energy Found: {result.best_energy}")

if result.timeout_reached and (end - start) < 3.0:
    print("\n✅ ANYTIME MECHANISM VERIFIED: Solver exited early and returned best solution.")
else:
    print("\n❌ VERIFICATION FAILED.")
