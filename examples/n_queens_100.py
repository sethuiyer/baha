import sys
import os
sys.path.append(os.getcwd())
import pybaha
import random

N = 100

def energy(state):
    # state is list where state[i] is row of queen in column i
    conflicts = 0
    row_counts = [0] * N
    diag1_counts = [0] * (2 * N)
    diag2_counts = [0] * (2 * N)
    
    for col, row in enumerate(state):
        row_counts[row] += 1
        diag1_counts[row + col] += 1
        diag2_counts[row - col + N] += 1
        
    for c in row_counts:
        if c > 1: conflicts += c * (c - 1) // 2
    for c in diag1_counts:
        if c > 1: conflicts += c * (c - 1) // 2
    for c in diag2_counts:
        if c > 1: conflicts += c * (c - 1) // 2
        
    return float(conflicts)

def sampler():
    # Start with a permutation (guarantees row uniqueness)
    state = list(range(N))
    random.shuffle(state)
    return state

def neighbors(state):
    nbrs = []
    # Swap two random columns (keeps it a permutation)
    for _ in range(64):
        i, j = random.sample(range(N), 2)
        nbr = list(state)
        nbr[i], nbr[j] = nbr[j], nbr[i]
        nbrs.append(nbr)
    return nbrs

print(f"👑 SOLVING {N}-QUEENS USING BAHA 👑")
print(f"Search Space: {N}^{N} (Permutation subset: {N}!)")

opt = pybaha.Optimizer(energy, sampler, neighbors)
config = pybaha.Config()
config.beta_steps = 1000
config.beta_end = 20.0
config.fracture_threshold = 1.3
config.verbose = True
config.max_branches = 8

result = opt.optimize(config)

print(f"\nFinal Conflicts: {result.best_energy}")
print(f"Fractures: {result.fractures_detected}, Jumps: {result.branch_jumps}")
print(f"Time: {result.time_ms/1000.0:.2f}s")

if result.best_energy == 0:
    print("✅ PERFECT SOLUTION FOUND FOR 100-QUEENS!")
else:
    print("❌ FAILED. TRYING AGAIN...")
