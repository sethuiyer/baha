import sys
import os
sys.path.append(os.getcwd())
import pybaha
import random

items = [10, 20, 30, 40, 50, 60, 70, 80] # Weights
BIN_CAPACITY = 100

def energy(state):
    bin_totals = {}
    for i, bin_idx in enumerate(state):
        bin_totals[bin_idx] = bin_totals.get(bin_idx, 0) + items[i]
    
    violations = 0
    for total in bin_totals.values():
        if total > BIN_CAPACITY:
            violations += (total - BIN_CAPACITY)
    
    num_bins = len(bin_totals)
    return float(violations * 1000 + num_bins)

def sampler():
    return [random.randint(0, len(items)-1) for _ in range(len(items))]

def neighbors(state):
    nbrs = []
    for i in range(len(state)):
        for bin_idx in range(len(items)):
            if bin_idx != state[i]:
                nbr = list(state)
                nbr[i] = bin_idx
                nbrs.append(nbr)
    return nbrs

print("📦 RUNNING BIN PACKING TUTORIAL 📦")
opt = pybaha.Optimizer(energy, sampler, neighbors)
config = pybaha.Config()
config.verbose = True
config.beta_steps = 1000
result = opt.optimize(config)

print(f"\nFinal Bins Used: {int(result.best_energy % 1000)}")
print(f"Energy: {result.best_energy}")
print(f"Fractures: {result.fractures_detected}, Jumps: {result.branch_jumps}")
