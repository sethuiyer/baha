#!/usr/bin/env python3
"""Test AdaptiveOptimizer on 100-Queens - should auto-select BranchAware"""
import sys
import os
import random

sys.path.append(os.getcwd())
import pybaha

N = 100

def energy(state):
    """Count queen conflicts"""
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
    """Random permutation (guarantees row uniqueness)"""
    state = list(range(N))
    random.shuffle(state)
    return state

def neighbors(state):
    """Swap two random columns"""
    nbrs = []
    for _ in range(64):
        i, j = random.sample(range(N), 2)
        nbr = list(state)
        nbr[i], nbr[j] = nbr[j], nbr[i]
        nbrs.append(nbr)
    return nbrs

print("=" * 60)
print(f"🔄 AdaptiveOptimizer: {N}-Queens")
print("=" * 60)
print("Testing auto-switching based on fracture density...")
print("(density > 0.3 → BranchAware, else Zeta)")
print()

# Create AdaptiveOptimizer (no continuous functions = will fallback to BranchAware)
opt = pybaha.AdaptiveOptimizer(energy, sampler, neighbors)

config = pybaha.AdaptiveConfig()
config.fracture_threshold = 0.3
config.probe_steps = 100
config.probe_samples = 10
config.ba_beta_steps = 1000
config.ba_beta_end = 20.0
config.ba_samples_per_beta = 64
config.ba_max_branches = 8
config.verbose = True

result = opt.optimize(config)

print()
print("=" * 60)
print(f"⚡ Results:")
print(f"   Best Energy: {result.best_energy}")
print(f"   Time: {result.time_ms:.2f} ms ({result.time_ms/1000:.2f}s)")
print(f"   Steps: {result.steps_taken}")
print(f"   Fracture Density: {result.fracture_density:.3f}")
print(f"   Used BranchAware: {result.used_branch_aware}")
print(f"   Timeout: {result.timeout_reached}")

if result.best_energy == 0:
    print("✅ PERFECT SOLUTION FOUND!")
else:
    print(f"⚠️ Best found has {int(result.best_energy)} conflicts")
print("=" * 60)
