#!/usr/bin/env python3
"""Test AutoZetaOptimizer on N-Queens problem."""

import pybaha
import random

N = 20  # 20-Queens

def energy(state):
    """Count conflicts (diagonal attacks)."""
    conflicts = 0
    for i in range(N):
        for j in range(i + 1, N):
            if abs(state[i] - state[j]) == j - i:
                conflicts += 1
    return float(conflicts)

def sampler():
    """Random permutation."""
    return list(random.sample(range(N), N))

def neighbors(state):
    """Swap two random positions."""
    nbrs = []
    for _ in range(20):
        s = list(state)
        i, j = random.sample(range(N), 2)
        s[i], s[j] = s[j], s[i]
        nbrs.append(s)
    return nbrs

print("=" * 50)
print("AutoZetaOptimizer Test: {}-Queens".format(N))
print("=" * 50)

# Create optimizer with domain_size = N (each queen can be in rows 0..N-1)
opt = pybaha.AutoZetaOptimizer(energy, sampler, neighbors, N)

config = pybaha.AutoZetaConfig()
config.total_steps = 5000
config.period = 500
config.verbose = True
config.timeout_ms = 30000.0

print("Running AutoZetaOptimizer (30s timeout)...")
result = opt.optimize(config)

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"Conflicts: {int(result.best_energy)}")
print(f"Time: {result.time_ms:.0f}ms")
print(f"Peaks harvested: {result.peaks_harvested}")

if result.best_energy == 0:
    print("\n✅ SUCCESS!")
else:
    print(f"\n⚠️ {int(result.best_energy)} conflicts remaining")
