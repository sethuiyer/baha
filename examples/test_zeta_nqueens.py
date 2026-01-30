#!/usr/bin/env python3
"""Test ZetaOptimizer on N-Queens problem"""
import sys
import os
import random

sys.path.append(os.getcwd())
import pybaha

N = 20

def discrete_energy(state):
    """Count queen conflicts - discrete energy function"""
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
    """Random initial state"""
    return list(range(N))

def neighbors(state):
    """Generate neighbor states by swapping two queens"""
    nbrs = []
    for i in range(N):
        for j in range(i+1, N):
            s = list(state)
            s[i], s[j] = s[j], s[i]
            nbrs.append(s)
    return nbrs

def encode(state):
    """Encode discrete state as continuous (normalize position to [0,1])"""
    return [float(x) / (N - 1) for x in state]

def decode(x):
    """Decode continuous to discrete (round to nearest integer)"""
    return [min(N-1, max(0, int(round(v * (N - 1))))) for v in x]

def continuous_energy(x, beta):
    """Continuous energy with quadratic penalty for conflicts"""
    # Softmax-like row/diagonal conflict modeling
    energy = 0.0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            # Row conflict (similar values)
            row_diff = abs(x[i] - x[j])
            energy += 1.0 / (row_diff + 0.01) * (1.0 / N)
            
            # Diagonal conflict
            diag_diff = abs(x[i] - x[j]) - abs(i - j) / (N - 1)
            energy += 1.0 / (abs(diag_diff) + 0.01) * (0.1 / N)
    return energy

def continuous_gradient(x, beta):
    """Gradient of the continuous energy"""
    grad = [0.0] * len(x)
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            # Gradient from row penalty
            row_diff = x[i] - x[j]
            sign = 1.0 if row_diff >= 0 else -1.0
            penalty = -1.0 / ((abs(row_diff) + 0.01) ** 2) * sign * (1.0 / N)
            grad[i] += penalty
            grad[j] -= penalty
    return grad

print("=" * 60)
print(f"🔮 ZetaOptimizer: N-Queens (N={N})")
print("=" * 60)

# Create ZetaOptimizer
opt = pybaha.ZetaOptimizer(
    discrete_energy, sampler, neighbors,
    encode, decode,
    continuous_energy, continuous_gradient
)

# Configure
config = pybaha.ZetaConfig()
config.beta_min = 0.3
config.beta_max = 2.0
config.period = 500
config.total_steps = 5000
config.polish_steps = 50
config.polish_samples = 10
config.learning_rate = 0.02
config.verbose = True
config.timeout_ms = 30000

# Run optimization
result = opt.optimize(config)

print("=" * 60)
print(f"⚡ Results:")
print(f"   Best Energy: {result.best_energy}")
print(f"   Time: {result.time_ms:.2f} ms")
print(f"   Steps: {result.steps_taken}")
print(f"   Peaks Harvested: {result.peaks_harvested}")
print(f"   Timeout: {result.timeout_reached}")

if result.best_energy == 0:
    print("✅ PERFECT SOLUTION FOUND!")
    # Visualize the board
    state = list(result.best_state)
    for row in range(N):
        line = ""
        for col in range(N):
            if state[col] == row:
                line += "Q "
            else:
                line += ". "
        print(line)
else:
    print(f"⚠️ Best found has {int(result.best_energy)} conflicts")
print("=" * 60)
