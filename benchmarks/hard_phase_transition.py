#!/usr/bin/env python3
"""
Hard Phase Transition Problems - Testing BAHA's Limits

These problems have clear phase transitions but are designed to be 
challenging even for branch-aware optimization.
"""

import pybaha
import random
import time
import math

random.seed(42)

print("=" * 70)
print("🔥 Hard Phase Transition Problems - BAHA Stress Test")
print("=" * 70)
print()

results = []

def run_test(name, energy_fn, sampler_fn, neighbors_fn, timeout_ms=10000, runs=5):
    """Run a test and collect detailed results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    all_results = []
    for run_idx in range(runs):
        random.seed(42 + run_idx * 17)
        
        opt = pybaha.AdaptiveOptimizer(energy_fn, sampler_fn, neighbors_fn)
        config = pybaha.AdaptiveConfig()
        config.timeout_ms = timeout_ms
        
        start = time.time()
        result = opt.optimize(config)
        elapsed = (time.time() - start) * 1000
        
        all_results.append({
            'energy': result.best_energy,
            'density': result.fracture_density,
            'time_ms': elapsed,
        })
        print(f"  Run {run_idx+1}: E={result.best_energy:.2f}, ρ={result.fracture_density:.2f}, t={elapsed:.0f}ms")
    
    avg_energy = sum(r['energy'] for r in all_results) / runs
    avg_density = sum(r['density'] for r in all_results) / runs
    best_energy = min(r['energy'] for r in all_results)
    
    record = {
        'name': name,
        'avg_energy': avg_energy,
        'best_energy': best_energy,
        'avg_density': avg_density,
        'runs': runs,
    }
    results.append(record)
    print(f"\n  Summary: Avg E={avg_energy:.2f}, Best E={best_energy:.2f}, Avg ρ={avg_density:.2f}")
    return record

# ============================================================================
# 1. PLANTED 3-SAT (α=4.26, N=100)
# 
# A 3-SAT instance with a known satisfying assignment (planted).
# The planted solution creates a "hidden" basin that's hard to find.
# ============================================================================
print("\n" + "="*70)
print("1. PLANTED 3-SAT (α=4.26, N=100) - Hidden Solution Basin")
print("="*70)

N_SAT = 100
ALPHA = 4.26
M_SAT = int(N_SAT * ALPHA)

# Plant a solution
planted_solution = [random.randint(0, 1) for _ in range(N_SAT)]

# Generate clauses that are satisfied by the planted solution
sat_clauses = []
random.seed(12345)
for _ in range(M_SAT):
    # Pick 3 random variables
    vars = random.sample(range(N_SAT), 3)
    # Generate signs such that at least one literal is satisfied by planted
    signs = []
    for v in vars:
        if random.random() < 0.5:
            # Make this literal satisfied by planted
            signs.append(1 if planted_solution[v] == 1 else -1)
        else:
            signs.append(random.choice([1, -1]))
    
    # Ensure at least one literal is satisfied
    satisfied = False
    for v, s in zip(vars, signs):
        val = planted_solution[v] if s == 1 else (1 - planted_solution[v])
        if val == 1:
            satisfied = True
            break
    if not satisfied:
        # Fix by flipping one sign
        idx = random.randint(0, 2)
        signs[idx] = 1 if planted_solution[vars[idx]] == 1 else -1
    
    sat_clauses.append(list(zip(vars, signs)))

def planted_sat_energy(state):
    unsat = 0
    for clause in sat_clauses:
        satisfied = False
        for var, sign in clause:
            val = state[var] if sign == 1 else (1 - state[var])
            if val == 1:
                satisfied = True
                break
        if not satisfied:
            unsat += 1
    return float(unsat)

def planted_sat_sampler():
    return [random.randint(0, 1) for _ in range(N_SAT)]

def planted_sat_neighbors(state):
    nbrs = []
    for _ in range(50):
        s = list(state)
        s[random.randint(0, N_SAT-1)] ^= 1
        nbrs.append(s)
    return nbrs

# Verify planted solution works
print(f"Planted solution energy: {planted_sat_energy(planted_solution)}")
run_test("Planted 3-SAT (N=100, α=4.26)", planted_sat_energy, planted_sat_sampler, 
         planted_sat_neighbors, timeout_ms=15000, runs=5)

# ============================================================================
# 2. FRUSTRATED SPIN GLASS (2D, N=10x10)
#
# Spin glass with random ±1 couplings. Has exponentially many local minima
# and clear phase transitions, but ground state is NP-hard to find.
# ============================================================================
print("\n" + "="*70)
print("2. FRUSTRATED 2D SPIN GLASS (10x10) - Exponential Local Minima")
print("="*70)

GRID_SIZE = 10
N_SPINS = GRID_SIZE * GRID_SIZE

# Random ±1 couplings (horizontal and vertical)
random.seed(54321)
h_couplings = {}  # horizontal
v_couplings = {}  # vertical

for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        idx = i * GRID_SIZE + j
        # Horizontal coupling
        if j < GRID_SIZE - 1:
            h_couplings[(idx, idx + 1)] = random.choice([-1, 1])
        # Vertical coupling
        if i < GRID_SIZE - 1:
            v_couplings[(idx, idx + GRID_SIZE)] = random.choice([-1, 1])

def spin_glass_energy(state):
    # Convert 0/1 to -1/+1
    spins = [2*s - 1 for s in state]
    energy = 0.0
    
    for (i, j), J in h_couplings.items():
        energy -= J * spins[i] * spins[j]
    for (i, j), J in v_couplings.items():
        energy -= J * spins[i] * spins[j]
    
    return energy

def spin_glass_sampler():
    return [random.randint(0, 1) for _ in range(N_SPINS)]

def spin_glass_neighbors(state):
    nbrs = []
    for _ in range(50):
        s = list(state)
        # Single spin flip
        s[random.randint(0, N_SPINS-1)] ^= 1
        nbrs.append(s)
    return nbrs

run_test("2D Spin Glass (10x10)", spin_glass_energy, spin_glass_sampler,
         spin_glass_neighbors, timeout_ms=15000, runs=5)

# ============================================================================
# 3. RANDOM K-COLORING AT THRESHOLD (K=3, V=50, high edge density)
#
# Graph coloring near the colorability threshold. Has sharp phase transition
# but solutions become exponentially rare.
# ============================================================================
print("\n" + "="*70)
print("3. RANDOM 3-COLORING AT THRESHOLD (V=50, E=130)")
print("="*70)

V_COL = 50
E_COL = 130  # Near threshold for 3-colorability
K_COL = 3

random.seed(99999)
col_edges = []
while len(col_edges) < E_COL:
    u, v = random.sample(range(V_COL), 2)
    if (u, v) not in col_edges and (v, u) not in col_edges:
        col_edges.append((u, v))

def coloring_energy(state):
    conflicts = sum(1 for u, v in col_edges if state[u] == state[v])
    return float(conflicts)

def coloring_sampler():
    return [random.randint(0, K_COL-1) for _ in range(V_COL)]

def coloring_neighbors(state):
    nbrs = []
    for _ in range(50):
        s = list(state)
        s[random.randint(0, V_COL-1)] = random.randint(0, K_COL-1)
        nbrs.append(s)
    return nbrs

run_test("3-Coloring (V=50, E=130)", coloring_energy, coloring_sampler,
         coloring_neighbors, timeout_ms=15000, runs=5)

# ============================================================================
# 4. NUMBER PARTITIONING (Large integers)
#
# Partition a set of large random integers into two subsets of equal sum.
# Has a sharp phase transition and is weakly NP-complete.
# ============================================================================
print("\n" + "="*70)
print("4. NUMBER PARTITIONING (N=40, large integers)")
print("="*70)

N_PART = 40
random.seed(77777)
# Large integers make the problem harder
part_numbers = [random.randint(10**6, 10**7) for _ in range(N_PART)]
total_sum = sum(part_numbers)
print(f"Total sum: {total_sum}, Target per subset: {total_sum // 2}")

def partition_energy(state):
    s0 = sum(n for i, n in enumerate(part_numbers) if state[i] == 0)
    s1 = total_sum - s0
    return float(abs(s0 - s1))

def partition_sampler():
    return [random.randint(0, 1) for _ in range(N_PART)]

def partition_neighbors(state):
    nbrs = []
    for _ in range(50):
        s = list(state)
        s[random.randint(0, N_PART-1)] ^= 1
        nbrs.append(s)
    return nbrs

run_test("Number Partitioning (N=40, large)", partition_energy, partition_sampler,
         partition_neighbors, timeout_ms=15000, runs=5)

# ============================================================================
# 5. RANDOM MAX-2-SAT (dense, N=80, α=3.0)
#
# Max-2-SAT at high density. Has continuous phase transition (different
# from 3-SAT's sharp transition), testing if BAHA's fracture detection
# works for "softer" transitions.
# ============================================================================
print("\n" + "="*70)
print("5. DENSE MAX-2-SAT (N=80, α=3.0)")
print("="*70)

N_2SAT = 80
ALPHA_2SAT = 3.0
M_2SAT = int(N_2SAT * ALPHA_2SAT)

random.seed(88888)
sat2_clauses = []
for _ in range(M_2SAT):
    vars = random.sample(range(N_2SAT), 2)
    signs = [random.choice([1, -1]) for _ in range(2)]
    sat2_clauses.append(list(zip(vars, signs)))

def max2sat_energy(state):
    unsat = 0
    for clause in sat2_clauses:
        satisfied = False
        for var, sign in clause:
            val = state[var] if sign == 1 else (1 - state[var])
            if val == 1:
                satisfied = True
                break
        if not satisfied:
            unsat += 1
    return float(unsat)

def max2sat_sampler():
    return [random.randint(0, 1) for _ in range(N_2SAT)]

def max2sat_neighbors(state):
    nbrs = []
    for _ in range(50):
        s = list(state)
        s[random.randint(0, N_2SAT-1)] ^= 1
        nbrs.append(s)
    return nbrs

run_test("Dense Max-2-SAT (N=80, α=3.0)", max2sat_energy, max2sat_sampler,
         max2sat_neighbors, timeout_ms=15000, runs=5)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("📊 HARD PHASE TRANSITION SUMMARY")
print("="*70)
print(f"{'Problem':<40} {'Avg E':>10} {'Best E':>10} {'Avg ρ':>8}")
print("-"*70)
for r in results:
    print(f"{r['name']:<40} {r['avg_energy']:>10.2f} {r['best_energy']:>10.2f} {r['avg_density']:>8.2f}")
print("-"*70)

# Assess performance
print("\n🎯 Assessment:")
for r in results:
    if r['best_energy'] <= 0.001:
        status = "✅ SOLVED"
    elif r['best_energy'] < 5:
        status = "⚠️ CLOSE"
    else:
        status = "❌ STRUGGLED"
    print(f"  {r['name']}: {status} (best E={r['best_energy']:.2f})")

print("\n" + "="*70)
