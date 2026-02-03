#!/usr/bin/env python3
"""Detailed BAHA experiments for article data generation."""

import pybaha
import random
import time
import json
import sys

random.seed(42)

results = []

def run_detailed(name, energy_fn, sampler_fn, neighbors_fn, timeout_ms=5000, runs=5):
    """Run multiple trials and collect detailed metrics."""
    all_results = []
    
    for run_idx in range(runs):
        random.seed(42 + run_idx)
        
        opt = pybaha.AdaptiveOptimizer(energy_fn, sampler_fn, neighbors_fn)
        config = pybaha.AdaptiveConfig()
        config.timeout_ms = timeout_ms
        
        start = time.time()
        result = opt.optimize(config)
        elapsed_ms = (time.time() - start) * 1000
        
        all_results.append({
            'energy': result.best_energy,
            'fracture_density': result.fracture_density,
            'used_branch_aware': result.used_branch_aware,
            'time_ms': elapsed_ms,
        })
    
    avg_energy = sum(r['energy'] for r in all_results) / runs
    avg_density = sum(r['fracture_density'] for r in all_results) / runs
    avg_time = sum(r['time_ms'] for r in all_results) / runs
    branch_aware_rate = sum(1 for r in all_results if r['used_branch_aware']) / runs
    
    record = {
        'name': name,
        'avg_energy': avg_energy,
        'avg_fracture_density': avg_density,
        'avg_time_ms': avg_time,
        'branch_aware_rate': branch_aware_rate,
        'runs': runs,
        'all_results': all_results,
    }
    results.append(record)
    print(f"{name}: E={avg_energy:.2f}, ρ={avg_density:.2f}, BA_rate={branch_aware_rate:.0%}, t={avg_time:.0f}ms")
    return record

print("=" * 70)
print("🔬 Detailed BAHA Experiments for Article Data")
print("=" * 70)
print()

# ============================================================================
# 1. N-QUEENS scaling study
# ============================================================================
print("=== N-Queens Scaling ===")
for N in [8, 16, 32, 50, 100]:
    def make_nq(n):
        def energy(state):
            conflicts = 0
            for i in range(n):
                for j in range(i+1, n):
                    if abs(state[i] - state[j]) == j - i:
                        conflicts += 1
            return float(conflicts)
        def sampler():
            return list(random.sample(range(n), n))
        def neighbors(state):
            nbrs = []
            for _ in range(min(30, n)):
                s = list(state)
                i, j = random.sample(range(n), 2)
                s[i], s[j] = s[j], s[i]
                nbrs.append(s)
            return nbrs
        return energy, sampler, neighbors
    
    e, s, n = make_nq(N)
    run_detailed(f"N-Queens (N={N})", e, s, n, timeout_ms=min(5000 + N*50, 15000), runs=3)

print()

# ============================================================================
# 2. 3-SAT with varying alpha
# ============================================================================
print("=== 3-SAT α Sensitivity ===")
for alpha in [3.8, 4.0, 4.2, 4.26, 4.4, 4.6]:
    N_sat = 50
    M_sat = int(N_sat * alpha)
    
    sat_clauses = []
    random.seed(int(alpha * 100))
    for _ in range(M_sat):
        vars = random.sample(range(N_sat), 3)
        signs = [random.choice([1, -1]) for _ in range(3)]
        sat_clauses.append(list(zip(vars, signs)))
    
    def sat_energy(state):
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
    
    def sat_sampler():
        return [random.randint(0, 1) for _ in range(N_sat)]
    
    def sat_neighbors(state):
        nbrs = []
        for _ in range(30):
            s = list(state)
            s[random.randint(0, N_sat-1)] ^= 1
            nbrs.append(s)
        return nbrs
    
    run_detailed(f"3-SAT (α={alpha})", sat_energy, sat_sampler, sat_neighbors, 
                 timeout_ms=8000, runs=5)

print()

# ============================================================================
# 3. TSP scaling
# ============================================================================
print("=== TSP Scaling ===")
for N_tsp in [10, 15, 20, 25]:
    random.seed(100 + N_tsp)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(N_tsp)]
    
    def make_tsp(n, c):
        def dist(i, j):
            return ((c[i][0]-c[j][0])**2 + (c[i][1]-c[j][1])**2)**0.5
        def energy(state):
            return sum(dist(state[i], state[(i+1) % n]) for i in range(n))
        def sampler():
            return list(random.sample(range(n), n))
        def neighbors(state):
            nbrs = []
            for _ in range(30):
                s = list(state)
                i, j = random.sample(range(n), 2)
                s[i], s[j] = s[j], s[i]
                nbrs.append(s)
            return nbrs
        return energy, sampler, neighbors
    
    e, s, n = make_tsp(N_tsp, cities)
    run_detailed(f"TSP (N={N_tsp})", e, s, n, timeout_ms=10000, runs=3)

print()

# ============================================================================
# 4. Problem category comparison
# ============================================================================
print("=== Problem Category Comparison ===")

# Graph Coloring
V, E_count, K = 30, 60, 4
random.seed(200)
edges = []
while len(edges) < E_count:
    u, v = random.sample(range(V), 2)
    if (u, v) not in edges and (v, u) not in edges:
        edges.append((u, v))

def gc_energy(state):
    return float(sum(1 for u, v in edges if state[u] == state[v]))
def gc_sampler():
    return [random.randint(0, K-1) for _ in range(V)]
def gc_neighbors(state):
    nbrs = []
    for _ in range(30):
        s = list(state)
        s[random.randint(0, V-1)] = random.randint(0, K-1)
        nbrs.append(s)
    return nbrs

run_detailed("Graph Coloring (30V)", gc_energy, gc_sampler, gc_neighbors)

# Max Cut
V_mc, E_mc = 30, 80
random.seed(201)
edges_mc = []
while len(edges_mc) < E_mc:
    u, v = random.sample(range(V_mc), 2)
    if (u, v) not in edges_mc and (v, u) not in edges_mc:
        edges_mc.append((u, v))

def mc_energy(state):
    return -float(sum(1 for u, v in edges_mc if state[u] != state[v]))
def mc_sampler():
    return [random.randint(0, 1) for _ in range(V_mc)]
def mc_neighbors(state):
    nbrs = []
    for _ in range(30):
        s = list(state)
        s[random.randint(0, V_mc-1)] ^= 1
        nbrs.append(s)
    return nbrs

run_detailed("Max Cut (30V)", mc_energy, mc_sampler, mc_neighbors)

# Knapsack
N_ks = 30
CAP = 150
random.seed(202)
weights = [random.randint(5, 30) for _ in range(N_ks)]
values = [random.randint(10, 60) for _ in range(N_ks)]

def ks_energy(state):
    tw = sum(w for i, w in enumerate(weights) if state[i] == 1)
    tv = sum(v for i, v in enumerate(values) if state[i] == 1)
    return -float(tv) if tw <= CAP else 10000 + tw
def ks_sampler():
    s = [0] * N_ks
    rem = CAP
    for i in random.sample(range(N_ks), N_ks):
        if weights[i] <= rem:
            s[i] = 1
            rem -= weights[i]
    return s
def ks_neighbors(state):
    nbrs = []
    for _ in range(30):
        s = list(state)
        s[random.randint(0, N_ks-1)] ^= 1
        nbrs.append(s)
    return nbrs

run_detailed("Knapsack (30 items)", ks_energy, ks_sampler, ks_neighbors)

# Number Partitioning
N_part = 30
random.seed(203)
part_nums = [random.randint(100, 1000) for _ in range(N_part)]
total = sum(part_nums)

def part_energy(state):
    s1 = sum(n for i, n in enumerate(part_nums) if state[i] == 0)
    return abs(s1 - (total - s1))
def part_sampler():
    return [random.randint(0, 1) for _ in range(N_part)]
def part_neighbors(state):
    nbrs = []
    for _ in range(30):
        s = list(state)
        s[random.randint(0, N_part-1)] ^= 1
        nbrs.append(s)
    return nbrs

run_detailed("Number Partitioning (30)", part_energy, part_sampler, part_neighbors)

# LABS
N_labs = 25

def labs_energy(state):
    seq = [2*s - 1 for s in state]
    E = 0
    for k in range(1, N_labs):
        c_k = sum(seq[i] * seq[i+k] for i in range(N_labs - k))
        E += c_k * c_k
    return float(E)
def labs_sampler():
    return [random.randint(0, 1) for _ in range(N_labs)]
def labs_neighbors(state):
    nbrs = []
    for _ in range(30):
        s = list(state)
        s[random.randint(0, N_labs-1)] ^= 1
        nbrs.append(s)
    return nbrs

run_detailed("LABS (N=25)", labs_energy, labs_sampler, labs_neighbors, timeout_ms=10000)

print()

# ============================================================================
# Output JSON for data processing
# ============================================================================
print("=" * 70)
print("📊 Experiment Summary (JSON)")
print("=" * 70)
print(json.dumps(results, indent=2))
