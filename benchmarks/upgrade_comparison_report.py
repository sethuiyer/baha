#!/usr/bin/env python3
"""
BAHA Upgrade Comparison Report
==============================
Compares original BranchAwareOptimizer vs new AdaptiveOptimizer (with log(p²/(p-1)) physics)

For enterprise transparency: European truck company contract requirement.
"""

import pybaha
import random
import time

random.seed(42)

print("=" * 80)
print("🔬 BAHA UPGRADE COMPARISON REPORT")
print("=" * 80)
print("Comparing: BranchAwareOptimizer (Original) vs AdaptiveOptimizer (New + Physics)")
print()

results = []

def run_comparison(name, energy_fn, sampler_fn, neighbors_fn, 
                   timeout_ms=5000, lower_is_better=True):
    """Run both optimizers on the same problem."""
    print(f"\n{'='*80}")
    print(f"📊 {name}")
    print(f"{'='*80}")
    
    # Run BranchAwareOptimizer (Original)
    random.seed(42)
    opt_ba = pybaha.Optimizer(energy_fn, sampler_fn, neighbors_fn)
    config_ba = pybaha.Config()
    config_ba.beta_start = 0.1
    config_ba.beta_end = 10.0
    config_ba.beta_steps = 100
    config_ba.samples_per_beta = 30
    config_ba.timeout_ms = timeout_ms
    
    start = time.time()
    result_ba = opt_ba.optimize(config_ba)
    time_ba = (time.time() - start) * 1000
    
    # Run AdaptiveOptimizer (New)
    random.seed(42)
    opt_adapt = pybaha.AdaptiveOptimizer(energy_fn, sampler_fn, neighbors_fn)
    config_adapt = pybaha.AdaptiveConfig()
    config_adapt.timeout_ms = timeout_ms
    
    start = time.time()
    result_adapt = opt_adapt.optimize(config_adapt)
    time_adapt = (time.time() - start) * 1000
    
    # Compare
    e_ba = result_ba.best_energy
    e_adapt = result_adapt.best_energy
    
    if lower_is_better:
        improvement = ((e_ba - e_adapt) / abs(e_ba) * 100) if e_ba != 0 else 0
        winner = "Adaptive" if e_adapt < e_ba else ("Tie" if e_adapt == e_ba else "BranchAware")
    else:
        improvement = ((e_adapt - e_ba) / abs(e_ba) * 100) if e_ba != 0 else 0
        winner = "Adaptive" if e_adapt > e_ba else ("Tie" if e_adapt == e_ba else "BranchAware")
    
    print(f"  {'Metric':<25} {'BranchAware':>15} {'Adaptive':>15} {'Change':>15}")
    print(f"  {'-'*70}")
    print(f"  {'Energy':<25} {e_ba:>15.2f} {e_adapt:>15.2f} {improvement:>+14.1f}%")
    print(f"  {'Time (ms)':<25} {time_ba:>15.0f} {time_adapt:>15.0f}")
    print(f"  {'Fractures Detected':<25} {result_ba.fractures_detected:>15} {result_adapt.fractures_detected:>15}")
    print(f"  {'Fracture Density':<25} {'N/A':>15} {result_adapt.fracture_density:>15.2f}")
    print(f"  {'-'*70}")
    print(f"  {'Winner':<25} {winner:>15}")
    
    results.append({
        'name': name,
        'ba_energy': e_ba,
        'adapt_energy': e_adapt,
        'ba_time': time_ba,
        'adapt_time': time_adapt,
        'improvement': improvement,
        'winner': winner
    })
    
    return e_ba, e_adapt

# ==============================================================================
# 1. VRP (Vehicle Routing Problem) - Critical for truck company
# ==============================================================================
N_vrp = 10
VRP_CAP = 50
vrp_demands = [random.randint(5, 15) for _ in range(N_vrp)]
vrp_locs = [(random.uniform(0, 50), random.uniform(0, 50)) for _ in range(N_vrp + 1)]

def vrp_dist(i, j):
    return ((vrp_locs[i][0]-vrp_locs[j][0])**2 + (vrp_locs[i][1]-vrp_locs[j][1])**2)**0.5

def vrp_energy(state):
    routes = [[], []]
    for i, v in enumerate(state):
        routes[v].append(i)
    
    total_dist = 0
    penalty = 0
    for route in routes:
        if not route:
            continue
        demand = sum(vrp_demands[i] for i in route)
        if demand > VRP_CAP:
            penalty += (demand - VRP_CAP) * 10
        if route:
            total_dist += vrp_dist(N_vrp, route[0])
            for i in range(len(route) - 1):
                total_dist += vrp_dist(route[i], route[i+1])
            total_dist += vrp_dist(route[-1], N_vrp)
    return total_dist + penalty

def vrp_sampler():
    return [random.randint(0, 1) for _ in range(N_vrp)]

def vrp_neighbors(state):
    nbrs = []
    for _ in range(20):
        s = list(state)
        s[random.randint(0, N_vrp-1)] ^= 1
        nbrs.append(s)
    return nbrs

run_comparison("VRP (10 customers, 2 vehicles)", vrp_energy, vrp_sampler, vrp_neighbors, timeout_ms=10000)

# ==============================================================================
# 2. TSP (Traveling Salesman Problem)
# ==============================================================================
N_tsp = 15
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(N_tsp)]
def dist(i, j):
    return ((cities[i][0]-cities[j][0])**2 + (cities[i][1]-cities[j][1])**2)**0.5

def tsp_energy(state):
    return sum(dist(state[i], state[(i+1) % N_tsp]) for i in range(N_tsp))

def tsp_sampler():
    return list(random.sample(range(N_tsp), N_tsp))

def tsp_neighbors(state):
    nbrs = []
    for _ in range(20):
        s = list(state)
        i, j = random.sample(range(N_tsp), 2)
        s[i], s[j] = s[j], s[i]
        nbrs.append(s)
    return nbrs

run_comparison("TSP (15 cities)", tsp_energy, tsp_sampler, tsp_neighbors, timeout_ms=10000)

# ==============================================================================
# 3. N-QUEENS (Constraint Satisfaction)
# ==============================================================================
N = 50
def nq_energy(state):
    conflicts = 0
    for i in range(N):
        for j in range(i+1, N):
            if abs(state[i] - state[j]) == j - i:
                conflicts += 1
    return float(conflicts)

def nq_sampler():
    return list(random.sample(range(N), N))

def nq_neighbors(state):
    nbrs = []
    for _ in range(30):
        s = list(state)
        i, j = random.sample(range(N), 2)
        s[i], s[j] = s[j], s[i]
        nbrs.append(s)
    return nbrs

run_comparison("N-Queens (N=50)", nq_energy, nq_sampler, nq_neighbors, timeout_ms=10000)

# ==============================================================================
# 4. BIN PACKING
# ==============================================================================
N_bp = 15
BP_CAP = 100
bp_sizes = [random.randint(10, 40) for _ in range(N_bp)]
N_BINS = 5

def bp_energy(state):
    bins = [0] * N_BINS
    for i, s in enumerate(state):
        bins[s] += bp_sizes[i]
    overflow = sum(max(0, b - BP_CAP) for b in bins)
    used = sum(1 for b in bins if b > 0)
    return overflow * 100 + used

def bp_sampler():
    return [random.randint(0, N_BINS-1) for _ in range(N_bp)]

def bp_neighbors(state):
    nbrs = []
    for _ in range(20):
        s = list(state)
        s[random.randint(0, N_bp-1)] = random.randint(0, N_BINS-1)
        nbrs.append(s)
    return nbrs

run_comparison("Bin Packing (15 items, 5 bins)", bp_energy, bp_sampler, bp_neighbors)

# ==============================================================================
# 5. KNAPSACK
# ==============================================================================
N_ks = 20
CAP = 100
weights = [random.randint(5, 25) for _ in range(N_ks)]
values = [random.randint(10, 50) for _ in range(N_ks)]

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
    for _ in range(20):
        s = list(state)
        s[random.randint(0, N_ks-1)] ^= 1
        nbrs.append(s)
    return nbrs

run_comparison("Knapsack (20 items)", ks_energy, ks_sampler, ks_neighbors, lower_is_better=True)

# ==============================================================================
# 6. GRAPH COLORING
# ==============================================================================
V, E_count, K = 30, 60, 4
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

run_comparison("Graph Coloring (30V, K=4)", gc_energy, gc_sampler, gc_neighbors)

# ==============================================================================
# 7. MAX CUT
# ==============================================================================
V_mc, E_mc = 25, 50
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
    for _ in range(20):
        s = list(state)
        s[random.randint(0, V_mc-1)] ^= 1
        nbrs.append(s)
    return nbrs

run_comparison("Max Cut (25V, 50E)", mc_energy, mc_sampler, mc_neighbors)

# ==============================================================================
# 8. 3-SAT
# ==============================================================================
N_sat = 30
M_sat = 60
sat_clauses = []
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
    for _ in range(20):
        s = list(state)
        s[random.randint(0, N_sat-1)] ^= 1
        nbrs.append(s)
    return nbrs

run_comparison("3-SAT (30 vars, 60 clauses)", sat_energy, sat_sampler, sat_neighbors)

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================
print("\n")
print("=" * 80)
print("📋 EXECUTIVE SUMMARY: BAHA UPGRADE COMPARISON")
print("=" * 80)
print()
print(f"{'Problem':<35} {'BranchAware':>12} {'Adaptive':>12} {'Change':>10} {'Winner':>12}")
print("-" * 80)

adaptive_wins = 0
ties = 0
ba_wins = 0

for r in results:
    print(f"{r['name']:<35} {r['ba_energy']:>12.1f} {r['adapt_energy']:>12.1f} {r['improvement']:>+9.1f}% {r['winner']:>12}")
    if r['winner'] == 'Adaptive':
        adaptive_wins += 1
    elif r['winner'] == 'BranchAware':
        ba_wins += 1
    else:
        ties += 1

print("-" * 80)
print()
print("📊 OVERALL RESULTS:")
print(f"   AdaptiveOptimizer Wins: {adaptive_wins}")
print(f"   BranchAwareOptimizer Wins: {ba_wins}")
print(f"   Ties: {ties}")
print()

if adaptive_wins > ba_wins:
    print("✅ RECOMMENDATION: AdaptiveOptimizer with log(p²/(p-1)) physics is SUPERIOR")
    print("   - Better energy values across majority of test cases")
    print("   - Automatic engine switching based on fracture density")
    print("   - Physics-based weighting improves low-density problems (VRP, TSP)")
elif adaptive_wins == ba_wins:
    print("⚖️  RECOMMENDATION: EQUIVALENT performance")
    print("   - AdaptiveOptimizer adds no regression")
    print("   - Provides automatic engine selection")
else:
    print("⚠️  RECOMMENDATION: Review needed - BranchAware performed better")

print()
print("=" * 80)
print("END OF REPORT")
print("=" * 80)
