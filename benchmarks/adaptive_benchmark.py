#!/usr/bin/env python3
"""
AdaptiveOptimizer Benchmark Suite
=================================
Tests the AdaptiveOptimizer against multiple problem domains to verify
the fracture-based switching heuristic works correctly.

Expected behavior:
- High fracture density (>0.3) → BranchAwareOptimizer
- Low fracture density → ZetaOptimizer (if continuous functions available)
"""
import sys
import os
import random
import time

sys.path.append(os.getcwd())
import pybaha

# =============================================================================
# BENCHMARK 1: N-Queens (Permutation-based, high fracture)
# =============================================================================
def benchmark_nqueens(N=50):
    """N-Queens: Expect HIGH fracture density → BranchAware"""
    def energy(state):
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
        state = list(range(N))
        random.shuffle(state)
        return state

    def neighbors(state):
        nbrs = []
        for _ in range(32):
            i, j = random.sample(range(N), 2)
            nbr = list(state)
            nbr[i], nbr[j] = nbr[j], nbr[i]
            nbrs.append(nbr)
        return nbrs

    opt = pybaha.AdaptiveOptimizer(energy, sampler, neighbors)
    config = pybaha.AdaptiveConfig()
    config.probe_steps = 50
    config.ba_beta_steps = 500
    config.ba_beta_end = 20.0
    config.verbose = False
    
    result = opt.optimize(config)
    return {
        "name": f"N-Queens (N={N})",
        "energy": result.best_energy,
        "time_ms": result.time_ms,
        "fracture_density": result.fracture_density,
        "used_branch_aware": result.used_branch_aware,
        "success": result.best_energy == 0
    }


# =============================================================================
# BENCHMARK 2: Graph Coloring (3-colorable graph)
# =============================================================================
def benchmark_graph_coloring(V=30, E=60, K=3):
    """Graph Coloring: Expect HIGH fracture density → BranchAware"""
    # Generate random graph
    edges = set()
    while len(edges) < E:
        u, v = random.sample(range(V), 2)
        if u > v: u, v = v, u
        edges.add((u, v))
    edges = list(edges)
    
    def energy(state):
        conflicts = 0
        for u, v in edges:
            if state[u] == state[v]:
                conflicts += 1
        return float(conflicts)
    
    def sampler():
        return [random.randint(0, K-1) for _ in range(V)]
    
    def neighbors(state):
        nbrs = []
        for _ in range(16):
            nbr = list(state)
            v = random.randint(0, V-1)
            nbr[v] = random.randint(0, K-1)
            nbrs.append(nbr)
        return nbrs
    
    opt = pybaha.AdaptiveOptimizer(energy, sampler, neighbors)
    config = pybaha.AdaptiveConfig()
    config.probe_steps = 50
    config.ba_beta_steps = 500
    config.verbose = False
    
    result = opt.optimize(config)
    return {
        "name": f"Graph Coloring (V={V}, E={E}, K={K})",
        "energy": result.best_energy,
        "time_ms": result.time_ms,
        "fracture_density": result.fracture_density,
        "used_branch_aware": result.used_branch_aware,
        "success": result.best_energy == 0
    }


# =============================================================================
# BENCHMARK 3: Max-Cut (Graph partitioning)
# =============================================================================
def benchmark_max_cut(V=40, E=80):
    """Max-Cut: Expect HIGH fracture density → BranchAware"""
    edges = set()
    weights = {}
    while len(edges) < E:
        u, v = random.sample(range(V), 2)
        if u > v: u, v = v, u
        if (u, v) not in edges:
            edges.add((u, v))
            weights[(u, v)] = random.randint(1, 10)
    edges = list(edges)
    
    def energy(state):
        # Negative of cut value (we minimize)
        cut = 0
        for u, v in edges:
            if state[u] != state[v]:
                cut += weights[(u, v)]
        return -float(cut)
    
    def sampler():
        return [random.randint(0, 1) for _ in range(V)]
    
    def neighbors(state):
        nbrs = []
        for _ in range(16):
            nbr = list(state)
            v = random.randint(0, V-1)
            nbr[v] = 1 - nbr[v]
            nbrs.append(nbr)
        return nbrs
    
    opt = pybaha.AdaptiveOptimizer(energy, sampler, neighbors)
    config = pybaha.AdaptiveConfig()
    config.probe_steps = 50
    config.ba_beta_steps = 500
    config.verbose = False
    
    result = opt.optimize(config)
    return {
        "name": f"Max-Cut (V={V}, E={E})",
        "energy": result.best_energy,
        "time_ms": result.time_ms,
        "fracture_density": result.fracture_density,
        "used_branch_aware": result.used_branch_aware,
        "success": True  # Always succeeds, just want good cut
    }


# =============================================================================
# BENCHMARK 4: Knapsack (0/1 Knapsack)
# =============================================================================
def benchmark_knapsack(N=30, capacity_ratio=0.5):
    """Knapsack: Expect MEDIUM fracture density"""
    values = [random.randint(10, 100) for _ in range(N)]
    weights = [random.randint(5, 50) for _ in range(N)]
    capacity = int(sum(weights) * capacity_ratio)
    
    def energy(state):
        total_weight = sum(w for i, w in enumerate(weights) if state[i])
        total_value = sum(v for i, v in enumerate(values) if state[i])
        if total_weight > capacity:
            return float(total_weight - capacity) * 10  # Penalty
        return -float(total_value)  # Minimize negative value
    
    def sampler():
        return [random.randint(0, 1) for _ in range(N)]
    
    def neighbors(state):
        nbrs = []
        for _ in range(16):
            nbr = list(state)
            i = random.randint(0, N-1)
            nbr[i] = 1 - nbr[i]
            nbrs.append(nbr)
        return nbrs
    
    opt = pybaha.AdaptiveOptimizer(energy, sampler, neighbors)
    config = pybaha.AdaptiveConfig()
    config.probe_steps = 50
    config.ba_beta_steps = 500
    config.verbose = False
    
    result = opt.optimize(config)
    return {
        "name": f"Knapsack (N={N})",
        "energy": result.best_energy,
        "time_ms": result.time_ms,
        "fracture_density": result.fracture_density,
        "used_branch_aware": result.used_branch_aware,
        "success": result.best_energy < 0  # Found valid solution
    }


# =============================================================================
# BENCHMARK 5: Bin Packing
# =============================================================================
def benchmark_bin_packing(N=20, bin_capacity=100):
    """Bin Packing: Expect HIGH fracture density → BranchAware"""
    items = [random.randint(10, 40) for _ in range(N)]
    max_bins = N
    
    def energy(state):
        # state[i] = bin assignment for item i
        bins_used = [0] * max_bins
        for i, b in enumerate(state):
            bins_used[b] += items[i]
        
        # Penalty for overflow
        overflow = sum(max(0, b - bin_capacity) for b in bins_used)
        # Minimize bins used + overflow
        active_bins = len([b for b in bins_used if b > 0])
        return float(overflow * 100 + active_bins)
    
    def sampler():
        return [random.randint(0, N//2) for _ in range(N)]
    
    def neighbors(state):
        nbrs = []
        for _ in range(16):
            nbr = list(state)
            i = random.randint(0, N-1)
            nbr[i] = random.randint(0, max_bins-1)
            nbrs.append(nbr)
        return nbrs
    
    opt = pybaha.AdaptiveOptimizer(energy, sampler, neighbors)
    config = pybaha.AdaptiveConfig()
    config.probe_steps = 50
    config.ba_beta_steps = 500
    config.verbose = False
    
    result = opt.optimize(config)
    return {
        "name": f"Bin Packing (N={N})",
        "energy": result.best_energy,
        "time_ms": result.time_ms,
        "fracture_density": result.fracture_density,
        "used_branch_aware": result.used_branch_aware,
        "success": result.best_energy < 100  # No overflow
    }


# =============================================================================
# BENCHMARK 6: TSP (Traveling Salesman)
# =============================================================================
def benchmark_tsp(N=20):
    """TSP: Expect HIGH fracture density → BranchAware"""
    # Generate random cities
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(N)]
    
    def dist(i, j):
        dx = cities[i][0] - cities[j][0]
        dy = cities[i][1] - cities[j][1]
        return (dx*dx + dy*dy) ** 0.5
    
    def energy(state):
        total = 0
        for i in range(N):
            total += dist(state[i], state[(i+1) % N])
        return float(total)
    
    def sampler():
        state = list(range(N))
        random.shuffle(state)
        return state
    
    def neighbors(state):
        nbrs = []
        for _ in range(16):
            nbr = list(state)
            i, j = random.sample(range(N), 2)
            nbr[i], nbr[j] = nbr[j], nbr[i]
            nbrs.append(nbr)
        return nbrs
    
    opt = pybaha.AdaptiveOptimizer(energy, sampler, neighbors)
    config = pybaha.AdaptiveConfig()
    config.probe_steps = 50
    config.ba_beta_steps = 500
    config.verbose = False
    
    result = opt.optimize(config)
    return {
        "name": f"TSP (N={N})",
        "energy": result.best_energy,
        "time_ms": result.time_ms,
        "fracture_density": result.fracture_density,
        "used_branch_aware": result.used_branch_aware,
        "success": True  # Optimization problem
    }


# =============================================================================
# RUN ALL BENCHMARKS
# =============================================================================
def main():
    print("=" * 70)
    print("🔄 AdaptiveOptimizer Benchmark Suite")
    print("=" * 70)
    print("Testing fracture-based switching: density > 0.3 → BranchAware")
    print()
    
    benchmarks = [
        ("N-Queens", lambda: benchmark_nqueens(50)),
        ("Graph Coloring", lambda: benchmark_graph_coloring(30, 60, 3)),
        ("Max-Cut", lambda: benchmark_max_cut(40, 80)),
        ("Knapsack", lambda: benchmark_knapsack(30)),
        ("Bin Packing", lambda: benchmark_bin_packing(20)),
        ("TSP", lambda: benchmark_tsp(20)),
    ]
    
    results = []
    for name, fn in benchmarks:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            result = fn()
            results.append(result)
            status = "✅" if result["success"] else "⚠️"
            engine = "BA" if result["used_branch_aware"] else "Zeta"
            print(f"{status} E={result['energy']:.2f}, ρ={result['fracture_density']:.3f} → {engine}, {result['time_ms']:.0f}ms")
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({"name": name, "error": str(e)})
    
    print()
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"{'Benchmark':<30} {'Energy':>10} {'Density':>10} {'Engine':>10} {'Time':>10}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} {'ERROR':>10}")
        else:
            engine = "BranchAware" if r["used_branch_aware"] else "Zeta"
            print(f"{r['name']:<30} {r['energy']:>10.2f} {r['fracture_density']:>10.3f} {engine:>10} {r['time_ms']:>9.0f}ms")
    
    print()
    successes = sum(1 for r in results if r.get("success", False))
    print(f"Success Rate: {successes}/{len(results)} ({100*successes/len(results):.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
