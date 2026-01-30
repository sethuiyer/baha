#!/usr/bin/env python3
"""Comprehensive benchmark for AutoZetaOptimizer across multiple problem domains."""

import pybaha
import random
import time

# Seed for reproducibility
random.seed(42)

print("=" * 70)
print("🔬 AutoZetaOptimizer Benchmark Suite")
print("=" * 70)
print("Testing physics-based auto-relaxation with log(p²/(p-1)) energy")
print()

results = []

# ==============================================================================
# 1. N-QUEENS (N=30)
# ==============================================================================
def run_nqueens():
    N = 30
    
    def energy(state):
        conflicts = 0
        for i in range(N):
            for j in range(i + 1, N):
                if abs(state[i] - state[j]) == j - i:
                    conflicts += 1
        return float(conflicts)
    
    def sampler():
        return list(random.sample(range(N), N))
    
    def neighbors(state):
        nbrs = []
        for _ in range(20):
            s = list(state)
            i, j = random.sample(range(N), 2)
            s[i], s[j] = s[j], s[i]
            nbrs.append(s)
        return nbrs
    
    print("Running N-Queens (N=30)...", end=" ", flush=True)
    opt = pybaha.AutoZetaOptimizer(energy, sampler, neighbors, N)
    config = pybaha.AutoZetaConfig()
    config.total_steps = 3000
    config.period = 300
    config.timeout_ms = 30000.0
    
    result = opt.optimize(config)
    success = result.best_energy == 0
    print(f"{'✅' if success else '⚠️'} E={int(result.best_energy)}, {result.time_ms:.0f}ms, peaks={result.peaks_harvested}")
    return ("N-Queens (N=30)", result.best_energy, result.time_ms, result.peaks_harvested, success)

# ==============================================================================
# 2. GRAPH COLORING (V=40, E=100, K=3)
# ==============================================================================
def run_graph_coloring():
    V, E_count, K = 40, 100, 3
    
    # Random graph
    edges = []
    while len(edges) < E_count:
        u, v = random.sample(range(V), 2)
        if (u, v) not in edges and (v, u) not in edges:
            edges.append((u, v))
    
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
        for _ in range(20):
            s = list(state)
            node = random.randint(0, V-1)
            s[node] = random.randint(0, K-1)
            nbrs.append(s)
        return nbrs
    
    print("Running Graph Coloring (V=40, K=3)...", end=" ", flush=True)
    opt = pybaha.AutoZetaOptimizer(energy, sampler, neighbors, K)
    config = pybaha.AutoZetaConfig()
    config.total_steps = 3000
    config.period = 300
    config.timeout_ms = 30000.0
    
    result = opt.optimize(config)
    success = result.best_energy == 0
    print(f"{'✅' if success else '⚠️'} E={int(result.best_energy)}, {result.time_ms:.0f}ms, peaks={result.peaks_harvested}")
    return ("Graph Coloring (V=40, K=3)", result.best_energy, result.time_ms, result.peaks_harvested, success)

# ==============================================================================
# 3. SUDOKU 4x4
# ==============================================================================
def run_sudoku():
    N = 4
    
    # Initial fixed cells (a valid 4x4 sudoku with some blanks)
    # 0 means blank
    fixed = [
        [1, 0, 0, 4],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [4, 0, 0, 1],
    ]
    
    def energy(state):
        # State is flat 16 values (1-4)
        grid = [state[i*N:(i+1)*N] for i in range(N)]
        
        conflicts = 0
        # Row conflicts
        for r in range(N):
            for i in range(N):
                for j in range(i+1, N):
                    if grid[r][i] == grid[r][j]:
                        conflicts += 1
        # Col conflicts
        for c in range(N):
            for i in range(N):
                for j in range(i+1, N):
                    if grid[i][c] == grid[j][c]:
                        conflicts += 1
        # 2x2 box conflicts
        for br in range(2):
            for bc in range(2):
                vals = []
                for dr in range(2):
                    for dc in range(2):
                        vals.append(grid[br*2+dr][bc*2+dc])
                for i in range(4):
                    for j in range(i+1, 4):
                        if vals[i] == vals[j]:
                            conflicts += 1
        # Fixed cell penalty
        for r in range(N):
            for c in range(N):
                if fixed[r][c] != 0 and grid[r][c] != fixed[r][c]:
                    conflicts += 10
        return float(conflicts)
    
    def sampler():
        state = []
        for r in range(N):
            for c in range(N):
                if fixed[r][c] != 0:
                    state.append(fixed[r][c])
                else:
                    state.append(random.randint(1, N))
        return state
    
    def neighbors(state):
        nbrs = []
        for _ in range(20):
            s = list(state)
            idx = random.randint(0, N*N - 1)
            r, c = idx // N, idx % N
            if fixed[r][c] == 0:  # Only modify non-fixed cells
                s[idx] = random.randint(1, N)
            nbrs.append(s)
        return nbrs
    
    print("Running Sudoku 4x4...", end=" ", flush=True)
    opt = pybaha.AutoZetaOptimizer(energy, sampler, neighbors, N + 1)  # domain 1-4 = 5 values (0-4)
    config = pybaha.AutoZetaConfig()
    config.total_steps = 2000
    config.period = 200
    config.timeout_ms = 15000.0
    
    result = opt.optimize(config)
    success = result.best_energy == 0
    print(f"{'✅' if success else '⚠️'} E={int(result.best_energy)}, {result.time_ms:.0f}ms, peaks={result.peaks_harvested}")
    return ("Sudoku 4x4", result.best_energy, result.time_ms, result.peaks_harvested, success)

# ==============================================================================
# 4. MAGIC SQUARE 3x3
# ==============================================================================
def run_magic_square():
    N = 3
    TARGET = 15  # Sum for 3x3 magic square
    
    def energy(state):
        grid = [state[i*N:(i+1)*N] for i in range(N)]
        
        # Check if all values 1-9 are used
        used = set(state)
        if len(used) != 9:
            return 100.0  # Penalty for duplicates
        
        error = 0
        # Rows
        for r in range(N):
            error += abs(sum(grid[r]) - TARGET)
        # Cols
        for c in range(N):
            col_sum = sum(grid[r][c] for r in range(N))
            error += abs(col_sum - TARGET)
        # Diagonals
        diag1 = sum(grid[i][i] for i in range(N))
        diag2 = sum(grid[i][N-1-i] for i in range(N))
        error += abs(diag1 - TARGET) + abs(diag2 - TARGET)
        
        return float(error)
    
    def sampler():
        return list(random.sample(range(1, 10), 9))  # Permutation of 1-9
    
    def neighbors(state):
        nbrs = []
        for _ in range(20):
            s = list(state)
            i, j = random.sample(range(9), 2)
            s[i], s[j] = s[j], s[i]
            nbrs.append(s)
        return nbrs
    
    print("Running Magic Square 3x3...", end=" ", flush=True)
    opt = pybaha.AutoZetaOptimizer(energy, sampler, neighbors, 10)  # domain 1-9
    config = pybaha.AutoZetaConfig()
    config.total_steps = 2000
    config.period = 200
    config.timeout_ms = 15000.0
    
    result = opt.optimize(config)
    success = result.best_energy == 0
    print(f"{'✅' if success else '⚠️'} E={int(result.best_energy)}, {result.time_ms:.0f}ms, peaks={result.peaks_harvested}")
    return ("Magic Square 3x3", result.best_energy, result.time_ms, result.peaks_harvested, success)

# ==============================================================================
# 5. SET COVER (simplified)
# ==============================================================================
def run_set_cover():
    N_ELEMENTS = 20
    N_SETS = 10
    
    # Random sets
    sets = []
    for _ in range(N_SETS):
        size = random.randint(3, 8)
        s = set(random.sample(range(N_ELEMENTS), size))
        sets.append(s)
    
    def energy(state):
        # state[i] = 1 if set i is selected, 0 otherwise
        covered = set()
        for i in range(N_SETS):
            if state[i] == 1:
                covered.update(sets[i])
        
        uncovered = N_ELEMENTS - len(covered)
        num_selected = sum(state)
        
        # Minimize: uncovered * 10 + num_selected (prefer fewer sets)
        return float(uncovered * 10 + num_selected)
    
    def sampler():
        return [random.randint(0, 1) for _ in range(N_SETS)]
    
    def neighbors(state):
        nbrs = []
        for _ in range(20):
            s = list(state)
            i = random.randint(0, N_SETS - 1)
            s[i] = 1 - s[i]  # Flip bit
            nbrs.append(s)
        return nbrs
    
    print("Running Set Cover (20 elements, 10 sets)...", end=" ", flush=True)
    opt = pybaha.AutoZetaOptimizer(energy, sampler, neighbors, 2)  # binary
    config = pybaha.AutoZetaConfig()
    config.total_steps = 2000
    config.period = 200
    config.timeout_ms = 15000.0
    
    result = opt.optimize(config)
    # Success if all covered (energy < 10 means all covered)
    success = result.best_energy < 10
    print(f"{'✅' if success else '⚠️'} E={result.best_energy:.0f}, {result.time_ms:.0f}ms, peaks={result.peaks_harvested}")
    return ("Set Cover", result.best_energy, result.time_ms, result.peaks_harvested, success)

# ==============================================================================
# RUN ALL BENCHMARKS
# ==============================================================================
results.append(run_nqueens())
results.append(run_graph_coloring())
results.append(run_sudoku())
results.append(run_magic_square())
results.append(run_set_cover())

# ==============================================================================
# SUMMARY
# ==============================================================================
print()
print("=" * 70)
print("📊 SUMMARY")
print("=" * 70)
print(f"{'Problem':<30} {'Energy':>10} {'Time':>10} {'Peaks':>8} {'Status':>8}")
print("-" * 70)

success_count = 0
for name, energy, time_ms, peaks, success in results:
    status = "✅" if success else "⚠️"
    print(f"{name:<30} {energy:>10.0f} {time_ms:>9.0f}ms {peaks:>8} {status:>8}")
    if success:
        success_count += 1

print("-" * 70)
print(f"Success Rate: {success_count}/{len(results)} ({100*success_count//len(results)}%)")
print("=" * 70)
