# Complexity Analysis of BAHA

## Overview
The BAHA (Branch‑Aware Optimizer) is a heuristic optimizer that combines fracture detection, Lambert‑W branch enumeration, and Monte‑Carlo sampling. It targets **NP‑hard** problems (graph coloring, SAT, MILP, etc.) and continuous non‑convex landscapes.

## High‑Level Algorithmic Flow
```
for each β in schedule:
    estimate log Z (Monte‑Carlo sampling)                → O(S)
    record (β, log Z) in FractureDetector
    if fracture detected:
        enumerate branches (Lambert‑W)                  → O(1)
        score each branch (Monte‑Carlo sampling)       → O(M·S)
        jump to best branch (sample + local search)    → O(N·S)
    else:
        perform standard local‑search step (neighbors) → O(N)
```
*`S` = `samples_per_beta`, `M` = `max_branches` (≤ 5), `N` = average neighbor count, `B` = `beta_steps`.

## Detailed Complexity
- **Per‑iteration cost (worst case)**: `O((M + N)·S)`. Since `M` is a tiny constant, the dominant term is `O(N·S)`.
- **Overall runtime**: `O(B·N·S)`. The algorithm is linear in the annealing schedule length, the number of neighbor evaluations, and the Monte‑Carlo sample count.
- **Space usage**: `O(B + S)` – schedule vector plus temporary sample vectors.

## Underlying Problem Complexity
| Problem class | Decision version | Complexity |
|---------------|------------------|------------|
| Graph coloring, SAT, MILP, TSP, DNA‑barcode design | “Is there a solution with ≤ k conflicts?” | **NP‑complete** (optimization variants are NP‑hard) |
| Continuous non‑convex landscapes (Rosenbrock, Rastrigin, etc.) | “Find a point with objective ≤ ε” | Generally **NP‑hard** to guarantee a global optimum |

Because BAHA is a stochastic heuristic, its **worst‑case** runtime is unbounded, but the **expected** runtime follows the `O(B·N·S)` bound above.

## Empirical Observations (from benchmarks)
- **Mega Landscape Benchmark** shows fast convergence on 2‑D Rosenbrock, slower on high‑dimensional Rastrigin where many fractures occur.
- **Graph‑coloring benchmark** reports a 2‑3× speed‑up over plain Simulated Annealing, confirming the reduction in effective `B` when fractures trigger jumps.

## Practical Tuning Recommendations
| Goal | Parameter | Effect |
|------|-----------|--------|
| Reduce runtime | Lower `beta_steps` (`B`) or increase `beta_critical` to trigger fractures earlier | Fewer annealing steps → lower `O(B·N·S)` |
| Improve solution quality | Increase `samples_per_beta` (`S`) and/or `max_branches` (`M`) | More accurate `log Z` and branch scores, at linear cost increase |
| Scale to large discrete problems | Optimize `neighbors_` to return a sparse subset (e.g., random `k << N`) | Directly reduces dominant `N·S` factor |
| GPU acceleration | Offload Monte‑Carlo sampling to CUDA (`baha_gpu.cu`) | Parallelism reduces constant factor of `O(N·S)` |
| Memory footprint | Keep `S` modest (≤ 200) and reuse vectors | Space stays `O(B+S)` and cache locality improves |

## AdaptiveOptimizer Complexity (`pybaha.optimize()`)

The `AdaptiveOptimizer` (default via `pybaha.optimize()`) adds a probe phase before selecting the optimal engine:

```
Phase 1: Probe (detect fracture density ρ)
    → O(probe_steps × probe_samples) = O(100 × 10) = O(1000 evals)

Phase 2: Engine selection based on ρ
    If ρ > 0.3 (fracture-rich): BranchAwareOptimizer
        → O(B·N·S) where B=200, S=50 (defaults)
        
    If ρ ≤ 0.3 (smooth landscape): ZetaBreatherOptimizer
        → O(total_steps × K) + O(polish_steps × polish_samples)
        where K = gradient computation cost
```

### Default Parameters (AdaptiveOptimizer)

| Parameter | Value | Effect on Complexity |
|-----------|-------|---------------------|
| `probe_steps` | 100 | O(100) probe overhead |
| `probe_samples` | 10 | O(10) samples per probe step |
| `ba_beta_steps` | 200 | O(200) annealing iterations |
| `ba_samples_per_beta` | 50 | O(50) MC samples per β |
| `timeout_ms` | 5000 | Hard cutoff (overrides steps) |

### Per-Problem Complexity Examples

| Problem | Neighbors (N) | Total Evals | Time (empirical) |
|---------|---------------|-------------|------------------|
| N-Queens (N=100) | 50 | ~500K | 19-30s |
| Graph Coloring (30V) | 30 | ~300K | 300ms |
| 3-SAT (20v, 40c) | 20 | ~200K | 1.1s |
| Sudoku (4×4) | 20 | ~100K | 5s |

### Timeout vs Steps

When `timeout_ms` is set, the algorithm may terminate before completing all `beta_steps`. The effective complexity becomes:

```
O(min(B·N·S, timeout_ms × eval_rate))
```

where `eval_rate` ≈ 10K-100K evals/second depending on energy function cost.

## Summary
- **Runtime:** `O(B·N·S)` (linear in schedule length, neighbor count, and Monte-Carlo samples). 
- **Space:** `O(B + S)`. 
- **Problem class:** NP-hard / NP-complete. 
- **AdaptiveOptimizer:** Adds O(1000) probe overhead, then routes to optimal engine.
- **Empirical speed-up:** 2-3× over vanilla SA on medium-size discrete benchmarks; larger gains when fractures are frequent. 
- **Scalability:** Linear scaling with problem size; can be accelerated via parallel sampling and smarter neighbor generation.

