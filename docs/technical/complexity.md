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

## Summary
- **Runtime:** `O(B·N·S)` (linear in schedule length, neighbor count, and Monte‑Carlo samples). 
- **Space:** `O(B + S)`. 
- **Problem class:** NP‑hard / NP‑complete. 
- **Empirical speed‑up:** 2‑3× over vanilla SA on medium‑size discrete benchmarks; larger gains when fractures are frequent. 
- **Scalability:** Linear scaling with problem size; can be accelerated via parallel sampling and smarter neighbor generation.
