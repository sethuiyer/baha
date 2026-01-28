<p align="center">
  <img src="logo.png" alt="BAHA Logo" width="300"/>
</p>

# BAHA: Branch-Aware Holonomy Annealing

A simulated annealing variant that uses phase transition detection (fractures) and Lambert-W branch enumeration to escape local minima.

> **Better than Simulated Annealing on structured problems. Handles instances where exact solvers are impractical.**

[![Watch the Presentation](https://img.shields.io/badge/YouTube-Watch%20Presentation-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=jVKetFO7SgM)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18373732.svg)](https://doi.org/10.5281/zenodo.18373732)

> **🏆 Notable Results:**
> - **R(5,5,5) @ N=52**: Perfect 3-coloring with 2.6M constraints solved in <1s (RTX 3050) — [verifiable witness](data/ramsey_52_witness.csv)
> - **Graph Isomorphism (N=50)**: **100% success rate** vs 20% for Simulated Annealing
> - **Spectrum Auction**: **+102% revenue** ($2.4B vs $1.18B) in 1.657ms
> - **Number Partitioning N=100k**: Near-optimal in 13.6s — **1.5M× faster** than random
> - **DNA Barcode Design**: Perfect solution (0 violations) — **first application** of fracture-aware optimization to bioinformatics
> - **20+ Problem Types Validated**: Constraint satisfaction, graph problems, scheduling, physics, security, biology
> - **Graph Coloring (N=200)**: Solved 200-vertex graph with 0.5 edge probability in ~36s (54 colors, 0 conflicts) — demonstrates scalability on NP-hard problems

<p align="center">
  <img src="data/ramsey_102.webp" alt="Ramsey N=102 3-Coloring" width="500"/>
  <br/>
  <em>3-Coloring of K₁₀₂ — 5,151 edges navigating 83.2M clique constraints</em>
</p>

## Overview

**BAHA is a general phase-aware optimization framework with demonstrated cross-domain fracture structure.**

### What Makes BAHA Different

Most optimizers do one of these:

- **Exploit locality** (local search, hill climbing)
- **Exploit gradients** (gradient descent, Newton methods)
- **Exploit heuristics** (greedy algorithms, domain-specific rules)
- **Exploit problem structure manually** (hand-crafted relaxations, problem-specific solvers)

**BAHA does something rarer:**

> **It exploits changes in structure.**

That's why:

- ✅ **It generalizes across domains** — fracture detection works regardless of problem type
- ✅ **It reduces to SA when nothing interesting happens** — no overhead when structure is absent
- ✅ **It doesn't overjump on most problems** — selective exploitation (typically <2% jump rate)
- ✅ **It shines when classical methods stall** — nonlocal jumps escape deep local minima

### Core Invariant

**Across 26 diverse optimization domains, BAHA consistently detects thermodynamic fractures in the solution landscape; however, only a small subset of these fractures correspond to actionable phase transitions that justify nonlocal branch jumps.**

Every problem type exhibits the same **five-phase pattern**:

1. **Smooth annealing phase**: Standard Metropolis sampling with gradual β increase
2. **Sharp change in log-partition slope**: `ρ = |d/dβ log Z|` spikes beyond threshold
3. **Fracture detected**: Thermodynamic discontinuity identified
4. **Nonlocal jump in control parameter**: Lambert-W branch enumeration generates alternative β values
5. **Qualitatively different basin entered**: Solution quality improves after jump

### Fractures ≠ Branch Jumps

**Most fractures are local thermodynamic perturbations; only a small subset exceed the jump-selection criterion and trigger branch enumeration.**

| Problem Type | Typical Fractures | Typical Jumps | Jump Rate |
|--------------|-------------------|---------------|-----------|
| VRP / Bin Packing | 450-599 | 1-4 | 0.2-0.7% |
| Network Design | 499 | 8 | 1.6% |
| Course Scheduling | 499 | 2 | 0.4% |
| Maximum Clique | 4 | 1 | 25% |
| TSP | 498 | 5 | 1.0% |

**Interpretation**: Fracture detection is **sensitive** (many events), but branch jumping is **selective** (rare, high-impact). BAHA reduces to near-SA behavior when fractures don't matter, but exploits structure when it exists.

### What This Proves

- Fracture signals are **not domain-specific** (appear in routing, packing, scheduling, graph structure, economics, infrastructure)
- Branch jumping is **selectively useful**, not spammy (typically <2% of fractures trigger jumps)
- The thermodynamic framework generalizes across discrete and continuous-relaxed problems
- BAHA is a **structure exploitation framework**, not a universal solver

## Technical Specification: B.A.H.A.

**B.A.H.A.** — **Branch-Adaptive Hardness Aligner**

1. **Branch-Adaptive**: Uses Lambert-W function to enumerate alternative solution branches when phase transitions are detected.
2. **Hardness-Aware**: Designed for NP-hard combinatorial problems with rugged energy landscapes.
3. **Hybrid Sampling**: Combines Metropolis MCMC with optional O(N) analytical moment computation for phase detection.
4. **Observable**: Fracture events and branch jumps are logged, making the optimization trajectory interpretable.

**Key Insight**: BAHA isn't just an optimizer—it's a **hardness detector**. If BAHA finds fractures, the problem has exploitable structure. If it doesn't, you've proven the landscape is genuinely random.

## Key Features

- **Fracture Detection**: Monitors specific heat to identify phase transitions
- **Branch Navigation**: Uses Lambert-W functions to enumerate alternative restart points
- **GPU Acceleration**: CUDA kernels for parallel constraint evaluation
- **Spectral Analysis**: Optional O(N) analytical moments for partition-like problems
- **Header-Only**: Single `baha.hpp` with no dependencies beyond C++17

## Installation

### Prerequisites
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.12+
- CUDA Toolkit (optional, for GPU acceleration)

### Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Quick Start

```cpp
#include "baha/baha.hpp"

// Define your optimization problem
struct MyState {
    std::vector<double> variables;
};

// Energy function
auto energy = [](const MyState& s) -> double {
    // Compute energy based on state
    return compute_energy(s);
};

// Sampler function
auto sampler = []() -> MyState {
    // Generate random state
    return generate_random_state();
};

// Neighbor function
auto neighbors = [](const MyState& s) -> std::vector<MyState> {
    // Generate neighboring states
    return generate_neighbors(s);
};

// Create optimizer
navokoj::BranchAwareOptimizer<MyState> optimizer(energy, sampler, neighbors);

// Configure and run
typename navokoj::BranchAwareOptimizer<MyState>::Config config;
config.beta_steps = 1000;
config.beta_end = 10.0;

auto result = optimizer.optimize(config);
```

## Examples

See the `examples/` directory for complete examples:

- `spectrum_auction.cpp` - Combinatorial spectrum auction optimization (+102% revenue)
- `list_coloring.cpp` - Constrained graph coloring (80% improvement)
- `n_queens.cpp` - N-Queens constraint satisfaction (100% success rate, symmetry-breaking fractures)
- `max_cut.cpp` - Maximum cut graph partitioning (84% cut ratio, basin transition fractures)
- `knapsack.cpp` - 0/1 Knapsack optimization (100% capacity utilization, boundary navigation)
- `traveling_salesman.cpp` - TSP permutation optimization (498 fractures, 5 jumps, 1% rate)
- `vehicle_routing.cpp` - Vehicle routing (route restructuring fractures, 0.67% jump rate)
- `bin_packing.cpp` - Bin packing (feasibility regime transitions, 0.22% jump rate)
- `max_clique.cpp` - Maximum clique (rare high-signal fractures, 25% jump rate)
- `course_scheduling.cpp` - Course scheduling (symmetry-breaking, 0.4% jump rate)
- `network_design.cpp` - Network design (connectivity phase transition, 1.6% jump rate)
- `resource_allocation.cpp` - Resource allocation (utility landscape, rare but high-signal)
- `isr_benchmarks.cpp` - High-signal ISR problems

## Benchmarks

BAHA has been validated on **26+ problem types** across diverse domains, demonstrating that **fracture structure is not domain-specific**:

- **Combinatorial**: VRP (route restructuring), Bin Packing (feasibility boundaries), TSP (multi-basin)
- **Graph-Theoretic**: Max Clique (rare high-signal), Max Cut (basin transitions), Graph Isomorphism (100% vs 20% SA)
- **Constraint-Based**: N-Queens (symmetry breaking, 100% success), Course Scheduling (constraint resolution), SAT/5-SAT (phase transition)
- **Real-World**: Spectrum auctions (+102% revenue), DNA barcode design (perfect solution), side-channel attacks
- **Physics**: LABS sequences (glassy landscape), protein folding (GPU-accelerated)
- **Ramsey Theory**: $R(5,5,5) > 52$ proven, scaled to **83.2M constraints** @ N=102

**Each domain exhibits different fracture types** (entropy-driven, feasibility-driven, symmetry-breaking, phase transitions, utility collapse, rare high-signal), but the **detection mechanism** (log-partition slope) and **response mechanism** (Lambert-W branch enumeration) are universal.

## Architecture

```
include/          - Header files (optimized)
├── baha/
│   ├── baha.hpp  - Optimized CPU library (SIMD, fast math)
│   └── baha.cuh  - Optimized GPU kernels (warp shuffles)
└── baha.h        - Convenience wrapper

src/              - Source files
├── baha.cpp      - CPU implementation
└── baha_gpu.cu   - GPU implementation

examples/         - Usage examples
├── spectrum_auction.cpp
├── list_coloring.cpp
└── hybrid_solver.cpp

benchmarks/       - Benchmark suites
├── optimization_benchmark.cpp
├── ramsey_benchmark.cpp
├── hardness_barrier.cpp
└── [30+ benchmarks]

docs/             - Documentation
├── README.md                 - Documentation index
├── OPTIMIZATION_GUIDE.md     - Performance tuning guide
├── SPECTRAL_ANALYSIS.md      - O(N) hardness detection
├── COMPLETE_ANALYSIS.md      - Technical deep dive
└── CASE_STUDY.md             - Real-world examples

tests/            - Unit tests
cmake/            - CMake modules
scripts/          - Visualization tools
data/             - Benchmark data & results
```

## Documentation

Complete documentation is available in the `docs/` directory:

- **[docs/README.md](docs/README.md)** - Documentation index
- **[docs/CASE_STUDY.md](docs/CASE_STUDY.md)** - Real-world performance benchmarks (26+ problems)
- **[docs/PROBLEM_LIST.md](docs/PROBLEM_LIST.md)** - Complete catalog of all tested problems
- **[docs/OPTIMIZATION_GUIDE.md](docs/OPTIMIZATION_GUIDE.md)** - CPU/GPU optimization techniques

## Performance Highlights

- **Graph Isomorphism**: **100% success** (vs 20% for Simulated Annealing) in 0.2ms
- **Spectrum Auction**: **+102% revenue** ($2.4B vs $1.18B) in 1.657ms
- **Ramsey Theory**: Perfect solution $R(5,5,5) > 52$ with 2.6M constraints in <1s
- **Number Partitioning**: **1.5M× faster** than random at N=100k (13.6s via O(N) spectral mode)
- **DNA Barcode Design**: Perfect solution (0 violations) — **first application** to bioinformatics
- **Cross-Platform**: CPU, CUDA, and MPS (Metal) backends — unified API
- **GPU Acceleration**: 50-100x speedup on constraint-heavy problems

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use BAHA/Fracture in your research, please cite the original paper:

**Sethurathienam Iyer. (2026). Multiplicative Calculus for Hardness Detection and Branch-Aware Optimization: A Computational Framework for Detecting Phase Transitions via Non-Integrable Log-Derivatives. Zenodo. [https://doi.org/10.5281/zenodo.18373732](https://doi.org/10.5281/zenodo.18373732)**

```bibtex
@article{iyer2026multiplicative,
  title={Multiplicative Calculus for Hardness Detection and Branch-Aware Optimization},
  author={Iyer, Sethurathienam},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18373732},
  url={https://doi.org/10.5281/zenodo.18373732}
}
```