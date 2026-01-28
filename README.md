<p align="center">
  <img src="logo.png" alt="BAHA Logo" width="300"/>
</p>

# BAHA: Branch-Aware Holonomy Annealing

A simulated annealing variant that uses phase transition detection (fractures) and Lambert-W branch enumeration to escape local minima.

[![Watch the Presentation](https://img.shields.io/badge/YouTube-Watch%20Presentation-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=jVKetFO7SgM)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18373732.svg)](https://doi.org/10.5281/zenodo.18373732)

> **🏆 Notable Results:**
> - **R(5,5,5) @ N=52**: Perfect 3-coloring with 2.6M constraints solved in <1s (RTX 3050) — [verifiable witness](data/ramsey_52_witness.csv)
> - **Graph Isomorphism (N=50)**: **100% success rate** vs 20% for Simulated Annealing
> - **Spectrum Auction**: **+102% revenue** ($2.4B vs $1.18B) in 1.657ms
> - **Number Partitioning N=100k**: Near-optimal in 13.6s — **1.5M× faster** than random
> - **DNA Barcode Design**: Perfect solution (0 violations) — **first application** of fracture-aware optimization to bioinformatics
> - **20+ Problem Types Validated**: Constraint satisfaction, graph problems, scheduling, physics, security, biology

<p align="center">
  <img src="data/ramsey_102.webp" alt="Ramsey N=102 3-Coloring" width="500"/>
  <br/>
  <em>3-Coloring of K₁₀₂ — 5,151 edges navigating 83.2M clique constraints</em>
</p>

## Overview

BAHA is a phase-aware optimization algorithm that monitors the specific heat (ρ = |d/dβ log Z|) during the annealing schedule. When ρ spikes (indicating a phase transition), BAHA uses Lambert-W branch enumeration to strategically select new starting points, exploiting the phase structure rather than getting trapped in local minima.

## Technical Specification: B.A.H.A.

**B.A.H.A.** — **Branch-Adaptive Hardness Aligner**

1. **Branch-Adaptive**: Uses Lambert-W function to enumerate alternative solution branches when phase transitions are detected.
2. **Hardness-Aware**: Designed for NP-hard combinatorial problems with rugged energy landscapes.
3. **Hybrid Sampling**: Combines Metropolis MCMC with optional O(N) analytical moment computation for phase detection.
4. **Observable**: Fracture events and branch jumps are logged, making the optimization trajectory interpretable.

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
- `n_queens.cpp` - N-Queens constraint satisfaction (100% success rate)
- `max_cut.cpp` - Maximum cut graph partitioning (84% cut ratio)
- `knapsack.cpp` - 0/1 Knapsack optimization (100% capacity utilization)
- `traveling_salesman.cpp` - TSP permutation optimization
- `isr_benchmarks.cpp` - High-signal ISR problems

## Benchmarks

BAHA has been validated on **20+ problem types** across diverse domains:

- **Graph Problems**: Graph isomorphism (100% vs 20% SA), max cut, max independent set
- **Constraint Satisfaction**: N-Queens (100% success), SAT/5-SAT (solved at phase transition)
- **Combinatorial Optimization**: TSP, knapsack, job shop scheduling (30%+ improvement)
- **Ramsey Theory**: $R(5,5,5) > 52$ proven, scaled to **83.2M constraints** @ N=102
- **Real-World**: Spectrum auctions (+102% revenue), DNA barcode design (perfect solution), side-channel attacks
- **Physics**: LABS sequences, protein folding (GPU-accelerated)

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
- **[docs/OPTIMIZATION_GUIDE.md](docs/OPTIMIZATION_GUIDE.md)** - CPU/GPU optimization techniques
- **[docs/CASE_STUDY.md](docs/CASE_STUDY.md)** - Real-world performance benchmarks

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