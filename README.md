<p align="center">
  <img src="logo.png" alt="BAHA Logo" width="300"/>
</p>

# BAHA: Branch-Aware Holonomy Annealing

A simulated annealing variant that uses phase transition detection (fractures) and Lambert-W branch enumeration to escape local minima.

[![Watch the Presentation](https://img.shields.io/badge/YouTube-Watch%20Presentation-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=jVKetFO7SgM)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18373732.svg)](https://doi.org/10.5281/zenodo.18373732)

> **🏆 Notable Results:**
> - **R(5,5,5) @ N=52**: 3-coloring of K₅₂ with zero monochromatic K₅ — [verifiable witness](data/ramsey_52_witness.csv) solved in <1s (RTX 3050). Run `python data/verify_ramsey.py` to validate.
> - **Scale Test @ N=102**: 83.2M clique constraints, reduced violations from 4,200+ to ~150 (not solved, demonstrates scaling).
> - **Number Partitioning N=100k**: Found near-optimal partition in 13.6s using O(N) spectral moment computation for phase detection.

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

- `spectrum_auction.cpp` - Combinatorial spectrum auction optimization
- `list_coloring.cpp` - Constrained graph coloring
- `isr_benchmarks.cpp` - High-signal ISR problems

## Benchmarks

BAHA has been tested on numerous challenging problems:

- **SAT Problems**: Superior performance on phase transitions
- **Ramsey Theory**: 
    - **Constructive Proof**: $R(5,5,5) > 52$ (Perfect zero-energy coloring).
    - **Cosmic Scale**: $N=102$ handled **83.2 Million** constraints in minutes.
- **Spectrum Auctions**: Real-world optimization with billions in stake

## Architecture

```
include/          - Header files
├── baha/
│   └── baha.hpp  - Main library header

src/              - Source files
├── baha.cpp      - CPU implementation
└── baha_gpu.cu   - GPU implementation

examples/         - Usage examples
├── spectrum_auction.cpp
├── list_coloring.cpp
└── ...

benchmarks/       - Benchmark suites
├── casimir_sat.cpp
├── graph_iso_benchmark.cpp
└── ...

docs/             - Documentation
tests/            - Unit tests
cmake/            - CMake modules
```

## Performance Highlights

- **Spectrum Auction**: Solved in 1.657ms with 102% revenue improvement
- **List Coloring**: 80% improvement over random solutions
- **Ramsey Theory**: $R(5,5,5) > 52$ proven in < 30 seconds on RTX 3050.
- **Spectral Scaling**: Number Partitioning solved at **$N=100,000$** in 13.6 seconds via $O(N \log N)$ moments.

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