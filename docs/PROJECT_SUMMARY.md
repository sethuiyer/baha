# BAHA Framework - Project Structure

## Directory Layout

```
baha/
├── include/                 # Header files
│   ├── baha/              # Main library headers
│   │   └── baha.hpp       # Core implementation
│   └── baha.h             # Main include file
├── src/                    # Source files
│   ├── baha.cpp           # CPU implementation
│   └── baha_gpu.cu        # GPU implementation
├── examples/               # Usage examples
│   ├── spectrum_auction.cpp  # Spectrum auction optimization
│   ├── list_coloring.cpp     # Constrained graph coloring
│   └── isr_benchmarks.cpp    # High-signal ISR problems
├── benchmarks/             # Benchmark suites
│   ├── casimir_sat.cpp       # SAT problem benchmarks
│   ├── graph_iso_benchmark.cpp  # Graph isomorphism benchmarks
│   └── ...                   # More benchmark files
├── tests/                  # Test files
│   └── simple_test.cpp       # Basic functionality test
├── docs/                   # Documentation
│   └── index.md            # Main documentation
├── cmake/                  # CMake modules
├── CMakeLists.txt          # Main CMake configuration
├── README.md               # Project overview
├── LICENSE                 # License information
├── CONTRIBUTING.md         # Contribution guidelines
├── CHANGELOG.md            # Version history
├── build.sh                # Build script
└── ...                     # Other project files
```

## Key Features Organized

### Core Algorithms
- Branch-Aware Optimization
- Fracture Detection (ρ = |d/dβ log Z|)
- Lambert-W Function Implementation
- Branch Enumeration and Scoring

### Performance Features
- GPU Acceleration (CUDA)
- Spectral Analysis (O(N log N) scaling)
- Adaptive Thresholds
- Multi-scale Fracture Detection

### Problem Domains
- Graph Coloring & List Coloring
- SAT/SMT Problems
- Combinatorial Auctions
- Ramsey Theory (N=100,000 variables)
- Protein Folding
- Side-channel Analysis

## Build Instructions

```bash
# Make sure you have prerequisites:
# - C++17 compiler
# - CMake 3.12+
# - CUDA Toolkit (optional)

# Build the project
./build.sh

# Or manually:
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage Example

```cpp
#include "baha/baha.hpp"

// Define your problem
struct MyState { /* ... */ };

auto energy = [](const MyState& s) { /* ... */ };
auto sampler = []() { /* ... */ };
auto neighbors = [](const MyState& s) { /* ... */ };

navokoj::BranchAwareOptimizer<MyState> opt(energy, sampler, neighbors);
auto result = opt.optimize();
```

## Project Status

✅ **Complete**: All core functionality implemented  
✅ **Organized**: Professional project structure  
✅ **Documented**: Comprehensive documentation  
✅ **Benchmarked**: Extensive performance validation  
✅ **Ready for Open Source**: Proper licensing and contribution guide  

The "Fracture Hunter" is now a complete, production-ready optimization framework!