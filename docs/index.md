# BAHA Documentation

Welcome to the documentation for BAHA (Branch-Aware Optimizer), the revolutionary optimization framework that detects and navigates structural discontinuities in optimization landscapes.

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Installation](#installation)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Performance Guide](#performance-guide)
7. [Troubleshooting](#troubleshooting)

## Introduction

BAHA (Branch-Aware Optimizer) is a next-generation optimization algorithm that fundamentally differs from traditional approaches like simulated annealing. Instead of slowly cooling and hoping the system finds the ground state, BAHA:

1. **Detects fractures** via ρ = |d/dβ log Z|
2. **Enumerates branches** via Lambert-W function
3. **Jumps to optimal branch** instead of slow annealing

This approach results in orders-of-magnitude performance improvements on many optimization problems.

## Core Concepts

### Fracture Detection

The core innovation of BAHA is its ability to detect "fractures" in the optimization landscape. These fractures represent structural discontinuities where the system's behavior changes dramatically.

The fracture rate is computed as:
```
ρ = |d/dβ log Z|
```

Where Z is the partition function of the system.

### Branch Navigation

Once a fracture is detected, BAHA uses the Lambert-W function to enumerate possible solution branches and selects the most promising one to jump to.

### Phase Awareness

BAHA operates differently depending on whether solutions exist in connected basins:
- Below phase transition: Locks onto solution basins quickly
- At transition: Hops between fractures efficiently  
- Above transition: Honest stall with no false positives

## Installation

See the [README.md](../README.md) for installation instructions.

## API Reference

### Main Template Class

```cpp
template<typename State>
class BranchAwareOptimizer {
public:
    using EnergyFn = std::function<double(const State&)>;
    using SamplerFn = std::function<State()>;
    using NeighborFn = std::function<std::vector<State>(const State&)>;

    struct Config {
        double beta_start = 0.01;
        double beta_end = 10.0;
        int beta_steps = 500;
        double fracture_threshold = 1.5;
        double beta_critical = 1.0;
        int max_branches = 5;
        int samples_per_beta = 100;
        bool verbose = false;
        ScheduleType schedule_type = ScheduleType::LINEAR;
    };

    struct Result {
        State best_state;
        double best_energy;
        int fractures_detected;
        int branch_jumps;
        double beta_at_solution;
        int steps_taken;
        double time_ms;
    };

    BranchAwareOptimizer(EnergyFn energy, SamplerFn sampler, NeighborFn neighbors = nullptr);
    Result optimize(const Config& config = Config());
};
```

### Configuration Options

- `beta_start`: Starting inverse temperature
- `beta_end`: Ending inverse temperature  
- `beta_steps`: Number of steps in β schedule
- `fracture_threshold`: Threshold for detecting fractures
- `max_branches`: Maximum number of branches to consider
- `samples_per_beta`: Samples for partition function estimation
- `schedule_type`: LINEAR or GEOMETRIC β schedule

## Examples

### Basic Usage

```cpp
#include "baha/baha.hpp"
#include <vector>

struct MyState {
    std::vector<int> values;
};

// Define your energy function
auto energy_fn = [](const MyState& s) -> double {
    // Calculate energy based on state
    double energy = 0.0;
    // ... your calculation here ...
    return energy;
};

// Define your sampler
auto sampler_fn = []() -> MyState {
    MyState state;
    // ... initialize random state ...
    return state;
};

// Define your neighbor function
auto neighbor_fn = [](const MyState& s) -> std::vector<MyState> {
    std::vector<MyState> neighbors;
    // ... generate neighboring states ...
    return neighbors;
};

// Create and run optimizer
navokoj::BranchAwareOptimizer<MyState> optimizer(energy_fn, sampler_fn, neighbor_fn);

auto result = optimizer.optimize();
```

## Performance Guide

### When BAHA Excels

BAHA performs exceptionally well on:
- Discrete constraint satisfaction problems
- Problems with clear solution basins
- Combinatorial optimization with structure
- Problems below phase transitions

### When to Use Alternatives

Consider alternatives for:
- Smooth, continuous landscapes without clear structure
- Problems where SA already performs well
- Very small problems where overhead isn't justified

### Optimization Tips

1. **Tune the fracture threshold** based on your problem
2. **Adjust samples_per_beta** for accuracy vs speed tradeoff
3. **Experiment with schedule types** (linear vs geometric)
4. **Monitor fracture detection** to understand problem structure

## Troubleshooting

### Common Issues

**Q: BAHA is not finding good solutions**
A: Check your energy function and neighbor generation. Ensure the problem has structure that BAHA can exploit.

**Q: BAHA is taking too long**
A: Reduce samples_per_beta or adjust the β schedule. Consider if the problem is suitable for BAHA.

**Q: BAHA detects too many fractures**
A: Increase the fracture_threshold parameter.

### Debugging

Enable verbose mode in the configuration to see detailed progress information:

```cpp
config.verbose = true;
```

## Advanced Topics

### GPU Acceleration

For problems that can benefit from parallelization, BAHA offers CUDA support through the GPU-accelerated variants.

### Spectral Analysis

For large-scale problems, spectral analysis enables O(N log N) scaling instead of O(N²).

## Support

For support, please open an issue on the GitHub repository or consult the community forums.