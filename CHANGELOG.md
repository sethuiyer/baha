# Changelog

All notable changes to the BAHA project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of BAHA (Branch-Aware Optimizer)
- Core BranchAwareOptimizer template class
- Lambert-W function implementation with numerical stability
- Fracture detection via ρ = |d/dβ log Z|
- Branch enumeration using Lambert-W function
- GPU acceleration support via CUDA
- Spectral analysis for O(N log N) scaling
- Comprehensive benchmark suite
- Examples for various optimization problems
- RAMSEY theory solver capability
- Side-channel attack analysis tools
- Combinatorial auction optimization

### Changed
- Original research implementation to production-ready library
- Single-header to organized include structure
- Basic examples to comprehensive tutorial suite

### Deprecated
- None

### Removed
- None

### Fixed
- Numerical stability in Lambert-W computation
- Memory management in GPU implementation
- Thread safety in multi-instance scenarios

### Security
- None

## [1.0.0] - 2026-01-27

### Added
- Initial public release
- Core optimization algorithms
- Documentation and examples
- Build system (CMake)
- License and contributing guidelines