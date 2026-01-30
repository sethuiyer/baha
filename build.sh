#!/bin/bash

# Build script for BAHA framework
set -e  # Exit on any error

echo "Building BAHA Framework..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_BENCHMARKS=ON \
    -DENABLE_CUDA=ON

# Build the project
echo "Building project..."
# Portable parallelism (Linux + macOS)
JOBS=1
if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
elif [ "$(uname -s)" = "Darwin" ] && command -v sysctl >/dev/null 2>&1; then
    JOBS="$(sysctl -n hw.ncpu)"
elif command -v getconf >/dev/null 2>&1; then
    JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
fi

echo "Using ${JOBS} parallel jobs"
make -j"${JOBS}"

echo ""
echo "Build completed successfully!"
echo ""
echo "Built executables:"
find . -maxdepth 1 -type f -executable -exec basename {} \;
echo ""
echo "To run examples, use: ./examples/<example_name>"
echo "To run benchmarks, use: ./benchmarks/<benchmark_name>"