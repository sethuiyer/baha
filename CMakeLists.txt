cmake_minimum_required(VERSION 3.12)
project(baha VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_BENCHMARKS "Build benchmark programs" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -Wall -Wextra")

# Find CUDA if enabled
if(ENABLE_CUDA)
  find_package(CUDA QUIET)
  if(CUDA_FOUND)
    enable_language(CUDA)
    set(HAVE_CUDA TRUE)
    message(STATUS "CUDA support enabled")
  else()
    set(HAVE_CUDA FALSE)
    message(WARNING "CUDA not found, disabling CUDA support")
  endif()
endif()

# Main library
add_library(baha INTERFACE)
target_include_directories(baha INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Examples
if(BUILD_EXAMPLES)
  # Spectrum auction example
  add_executable(spectrum_auction examples/spectrum_auction.cpp)
  target_link_libraries(spectrum_auction baha)
  set_property(TARGET spectrum_auction PROPERTY CXX_STANDARD 17)

  # List coloring example
  add_executable(list_coloring examples/list_coloring.cpp)
  target_link_libraries(list_coloring baha)
  set_property(TARGET list_coloring PROPERTY CXX_STANDARD 17)

  # ISR benchmarks
  add_executable(isr_benchmarks examples/isr_benchmarks.cpp)
  target_link_libraries(isr_benchmarks baha)
  set_property(TARGET isr_benchmarks PROPERTY CXX_STANDARD 17)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
  # SAT benchmark
  add_executable(sat_benchmark benchmarks/casimir_sat.cpp)
  target_link_libraries(sat_benchmark baha)
  set_property(TARGET sat_benchmark PROPERTY CXX_STANDARD 17)

  # Chaotic benchmark
  add_executable(chaotic_benchmark benchmarks/chaotic_benchmark.cpp)
  target_link_libraries(chaotic_benchmark baha)
  set_property(TARGET chaotic_benchmark PROPERTY CXX_STANDARD 17)

  # Graph isomorphism benchmark
  add_executable(graph_iso_benchmark benchmarks/graph_iso_benchmark.cpp)
  target_link_libraries(graph_iso_benchmark baha)
  set_property(TARGET graph_iso_benchmark PROPERTY CXX_STANDARD 17)

  # Graph isomorphism hard benchmark
  add_executable(graph_iso_benchmark_hard benchmarks/graph_iso_benchmark_hard.cpp)
  target_link_libraries(graph_iso_benchmark_hard baha)
  set_property(TARGET graph_iso_benchmark_hard PROPERTY CXX_STANDARD 17)

  # Diagnostic benchmark
  add_executable(diagnostic benchmarks/diagnostic.cpp)
  target_link_libraries(diagnostic baha)
  set_property(TARGET diagnostic PROPERTY CXX_STANDARD 17)
endif()

# GPU support if available
if(HAVE_CUDA)
  # GPU accelerated optimizer
  add_executable(baha_gpu src/baha_gpu.cu)
  set_property(TARGET baha_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET baha_gpu PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# Install targets
install(DIRECTORY include/ DESTINATION include)
install(TARGETS baha EXPORT baha_targets)
install(EXPORT baha_targets FILE baha-config.cmake DESTINATION lib/cmake/baha)

# Export package
include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/baha-config.cmake.in"
  "${PROJECT_BINARY_DIR}/baha-config.cmake"
  INSTALL_DESTINATION lib/cmake/baha
)