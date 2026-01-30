cmake_minimum_required(VERSION 3.12)
project(baha VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_BENCHMARKS "Build benchmark programs" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_MPS "Enable Metal/MPS support" ON)
option(BUILD_TESTS "Build test programs" ON)

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

# Enable Metal/MPS on macOS
if(ENABLE_MPS AND APPLE)
  enable_language(OBJCXX)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    set(HAVE_MPS TRUE)
    message(STATUS "Metal/MPS support enabled")
  else()
    set(HAVE_MPS FALSE)
    message(WARNING "Metal frameworks not found, disabling MPS support")
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

  # New problem types
  add_executable(n_queens examples/n_queens.cpp)
  target_link_libraries(n_queens baha)
  set_property(TARGET n_queens PROPERTY CXX_STANDARD 17)

  add_executable(max_cut examples/max_cut.cpp)
  target_link_libraries(max_cut baha)
  set_property(TARGET max_cut PROPERTY CXX_STANDARD 17)

  add_executable(knapsack examples/knapsack.cpp)
  target_link_libraries(knapsack baha)
  set_property(TARGET knapsack PROPERTY CXX_STANDARD 17)

  add_executable(traveling_salesman examples/traveling_salesman.cpp)
  target_link_libraries(traveling_salesman baha)
  set_property(TARGET traveling_salesman PROPERTY CXX_STANDARD 17)

  add_executable(vehicle_routing examples/vehicle_routing.cpp)
  target_link_libraries(vehicle_routing baha)
  set_property(TARGET vehicle_routing PROPERTY CXX_STANDARD 17)

  add_executable(bin_packing examples/bin_packing.cpp)
  target_link_libraries(bin_packing baha)
  set_property(TARGET bin_packing PROPERTY CXX_STANDARD 17)

  add_executable(max_clique examples/max_clique.cpp)
  target_link_libraries(max_clique baha)
  set_property(TARGET max_clique PROPERTY CXX_STANDARD 17)

  add_executable(course_scheduling examples/course_scheduling.cpp)
  target_link_libraries(course_scheduling baha)
  set_property(TARGET course_scheduling PROPERTY CXX_STANDARD 17)

  add_executable(network_design examples/network_design.cpp)
  target_link_libraries(network_design baha)
  set_property(TARGET network_design PROPERTY CXX_STANDARD 17)

  add_executable(resource_allocation examples/resource_allocation.cpp)
  target_link_libraries(resource_allocation baha)
  set_property(TARGET resource_allocation PROPERTY CXX_STANDARD 17)

  add_executable(adversarial_ml examples/adversarial_ml.cpp)
  target_link_libraries(adversarial_ml baha)
  set_property(TARGET adversarial_ml PROPERTY CXX_STANDARD 17)

  add_executable(graph_coloring examples/graph_coloring.cpp)
  target_link_libraries(graph_coloring baha)
  set_property(TARGET graph_coloring PROPERTY CXX_STANDARD 17)

  # Incident response playbook example
  add_executable(incident_response examples/incident_response.cpp)
  target_link_libraries(incident_response baha)
  set_property(TARGET incident_response PROPERTY CXX_STANDARD 17)
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

  # Graph coloring distribution test
  add_executable(graph_coloring_distribution benchmarks/graph_coloring_distribution.cpp)
  target_link_libraries(graph_coloring_distribution baha)
  set_property(TARGET graph_coloring_distribution PROPERTY CXX_STANDARD 17)

  # Diagnostic benchmark
  add_executable(diagnostic benchmarks/diagnostic.cpp)
  target_link_libraries(diagnostic baha)
  set_property(TARGET diagnostic PROPERTY CXX_STANDARD 17)

  # Mega Landscape Benchmark
  add_executable(mega_benchmark benchmarks/mega_landscape_benchmark.cpp)
  target_link_libraries(mega_benchmark baha)
  set_property(TARGET mega_benchmark PROPERTY CXX_STANDARD 17)

  # MILP Benchmark (Facility Location)
  add_executable(milp_benchmark benchmarks/milp_benchmark.cpp)
  target_link_libraries(milp_benchmark baha)
  set_property(TARGET milp_benchmark PROPERTY CXX_STANDARD 17)

  # Unified Ramsey solver (CPU/CUDA/MPS)
  set(RAMSEY_UNIFIED_SOURCES
    benchmarks/ramsey_unified.cpp
    benchmarks/ramsey_backend_cpu.cpp
  )

  if(HAVE_CUDA)
    list(APPEND RAMSEY_UNIFIED_SOURCES benchmarks/ramsey_backend_cuda.cu)
  endif()

  if(HAVE_MPS)
    list(APPEND RAMSEY_UNIFIED_SOURCES benchmarks/ramsey_backend_mps.mm)
  endif()

  add_executable(ramsey_unified ${RAMSEY_UNIFIED_SOURCES})
  target_link_libraries(ramsey_unified baha)
  target_include_directories(ramsey_unified PRIVATE ${PROJECT_SOURCE_DIR}/benchmarks)
  set_property(TARGET ramsey_unified PROPERTY CXX_STANDARD 17)

  if(HAVE_CUDA)
    set_property(TARGET ramsey_unified PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ramsey_unified PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_definitions(ramsey_unified PRIVATE BAHA_HAVE_CUDA_BACKEND=1)
  endif()

  if(HAVE_MPS)
    target_compile_definitions(ramsey_unified PRIVATE BAHA_HAVE_MPS_BACKEND=1)
    target_link_libraries(ramsey_unified ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
  endif()
endif()

# Tests
add_executable(unit_tests tests/unit_tests.cpp)
target_link_libraries(unit_tests baha)
set_property(TARGET unit_tests PROPERTY CXX_STANDARD 17)

# GPU support if available
if(HAVE_CUDA)
  # GPU accelerated optimizer
  add_executable(baha_gpu src/baha_gpu.cu)
  set_property(TARGET baha_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET baha_gpu PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# Tests
if(BUILD_TESTS)
  set(RAMSEY_TEST_SOURCES
    tests/ramsey_backend_test.cpp
    benchmarks/ramsey_backend_cpu.cpp
  )

  if(HAVE_CUDA)
    list(APPEND RAMSEY_TEST_SOURCES benchmarks/ramsey_backend_cuda.cu)
  endif()

  if(HAVE_MPS)
    list(APPEND RAMSEY_TEST_SOURCES benchmarks/ramsey_backend_mps.mm)
  endif()

  add_executable(ramsey_backend_test ${RAMSEY_TEST_SOURCES})
  target_link_libraries(ramsey_backend_test baha)
  target_include_directories(ramsey_backend_test PRIVATE ${PROJECT_SOURCE_DIR}/benchmarks)
  set_property(TARGET ramsey_backend_test PROPERTY CXX_STANDARD 17)

  if(HAVE_CUDA)
    set_property(TARGET ramsey_backend_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ramsey_backend_test PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_definitions(ramsey_backend_test PRIVATE BAHA_HAVE_CUDA_BACKEND=1)
  endif()

  if(HAVE_MPS)
    target_compile_definitions(ramsey_backend_test PRIVATE BAHA_HAVE_MPS_BACKEND=1)
    target_link_libraries(ramsey_backend_test ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
  endif()

  # Cross-platform BAHA test
  add_executable(cross_platform_test tests/cross_platform_test.cpp)
  target_link_libraries(cross_platform_test baha)
  set_property(TARGET cross_platform_test PROPERTY CXX_STANDARD 17)

  if(HAVE_CUDA)
    target_compile_definitions(cross_platform_test PRIVATE BAHA_HAVE_CUDA_BACKEND=1)
  endif()

  if(HAVE_MPS)
    target_compile_definitions(cross_platform_test PRIVATE BAHA_HAVE_MPS_BACKEND=1)
    target_link_libraries(cross_platform_test ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
  endif()
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