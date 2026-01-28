/*
 * Author: Sethurathienam Iyer
 */
#include "baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

// ============================================================================
// CUDA RAMSEY BENCHMARK: R(5,5,5) @ N=52
// ============================================================================

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d
", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

// Global variable for the count (simplest for atomicAdd)
__device__ int g_mono_count = 0;

// CUDA Kernel: Each thread checks one clique
__global__ void count_mono_cliques_kernel(const int* d_edges, const int* d_clique_edges, int num_cliques, int k_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cliques) {
        // Offset into d_clique_edges
        const int* c_edges = &d_clique_edges[idx * k_edges];
        
        int first_color = d_edges[c_edges[0]];
        bool is_mono = true;
        for (int i = 1; i < k_edges; ++i) {
            if (d_edges[c_edges[i]] != first_color) {
                is_mono = false;
                break;
            }
        }
        
        if (is_mono) {
            atomicAdd(&g_mono_count, 1);
        }
    }
}

class RamseyCuda {
    int N;
    int K;
    int Colors;
    int num_edges;
    int num_cliques;
    int k_edges; // Number of edges in a K-clique (K*(K-1)/2)

    int* d_clique_edges; // Flattened list of edge indices for all cliques
    int* d_edges_state;  // Current edge colors

public:
    RamseyCuda(int n, int k, int colors) : N(n), K(k), Colors(colors) {
        num_edges = N * (N - 1) / 2;
        k_edges = K * (K - 1) / 2;
        
        std::cout << "Precomputing " << N << "-vertex Ramsey problem (K=" << K << ")..." << std::endl;
        
        // 1. Generate Cliques on CPU
        std::vector<int> clique_indices;
        std::vector<int> selector(N);
        std::fill(selector.end() - K, selector.end(), 1);

        num_cliques = 0;
        do {
            std::vector<int> verts;
            for (int i = 0; i < N; ++i) if (selector[i]) verts.push_back(i);
            
            for (int i = 0; i < K; ++i) {
                for (int j = i + 1; j < K; ++j) {
                    clique_indices.push_back(edge_idx(verts[i], verts[j]));
                }
            }
            num_cliques++;
        } while (std::next_permutation(selector.begin(), selector.end()));

        std::cout << "Allocating GPU memory for " << num_cliques << " cliques (" << num_cliques * k_edges * 4 / (1024*1024) << " MB)..." << std::endl;

        // 2. Upload to GPU
        CHECK_CUDA(cudaMalloc(&d_clique_edges, clique_indices.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_clique_edges, clique_indices.data(), clique_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        
        CHECK_CUDA(cudaMalloc(&d_edges_state, num_edges * sizeof(int)));
    }

    ~RamseyCuda() {
        cudaFree(d_clique_edges);
        cudaFree(d_edges_state);
    }

    int edge_idx(int u, int v) const {
        if (u > v) std::swap(u, v);
        return (2 * N - 1 - u) * u / 2 + v - u - 1;
    }

    double energy(const std::vector<int>& host_edges) {
        // Upload state
        CHECK_CUDA(cudaMemcpy(d_edges_state, host_edges.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
        
        // Reset device counter
        int zero = 0;
        CHECK_CUDA(cudaMemcpyToSymbol(g_mono_count, &zero, sizeof(int)));

        // Launch Kernel
        int threadsPerBlock = 256;
        int blocks = (num_cliques + threadsPerBlock - 1) / threadsPerBlock;
        count_mono_cliques_kernel<<<blocks, threadsPerBlock>>>(d_edges_state, d_clique_edges, num_cliques, k_edges);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Download result
        int host_mono_count;
        CHECK_CUDA(cudaMemcpyFromSymbol(&host_mono_count, g_mono_count, sizeof(int)));

        return (double)host_mono_count;
    }

    std::vector<int> random_state() {
        std::vector<int> s(num_edges);
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, Colors - 1);
        for(int i=0; i<num_edges; ++i) s[i] = dist(rng);
        return s;
    }

    std::vector<std::vector<int>> neighbors(const std::vector<int>& s) {
        std::vector<std::vector<int>> nbrs;
        // Sample subset of neighbors for N=52 to keep things moving
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> edge_dist(0, num_edges - 1);
        std::uniform_int_distribution<int> color_dist(1, Colors - 1);

        for (int i = 0; i < 64; ++i) { // 64 random flips per step
            std::vector<int> n = s;
            int idx = edge_dist(rng);
            n[idx] = (n[idx] + color_dist(rng)) % Colors;
            nbrs.push_back(n);
        }
        return nbrs;
    }

    void verify(const std::vector<int>& edges) {
        int count = (int)energy(edges);
        if (count == 0) std::cout << "✅ VALID COLORING FOUND! R(5,5,5) > " << N << " Proven.
";
        else std::cout << "❌ Invalid. Mono cliques: " << count << "
";
    }
};

int main() {
    std::cout << "🚀 CUDA RAMSEY SOLVER: R(5,5,5) @ N=52 🚀" << std::endl;
    std::cout << "Utilizing GPU to parallelize 2.6M clique checks." << std::endl;
    
    RamseyCuda problem(52, 5, 3);

    auto energy = [&](const std::vector<int>& s) { return problem.energy(s); };
    auto sampler = [&]() { return problem.random_state(); };
    auto neighbors = [&](const std::vector<int>& s) { return problem.neighbors(s); };

    navokoj::BranchAwareOptimizer<std::vector<int>> baha(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<std::vector<int>>::Config config;
    
    config.beta_steps = 500;
    config.beta_end = 15.0; 
    config.samples_per_beta = 30;
    config.fracture_threshold = 1.6;
    config.max_branches = 6;
    config.verbose = true;
    config.schedule_type = navokoj::BranchAwareOptimizer<std::vector<int>>::ScheduleType::GEOMETRIC;

    std::cout << "
Starting BAHA (CUDA Accelerated)..." << std::endl;
    auto result = baha.optimize(config);

    std::cout << "
RESULT:" << std::endl;
    std::cout << "Final Energy: " << result.best_energy << std::endl;
    problem.verify(result.best_state);

    return 0;
}
