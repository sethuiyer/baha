/*
 * Author: Sethurathienam Iyer
 */
#include "baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <fstream>

// External CUDA wrapper functions
extern "C" {
    void cuda_malloc_int(int** ptr, size_t size);
    void cuda_free(void* ptr);
    void cuda_memcpy_h2d(int* dest, const int* src, size_t size);
    int launch_ramsey_kernel(int* d_edges, int* d_clique_edges, int num_cliques, int k_edges);
}

class RamseyCuda {
    int N;
    int K;
    int Colors;
    int num_edges;
    int num_cliques;
    int k_edges; 

    int* d_clique_edges; 
    int* d_edges_state;  

public:
    RamseyCuda(int n, int k, int colors) : N(n), K(k), Colors(colors) {
        num_edges = N * (N - 1) / 2;
        k_edges = K * (K - 1) / 2;
        
        std::cout << "Precomputing " << N << "-vertex Ramsey problem (K=" << K << ")..." << std::endl;
        
        std::vector<int> clique_indices;
        std::vector<int> selector(N);
        std::fill(selector.end() - K, selector.end(), 1);

        num_cliques = 0;
        // Helper to compute edge index
        auto edge_idx = [&](int u, int v) {
            if (u > v) std::swap(u, v);
            return (2 * N - 1 - u) * u / 2 + v - u - 1;
        };

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

        std::cout << "Allocating GPU memory for " << num_cliques << " cliques (" << (clique_indices.size() * 4) / (1024*1024) << " MB)..." << std::endl;

        cuda_malloc_int(&d_clique_edges, clique_indices.size() * sizeof(int));
        cuda_memcpy_h2d(d_clique_edges, clique_indices.data(), clique_indices.size() * sizeof(int));
        
        cuda_malloc_int(&d_edges_state, num_edges * sizeof(int));
    }

    ~RamseyCuda() {
        cuda_free(d_clique_edges);
        cuda_free(d_edges_state);
    }

    double energy(const std::vector<int>& host_edges) {
        cuda_memcpy_h2d(d_edges_state, host_edges.data(), num_edges * sizeof(int));
        return (double)launch_ramsey_kernel(d_edges_state, d_clique_edges, num_cliques, k_edges);
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
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> edge_dist(0, num_edges - 1);
        
        // Use 100 neighbors to keep GPU saturated but nimble
        for (int i = 0; i < 100; ++i) { 
            std::vector<int> n = s;
            int idx = edge_dist(rng);
            n[idx] = (n[idx] + 1) % Colors; // Simple flip
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
    
    config.beta_steps = 1000;
    config.beta_end = 20.0; 
    config.samples_per_beta = 30; // GPU is fast but let's not overdo sampling
    config.fracture_threshold = 2.0;
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

    // Export witness CSV for verification
    if (result.best_energy == 0) {
        std::ofstream csv("data/ramsey_52_witness.csv");
        csv << "edge_index,u,v,color
";
        int idx = 0;
        for (int u = 0; u < 52; ++u) {
            for (int v = u + 1; v < 52; ++v) {
                csv << idx << "," << u << "," << v << "," << result.best_state[idx] << "
";
                idx++;
            }
        }
        csv.close();
        std::cout << "✅ Witness exported to data/ramsey_52_witness.csv
";
    }

    return 0;
}
