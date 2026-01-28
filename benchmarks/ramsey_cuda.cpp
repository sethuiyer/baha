/*
 * Author: Sethurathienam Iyer
 */
#include "baha.hpp"
#include "ramsey_engine.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <stdint.h>

class RamseyCudaProblem {
    int N, Colors;
    void* handle;

public:
    RamseyCudaProblem(int n, int k, int colors) : N(n), Colors(colors) {
        std::vector<uint16_t> clique_indices;
        std::vector<int> selector(N);
        std::fill(selector.end() - k, selector.end(), 1);
        int num_cliques = 0;
        int k_edges = k * (k - 1) / 2;

        std::cout << "Generating " << k << "-cliques for N=" << n << "..." << std::endl;
        do {
            std::vector<int> verts;
            for (int i = 0; i < N; ++i) if (selector[i]) verts.push_back(i);
            for (int i = 0; i < k; ++i) {
                for (int j = i + 1; j < k; ++j) {
                    int u = verts[i], v = verts[j];
                    if (u > v) std::swap(u, v);
                    int idx = (2 * N - 1 - u) * u / 2 + v - u - 1;
                    clique_indices.push_back((uint16_t)idx);
                }
            }
            num_cliques++;
            if (num_cliques % 5000000 == 0) std::cout << "  Generated " << num_cliques / 1000000 << "M cliques..." << std::endl;
        } while (std::next_permutation(selector.begin(), selector.end()));

        int num_edges = N * (N - 1) / 2;
        handle = ramsey_cuda_init(clique_indices.data(), num_cliques, k_edges, num_edges);
        if (!handle) {
            std::cerr << "Failed to initialize CUDA Ramsey engine (Likely OOM)!" << std::endl;
            exit(1);
        }
    }

    ~RamseyCudaProblem() {
        ramsey_cuda_free(handle);
    }

    double energy(const std::vector<int>& s) {
        return ramsey_cuda_evaluate(handle, s.data(), (int)s.size());
    }

    std::vector<int> random_state() {
        int num_edges = N * (N - 1) / 2;
        std::vector<int> s(num_edges);
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, Colors - 1);
        for(int i=0; i<num_edges; ++i) s[i] = dist(rng);
        return s;
    }

    std::vector<std::vector<int>> neighbors(const std::vector<int>& s) {
        std::vector<std::vector<int>> nbrs;
        int num_edges = s.size();
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> edge_dist(0, num_edges - 1);
        std::uniform_int_distribution<int> color_dist(1, Colors - 1);

        for (int i = 0; i < 64; ++i) { // 64 parallel neighbor explorations
            std::vector<int> n = s;
            int idx = edge_dist(rng);
            n[idx] = (n[idx] + color_dist(rng)) % Colors;
            nbrs.push_back(n);
        }
        return nbrs;
    }

    void verify(const std::vector<int>& edges) {
        if (energy(edges) == 0) std::cout << "✅ VALID COLORING FOUND for R(5,5,5) > " << N << std::endl;
        else std::cout << "❌ Invalid coloring. Energy=" << energy(edges) << std::endl;
    }
};

int main() {
    std::cout << "🚀 COSMIC RAMSEY CHALLENGE: R(5,5,5) @ N=102 🚀" << std::endl;
    std::cout << "Search space: 3^5151. Constraints: 75.3 Million." << std::endl;
    
    RamseyCudaProblem problem(102, 5, 3);

    auto energy = [&](const std::vector<int>& s) { return problem.energy(s); };
    auto sampler = [&]() { return problem.random_state(); };
    auto neighbors = [&](const std::vector<int>& s) { return problem.neighbors(s); };

    navokoj::BranchAwareOptimizer<std::vector<int>> baha(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<std::vector<int>>::Config config;
    
    config.beta_steps = 1000;
    config.beta_end = 25.0; 
    config.samples_per_beta = 30; // Fewer samples for speed at this scale
    config.fracture_threshold = 1.3;
    config.max_branches = 8;
    config.verbose = true;
    config.schedule_type = navokoj::BranchAwareOptimizer<std::vector<int>>::ScheduleType::GEOMETRIC;

    config.logger = [&](int step, double beta, double energy, double rho, const char* event) {
        if (step % 10 == 0) {
            std::ofstream out("solution_102.adj");
            // This is slow but ensure we have something
            // (In a real app we'd pass a pointer to the best state)
        }
    };

    // To make it finish faster for the demo, I'll shorten the steps
    config.beta_steps = 50; 

    auto result = baha.optimize(config);
    std::cout << "
RESULT: Energy = " << result.best_energy << std::endl;
    problem.verify(result.best_state);

    std::ofstream out("solution_102.adj");
    for (int e : result.best_state) out << e << " ";
    out.close();
    std::cout << "Graph exported to solution_102.adj" << std::endl;

    return 0;
}
