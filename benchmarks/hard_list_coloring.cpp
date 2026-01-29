/*
 * HARD LIST COLORING BENCHMARK (N=500)
 * BAHA Only
 */
#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <set>
#include <algorithm>
#include <iomanip>
#include <chrono>

struct ListColoringState {
    std::vector<int> coloring;
};

class ListColoringProblem {
public:
    ListColoringProblem(int n_vertices, const std::vector<std::pair<int, int>>& edges, 
                       const std::vector<std::vector<int>>& allowed_colors, int n_total_colors, int seed)
        : n_vertices_(n_vertices), edges_(edges), allowed_colors_(allowed_colors), 
          n_total_colors_(n_total_colors), rng_(seed) {}

    double energy(const ListColoringState& state) const {
        int conflicts = 0;
        int unassigned = 0;
        for (const auto& [u, v] : edges_) {
            if (state.coloring[u] != -1 && state.coloring[v] != -1 && state.coloring[u] == state.coloring[v]) {
                conflicts++;
            }
        }
        for (int i = 0; i < n_vertices_; ++i) {
            if (state.coloring[i] == -1) {
                unassigned++;
            } else {
                bool valid_color = false;
                for (int c : allowed_colors_[i]) if (c == state.coloring[i]) valid_color = true;
                if (!valid_color) conflicts += 10;
            }
        }
        return static_cast<double>(conflicts * 1000 + unassigned * 10); // Heavy penalty for conflicts
    }

    ListColoringState random_state() const {
        ListColoringState state;
        state.coloring.resize(n_vertices_);
        std::uniform_int_distribution<> dist(0, 1000);
        for (int i = 0; i < n_vertices_; ++i) {
            if (dist(rng_) % 3 == 0) state.coloring[i] = -1;
            else {
                const auto& allowed = allowed_colors_[i];
                if (!allowed.empty()) state.coloring[i] = allowed[dist(rng_) % allowed.size()];
                else state.coloring[i] = -1;
            }
        }
        return state;
    }

    std::vector<ListColoringState> neighbors(const ListColoringState& state) const {
        std::vector<ListColoringState> nbrs;
        std::uniform_int_distribution<> v_dist(0, n_vertices_-1);
        
        // Optimize: Generate a subset of neighbors for speed on N=500
        // Or BAHA hot path will do it. Let's return full neighborhood for single-node flips.
        // N=500, roughly 3 colors/node -> ~1500 neighbors.
        // Evaluating all might be slow.
        // Let's rely on BAHA's internal caching if it had it, but here we construct.
        // For N=500, constructing 1500 vectors is expensive.
        // But baha.hpp iterates whatever we return.
        
        // Let's generate a smaller stochastic neighborhood? 
        // No, standard SA/BAHA expects full neighborhood or a representative sample.
        // Let's generate ~50 random neighbors to keep speed high.
        
        for(int k=0; k<50; ++k) {
             int i = v_dist(rng_);
             const auto& allowed = allowed_colors_[i];
             if(allowed.empty()) continue;
             
             int new_c = allowed[rng_() % allowed.size()];
             if(new_c == state.coloring[i]) {
                 // Try unsetting
                 if(state.coloring[i] != -1) new_c = -1;
                 else continue;
             }
             
             ListColoringState nbr = state;
             nbr.coloring[i] = new_c;
             nbrs.push_back(nbr);
        }
        return nbrs;
    }

    int n_vertices_;
    std::vector<std::pair<int, int>> edges_;
    std::vector<std::vector<int>> allowed_colors_;
    int n_total_colors_;
    mutable std::mt19937 rng_;
};

int main() {
    std::cout << "🎨 HARD LIST COLORING (BAHA ONLY) 🎨\n";
    std::cout << "N=500, Density=4.0, Colors=10, Allowed/Node=3\n";
    std::cout << "=============================================\n";

    // VERY hard setup
    int N = 500;
    int M = 2000; // Density 4.0
    int total_colors = 10;
    int allowed_per_node = 3;
    
    std::mt19937 gen(42);
    std::vector<std::pair<int, int>> edges;
    std::uniform_int_distribution<> v_dist(0, N-1);
    
    // Ensure connectivity base
    for (int i = 1; i < N; ++i) edges.push_back({i-1, i});
    
    // Random edges
    while(edges.size() < M) {
        int u = v_dist(gen);
        int v = v_dist(gen);
        if(u!=v) edges.push_back({u,v});
    }
    
    std::vector<std::vector<int>> allowed(N);
    std::uniform_int_distribution<> c_dist(0, total_colors-1);
    for(int i=0; i<N; ++i) {
        std::set<int> s;
        while(s.size() < allowed_per_node) s.insert(c_dist(gen));
        allowed[i].assign(s.begin(), s.end());
    }
    
    ListColoringProblem prob(N, edges, allowed, total_colors, 42);

    std::function<double(const ListColoringState&)> energy = [&](const ListColoringState& s) { return prob.energy(s); };
    std::function<ListColoringState()> sampler = [&]() { return prob.random_state(); };
    std::function<std::vector<ListColoringState>(const ListColoringState&)> neighbors = [&](const ListColoringState& s) { return prob.neighbors(s); };

    // Run BAHA
    std::cout << "Running BAHA...\n";
    navokoj::BranchAwareOptimizer<ListColoringState> ba(energy, sampler, neighbors);
    typename navokoj::BranchAwareOptimizer<ListColoringState>::Config ba_config;
    ba_config.beta_steps = 1000;
    ba_config.beta_end = 20.0;
    ba_config.fracture_threshold = 1.2; // Sensitive
    ba_config.max_branches = 5;
    ba_config.verbose = true;
    
    auto ba_start = std::chrono::high_resolution_clock::now();
    auto ba_res = ba.optimize(ba_config);
    auto ba_end = std::chrono::high_resolution_clock::now();
    double ba_time = std::chrono::duration<double>(ba_end - ba_start).count();

    std::cout << "\nRESULTS:\n";
    std::cout << "Final Energy: " << ba_res.best_energy << "\n";
    std::cout << "Time: " << ba_time << "s\n";
    std::cout << "Fractures: " << ba_res.fractures_detected << "\n";
    std::cout << "Jumps: " << ba_res.branch_jumps << "\n";
    
    if(ba_res.best_energy == 0) std::cout << "✅ PERFECT SOLUTION FOUND\n";
    else std::cout << "❌ CONFLICTS REMAIN\n";

    return 0;
}
