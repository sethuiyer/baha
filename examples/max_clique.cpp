/*
 * Maximum Clique Problem using BAHA
 * Find the largest complete subgraph (clique) in a graph
 */
#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <set>
#include <numeric>

struct CliqueState {
    std::vector<bool> in_clique;  // in_clique[i] = true if vertex i is in clique
    int n_vertices;
    
    CliqueState() : n_vertices(0) {}
    CliqueState(int n) : n_vertices(n), in_clique(n, false) {}
};

struct Graph {
    std::vector<std::vector<bool>> adj;  // adj[i][j] = true if edge exists
    int n;
    
    Graph(int vertices, double edge_prob = 0.3) : n(vertices) {
        adj.assign(n, std::vector<bool>(n, false));
        std::mt19937 rng(42);
        std::bernoulli_distribution dist(edge_prob);
        
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (dist(rng)) {
                    adj[i][j] = adj[j][i] = true;
                }
            }
        }
    }
    
    bool is_clique(const std::vector<int>& vertices) const {
        for (size_t i = 0; i < vertices.size(); ++i) {
            for (size_t j = i + 1; j < vertices.size(); ++j) {
                if (!adj[vertices[i]][vertices[j]]) {
                    return false;
                }
            }
        }
        return true;
    }
};

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::stoi(argv[1]) : 30;
    double edge_prob = (argc > 2) ? std::stod(argv[2]) : 0.3;
    
    Graph g(n, edge_prob);
    
    // Count edges
    int edges = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (g.adj[i][j]) edges++;
        }
    }
    
    std::cout << "============================================================\n";
    std::cout << "MAXIMUM CLIQUE PROBLEM: " << n << " vertices, " << edges << " edges\n";
    std::cout << "============================================================\n";
    
    // Energy: negative clique size + penalty for non-clique
    auto energy = [&g](const CliqueState& s) -> double {
        std::vector<int> clique;
        for (int i = 0; i < s.n_vertices; ++i) {
            if (s.in_clique[i]) {
                clique.push_back(i);
            }
        }
        
        if (clique.empty()) return 1000.0;
        
        // Check if it's a valid clique
        bool valid = g.is_clique(clique);
        if (!valid) {
            // Count missing edges
            int missing = 0;
            for (size_t i = 0; i < clique.size(); ++i) {
                for (size_t j = i + 1; j < clique.size(); ++j) {
                    if (!g.adj[clique[i]][clique[j]]) {
                        missing++;
                    }
                }
            }
            return 1000.0 + missing * 100.0 - clique.size();
        }
        
        return -static_cast<double>(clique.size());  // Negative because we minimize
    };
    
    // Random initial clique
    auto sampler = [n]() -> CliqueState {
        CliqueState s(n);
        std::mt19937 rng(std::random_device{}());
        std::bernoulli_distribution dist(0.2);  // Start with small clique
        for (int i = 0; i < n; ++i) {
            s.in_clique[i] = dist(rng);
        }
        return s;
    };
    
    // Neighbors: add/remove vertex, or swap vertex
    auto neighbors = [n](const CliqueState& s) -> std::vector<CliqueState> {
        std::vector<CliqueState> nbrs;
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> vertex_dist(0, n - 1);
        
        for (int k = 0; k < 30; ++k) {
            CliqueState nbr = s;
            int op = rng() % 2;
            
            if (op == 0) {
                // Toggle a vertex
                int v = vertex_dist(rng);
                nbr.in_clique[v] = !nbr.in_clique[v];
            } else {
                // Swap: remove one, add another
                std::vector<int> in_set, out_set;
                for (int i = 0; i < n; ++i) {
                    if (nbr.in_clique[i]) in_set.push_back(i);
                    else out_set.push_back(i);
                }
                if (!in_set.empty() && !out_set.empty()) {
                    int remove = in_set[rng() % in_set.size()];
                    int add = out_set[rng() % out_set.size()];
                    nbr.in_clique[remove] = false;
                    nbr.in_clique[add] = true;
                }
            }
            
            nbrs.push_back(nbr);
        }
        
        return nbrs;
    };
    
    navokoj::BranchAwareOptimizer<CliqueState> opt(energy, sampler, neighbors);
    
    typename navokoj::BranchAwareOptimizer<CliqueState>::Config config;
    config.beta_steps = 500;
    config.beta_end = 15.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.8;
    config.max_branches = 6;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<CliqueState>::ScheduleType::GEOMETRIC;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = opt.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    // Extract final clique
    std::vector<int> final_clique;
    for (int i = 0; i < n; ++i) {
        if (result.best_state.in_clique[i]) {
            final_clique.push_back(i);
        }
    }
    
    bool is_valid = g.is_clique(final_clique);
    int max_clique_size = static_cast<int>(-result.best_energy);
    
    std::cout << "\nResult:\n";
    std::cout << "Maximum clique size: " << max_clique_size << "\n";
    std::cout << "Valid clique: " << (is_valid ? "Yes" : "No") << "\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed << "s\n";
    
    if (is_valid && !final_clique.empty()) {
        std::cout << "\nClique vertices: ";
        for (int v : final_clique) {
            std::cout << v << " ";
        }
        std::cout << "\n";
    }
    
    return 0;
}
