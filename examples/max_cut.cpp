/*
 * Maximum Cut Problem using BAHA
 * Partition graph vertices into two sets to maximize edges between sets
 */
#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>

struct CutState {
    std::vector<bool> partition;  // true = set A, false = set B
    int n_vertices;
    
    CutState() : n_vertices(0) {}
    CutState(int n) : n_vertices(n), partition(n, false) {}
};

// Simple graph: complete graph or random edges
struct Graph {
    std::vector<std::pair<int, int>> edges;
    int n;
    
    Graph(int vertices, bool complete = false) : n(vertices) {
        if (complete) {
            // Complete graph
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    edges.push_back({i, j});
                }
            }
        } else {
            // Random sparse graph
            std::mt19937 rng(42);
            std::uniform_int_distribution<int> dist(0, n - 1);
            for (int i = 0; i < n * 2; ++i) {
                int u = dist(rng);
                int v = dist(rng);
                if (u != v) {
                    edges.push_back({std::min(u, v), std::max(u, v)});
                }
            }
            // Remove duplicates
            std::sort(edges.begin(), edges.end());
            edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        }
    }
};

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::stoi(argv[1]) : 20;
    bool complete = (argc > 2 && std::string(argv[2]) == "complete");
    
    Graph g(n, complete);
    
    std::cout << "============================================================\n";
    std::cout << "MAX CUT PROBLEM: " << n << " vertices, " << g.edges.size() << " edges\n";
    std::cout << "============================================================\n";
    
    // Energy: negative of cut size (we minimize, so max cut = min -cut)
    auto energy = [&g](const CutState& s) -> double {
        int cut_size = 0;
        for (const auto& e : g.edges) {
            if (s.partition[e.first] != s.partition[e.second]) {
                cut_size++;
            }
        }
        return -static_cast<double>(cut_size);  // Negative because we minimize
    };
    
    // Random partition
    auto sampler = [n]() -> CutState {
        CutState s(n);
        std::mt19937 rng(std::random_device{}());
        std::bernoulli_distribution dist(0.5);
        for (int i = 0; i < n; ++i) {
            s.partition[i] = dist(rng);
        }
        return s;
    };
    
    // Neighbors: flip one vertex
    auto neighbors = [n](const CutState& s) -> std::vector<CutState> {
        std::vector<CutState> nbrs;
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, n - 1);
        
        for (int k = 0; k < 30; ++k) {
            CutState nbr = s;
            int flip = dist(rng);
            nbr.partition[flip] = !nbr.partition[flip];
            nbrs.push_back(nbr);
        }
        return nbrs;
    };
    
    navokoj::BranchAwareOptimizer<CutState> opt(energy, sampler, neighbors);
    
    typename navokoj::BranchAwareOptimizer<CutState>::Config config;
    config.beta_steps = 300;
    config.beta_end = 10.0;
    config.samples_per_beta = 40;
    config.fracture_threshold = 1.6;
    config.max_branches = 6;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<CutState>::ScheduleType::GEOMETRIC;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = opt.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    int max_cut = static_cast<int>(-result.best_energy);
    int total_edges = g.edges.size();
    
    std::cout << "\nResult:\n";
    std::cout << "Max cut size: " << max_cut << " / " << total_edges << " edges\n";
    std::cout << "Cut ratio: " << std::fixed << std::setprecision(2) 
              << (100.0 * max_cut / total_edges) << "%\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed << "s\n";
    
    // Show partition
    std::cout << "\nPartition:\n";
    std::cout << "Set A: ";
    for (int i = 0; i < n; ++i) {
        if (result.best_state.partition[i]) std::cout << i << " ";
    }
    std::cout << "\nSet B: ";
    for (int i = 0; i < n; ++i) {
        if (!result.best_state.partition[i]) std::cout << i << " ";
    }
    std::cout << "\n";
    
    return 0;
}
