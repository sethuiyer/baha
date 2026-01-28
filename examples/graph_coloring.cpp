/*
 * General Graph Coloring Problem using BAHA
 * Minimize the number of colors used such that no adjacent vertices share the same color.
 * 
 * This demonstrates BAHA's ability to handle entropy-driven fractures (solution space collapse)
 * and symmetry-breaking fractures (many equivalent colorings).
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

struct GraphColoringState {
    std::vector<int> colors;  // colors[i] = color of vertex i
    int n_vertices;
    
    GraphColoringState() : n_vertices(0) {}
    GraphColoringState(int n, int initial_colors) : n_vertices(n), colors(n, 0) {
        // Initialize with random colors up to initial_colors
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, initial_colors - 1);
        for (int i = 0; i < n_vertices; ++i) {
            colors[i] = dist(rng);
        }
    }
};

struct Graph {
    std::vector<std::vector<bool>> adj;  // adj[i][j] = true if edge exists
    int n_vertices;
    
    Graph(int n, double edge_prob = 0.5) : n_vertices(n) {
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
};

struct GraphColoringProblem {
    Graph graph;
    
    GraphColoringProblem(int n_vertices, double edge_prob = 0.5)
        : graph(n_vertices, edge_prob) {}
    
    // Independent verification of coloring validity
    double verify_coloring(const GraphColoringState& state) const {
        double conflicts = 0;
        for (int i = 0; i < graph.n_vertices; ++i) {
            for (int j = i + 1; j < graph.n_vertices; ++j) {
                if (graph.adj[i][j] && state.colors[i] == state.colors[j]) {
                    conflicts += 1;
                }
            }
        }
        return conflicts;
    }

    // Energy function: minimize conflicts and number of colors used
    double energy(const GraphColoringState& state) const {
        double conflicts = 0;
        std::set<int> unique_colors;
        
        for (int i = 0; i < graph.n_vertices; ++i) {
            unique_colors.insert(state.colors[i]);
            for (int j = i + 1; j < graph.n_vertices; ++j) {
                if (graph.adj[i][j] && state.colors[i] == state.colors[j]) {
                    conflicts += 1;
                }
            }
        }
        
        // Penalize conflicts heavily, then penalize using more colors
        return conflicts * 1000.0 + unique_colors.size();
    }
};

int main() {
    std::cout << "🎨 General Graph Coloring with BAHA\n";
    std::cout << "====================================\n\n";
    
    const int n_vertices = 20;  // Number of vertices
    const double edge_prob = 0.5; // Probability of an edge existing
    const int initial_max_colors = n_vertices; // Start with enough colors
    
    GraphColoringProblem problem(n_vertices, edge_prob);
    
    // Energy function
    auto energy = [&](const GraphColoringState& s) {
        return problem.energy(s);
    };
    
    // Sampler: starts with random coloring
    auto sampler = [&]() {
        GraphColoringState state(n_vertices, initial_max_colors);
        return state;
    };
    
    // Neighbors: change a random vertex's color to a random existing color or a new one
    auto neighbors = [&](const GraphColoringState& s) {
        std::vector<GraphColoringState> nbrs;
        std::mt19937 rng(std::random_device{}());
        
        // Find current max color used
        int max_color = 0;
        for (int c : s.colors) {
            max_color = std::max(max_color, c);
        }
        
        // Generate 10 neighbors
        for (int i = 0; i < 10; ++i) {
            GraphColoringState nbr = s;
            int vertex_to_change = rng() % n_vertices;
            
            // Option 1: change to an existing color
            if (max_color > 0) {
                nbr.colors[vertex_to_change] = rng() % (max_color + 1);
                nbrs.push_back(nbr);
            }
            
            // Option 2: change to a new color (if we allow more colors)
            if (max_color + 1 < initial_max_colors) { // Ensure we don't exceed initial_max_colors
                nbr = s;
                nbr.colors[vertex_to_change] = max_color + 1;
                nbrs.push_back(nbr);
            }
        }
        
        return nbrs;
    };
    
    // Create BAHA optimizer
    navokoj::BranchAwareOptimizer<GraphColoringState> optimizer(energy, sampler, neighbors);
    
    // Configure BAHA
    navokoj::BranchAwareOptimizer<GraphColoringState>::Config config;
    config.beta_start = 0.01;
    config.beta_end = 100.0;
    config.beta_steps = 3000;
    config.fracture_threshold = 2.0;
    config.samples_per_beta = 100;
    config.schedule_type = navokoj::BranchAwareOptimizer<GraphColoringState>::ScheduleType::GEOMETRIC;
    
    std::cout << "Running BAHA optimization...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = optimizer.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n=== RESULTS ===\n";
    std::cout << "Best energy: " << std::fixed << std::setprecision(6) 
              << result.best_energy << "\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Solve time: " << duration.count() << " ms\n\n";
    
    // Calculate final conflicts and unique colors using independent verifier
    double final_conflicts = problem.verify_coloring(result.best_state);
    std::set<int> final_unique_colors;
    for (int i = 0; i < problem.graph.n_vertices; ++i) {
        final_unique_colors.insert(result.best_state.colors[i]);
    }
    std::cout << "Final conflicts: " << final_conflicts << "\n";
    std::cout << "Colors used: " << final_unique_colors.size() << "\n";
    
    if (final_conflicts == 0) {
        std::cout << "\n✅ SUCCESS: Found a valid coloring with " 
                  << final_unique_colors.size() << " colors!\n";
    } else {
        std::cout << "\n⚠️  Did not find a valid coloring (conflicts remain: " 
                  << final_conflicts << ")\n";
    }
    
    return 0;
}
