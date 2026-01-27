#include "baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <set>
#include <algorithm>
#include <iomanip>

// =============================================================================
// LIST COLORING PROBLEM (More Constrained Than Regular Graph Coloring)
// Each vertex has its own list of allowed colors - harder than regular coloring
// =============================================================================
struct ListColoringState {
    std::vector<int> coloring;  // coloring[i] = color assigned to vertex i (-1 if unassigned)
};

class ListColoringProblem {
public:
    ListColoringProblem(int n_vertices, const std::vector<std::pair<int, int>>& edges, 
                       const std::vector<std::vector<int>>& allowed_colors, int n_total_colors, int seed)
        : n_vertices_(n_vertices), edges_(edges), allowed_colors_(allowed_colors), 
          n_total_colors_(n_total_colors), rng_(seed) {
        
        // Validate that each vertex has at least one allowed color
        for (const auto& colors : allowed_colors_) {
            if (colors.empty()) {
                throw std::runtime_error("Vertex has no allowed colors!");
            }
        }
    }

    double energy(const ListColoringState& state) const {
        int conflicts = 0;
        int unassigned = 0;
        
        // Count conflicts between adjacent vertices with same color
        for (const auto& [u, v] : edges_) {
            if (state.coloring[u] != -1 && state.coloring[v] != -1 && state.coloring[u] == state.coloring[v]) {
                conflicts++;
            }
        }
        
        // Count unassigned vertices (penalty for incompleteness)
        for (int i = 0; i < n_vertices_; ++i) {
            if (state.coloring[i] == -1) {
                unassigned++;
            } else {
                // Check if assigned color is in allowed list for this vertex
                bool valid_color = false;
                for (int allowed_color : allowed_colors_[i]) {
                    if (allowed_color == state.coloring[i]) {
                        valid_color = true;
                        break;
                    }
                }
                if (!valid_color) {
                    conflicts += 10;  // Heavy penalty for invalid color
                }
            }
        }
        
        // Return energy as combination of conflicts and unassigned vertices
        return static_cast<double>(conflicts * 100 + unassigned * 10);
    }

    ListColoringState random_state() const {
        ListColoringState state;
        state.coloring.resize(n_vertices_);
        
        std::uniform_int_distribution<> dist(0, 1000);  // For random decisions
        
        for (int i = 0; i < n_vertices_; ++i) {
            // Randomly decide whether to assign a color or leave unassigned
            if (dist(rng_) % 3 == 0) {  // 1/3 chance of leaving unassigned initially
                state.coloring[i] = -1;
            } else {
                // Pick a random allowed color for this vertex
                const auto& allowed = allowed_colors_[i];
                if (!allowed.empty()) {
                    int random_idx = dist(rng_) % allowed.size();
                    state.coloring[i] = allowed[random_idx];
                } else {
                    state.coloring[i] = -1;  // Fallback if no allowed colors
                }
            }
        }
        
        return state;
    }

    std::vector<ListColoringState> neighbors(const ListColoringState& state) const {
        std::vector<ListColoringState> nbrs;
        
        // For each vertex, try changing its color to any allowed color
        for (int i = 0; i < n_vertices_; ++i) {
            const auto& allowed = allowed_colors_[i];
            
            // Try each allowed color for this vertex
            for (int allowed_color : allowed) {
                if (allowed_color != state.coloring[i]) {  // Only if different from current
                    ListColoringState nbr = state;
                    nbr.coloring[i] = allowed_color;
                    nbrs.push_back(nbr);
                }
            }
            
            // Also try unassigning if currently assigned
            if (state.coloring[i] != -1) {
                ListColoringState nbr = state;
                nbr.coloring[i] = -1;
                nbrs.push_back(nbr);
            }
            
            // Try assigning if currently unassigned
            if (state.coloring[i] == -1) {
                for (int allowed_color : allowed) {
                    ListColoringState nbr = state;
                    nbr.coloring[i] = allowed_color;
                    nbrs.push_back(nbr);
                }
            }
        }
        
        return nbrs;
    }

    void print_solution(const ListColoringState& state) const {
        std::cout << "\nLIST COLORING SOLUTION:\n";
        std::cout << "=======================\n";
        
        int conflicts = 0;
        int unassigned = 0;
        int valid_assignment = 0;
        
        for (int i = 0; i < n_vertices_; ++i) {
            if (state.coloring[i] == -1) {
                std::cout << "Vertex " << i << ": UNASSIGNED\n";
                unassigned++;
            } else {
                // Check if color is valid
                bool valid_color = false;
                for (int allowed_color : allowed_colors_[i]) {
                    if (allowed_color == state.coloring[i]) {
                        valid_color = true;
                        break;
                    }
                }
                
                std::cout << "Vertex " << i << ": Color " << state.coloring[i] 
                          << " (allowed: ";
                for (int c : allowed_colors_[i]) {
                    std::cout << c << " ";
                }
                std::cout << ") " << (valid_color ? "✓" : "✗INVALID") << "\n";
                
                if (valid_color) valid_assignment++;
            }
        }
        
        // Count edge conflicts
        for (const auto& [u, v] : edges_) {
            if (state.coloring[u] != -1 && state.coloring[v] != -1 && state.coloring[u] == state.coloring[v]) {
                conflicts++;
            }
        }
        
        std::cout << "\nStatistics:\n";
        std::cout << "- Valid assignments: " << valid_assignment << "/" << n_vertices_ << "\n";
        std::cout << "- Unassigned: " << unassigned << "\n";
        std::cout << "- Edge conflicts: " << conflicts << "\n";
        std::cout << "- Total energy: " << energy(state) << "\n";
    }

private:
    int n_vertices_;
    std::vector<std::pair<int, int>> edges_;
    std::vector<std::vector<int>> allowed_colors_;  // allowed_colors_[i] = list of colors vertex i can take
    int n_total_colors_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// GENERATE A CHALLENGING LIST COLORING INSTANCE
// =============================================================================
ListColoringProblem generate_list_coloring_instance(int n_vertices, int n_colors, int seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> edge_dist(0, n_vertices - 1);
    std::uniform_int_distribution<> color_dist(0, n_colors - 1);
    std::uniform_int_distribution<> list_size_dist(1, std::min(n_colors, 4));  // 1-4 allowed colors per vertex
    
    // Generate random graph (Erdős–Rényi style)
    std::vector<std::pair<int, int>> edges;
    int n_edges = n_vertices * 1.5;  // Dense enough to be challenging
    
    // Create edges ensuring graph connectivity
    for (int i = 1; i < n_vertices; ++i) {
        edges.push_back({i-1, i});  // Connect consecutive vertices to ensure connectivity
    }
    
    // Add random edges
    for (int i = 0; i < n_edges - (n_vertices - 1); ++i) {
        int u = edge_dist(gen);
        int v = edge_dist(gen);
        if (u != v) {
            edges.push_back({u, v});
        }
    }
    
    // Generate allowed color lists for each vertex (more constrained than regular coloring)
    std::vector<std::vector<int>> allowed_colors(n_vertices);
    for (int i = 0; i < n_vertices; ++i) {
        int list_size = list_size_dist(gen);
        std::set<int> chosen_colors;
        
        while (chosen_colors.size() < list_size) {
            chosen_colors.insert(color_dist(gen));
        }
        
        for (int color : chosen_colors) {
            allowed_colors[i].push_back(color);
        }
    }
    
    return ListColoringProblem(n_vertices, edges, allowed_colors, n_colors, seed);
}

// =============================================================================
// MAIN RUNNER
// =============================================================================
int main() {
    std::cout << "🎨 LIST COLORING OPTIMIZATION USING BAHA 🎨\n";
    std::cout << "Hunting for valid colorings in constrained spaces...\n";
    std::cout << "==================================================\n\n";

    // Create a challenging list coloring instance: 20 vertices, 5 colors, each vertex has 1-4 allowed colors
    auto problem = generate_list_coloring_instance(20, 5, 42);

    // Wrap in BAHA-compatible functions
    std::function<double(const ListColoringState&)> energy = 
        [&](const ListColoringState& s) { return problem.energy(s); };
    
    std::function<ListColoringState()> sampler = 
        [&]() { return problem.random_state(); };
    
    std::function<std::vector<ListColoringState>(const ListColoringState&)> neighbors = 
        [&](const ListColoringState& s) { return problem.neighbors(s); };

    // Configure BAHA for this challenging problem
    typename navokoj::BranchAwareOptimizer<ListColoringState>::Config config;
    config.beta_steps = 1500;           // More steps for complex constraint satisfaction
    config.beta_end = 20.0;             // Higher beta for precision
    config.samples_per_beta = 60;       // More samples for accuracy
    config.fracture_threshold = 1.8;    // Adjusted for this problem type
    config.max_branches = 7;            // More branches for complex landscape
    config.verbose = true;              // Show progress
    config.schedule_type = navokoj::BranchAwareOptimizer<ListColoringState>::ScheduleType::GEOMETRIC;

    // Run optimization
    navokoj::BranchAwareOptimizer<ListColoringState> ba(energy, sampler, neighbors);
    std::cout << "Starting BAHA optimization on List Coloring...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = ba.optimize(config);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    std::cout << "\n🎯 LIST COLORING OPTIMIZATION COMPLETE 🎯\n";
    std::cout << "Best energy found: " << result.best_energy << "\n";
    std::cout << "Time taken: " << duration << " ms\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Beta at solution: " << result.beta_at_solution << "\n\n";
    
    // Print detailed solution
    problem.print_solution(result.best_state);
    
    // Compare with random baseline
    std::cout << "\n📊 COMPARISON WITH RANDOM SOLUTIONS 📊\n";
    std::cout << "=====================================\n";
    
    double best_random_energy = 1e9;
    ListColoringState best_random_state;
    
    for (int i = 0; i < 100; ++i) {  // Try 100 random solutions
        auto random_state = problem.random_state();
        double random_energy = problem.energy(random_state);
        if (random_energy < best_random_energy) {
            best_random_energy = random_energy;
            best_random_state = random_state;
        }
    }
    
    std::cout << "Best random solution energy: " << best_random_energy << "\n";
    std::cout << "BAHA improvement: " << (best_random_energy - result.best_energy) << " energy units\n";
    std::cout << "Relative improvement: " << ((best_random_energy - result.best_energy) / std::abs(best_random_energy)) * 100.0 << "%\n\n";
    
    std::cout << "Random solution summary:\n";
    problem.print_solution(best_random_state);

    return 0;
}