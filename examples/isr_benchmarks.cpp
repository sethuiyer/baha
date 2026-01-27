#include "baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <set>
#include <algorithm>
#include <cmath>
#include <iomanip>

// =============================================================================
// EXACT COVER WITH SECONDARY CONSTRAINTS (SUDOKU VARIANT)
// =============================================================================
struct ExactCoverState {
    std::vector<int> assignment;  // assignment[i] = value assigned to cell i
};

class ExactCoverProblem {
public:
    ExactCoverProblem(int board_size, int box_rows, int box_cols, int seed) 
        : board_size_(board_size), box_rows_(box_rows), box_cols_(box_cols), rng_(seed) {
        
        total_cells_ = board_size_ * board_size_;
        assignment.resize(total_cells_, -1);
        
        // Precompute constraints: rows, cols, boxes
        row_constraints_.resize(board_size_);
        col_constraints_.resize(board_size_);
        box_constraints_.resize(board_size_);
        
        for (int r = 0; r < board_size_; ++r) {
            for (int c = 0; c < board_size_; ++c) {
                int cell = r * board_size_ + c;
                row_constraints_[r].push_back(cell);
                col_constraints_[c].push_back(cell);
                int box_idx = (r / box_rows_) * box_cols_ + (c / box_cols_);
                box_constraints_[box_idx].push_back(cell);
            }
        }
    }

    double energy(const ExactCoverState& state) const {
        double penalty = 0.0;
        
        // Check row constraints: each number appears exactly once per row
        for (int r = 0; r < board_size_; ++r) {
            std::vector<int> counts(board_size_ + 1, 0);
            for (int cell : row_constraints_[r]) {
                if (state.assignment[cell] != -1) {
                    counts[state.assignment[cell]]++;
                }
            }
            for (int val = 1; val <= board_size_; ++val) {
                if (counts[val] != 1) {
                    penalty += 1.0;  // Penalize for not having exactly one of each value
                }
            }
        }
        
        // Check column constraints
        for (int c = 0; c < board_size_; ++c) {
            std::vector<int> counts(board_size_ + 1, 0);
            for (int cell : col_constraints_[c]) {
                if (state.assignment[cell] != -1) {
                    counts[state.assignment[cell]]++;
                }
            }
            for (int val = 1; val <= board_size_; ++val) {
                if (counts[val] != 1) {
                    penalty += 1.0;
                }
            }
        }
        
        // Check box constraints
        for (int b = 0; b < board_size_; ++b) {
            std::vector<int> counts(board_size_ + 1, 0);
            for (int cell : box_constraints_[b]) {
                if (state.assignment[cell] != -1) {
                    counts[state.assignment[cell]]++;
                }
            }
            for (int val = 1; val <= board_size_; ++val) {
                if (counts[val] != 1) {
                    penalty += 1.0;
                }
            }
        }
        
        return penalty;
    }

    ExactCoverState random_state() const {
        ExactCoverState state;
        state.assignment.resize(total_cells_);
        
        // Fill randomly but respecting constraints as much as possible
        std::uniform_int_distribution<> dist(1, board_size_);
        
        for (int i = 0; i < total_cells_; ++i) {
            state.assignment[i] = (dist(rng_) % 3 == 0) ? dist(rng_) : -1;  // Sometimes leave empty
        }
        
        return state;
    }

    std::vector<ExactCoverState> neighbors(const ExactCoverState& state) const {
        std::vector<ExactCoverState> nbrs;
        
        // Try changing one cell's value
        for (int cell = 0; cell < total_cells_; ++cell) {
            for (int val = 1; val <= board_size_; ++val) {
                if (state.assignment[cell] != val) {
                    ExactCoverState nbr = state;
                    nbr.assignment[cell] = val;
                    nbrs.push_back(nbr);
                }
            }
            
            // Also try unassigning
            if (state.assignment[cell] != -1) {
                ExactCoverState nbr = state;
                nbr.assignment[cell] = -1;
                nbrs.push_back(nbr);
            }
        }
        
        return nbrs;
    }

private:
    int board_size_;
    int box_rows_;
    int box_cols_;
    int total_cells_;
    std::vector<int> assignment;
    std::vector<std::vector<int>> row_constraints_;
    std::vector<std::vector<int>> col_constraints_;
    std::vector<std::vector<int>> box_constraints_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// PLANTED CLIQUE PROBLEM
// =============================================================================
struct CliqueState {
    std::vector<bool> nodes_selected;  // nodes_selected[i] = true if node i is in clique
};

class PlantedCliqueProblem {
public:
    PlantedCliqueProblem(int n, int k, int seed) : n_(n), k_(k), rng_(seed) {
        // Generate random graph with planted clique
        adj_matrix_.assign(n * n, 0);
        
        // Generate random Erdos-Renyi graph
        std::uniform_real_distribution<> dist(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (dist(rng_) < 0.5) {  // p = 0.5
                    adj_matrix_[i * n + j] = 1;
                    adj_matrix_[j * n + i] = 1;
                }
            }
        }
        
        // Plant a clique of size k
        std::vector<int> clique_nodes(k);
        std::iota(clique_nodes.begin(), clique_nodes.end(), 0);
        std::shuffle(clique_nodes.begin(), clique_nodes.end(), rng_);
        
        for (int i = 0; i < k; ++i) {
            for (int j = i + 1; j < k; ++j) {
                int u = clique_nodes[i];
                int v = clique_nodes[j];
                adj_matrix_[u * n + v] = 1;
                adj_matrix_[v * n + u] = 1;
            }
        }
    }

    double energy(const CliqueState& state) const {
        int selected_count = 0;
        int edges_inside = 0;
        int missing_edges = 0;
        
        // Count selected nodes
        for (bool selected : state.nodes_selected) {
            if (selected) selected_count++;
        }
        
        // Count edges inside selected subgraph
        for (int i = 0; i < n_; ++i) {
            if (!state.nodes_selected[i]) continue;
            for (int j = i + 1; j < n_; ++j) {
                if (!state.nodes_selected[j]) continue;
                if (adj_matrix_[i * n_ + j]) {
                    edges_inside++;
                } else {
                    missing_edges++;  // Penalize for missing edges in potential clique
                }
            }
        }
        
        // Energy = -edges_inside + penalty_for_wrong_size + penalty_for_missing_edges
        double size_penalty = std::abs(selected_count - k_) * 10.0;
        return -edges_inside * 1.0 + size_penalty + missing_edges * 2.0;
    }

    CliqueState random_state() const {
        CliqueState state;
        state.nodes_selected.resize(n_);
        
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n_; ++i) {
            state.nodes_selected[i] = (dist(rng_) == 1);
        }
        
        return state;
    }

    std::vector<CliqueState> neighbors(const CliqueState& state) const {
        std::vector<CliqueState> nbrs;
        
        // Flip one node selection
        for (int i = 0; i < n_; ++i) {
            CliqueState nbr = state;
            nbr.nodes_selected[i] = !nbr.nodes_selected[i];
            nbrs.push_back(nbr);
        }
        
        return nbrs;
    }

private:
    int n_;  // Total nodes
    int k_;  // Size of planted clique
    std::vector<int> adj_matrix_;  // Adjacency matrix
    mutable std::mt19937 rng_;
};

// =============================================================================
// LOW-AUTOCORRELATION BINARY SEQUENCES (LABS)
// =============================================================================
struct LABSState {
    std::vector<int> sequence;  // +1 or -1
};

class LABSProblem {
public:
    LABSProblem(int n, int seed) : n_(n), rng_(seed) {
        // Nothing special needed for initialization
    }

    double energy(const LABSState& state) const {
        double total_energy = 0.0;
        
        // Compute autocorrelations
        for (int k = 1; k < n_; ++k) {
            double correlation = 0.0;
            for (int i = 0; i < n_ - k; ++i) {
                correlation += state.sequence[i] * state.sequence[i + k];
            }
            total_energy += correlation * correlation;
        }
        
        return total_energy;
    }

    LABSState random_state() const {
        LABSState state;
        state.sequence.resize(n_);
        
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n_; ++i) {
            state.sequence[i] = (dist(rng_) == 0) ? -1 : 1;
        }
        
        return state;
    }

    std::vector<LABSState> neighbors(const LABSState& state) const {
        std::vector<LABSState> nbrs;
        
        // Flip one bit
        for (int i = 0; i < n_; ++i) {
            LABSState nbr = state;
            nbr.sequence[i] = -nbr.sequence[i];  // Flip sign
            nbrs.push_back(nbr);
        }
        
        return nbrs;
    }

private:
    int n_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// RUNNER FOR ALL PROBLEMS
// =============================================================================
void run_exact_cover() {
    std::cout << "\n🎨 EXACT COVER PROBLEM (SUDOKU VARIANT) 🎨\n";
    std::cout << "=========================================\n";
    
    ExactCoverProblem problem(4, 2, 2, 42);  // 4x4 Sudoku
    
    std::function<double(const ExactCoverState&)> energy = 
        [&](const ExactCoverState& s) { return problem.energy(s); };
    
    std::function<ExactCoverState()> sampler = 
        [&]() { return problem.random_state(); };
    
    std::function<std::vector<ExactCoverState>(const ExactCoverState&)> neighbors = 
        [&](const ExactCoverState& s) { return problem.neighbors(s); };

    typename navokoj::BranchAwareOptimizer<ExactCoverState>::Config config;
    config.beta_steps = 1000;
    config.beta_end = 15.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.5;
    config.max_branches = 5;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<ExactCoverState>::ScheduleType::GEOMETRIC;

    navokoj::BranchAwareOptimizer<ExactCoverState> ba(energy, sampler, neighbors);
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = ba.optimize(config);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    std::cout << "BAHA - Time: " << duration << "ms, Energy: " << result.best_energy 
              << ", Fractures: " << result.fractures_detected << ", Jumps: " << result.branch_jumps << "\n";
    
    // Compare with random
    double best_random_energy = 1e9;
    for (int i = 0; i < 100; ++i) {
        auto random_state = problem.random_state();
        double random_energy = problem.energy(random_state);
        if (random_energy < best_random_energy) {
            best_random_energy = random_energy;
        }
    }
    std::cout << "Random - Best Energy: " << best_random_energy << "\n";
    std::cout << "Improvement: " << (best_random_energy - result.best_energy) << "\n\n";
}

void run_planted_clique() {
    std::cout << "/cliq PLANTED CLIQUE PROBLEM (n=20, k=6) /cliq\n";
    std::cout << "===========================================\n";
    
    PlantedCliqueProblem problem(20, 6, 42);  // n=20, k=6 (around 2*log2(n))
    
    std::function<double(const CliqueState&)> energy = 
        [&](const CliqueState& s) { return problem.energy(s); };
    
    std::function<CliqueState()> sampler = 
        [&]() { return problem.random_state(); };
    
    std::function<std::vector<CliqueState>(const CliqueState&)> neighbors = 
        [&](const CliqueState& s) { return problem.neighbors(s); };

    typename navokoj::BranchAwareOptimizer<CliqueState>::Config config;
    config.beta_steps = 1000;
    config.beta_end = 15.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.5;
    config.max_branches = 5;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<CliqueState>::ScheduleType::GEOMETRIC;

    navokoj::BranchAwareOptimizer<CliqueState> ba(energy, sampler, neighbors);
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = ba.optimize(config);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    std::cout << "BAHA - Time: " << duration << "ms, Energy: " << result.best_energy 
              << ", Fractures: " << result.fractures_detected << ", Jumps: " << result.branch_jumps << "\n";
    
    // Compare with random
    double best_random_energy = 1e9;
    for (int i = 0; i < 100; ++i) {
        auto random_state = problem.random_state();
        double random_energy = problem.energy(random_state);
        if (random_energy < best_random_energy) {
            best_random_energy = random_energy;
        }
    }
    std::cout << "Random - Best Energy: " << best_random_energy << "\n";
    std::cout << "Improvement: " << (best_random_energy - result.best_energy) << "\n\n";
}

void run_labs() {
    std::cout << "📡 LOW-AUTOCORRELATION BINARY SEQUENCES (LABS) 📡\n";
    std::cout << "===============================================\n";
    
    LABSProblem problem(16, 42);  // Length 16 sequence
    
    std::function<double(const LABSState&)> energy = 
        [&](const LABSState& s) { return problem.energy(s); };
    
    std::function<LABSState()> sampler = 
        [&]() { return problem.random_state(); };
    
    std::function<std::vector<LABSState>(const LABSState&)> neighbors = 
        [&](const LABSState& s) { return problem.neighbors(s); };

    typename navokoj::BranchAwareOptimizer<LABSState>::Config config;
    config.beta_steps = 1000;
    config.beta_end = 15.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.5;
    config.max_branches = 5;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<LABSState>::ScheduleType::GEOMETRIC;

    navokoj::BranchAwareOptimizer<LABSState> ba(energy, sampler, neighbors);
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = ba.optimize(config);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    std::cout << "BAHA - Time: " << duration << "ms, Energy: " << result.best_energy 
              << ", Fractures: " << result.fractures_detected << ", Jumps: " << result.branch_jumps << "\n";
    
    // Compare with random
    double best_random_energy = 1e9;
    for (int i = 0; i < 100; ++i) {
        auto random_state = problem.random_state();
        double random_energy = problem.energy(random_state);
        if (random_energy < best_random_energy) {
            best_random_energy = random_energy;
        }
    }
    std::cout << "Random - Best Energy: " << best_random_energy << "\n";
    std::cout << "Improvement: " << (best_random_energy - result.best_energy) << "\n\n";
}

int main() {
    std::cout << "🔬 ISR PROBLEM BENCHMARK SUITE 🔬\n";
    std::cout << "Testing BAHA on high-signal problems...\n";
    
    run_exact_cover();
    run_planted_clique();
    run_labs();
    
    std::cout << "Benchmark complete!\n";
    
    return 0;
}