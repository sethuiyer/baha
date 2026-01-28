/*
 * Author: Sethurathienam Iyer
 */
#include "baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <set>
#include <iomanip>

// =============================================================================
// MAXIMUM INDEPENDENT SET (MIS)
// One of Karp's 21 NP-Complete Problems
// =============================================================================

struct MISState {
    std::vector<bool> in_set;  // in_set[i] = true if vertex i is in the independent set
};

class MaxIndependentSetProblem {
public:
    MaxIndependentSetProblem(int n_vertices, double edge_prob, int seed)
        : N_(n_vertices), rng_(seed) {
        
        // Generate Erdős–Rényi random graph G(n, p)
        adj_.resize(N_);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        for (int i = 0; i < N_; ++i) {
            for (int j = i + 1; j < N_; ++j) {
                if (dist(rng_) < edge_prob) {
                    adj_[i].push_back(j);
                    adj_[j].push_back(i);
                    n_edges_++;
                }
            }
        }
        
        std::cout << "Generated G(" << N_ << ", " << edge_prob << ") with " 
                  << n_edges_ << " edges\n";
    }

    // FRACTURE-INDUCING ENERGY FUNCTION
    // Goal: Create sharp phase transitions at the feasibility boundary
    double energy(const MISState& state) const {
        double reward = 0.0;
        double penalty = 0.0;
        
        for (int i = 0; i < N_; ++i) {
            if (state.in_set[i]) {
                // Reward for each vertex in set
                reward += 1.0;
                
                // Count conflicting neighbors
                int neighbor_conflicts = 0;
                int neighbor_in_set = 0;
                for (int j : adj_[i]) {
                    if (state.in_set[j]) {
                        neighbor_conflicts++;
                        neighbor_in_set++;
                    }
                }
                
                // FRACTURE MECHANISM: Exponential penalty creates sharp cliff
                // When conflicts > 0, energy spikes dramatically
                if (neighbor_conflicts > 0) {
                    penalty += std::exp(neighbor_conflicts * 2.0);  // Exponential spike = fracture
                }
                
                // Neighbor pressure: bonus for being isolated in the set
                reward += 0.1 * (adj_[i].size() - neighbor_in_set) / (adj_[i].size() + 1);
            } else {
                // Potential energy: how valuable is this vertex if added?
                int addable_neighbors = 0;
                for (int j : adj_[i]) {
                    if (!state.in_set[j]) addable_neighbors++;
                }
                // Small bonus for leaving "valuable" vertices out (creates gradient)
                reward -= 0.01 * addable_neighbors / (adj_[i].size() + 1);
            }
        }
        
        // Final energy: minimize (we want to MAXIMIZE set size, so use negative)
        return penalty - reward;
    }

    int get_set_size(const MISState& state) const {
        int size = 0;
        for (int i = 0; i < N_; ++i) {
            if (state.in_set[i]) size++;
        }
        return size;
    }

    int get_conflicts(const MISState& state) const {
        int conflicts = 0;
        for (int i = 0; i < N_; ++i) {
            if (state.in_set[i]) {
                for (int j : adj_[i]) {
                    if (j > i && state.in_set[j]) {
                        conflicts++;
                    }
                }
            }
        }
        return conflicts;
    }

    MISState random_state() {
        // 50% chance to start from greedy, 50% random
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng_) < 0.5) {
            return greedy_solution();  // Start from greedy baseline
        }
        
        MISState state;
        state.in_set.resize(N_);
        
        // Random subset with ~30% inclusion probability
        for (int i = 0; i < N_; ++i) {
            state.in_set[i] = (dist(rng_) < 0.3);
        }
        return state;
    }

    std::vector<MISState> neighbors(const MISState& state) {
        std::vector<MISState> nbrs;
        std::uniform_int_distribution<int> dist(0, N_ - 1);
        
        // Generate 32 random flip neighbors
        for (int k = 0; k < 32; ++k) {
            MISState nbr = state;
            int idx = dist(rng_);
            nbr.in_set[idx] = !nbr.in_set[idx];
            nbrs.push_back(nbr);
        }
        
        // Also try adding vertices that don't conflict
        for (int i = 0; i < N_; ++i) {
            if (!state.in_set[i]) {
                bool can_add = true;
                for (int j : adj_[i]) {
                    if (state.in_set[j]) {
                        can_add = false;
                        break;
                    }
                }
                if (can_add) {
                    MISState nbr = state;
                    nbr.in_set[i] = true;
                    nbrs.push_back(nbr);
                }
            }
        }
        
        return nbrs;
    }

    MISState greedy_solution() {
        MISState state;
        state.in_set.resize(N_, false);
        
        // Sort vertices by degree (ascending)
        std::vector<std::pair<int, int>> degree_vertex;
        for (int i = 0; i < N_; ++i) {
            degree_vertex.push_back({adj_[i].size(), i});
        }
        std::sort(degree_vertex.begin(), degree_vertex.end());
        
        std::set<int> excluded;
        for (auto& [deg, v] : degree_vertex) {
            if (excluded.find(v) == excluded.end()) {
                state.in_set[v] = true;
                // Exclude all neighbors
                for (int u : adj_[v]) {
                    excluded.insert(u);
                }
            }
        }
        
        return state;
    }

    void print_solution(const MISState& state, const std::string& label) const {
        int size = get_set_size(state);
        int conflicts = get_conflicts(state);
        
        std::cout << label << ":\n";
        std::cout << "  Independent Set Size: " << size << "\n";
        std::cout << "  Conflicts: " << conflicts << "\n";
        std::cout << "  Valid: " << (conflicts == 0 ? "YES ✅" : "NO ❌") << "\n";
        
        if (N_ <= 50) {
            std::cout << "  Vertices in set: ";
            for (int i = 0; i < N_; ++i) {
                if (state.in_set[i]) std::cout << i << " ";
            }
            std::cout << "\n";
        }
    }

    int get_n() const { return N_; }
    int get_edges() const { return n_edges_; }

private:
    int N_;
    int n_edges_ = 0;
    std::vector<std::vector<int>> adj_;
    mutable std::mt19937 rng_;
};

int main() {
    std::cout << "⚡ MAXIMUM INDEPENDENT SET USING BAHA ⚡\n";
    std::cout << "One of Karp's 21 NP-Complete Problems\n";
    std::cout << "========================================\n\n";

    // Test on larger, sparser graph where greedy fails
    int N = 200;
    double edge_prob = 0.3;  // Sparser = larger MIS, harder to find

    std::cout << "Configuration:\n";
    std::cout << "  Vertices: " << N << "\n";
    std::cout << "  Edge Probability: " << edge_prob << " (Phase Transition)\n";
    std::cout << "  Expected Edges: ~" << (int)(N * (N-1) / 2 * edge_prob) << "\n\n";

    MaxIndependentSetProblem problem(N, edge_prob, 123);  // Different seed

    auto energy = [&](const MISState& s) { return problem.energy(s); };
    auto sampler = [&]() { return problem.random_state(); };
    auto neighbors = [&](const MISState& s) { return problem.neighbors(s); };

    navokoj::BranchAwareOptimizer<MISState> baha(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<MISState>::Config config;

    config.beta_steps = 3000;
    config.beta_end = 30.0;
    config.samples_per_beta = 80;
    config.fracture_threshold = 1.2;
    config.max_branches = 8;
    config.verbose = true;
    config.schedule_type = navokoj::BranchAwareOptimizer<MISState>::ScheduleType::GEOMETRIC;

    std::cout << "Starting BAHA optimization...\n\n";
    auto start = std::chrono::high_resolution_clock::now();
    auto result = baha.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\n========================================\n";
    std::cout << "OPTIMIZATION COMPLETE\n";
    std::cout << "========================================\n";
    std::cout << "Time: " << std::fixed << std::setprecision(1) << time_ms / 1000.0 << " seconds\n";
    std::cout << "Fractures Detected: " << result.fractures_detected << "\n";
    std::cout << "Branch Jumps: " << result.branch_jumps << "\n\n";

    problem.print_solution(result.best_state, "BAHA Solution (raw)");

    // LOCAL SEARCH REFINEMENT: Try to add more vertices without conflicts
    MISState refined = result.best_state;
    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 0; i < problem.get_n(); ++i) {
            if (!refined.in_set[i]) {
                // Check if we can add this vertex
                MISState test = refined;
                test.in_set[i] = true;
                if (problem.get_conflicts(test) == 0) {
                    refined = test;
                    improved = true;
                }
            }
        }
    }
    std::cout << "\n";
    problem.print_solution(refined, "BAHA + Local Search");
    int baha_refined_size = problem.get_set_size(refined);

    // Compare with greedy
    std::cout << "\n";
    auto greedy = problem.greedy_solution();
    problem.print_solution(greedy, "Greedy Solution");

    // Compare with random
    int best_random_size = 0;
    MISState best_random_state;
    for (int i = 0; i < 1000; ++i) {
        auto random_state = problem.random_state();
        // Repair conflicts greedily
        for (int v = 0; v < problem.get_n(); ++v) {
            if (random_state.in_set[v] && problem.get_conflicts(random_state) > 0) {
                random_state.in_set[v] = false;
            }
        }
        int size = problem.get_set_size(random_state);
        if (size > best_random_size && problem.get_conflicts(random_state) == 0) {
            best_random_size = size;
            best_random_state = random_state;
        }
    }
    std::cout << "\n";
    problem.print_solution(best_random_state, "Best Random (1000 trials)");

    // Summary
    int baha_size = problem.get_set_size(result.best_state);
    int greedy_size = problem.get_set_size(greedy);
    int baha_conflicts = problem.get_conflicts(refined);

    std::cout << "\n📊 SUMMARY 📊\n";
    std::cout << "=============\n";
    std::cout << "| Method           | Set Size | Valid |\n";
    std::cout << "|------------------|----------|-------|\n";
    std::cout << "| BAHA (raw)       | " << std::setw(8) << baha_size << " | " << (problem.get_conflicts(result.best_state) == 0 ? "YES" : "NO") << "   |\n";
    std::cout << "| BAHA + LocalSrch | " << std::setw(8) << baha_refined_size << " | " << (baha_conflicts == 0 ? "YES" : "NO") << "   |\n";
    std::cout << "| Greedy           | " << std::setw(8) << greedy_size << " | YES   |\n";
    std::cout << "| Random Best      | " << std::setw(8) << best_random_size << " | YES   |\n";

    if (baha_conflicts == 0) {
        double improvement = ((double)baha_refined_size - greedy_size) / greedy_size * 100.0;
        std::cout << "\nBAHA+LS improvement over Greedy: " << std::showpos << std::fixed 
                  << std::setprecision(1) << improvement << "%\n";
    }

    return 0;
}
