/*
 * Author: Sethurathienam Iyer
 */
#include "baha.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>

// =============================================================================
// GRAPH ISOMORPHISM (N=20)
// Problem: Find permutation P such that A = P^T * B * P
// =============================================================================

struct PermutationState {
    std::vector<int> p; // p[i] maps node i of Graph A to node p[i] of Graph B
};

class GraphIsomorphismProblem {
public:
    GraphIsomorphismProblem(int n, int seed) : n_(n), rng_(seed) {
        // Generate random Graph A (Erdos-Renyi with p=0.5)
        adj_A_.assign(n * n, 0);
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int edge = dist(rng_);
                adj_A_[i * n + j] = edge;
                adj_A_[j * n + i] = edge;
            }
        }

        // Generate Target Permutation P_true
        std::vector<int> p_true(n);
        std::iota(p_true.begin(), p_true.end(), 0);
        std::shuffle(p_true.begin(), p_true.end(), rng_);

        // Generate Graph B = Permuted A
        // if A has edge (i, j), B has edge (p[i], p[j])
        adj_B_.assign(n * n, 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (adj_A_[i * n + j]) {
                    int u = p_true[i];
                    int v = p_true[j];
                    adj_B_[u * n + v] = 1;
                }
            }
        }
    }

    // Energy = Hamming distance between A and InversePermuted(B)
    // Actually simpler: sum | A_ij - B_p(i)p(j) | for i < j
    double energy(const PermutationState& s) const {
        int mismatches = 0;
        for (int i = 0; i < n_; ++i) {
            for (int j = i + 1; j < n_; ++j) {
                int u = s.p[i];
                int v = s.p[j];
                if (adj_A_[i * n_ + j] != adj_B_[u * n_ + v]) {
                    mismatches++;
                }
            }
        }
        return static_cast<double>(mismatches);
    }

    PermutationState random_state() {
        PermutationState s;
        s.p.resize(n_);
        std::iota(s.p.begin(), s.p.end(), 0);
        std::shuffle(s.p.begin(), s.p.end(), rng_);
        return s;
    }

    std::vector<PermutationState> neighbors(const PermutationState& s) {
        std::vector<PermutationState> nbrs;
        // Swap 2 elements
        for (int i = 0; i < n_; ++i) {
            for (int j = i + 1; j < n_; ++j) {
                PermutationState nbr = s;
                std::swap(nbr.p[i], nbr.p[j]);
                nbrs.push_back(nbr);
            }
        }
        return nbrs;
    }

private:
    int n_;
    std::vector<int> adj_A_; // Flattened adjacency matrix
    std::vector<int> adj_B_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// MAIN RUNNER
// =============================================================================

template<typename Problem, typename State>
void run_gi(const std::string& name, int n_trials, std::function<Problem(int)> factory) {
    std::cout << "\nRunning Graph Isomorphism: " << name << "\n";
    std::cout << "============================================================\n";
    
    int ba_wins = 0;
    int sa_wins = 0; // Wins means finding E=0 (exact isomorphism)
    double ba_total_e = 0;
    double sa_total_e = 0;

    for (int t = 0; t < n_trials; ++t) {
        std::cout << "Trial " << (t+1) << "... ";
        std::cout.flush();
        
        auto prob = factory(t);
        
        std::function<double(const State&)> energy = [&](const State& s) { return prob.energy(s); };
        std::function<State()> sampler = [&]() { return prob.random_state(); };
        std::function<std::vector<State>(const State&)> neighbors = [&](const State& s) { return prob.neighbors(s); };

        typename navokoj::BranchAwareOptimizer<State>::Config ba_config;
        ba_config.beta_steps = 2000; 
        ba_config.beta_end = 20.0;   
        ba_config.max_branches = 10;
        ba_config.schedule_type = navokoj::BranchAwareOptimizer<State>::ScheduleType::GEOMETRIC;

        navokoj::BranchAwareOptimizer<State> ba(energy, sampler, neighbors);
        auto ba_res = ba.optimize(ba_config);

        typename navokoj::SimulatedAnnealing<State>::Config sa_config;
        sa_config.beta_steps = 2000;
        sa_config.beta_end = 20.0;
        sa_config.steps_per_beta = 50; 

        navokoj::SimulatedAnnealing<State> sa(energy, sampler, neighbors);
        auto sa_res = sa.optimize(sa_config);

        ba_total_e += ba_res.best_energy;
        sa_total_e += sa_res.best_energy;
        
        bool ba_success = (ba_res.best_energy == 0);
        bool sa_success = (sa_res.best_energy == 0);

        if (ba_success) ba_wins++;
        if (sa_success) sa_wins++;

        if (ba_res.best_energy < sa_res.best_energy) {
            std::cout << "BA Wins (E=" << ba_res.best_energy << " vs " << sa_res.best_energy << ")\n";
        } else if (sa_res.best_energy < ba_res.best_energy) {
            std::cout << "SA Wins (E=" << ba_res.best_energy << " vs " << sa_res.best_energy << ")\n";
        } else {
            std::cout << "Tie (E=" << ba_res.best_energy << ")\n";
        }
    }

    std::cout << "------------------------------------------------------------\n";
    std::cout << "Summary for " << name << ":\n";
    std::cout << "  BA Exact Solves: " << ba_wins << "/" << n_trials << "\n";
    std::cout << "  SA Exact Solves: " << sa_wins << "/" << n_trials << "\n";
    std::cout << "  BA Avg Energy: " << ba_total_e / n_trials << "\n";
    std::cout << "  SA Avg Energy: " << sa_total_e / n_trials << "\n";
    std::cout << "------------------------------------------------------------\n";
}

int main() {
    std::cout << "🕸️ GRAPH ISOMORPHISM BENCHMARK 🕸️\n";
    std::cout << "Checking if BAHA can find the 'needle in the haystack' permutation.\n";

    // N=20 is standard hard for plain SA. N=18 claimed in back.md, let's try 20.
    run_gi<GraphIsomorphismProblem, PermutationState>(
        "Graph Isomorphism (N=20)", 
        5,
        [](int seed) { return GraphIsomorphismProblem(20, seed + 777); }
    );

    return 0;
}
