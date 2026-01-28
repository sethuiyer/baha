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

struct PermutationState {
    std::vector<int> p;
};

class GraphIsomorphismProblem {
public:
    GraphIsomorphismProblem(int n, int seed) : n_(n), rng_(seed) {
        adj_A_.assign(n * n, 0);
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int edge = dist(rng_);
                adj_A_[i * n + j] = edge;
                adj_A_[j * n + i] = edge;
            }
        }

        std::vector<int> p_true(n);
        std::iota(p_true.begin(), p_true.end(), 0);
        std::shuffle(p_true.begin(), p_true.end(), rng_);

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
        // Reduced neighborhood for speed at N=50? No, stick to swap.
        // N=50 means N*(N-1)/2 = 1225 neighbors. Large but manageable.
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
    std::vector<int> adj_A_;
    std::vector<int> adj_B_;
    mutable std::mt19937 rng_;
};

int main() {
    std::cout << "🕸️ HARD GRAPH ISOMORPHISM BENCHMARK (N=50) 🕸️\n";
    
    int n_trials = 5;
    int prob_size = 50;

    int ba_wins = 0, sa_wins = 0;
    double ba_total_e = 0, sa_total_e = 0;

    for (int t = 0; t < n_trials; ++t) {
        std::cout << "Trial " << (t+1) << "... ";
        std::cout.flush();
        
        GraphIsomorphismProblem prob(prob_size, t + 9999);
        
        std::function<double(const PermutationState&)> energy = [&](const PermutationState& s) { return prob.energy(s); };
        std::function<PermutationState()> sampler = [&]() { return prob.random_state(); };
        std::function<std::vector<PermutationState>(const PermutationState&)> neighbors = [&](const PermutationState& s) { return prob.neighbors(s); };

        typename navokoj::BranchAwareOptimizer<PermutationState>::Config ba_config;
        ba_config.beta_steps = 3000; 
        ba_config.beta_end = 30.0;   
        ba_config.max_branches = 10;
        ba_config.schedule_type = navokoj::BranchAwareOptimizer<PermutationState>::ScheduleType::GEOMETRIC;

        navokoj::BranchAwareOptimizer<PermutationState> ba(energy, sampler, neighbors);
        auto ba_res = ba.optimize(ba_config);

        typename navokoj::SimulatedAnnealing<PermutationState>::Config sa_config;
        sa_config.beta_steps = 3000;
        sa_config.beta_end = 30.0;
        sa_config.steps_per_beta = 50; 

        navokoj::SimulatedAnnealing<PermutationState> sa(energy, sampler, neighbors);
        auto sa_res = sa.optimize(sa_config);

        ba_total_e += ba_res.best_energy;
        sa_total_e += sa_res.best_energy;
        
        if (ba_res.best_energy < sa_res.best_energy) {
            std::cout << "BA Wins (E=" << ba_res.best_energy << " vs " << sa_res.best_energy << ")\n";
        } else if (sa_res.best_energy < ba_res.best_energy) {
            std::cout << "SA Wins (E=" << ba_res.best_energy << " vs " << sa_res.best_energy << ")\n";
        } else {
            std::cout << "Tie (E=" << ba_res.best_energy << ")\n";
        }
        
        if(ba_res.best_energy == 0) ba_wins++;
        if(sa_res.best_energy == 0) sa_wins++;
    }

    std::cout << "------------------------------------------------------------\n";
    std::cout << "Summary for Graph Isomorphism (N=" << prob_size << "):\n";
    std::cout << "  BA Exact Solves: " << ba_wins << "/" << n_trials << "\n";
    std::cout << "  SA Exact Solves: " << sa_wins << "/" << n_trials << "\n";
    std::cout << "  BA Avg Energy: " << ba_total_e / n_trials << "\n";
    std::cout << "  SA Avg Energy: " << sa_total_e / n_trials << "\n";
    std::cout << "------------------------------------------------------------\n";

    return 0;
}
