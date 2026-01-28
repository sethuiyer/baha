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
// CHAOTIC NUMBER PARTITIONING
// Generator: x_{n+1} = (x_n * 9973) mod 1
// This creates a deterministic, highly structured but "random-looking" sequence.
// =============================================================================

struct ChaoticPartitionState {
    std::vector<int> signs; // +1 or -1
};

class ChaoticNumberPartitioning {
public:
    ChaoticNumberPartitioning(int n, double seed_val) : n_(n) {
        numbers_.resize(n);
        double x = seed_val;
        for (int i = 0; i < n; ++i) {
            // Chaotic map
            x = std::fmod(x * 9973.0, 1.0);
            
            // Scale to large integer (10^14)
            numbers_[i] = static_cast<long long>(x * 100000000000000.0);
        }
        // Use a dummy RNG for internal shuffling if needed, but the problem itself is deterministic given seed.
        rng_.seed(static_cast<unsigned int>(seed_val * 100000)); 
    }

    double energy(const ChaoticPartitionState& s) const {
        long long sum = 0;
        for (size_t i = 0; i < numbers_.size(); ++i) {
            sum += s.signs[i] * numbers_[i];
        }
        return static_cast<double>(std::abs(sum));
    }

    ChaoticPartitionState random_state() {
        ChaoticPartitionState s;
        s.signs.resize(n_);
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n_; ++i) s.signs[i] = dist(rng_) ? 1 : -1;
        return s;
    }

    std::vector<ChaoticPartitionState> neighbors(const ChaoticPartitionState& s) {
        std::vector<ChaoticPartitionState> nbrs;
        for (int i = 0; i < n_; ++i) {
            ChaoticPartitionState nbr = s;
            nbr.signs[i] *= -1;
            nbrs.push_back(nbr);
        }
        return nbrs;
    }

private:
    int n_;
    std::vector<long long> numbers_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// MAIN RUNNER
// =============================================================================

template<typename Problem, typename State>
void run_chaotic(const std::string& name, int n_trials, std::function<Problem(int)> factory) {
    std::cout << "\nRunning Chaotic Challenge: " << name << "\n";
    std::cout << "============================================================\n";
    
    int ba_wins = 0;
    double ba_total_e = 0;
    double sa_total_e = 0;

    for (int t = 0; t < n_trials; ++t) {
        std::cout << "Trial " << (t+1) << "... ";
        std::cout.flush();
        
        auto prob = factory(t); // Seed with trial index (but converted to double inside factory)
        
        std::function<double(const State&)> energy = [&](const State& s) { return prob.energy(s); };
        std::function<State()> sampler = [&]() { return prob.random_state(); };
        std::function<std::vector<State>(const State&)> neighbors = [&](const State& s) { return prob.neighbors(s); };

        // Tune config
        typename navokoj::BranchAwareOptimizer<State>::Config ba_config;
        ba_config.beta_steps = 1000; 
        ba_config.beta_end = 20.0;
        // Use Geometric just in case, though Linear worked fine for Partitioning.
        // Actually, let's stick to Linear which we know worked for Partitioning to be safe/consistent.
        ba_config.schedule_type = navokoj::BranchAwareOptimizer<State>::ScheduleType::LINEAR;  
        
        navokoj::BranchAwareOptimizer<State> ba(energy, sampler, neighbors);
        auto ba_res = ba.optimize(ba_config);

        typename navokoj::SimulatedAnnealing<State>::Config sa_config;
        sa_config.beta_steps = 1000;
        sa_config.beta_end = 20.0;
        sa_config.steps_per_beta = 20;

        navokoj::SimulatedAnnealing<State> sa(energy, sampler, neighbors);
        auto sa_res = sa.optimize(sa_config);

        ba_total_e += ba_res.best_energy;
        sa_total_e += sa_res.best_energy;

        if (ba_res.best_energy < sa_res.best_energy) {
            ba_wins++;
            std::cout << "BA Wins (E=" << ba_res.best_energy << " vs " << sa_res.best_energy << ")\n";
        } else if (sa_res.best_energy < ba_res.best_energy) {
            std::cout << "SA Wins (E=" << ba_res.best_energy << " vs " << sa_res.best_energy << ")\n";
        } else {
            std::cout << "Tie (E=" << ba_res.best_energy << ")\n";
        }
    }

    std::cout << "------------------------------------------------------------\n";
    std::cout << "Summary for " << name << ":\n";
    std::cout << "  BA Win Rate: " << ba_wins << "/" << n_trials << "\n";
    std::cout << "  BA Avg Energy: " << ba_total_e / n_trials << "\n";
    std::cout << "  SA Avg Energy: " << sa_total_e / n_trials << "\n";
    std::cout << "  Speedup Factor (Energy Reduction): " << (sa_total_e / (ba_total_e + 1e-9)) << "x\n"; 
    std::cout << "------------------------------------------------------------\n";
}

int main() {
    std::cout << "🌪️ CHAOTIC BENCHMARK (x' = x * 9973 mod 1) 🌪️\n";

    // CHAOTIC PARTITIONING
    run_chaotic<ChaoticNumberPartitioning, ChaoticPartitionState>(
        "Chaotic Number Partitioning (N=100, Scale=10^14)", 
        5,
        [](int t) { 
            // Use different start seeds: 0.1, 0.2, 0.3...
            return ChaoticNumberPartitioning(100, 0.1 + t * 0.1); 
        }
    );

    return 0;
}
