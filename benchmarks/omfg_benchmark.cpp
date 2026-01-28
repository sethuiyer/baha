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
// 1. HARD NUMBER PARTITIONING (N=100, Very Large Numbers)
// =============================================================================
// N=100, Range=[1, 10^14]
// This fits in int64 (max ~9 * 10^18), sum ~ 100 * 10^14 = 10^16. Safe.
// But the landscape is incredibly rugged.
struct HardPartitionState {
    std::vector<int> signs; // +1 or -1
};

class HardNumberPartitioning {
public:
    HardNumberPartitioning(int n, int seed) : n_(n), rng_(seed) {
        // Log-uniform distribution to create multi-scale difficulty?
        // Or just uniform large numbers. Uniform large numbers are usually hardest for NPP.
        std::uniform_int_distribution<long long> dist(1LL, 100000000000000LL); // 10^14
        numbers_.resize(n);
        for (int i = 0; i < n; ++i) numbers_[i] = dist(rng_);
    }

    double energy(const HardPartitionState& s) const {
        long long sum = 0;
        for (size_t i = 0; i < numbers_.size(); ++i) {
            sum += s.signs[i] * numbers_[i];
        }
        return static_cast<double>(std::abs(sum));
    }

    HardPartitionState random_state() {
        HardPartitionState s;
        s.signs.resize(n_);
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n_; ++i) s.signs[i] = dist(rng_) ? 1 : -1;
        return s;
    }

    std::vector<HardPartitionState> neighbors(const HardPartitionState& s) {
        std::vector<HardPartitionState> nbrs;
        // Flip one sign (Hamming distance 1)
        for (int i = 0; i < n_; ++i) {
            HardPartitionState nbr = s;
            nbr.signs[i] *= -1;
            nbrs.push_back(nbr);
        }
        return nbrs;
    }

private:
    int n_;
    std::vector<long long> numbers_;
    mutable std::mt19937_64 rng_; // 64-bit RNG
};

// =============================================================================
// 2. RANDOM XORSAT (N=100, M=100) - The Frozen Core Nightmare
// =============================================================================
// System of linear equations over GF(2). 
// Each clause: x_i XOR x_j XOR x_k = b (0 or 1)
// Extremely brittle landscape.
struct XorSatState {
    std::vector<int> assignment; // 0 or 1
};

class RandomXorSat {
public:
    RandomXorSat(int n, int m, int seed) : n_(n), rng_(seed) {
        std::uniform_int_distribution<> var_dist(0, n - 1);
        std::uniform_int_distribution<> bit_dist(0, 1);
        
        for (int i = 0; i < m; ++i) {
            Clause c;
            for (int k = 0; k < 3; ++k) c.vars[k] = var_dist(rng_);
            c.target = bit_dist(rng_);
            clauses_.push_back(c);
        }
    }

    double energy(const XorSatState& s) const {
        int unsatisfied = 0;
        for (const auto& c : clauses_) {
            int sum = 0;
            for (int k = 0; k < 3; ++k) sum += s.assignment[c.vars[k]];
            if ((sum % 2) != c.target) unsatisfied++;
        }
        return static_cast<double>(unsatisfied);
    }

    XorSatState random_state() {
        XorSatState s;
        s.assignment.resize(n_);
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n_; ++i) s.assignment[i] = dist(rng_);
        return s;
    }

    std::vector<XorSatState> neighbors(const XorSatState& s) {
        std::vector<XorSatState> nbrs;
        for (int i = 0; i < n_; ++i) {
            XorSatState nbr = s;
            nbr.assignment[i] = 1 - nbr.assignment[i]; // Flip bit
            nbrs.push_back(nbr);
        }
        return nbrs;
    }

private:
    struct Clause {
        int vars[3];
        int target;
    };
    int n_;
    std::vector<Clause> clauses_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// 3. HARD RANDOM 3-SAT (N=50, Alpha=4.26) - The Phase Transition
// =============================================================================
// Not planted. True random SAT at the critical difficulty threshold.
// Approx 213 clauses for N=50.
struct SatState {
    std::vector<bool> assignment;
};

class HardRandom3Sat {
public:
    HardRandom3Sat(int n, int m, int seed) : n_(n), rng_(seed) {
        std::uniform_int_distribution<> var_dist(0, n - 1);
        std::uniform_int_distribution<> bool_dist(0, 1);
        
        for (int i = 0; i < m; ++i) {
            Clause c;
            for (int j = 0; j < 3; ++j) c.lits[j] = var_dist(rng_);
            for (int j = 0; j < 3; ++j) c.neg[j] = bool_dist(rng_);
            clauses_.push_back(c);
        }
    }

    double energy(const SatState& s) const {
        int unsatisfied = 0;
        for (const auto& c : clauses_) {
            bool sat = false;
            for (int k = 0; k < 3; ++k) {
                bool val = s.assignment[c.lits[k]];
                if (c.neg[k]) val = !val;
                if (val) {
                    sat = true;
                    break;
                }
            }
            if (!sat) unsatisfied++;
        }
        return static_cast<double>(unsatisfied);
    }

    SatState random_state() {
        SatState s;
        s.assignment.resize(n_);
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n_; ++i) s.assignment[i] = dist(rng_);
        return s;
    }

    std::vector<SatState> neighbors(const SatState& s) {
        std::vector<SatState> nbrs;
        for (int i = 0; i < n_; ++i) {
            SatState nbr = s;
            nbr.assignment[i] = !nbr.assignment[i];
            nbrs.push_back(nbr);
        }
        return nbrs;
    }

private:
    struct Clause {
        int lits[3];
        bool neg[3];
    };
    int n_;
    std::vector<Clause> clauses_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// MAIN RUNNER
// =============================================================================

template<typename Problem, typename State>
void run_omfg(const std::string& name, int n_trials, std::function<Problem(int)> factory) {
    std::cout << "\nRunning OMFG Challenge: " << name << "\n";
    std::cout << "============================================================\n";
    
    int ba_wins = 0;
    double ba_total_e = 0;
    double sa_total_e = 0;

    for (int t = 0; t < n_trials; ++t) {
        std::cout << "Trial " << (t+1) << "... ";
        std::cout.flush();
        
        // Factory returns by value, so prob is a local object.
        auto prob = factory(t);
        
        // Explicitly define functions to avoid ambiguous lambda type deductions
        std::function<double(const State&)> energy = [&](const State& s) { return prob.energy(s); };
        std::function<State()> sampler = [&]() { return prob.random_state(); };
        std::function<std::vector<State>(const State&)> neighbors = [&](const State& s) { return prob.neighbors(s); };

        auto start = std::chrono::high_resolution_clock::now();

        // Tune config for HARD problems
        typename navokoj::BranchAwareOptimizer<State>::Config ba_config;
        ba_config.beta_steps = 1000; 
        ba_config.beta_end = 20.0;   
        ba_config.max_branches = 10; // Explore more branches for these harder problems
        ba_config.schedule_type = navokoj::BranchAwareOptimizer<State>::ScheduleType::GEOMETRIC; // Reparametrization

        navokoj::BranchAwareOptimizer<State> ba(energy, sampler, neighbors);
        auto ba_res = ba.optimize(ba_config);

        typename navokoj::SimulatedAnnealing<State>::Config sa_config;
        sa_config.beta_steps = 1000;
        sa_config.beta_end = 20.0;
        sa_config.steps_per_beta = 50; // Give SA even more steps (50 * 1000 = 50k steps)

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
    std::cout << "  Speedup Factor (Energy Reduction): " << (sa_total_e / (ba_total_e + 1e-9)) << "x\n"; // +epsilon to avoid div0
    std::cout << "------------------------------------------------------------\n";
}

int main() {
    std::cout << "💀 WELCOME TO THE OMFG BENCHMARK (REDEMPTION ARC) 💀\n";
    std::cout << "(Replaced Planted SAT with actual hard problems)\n";

    // 1. RANDOM XORSAT
    // The solution space is shattered into small clusters.
    // Constraints are rigid.
    run_omfg<RandomXorSat, XorSatState>(
        "Random XORSAT (N=100, M=95)", // Near transition M/N=1 (actually transition moves with k, for 3-XUORSat it's different but usually harder)
        5,
        [](int seed) { return RandomXorSat(100, 95, seed); }
    );

    // 2. HARD RANDOM 3-SAT (Phase Transition)
    // alpha = 4.26, N=50 -> M = 213 (rounded)
    run_omfg<HardRandom3Sat, SatState>(
        "Hard Random 3-SAT (N=50, M=213, alpha=4.26)", 
        5,
        [](int seed) { return HardRandom3Sat(50, 213, seed + 100); }
    );

    return 0;
}
