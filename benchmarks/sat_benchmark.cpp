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
// HARD RANDOM 5-SAT (N=50, Alpha=30 -> M=1500)
// =============================================================================

struct SatState {
    std::vector<bool> assignment;
};

class HardRandom5Sat {
public:
    struct Clause {
        int lits[5];
        bool neg[5];
    };

    HardRandom5Sat(int n, int m, int seed) : n_(n), rng_(seed) {
        std::uniform_int_distribution<> var_dist(0, n - 1);
        std::uniform_int_distribution<> bool_dist(0, 1);
        
        for (int i = 0; i < m; ++i) {
            Clause c;
            for (int j = 0; j < 5; ++j) c.lits[j] = var_dist(rng_);
            for (int j = 0; j < 5; ++j) c.neg[j] = bool_dist(rng_);
            clauses_.push_back(c);
        }
    }

    double energy(const SatState& s) const {
        int unsatisfied = 0;
        for (const auto& c : clauses_) {
            bool sat = false;
            for (int k = 0; k < 5; ++k) {
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
    int n_;
    std::vector<Clause> clauses_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// MAIN RUNNER
// =============================================================================

template<typename Problem, typename State>
void run_sat_bench(const std::string& name, int n_trials, std::function<Problem(int)> factory) {
    std::cout << "\nRunning High-Density Challenge: " << name << "\n";
    std::cout << "============================================================\n";
    
    int ba_wins = 0;
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
        ba_config.beta_steps = 1000; 
        ba_config.beta_end = 20.0;   
        ba_config.max_branches = 10;
        ba_config.schedule_type = navokoj::BranchAwareOptimizer<State>::ScheduleType::GEOMETRIC;

        navokoj::BranchAwareOptimizer<State> ba(energy, sampler, neighbors);
        auto ba_res = ba.optimize(ba_config);

        typename navokoj::SimulatedAnnealing<State>::Config sa_config;
        sa_config.beta_steps = 1000;
        sa_config.beta_end = 20.0;
        sa_config.steps_per_beta = 50; 

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
    std::cout << "HIGH-DENSITY 5-SAT BENCHMARK\n";
    std::cout << "Testing Deep UNSAT Phase (Constraint Density Alpha=30)\n";

    run_sat_bench<HardRandom5Sat, SatState>(
        "Random 5-SAT (N=50, M=1500, Alpha=30)", 
        5,
        [](int seed) { return HardRandom5Sat(50, 1500, seed + 5555); }
    );

    return 0;
}
