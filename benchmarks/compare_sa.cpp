/*
 * Comparison: BAHA vs Standard Simulated Annealing
 */
#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>

// =============================================================================
// HARD RANDOM 3-SAT
// =============================================================================
struct SatState { std::vector<bool> assignment; };

class HardRandom3Sat {
public:
    struct Clause { int lits[3]; bool neg[3]; };

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
                if (val) { sat = true; break; }
            }
            if (!sat) unsatisfied++;
        }
        return static_cast<double>(unsatisfied);
    }

    SatState random_state() {
        SatState s; s.assignment.resize(n_);
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n_; ++i) s.assignment[i] = dist(rng_);
        return s;
    }

    std::vector<SatState> neighbors(const SatState& s) {
        std::vector<SatState> nbrs;
        // Just flip one variable. Full neighborhood might be large for SA if we generate all,
        // but for standard SA usually we pick ONE neighbor.
        // BAHA interface expects vector of neighbors.
        // For efficiency in this test we'll generate all N neighbors.
        for (int i = 0; i < n_; ++i) {
            SatState nbr = s; nbr.assignment[i] = !nbr.assignment[i]; nbrs.push_back(nbr);
        }
        return nbrs;
    }
    
    // Optimized neighbor for SA (single random pick) - but libraries use same interface.
    // The library SimualtedAnnealing takes neighbor function that returns vector.
    // It picks one randomly from vector.

private:
    int n_; std::vector<Clause> clauses_; mutable std::mt19937 rng_;
};

int main() {
    std::cout << "🔥 SA vs BAHA Benchmark 🔥\n";
    std::cout << "==========================\n";

    int n_trials = 10;
    int N = 50;
    int M = 213; // ratio 4.26 (hard phase)

    int ba_wins = 0, sa_wins = 0, ties = 0;
    double ba_total_e = 0, sa_total_e = 0;

    for (int t = 0; t < n_trials; ++t) {
        HardRandom3Sat prob(N, M, t + 100);
        
        std::function<double(const SatState&)> energy = [&](const SatState& s) { return prob.energy(s); };
        std::function<SatState()> sampler = [&]() { return prob.random_state(); };
        std::function<std::vector<SatState>(const SatState&)> neighbors = [&](const SatState& s) { return prob.neighbors(s); };

        // 1. Run SA
        // Match computational effort: steps * neighbors?
        // SA config: beta_steps=500, steps_per_beta=10 -> 5000 evals (roughly)
        
        navokoj::SimulatedAnnealing<SatState> sa(energy, sampler, neighbors);
        navokoj::SimulatedAnnealing<SatState>::Config sa_config;
        sa_config.beta_steps = 500;
        sa_config.steps_per_beta = 10; 
        sa_config.beta_end = 20.0;
        
        auto sa_res = sa.optimize(sa_config);

        // 2. Run BAHA
        // BAHA steps = 500. Samples per beta = 100? No, that's for estimation.
        // Local search in BAHA: performs local search "hot path".
        // In perform_local_search: it iterates over ALL neighbors?
        // Let's check perform_local_search in baha.hpp
        // "for (const auto& nbr : nbrs)" - yes, it sweeps ALL neighbors.
        // So for N=50, one BAHA step does 50 evaluations.
        // 500 steps * 50 evals = 25,000 evaluations.
        
        // SA: 500 * 10 steps. In each step "nbrs = neighbors_(current)", then picks ONE.
        // But neighbors_ generates ALL 50. So it evaluates 50 energies to generate the vector?
        // Wait, if neighbors() generates vector, it calls energy inside optimize loop?
        // In SA: "double nbr_energy = energy_eval(nbr);"
        // It consumes the vector generation cost but only evaluates energy for ONE.
        // BAHA "perform_local_search" iterates ALL neighbors and checks energy for ALL.
        
        // So BAHA does N times more work per step if N is the neighborhood size.
        // To be fair, we should give SA more steps.
        // BAHA: 500 steps * 50 evals = 25,000 evals.
        // SA: needs 25,000 steps.
        
        sa_config.beta_steps = 2500;
        sa_config.steps_per_beta = 10; // 25,000 steps total.
        
        // Wait, 2500 * 10 = 25,000 steps.
        // Is BAHA doing 25000 evals?
        // baha.hpp: perform_local_search iterates `nbrs`. nbrs size is 50.
        // So yes, 50 evals per step. + LogZ estimation?
        // estimate_log_Z_opt: samples `samples_per_beta` (default 100).
        // 500 steps * (50 + 100) = 75,000 evaluations.
        
        // So BAHA is doing ~75k evals.
        // Let's give SA 75k steps (or 7500 * 10).
        
        sa_config.beta_steps = 7500;
        sa_config.steps_per_beta = 10; // 75,000 queries.
        
        auto sa_res_fair = sa.optimize(sa_config);

        navokoj::BranchAwareOptimizer<SatState> ba(energy, sampler, neighbors);
        typename navokoj::BranchAwareOptimizer<SatState>::Config ba_config;
        ba_config.beta_steps = 500;
        ba_config.beta_end = 20.0;
        ba_config.samples_per_beta = 100;

        auto ba_res = ba.optimize(ba_config);

        if (ba_res.best_energy < sa_res_fair.best_energy) {
            ba_wins++;
        } else if (sa_res_fair.best_energy < ba_res.best_energy) {
            sa_wins++;
        } else {
            ties++;
        }
        
        ba_total_e += ba_res.best_energy;
        sa_total_e += sa_res_fair.best_energy;
        
        std::cout << "Trial " << t << ": BA=" << ba_res.best_energy << " SA=" << sa_res_fair.best_energy << "\n";
    }

    std::cout << "Summary:\n";
    std::cout << "BA Wins: " << ba_wins << "\n";
    std::cout << "SA Wins: " << sa_wins << "\n";
    std::cout << "Ties: " << ties << "\n";
    std::cout << "Avg BA Energy: " << ba_total_e / n_trials << "\n";
    std::cout << "Avg SA Energy: " << sa_total_e / n_trials << "\n";
    
    return 0;
}
