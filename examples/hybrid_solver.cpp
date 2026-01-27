#include "baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>

// =============================================================================
// HARD RANDOM 3-SAT (Same generator)
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

    int get_n() const { return n_; }
    const std::vector<Clause>& get_clauses() const { return clauses_; }

private:
    int n_; std::vector<Clause> clauses_; mutable std::mt19937 rng_;
};

// =============================================================================
// CASIMIR LANGEVIN BURST
// Runs a short burst of Langevin dynamics starting from a discrete state
// =============================================================================

class CasimirLangevin {
public:
    CasimirLangevin(const HardRandom3Sat& problem) : problem_(problem), rng_(std::random_device{}()) {}

    SatState refine(const SatState& initial_state, int steps = 100, double T = 0.5) {
        int N = problem_.get_n();
        std::vector<double> s(N);
        std::normal_distribution<> gauss(0.0, 1.0);

        // Initialize from discrete state (relaxed)
        for(int i=0; i<N; ++i) {
            s[i] = initial_state.assignment[i] ? 1.0 : -1.0;
            // Add slight noise to break symmetry if needed, but 1.0 is fine
        }

        double dt = 0.05;
        double lambda = 1.0; 

        for (int step = 0; step < steps; ++step) {
            std::vector<double> grad(N, 0.0);
            const auto& clauses = problem_.get_clauses();
            
            // Clause Gradients
            for (const auto& c : clauses) {
                double terms[3];
                int indices[3];
                double J[3]; 

                for(int k=0; k<3; ++k) {
                    indices[k] = c.lits[k];
                    J[k] = c.neg[k] ? -1.0 : 1.0;
                    terms[k] = (1.0 - J[k] * s[indices[k]]) / 2.0;
                }

                for(int k=0; k<3; ++k) {
                    double grad_contribution = -J[k] / 2.0;
                    for(int other=0; other<3; ++other) {
                        if(k != other) grad_contribution *= terms[other];
                    }
                    grad[indices[k]] += grad_contribution;
                }
            }

            // Confining Gradients
            for(int i=0; i<N; ++i) {
                 grad[i] += 4.0 * lambda * s[i] * (s[i]*s[i] - 1.0);
            }

            // Update
            double noise_scale = std::sqrt(2.0 * T * dt);
            for(int i=0; i<N; ++i) {
                s[i] += -grad[i] * dt + noise_scale * gauss(rng_);
                if (s[i] > 2.0) s[i] = 2.0;
                if (s[i] < -2.0) s[i] = -2.0;
            }
        }

        SatState result;
        result.assignment.resize(N);
        for(int i=0; i<N; ++i) result.assignment[i] = (s[i] > 0);
        return result;
    }

private:
    const HardRandom3Sat& problem_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// MAIN RUNNER
// =============================================================================

int main() {
    std::cout << "🧬 HYBRID SOLVER: BAHA + CASIMIR 🧬\n";
    std::cout << "Strategy: BAHA jumps to basins, Casimir refines with physics gradients.\n";
    std::cout << "=====================================================================\n";

    int n_trials = 5;
    int N = 50;
    int M = 213; // alpha = 4.26 (Hardest)

    int hybrid_wins = 0;
    double hybrid_total_e = 0;
    double baha_total_e = 0;

    for (int t = 0; t < n_trials; ++t) {
        std::cout << "Trial " << (t+1) << "... ";
        std::cout.flush();

        HardRandom3Sat prob(N, M, t + 2024);
        
        // 1. PURE BAHA (as baseline)
        std::function<double(const SatState&)> energy = [&](const SatState& s) { return prob.energy(s); };
        std::function<SatState()> sampler = [&]() { return prob.random_state(); };
        std::function<std::vector<SatState>(const SatState&)> discrete_neighbors = [&](const SatState& s) { 
            std::vector<SatState> nbrs;
            // Standard bit flip neighborhoods
             for (int i = 0; i < N; ++i) {
                SatState nbr = s; nbr.assignment[i] = !nbr.assignment[i]; nbrs.push_back(nbr);
            }
            return nbrs;
        };

        typename navokoj::BranchAwareOptimizer<SatState>::Config config;
        config.beta_steps = 1000;
        config.beta_end = 20.0;
        config.schedule_type = navokoj::BranchAwareOptimizer<SatState>::ScheduleType::GEOMETRIC;

        navokoj::BranchAwareOptimizer<SatState> baha(energy, sampler, discrete_neighbors);
        auto baha_res = baha.optimize(config);

        // 2. HYBRID BAHA-CASIMIR
        // We inject Casimir dynamics as the "Neighbor" function.
        // When BAHA asks for neighbors (local search), we give it the Casimir-refined state.
        // This is a powerful "Large Neighborhood Search" move.
        
        CasimirLangevin casimir_ops(prob);
        
        std::function<std::vector<SatState>(const SatState&)> hybrid_neighbors = [&](const SatState& s) {
            // Run a short burst of Langevin dynamics
            SatState refined = casimir_ops.refine(s, 50, 0.2); // 50 steps, low T
            return std::vector<SatState>{refined}; 
        };

        navokoj::BranchAwareOptimizer<SatState> hybrid(energy, sampler, hybrid_neighbors);
        
        // Hybrid allows for fewer beta steps but more expensive "neighbors"
        typename navokoj::BranchAwareOptimizer<SatState>::Config h_config = config;
        h_config.beta_steps = 500; // Fewer steps because each step is smarter
        
        auto hybrid_res = hybrid.optimize(h_config);

        baha_total_e += baha_res.best_energy;
        hybrid_total_e += hybrid_res.best_energy;

        if (hybrid_res.best_energy < baha_res.best_energy) {
            hybrid_wins++;
            std::cout << "Hybrid Wins (E=" << hybrid_res.best_energy << " vs " << baha_res.best_energy << ")\n";
        } else if (baha_res.best_energy < hybrid_res.best_energy) {
            std::cout << "Purist Wins (E=" << hybrid_res.best_energy << " vs " << baha_res.best_energy << ")\n";
        } else {
            std::cout << "Tie (E=" << hybrid_res.best_energy << ")\n";
        }
    }

    std::cout << "------------------------------------------------------------\n";
    std::cout << "Summary:\n";
    std::cout << "  Hybrid Win Rate: " << hybrid_wins << "/" << n_trials << "\n";
    std::cout << "  Hybrid Avg Energy: " << hybrid_total_e / n_trials << "\n";
    std::cout << "  Pure BAHA Avg Energy: " << baha_total_e / n_trials << "\n";
    std::cout << "------------------------------------------------------------\n";

    return 0;
}
