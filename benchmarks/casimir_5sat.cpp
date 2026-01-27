#include "baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>

// =============================================================================
// HARD RANDOM 5-SAT (N=50, M=1500)
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

    int get_n() const { return n_; }
    const std::vector<Clause>& get_clauses() const { return clauses_; }

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
// CASIMIR-5-SAT SOLVER
// =============================================================================

class CasimirSat5Solver {
public:
    struct Config {
        int steps;
        double dt;
        double temperature_start;
        double temperature_end;
        double lambda_aux;
    };

    CasimirSat5Solver(const HardRandom5Sat& problem) : problem_(problem), rng_(std::random_device{}()) {}

    SatState solve(const Config& config = {2000, 0.01, 1.0, 0.0, 1.0}) {
        int N = problem_.get_n();
        std::vector<double> s(N); // Continuos spins
        std::uniform_real_distribution<> dist(-1.0, 1.0);
        std::normal_distribution<> gauss(0.0, 1.0);

        // Initialize random continuous state
        for(int i=0; i<N; ++i) s[i] = dist(rng_);

        // Langevin Dynamics
        for (int step = 0; step < config.steps; ++step) {
            double progress = (double)step / config.steps;
            double T = config.temperature_start + (config.temperature_end - config.temperature_start) * progress;
            
            std::vector<double> grad(N, 0.0);
            
            // Clause Potential Gradients (5-SAT)
            // V_c = Product_{k=0..4} ( (1 - J_k * s_k) / 2 )
            
            const auto& clauses = problem_.get_clauses();
            for (const auto& c : clauses) {
                double terms[5];
                int indices[5];
                double J[5]; 

                for(int k=0; k<5; ++k) {
                    indices[k] = c.lits[k];
                    J[k] = c.neg[k] ? -1.0 : 1.0;
                    terms[k] = (1.0 - J[k] * s[indices[k]]) / 2.0;
                }

                // Gradient for each variable in clause
                for(int k=0; k<5; ++k) {
                    double grad_contribution = -J[k] / 2.0;
                    // Multiply by all OTHER terms
                    for(int other=0; other<5; ++other) {
                        if(k != other) grad_contribution *= terms[other];
                    }
                    grad[indices[k]] += grad_contribution;
                }
            }

            // Auxiliary Potential Gradients
            for(int i=0; i<N; ++i) {
                 grad[i] += 4.0 * config.lambda_aux * s[i] * (s[i]*s[i] - 1.0);
            }

            // Update
            double noise_scale = std::sqrt(2.0 * T * config.dt);
            for(int i=0; i<N; ++i) {
                s[i] += -grad[i] * config.dt + noise_scale * gauss(rng_);
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
    const HardRandom5Sat& problem_;
    mutable std::mt19937 rng_;
};

int main() {
    std::cout << "⚛️ CASIMIR-5-SAT vs BAHA ⚛️\n";
    std::cout << "Testing on High-Density 5-SAT (Alpha=30)\n";
    std::cout << "========================================\n";

    int n_trials = 5;
    int N = 50;
    int M = 1500; 

    int ba_better = 0, casimir_better = 0;
    double ba_total_e = 0, casimir_total_e = 0;

    for (int t = 0; t < n_trials; ++t) {
        std::cout << "Trial " << (t+1) << "... ";
        std::cout.flush();

        HardRandom5Sat prob(N, M, t + 5555);
        
        // 1. Run Casimir
        CasimirSat5Solver casimir(prob);
        CasimirSat5Solver::Config c_config;
        c_config.steps = 5000;
        c_config.dt = 0.05;
        
        SatState c_sol = casimir.solve(c_config);
        double c_energy = prob.energy(c_sol);

        // 2. Run BAHA
        std::function<double(const SatState&)> energy = [&](const SatState& s) { return prob.energy(s); };
        std::function<SatState()> sampler = [&]() { return prob.random_state(); };
        std::function<std::vector<SatState>(const SatState&)> neighbors = [&](const SatState& s) { return prob.neighbors(s); };

        typename navokoj::BranchAwareOptimizer<SatState>::Config ba_config;
        ba_config.beta_steps = 2000;
        ba_config.beta_end = 20.0;
        ba_config.schedule_type = navokoj::BranchAwareOptimizer<SatState>::ScheduleType::GEOMETRIC;

        navokoj::BranchAwareOptimizer<SatState> ba(energy, sampler, neighbors);
        auto ba_res = ba.optimize(ba_config);
        double ba_energy = ba_res.best_energy;

        ba_total_e += ba_energy;
        casimir_total_e += c_energy;

        if (ba_energy < c_energy) {
            ba_better++;
            std::cout << "BA Wins (E=" << ba_energy << " vs " << c_energy << ")\n";
        } else if (c_energy < ba_energy) {
            casimir_better++;
            std::cout << "Casimir Wins (E=" << ba_energy << " vs " << c_energy << ")\n";
        } else {
            std::cout << "Tie (E=" << ba_energy << ")\n";
        }
    }

    std::cout << "------------------------------------------------------------\n";
    std::cout << "Summary for 5-SAT (N=50, M=1500):\n";
    std::cout << "  BA Win Rate: " << ba_better << "/" << n_trials << "\n";
    std::cout << "  Casimir Win Rate: " << casimir_better << "/" << n_trials << "\n";
    std::cout << "  BA Avg Energy: " << ba_total_e / n_trials << "\n";
    std::cout << "  Casimir Avg Energy: " << casimir_total_e / n_trials << "\n";
    std::cout << "------------------------------------------------------------\n";

    return 0;
}
