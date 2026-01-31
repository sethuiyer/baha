/*
 * Spin Glass Optimization using Full BAHA (BranchAwareOptimizer)
 * 
 * Uses the same config as baha.cpp:
 * - Discrete state sampling
 * - Fracture detection via log Z estimation
 * - Lambert-W branch enumeration and jumping
 * 
 * Author: Sethurathienam Iyer, ShunyaBar Labs
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "baha/baha.hpp"

// Discrete spin glass state
struct SpinState {
    std::vector<int> spins; // {-1, +1}
};

// Sherrington-Kirkpatrick spin glass
class SKSpinGlass {
public:
    int N;
    std::vector<std::vector<double>> J;
    mutable std::mt19937 rng;

    SKSpinGlass(int n, unsigned seed = 42) : N(n), rng(seed) {
        std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(static_cast<double>(N)));

        J.resize(N, std::vector<double>(N, 0.0));
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                J[i][j] = dist(rng);
                J[j][i] = J[i][j];
            }
        }
    }

    double energy(const SpinState& s) const {
        double E = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                E -= J[i][j] * s.spins[i] * s.spins[j];
            }
        }
        return E;
    }

    SpinState random_state() {
        SpinState s;
        s.spins.resize(N);
        std::bernoulli_distribution coin(0.5);
        for (int i = 0; i < N; ++i) {
            s.spins[i] = coin(rng) ? 1 : -1;
        }
        return s;
    }

    std::vector<SpinState> neighbors(const SpinState& s) const {
        std::vector<SpinState> nbrs;
        nbrs.reserve(N);
        for (int i = 0; i < N; ++i) {
            SpinState nbr = s;
            nbr.spins[i] = -nbr.spins[i]; // Flip one spin
            nbrs.push_back(std::move(nbr));
        }
        return nbrs;
    }

    double magnetization(const SpinState& s) const {
        double M = 0.0;
        for (int i = 0; i < N; ++i) {
            M += s.spins[i];
        }
        return M / N;
    }
};

int main() {
    std::cout << "============================================================\n";
    std::cout << "SPIN GLASS OPTIMIZATION (FULL BAHA - DISCRETE SAMPLING)\n";
    std::cout << "============================================================\n\n";

    const int N = 64;
    SKSpinGlass problem(N, 12345);

    std::cout << "Spins: " << N << " (search space: 2^" << N << " discrete configs)\n\n";

    // Energy function
    auto energy_fn = [&problem](const SpinState& s) {
        return problem.energy(s);
    };

    // Sampler (random discrete state)
    auto sampler_fn = [&problem]() mutable {
        return problem.random_state();
    };

    // Neighbor function (single spin flips)
    auto neighbor_fn = [&problem](const SpinState& s) {
        return problem.neighbors(s);
    };

    // BAHA optimizer with same config as baha.cpp
    navokoj::BranchAwareOptimizer<SpinState> baha_opt(energy_fn, sampler_fn, neighbor_fn);
    
    typename navokoj::BranchAwareOptimizer<SpinState>::Config config;
    config.beta_start = 0.01;
    config.beta_end = 10.0;
    config.beta_steps = 500;
    config.fracture_threshold = 1.5;
    config.beta_critical = 1.0;
    config.max_branches = 5;
    config.samples_per_beta = 100;
    config.verbose = true;

    std::cout << "Config:\n";
    std::cout << "  beta: " << config.beta_start << " -> " << config.beta_end << "\n";
    std::cout << "  steps: " << config.beta_steps << "\n";
    std::cout << "  fracture_threshold: " << config.fracture_threshold << "\n";
    std::cout << "  samples_per_beta: " << config.samples_per_beta << "\n\n";

    auto result = baha_opt.optimize(config);

    std::cout << "\n============================================================\n";
    std::cout << "BAHA RESULTS\n";
    std::cout << "============================================================\n";
    std::cout << "Best energy:       " << std::fixed << std::setprecision(4) << result.best_energy << "\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps:       " << result.branch_jumps << "\n";
    std::cout << "Steps taken:        " << result.steps_taken << "\n";
    std::cout << "Time:               " << std::fixed << std::setprecision(2) << result.time_ms << " ms\n";
    std::cout << "Beta at solution:   " << std::fixed << std::setprecision(4) << result.beta_at_solution << "\n";

    // Compute magnetization
    double M = problem.magnetization(result.best_state);
    int up_count = 0, down_count = 0;
    for (int i = 0; i < N; ++i) {
        if (result.best_state.spins[i] == 1) up_count++;
        else down_count++;
    }

    std::cout << "\n============================================================\n";
    std::cout << "SOLUTION ANALYSIS\n";
    std::cout << "============================================================\n";
    std::cout << "Magnetization:      " << std::fixed << std::setprecision(4) << M << "\n";
    std::cout << "Spin distribution:  " << up_count << " up (+1), " << down_count << " down (-1)\n\n";

    std::cout << "Spin assignments (" << N << " spins):\n";
    for (int i = 0; i < N; ++i) {
        std::cout << (result.best_state.spins[i] == 1 ? "+" : "-");
        if ((i + 1) % 16 == 0) std::cout << "\n";
    }

    // Compare with Simulated Annealing
    std::cout << "\n============================================================\n";
    std::cout << "COMPARISON: SIMULATED ANNEALING\n";
    std::cout << "============================================================\n";

    navokoj::SimulatedAnnealing<SpinState> sa_opt(energy_fn, sampler_fn, neighbor_fn);
    typename navokoj::SimulatedAnnealing<SpinState>::Config sa_config;
    sa_config.beta_start = 0.01;
    sa_config.beta_end = 10.0;
    sa_config.beta_steps = 500;
    sa_config.steps_per_beta = 10;
    sa_config.verbose = false;

    auto sa_result = sa_opt.optimize(sa_config);

    std::cout << "SA Best energy:     " << std::fixed << std::setprecision(4) << sa_result.best_energy << "\n";
    std::cout << "SA Steps taken:     " << sa_result.steps_taken << "\n";
    std::cout << "SA Time:            " << std::fixed << std::setprecision(2) << sa_result.time_ms << " ms\n";
    std::cout << "SA Beta at solution:" << std::fixed << std::setprecision(4) << sa_result.beta_at_solution << "\n";

    std::cout << "\n============================================================\n";
    std::cout << "COMPARISON SUMMARY\n";
    std::cout << "============================================================\n";
    std::cout << "| Method | Energy    | Fractures | Jumps | Time (ms) |\n";
    std::cout << "|--------|-----------|-----------|-------|----------|\n";
    std::cout << "| BAHA   | " << std::setw(9) << std::fixed << std::setprecision(4) << result.best_energy
              << " | " << std::setw(9) << result.fractures_detected
              << " | " << std::setw(5) << result.branch_jumps
              << " | " << std::setw(8) << std::setprecision(2) << result.time_ms << " |\n";
    std::cout << "| SA     | " << std::setw(9) << std::fixed << std::setprecision(4) << sa_result.best_energy
              << " | " << std::setw(9) << "—"
              << " | " << std::setw(5) << "—"
              << " | " << std::setw(8) << std::setprecision(2) << sa_result.time_ms << " |\n";

    if (result.best_energy < sa_result.best_energy) {
        double improvement = 100.0 * (sa_result.best_energy - result.best_energy) / std::abs(sa_result.best_energy);
        std::cout << "\nBAHA found " << std::fixed << std::setprecision(1) << improvement << "% better solution!\n";
    } else if (sa_result.best_energy < result.best_energy) {
        double improvement = 100.0 * (result.best_energy - sa_result.best_energy) / std::abs(result.best_energy);
        std::cout << "\nSA found " << std::fixed << std::setprecision(1) << improvement << "% better solution.\n";
    } else {
        std::cout << "\nBoth methods found the same energy.\n";
    }

    std::cout << "\nInterpretation:\n";
    std::cout << "- BAHA uses fracture detection + Lambert-W branch jumping.\n";
    std::cout << "- SA uses standard Metropolis random walk.\n";
    std::cout << "- Branch jumps allow BAHA to escape local minima.\n";

    return 0;
}
