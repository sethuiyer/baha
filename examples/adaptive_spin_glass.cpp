/*
 * Spin Glass Optimization using AdaptiveOptimizer
 * 
 * The AdaptiveOptimizer auto-selects the best strategy:
 * - Probes fracture density first
 * - High fractures → BranchAwareOptimizer
 * - Low fractures → ZetaBreatherOptimizer (if continuous functions available)
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

// Continuous encoding of spin state
using ContinuousState = std::vector<double>;

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

    // Continuous energy (soft spins in [-1, 1])
    double continuous_energy(const ContinuousState& s, double beta) const {
        double E = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                E -= J[i][j] * s[i] * s[j];
            }
        }
        return beta * E;
    }

    // Gradient of continuous energy
    ContinuousState continuous_gradient(const ContinuousState& s, double beta) const {
        ContinuousState grad(N, 0.0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    grad[i] -= beta * J[i][j] * s[j];
                }
            }
        }
        return grad;
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
            nbr.spins[i] = -nbr.spins[i];
            nbrs.push_back(std::move(nbr));
        }
        return nbrs;
    }

    // Encode discrete → continuous
    ContinuousState encode(const SpinState& s) const {
        ContinuousState c(N);
        for (int i = 0; i < N; ++i) {
            c[i] = static_cast<double>(s.spins[i]);
        }
        return c;
    }

    // Decode continuous → discrete
    SpinState decode(const ContinuousState& c) const {
        SpinState s;
        s.spins.resize(N);
        for (int i = 0; i < N; ++i) {
            s.spins[i] = (c[i] >= 0.0) ? 1 : -1;
        }
        return s;
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
    std::cout << "SPIN GLASS OPTIMIZATION (ADAPTIVE OPTIMIZER)\n";
    std::cout << "============================================================\n\n";

    const int N = 64;
    SKSpinGlass problem(N, 12345);

    std::cout << "Spins: " << N << " (search space: 2^" << N << " discrete configs)\n\n";

    // Energy function
    auto energy_fn = [&problem](const SpinState& s) {
        return problem.energy(s);
    };

    // Sampler
    auto sampler_fn = [&problem]() mutable {
        return problem.random_state();
    };

    // Neighbors
    auto neighbor_fn = [&problem](const SpinState& s) {
        return problem.neighbors(s);
    };

    // Encode/decode for ZetaBreather
    auto encode_fn = [&problem](const SpinState& s) {
        return problem.encode(s);
    };

    auto decode_fn = [&problem](const ContinuousState& c) {
        return problem.decode(c);
    };

    // Continuous energy/gradient for ZetaBreather
    auto cont_energy_fn = [&problem](const ContinuousState& s, double beta) {
        return problem.continuous_energy(s, beta);
    };

    auto cont_gradient_fn = [&problem](const ContinuousState& s, double beta) {
        return problem.continuous_gradient(s, beta);
    };

    // AdaptiveOptimizer with all functions
    navokoj::AdaptiveOptimizer<SpinState> adaptive(
        energy_fn, sampler_fn, neighbor_fn,
        encode_fn, decode_fn,
        cont_energy_fn, cont_gradient_fn
    );

    typename navokoj::AdaptiveOptimizer<SpinState>::Config config;
    config.probe_steps = 100;
    config.probe_samples = 20;
    config.fracture_threshold = 0.3;
    config.ba_beta_steps = 300;
    config.ba_samples_per_beta = 80;
    config.verbose = true;

    std::cout << "Config:\n";
    std::cout << "  probe_steps: " << config.probe_steps << "\n";
    std::cout << "  fracture_threshold: " << config.fracture_threshold << "\n";
    std::cout << "  (Will auto-select BranchAware or Zeta based on probe)\n\n";

    auto result = adaptive.optimize(config);

    std::cout << "\n============================================================\n";
    std::cout << "ADAPTIVE OPTIMIZER RESULTS\n";
    std::cout << "============================================================\n";
    std::cout << "Best energy:        " << std::fixed << std::setprecision(4) << result.best_energy << "\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Fracture density:   " << std::fixed << std::setprecision(4) << result.fracture_density << "\n";
    std::cout << "Used BranchAware:   " << (result.used_branch_aware ? "YES" : "NO (used Zeta)") << "\n";
    std::cout << "Steps taken:        " << result.steps_taken << "\n";
    std::cout << "Time:               " << std::fixed << std::setprecision(2) << result.time_ms << " ms\n";

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

    std::cout << "\nInterpretation:\n";
    std::cout << "- AdaptiveOptimizer probed the landscape and detected fracture density.\n";
    if (result.used_branch_aware) {
        std::cout << "- High fracture density → switched to BranchAwareOptimizer.\n";
        std::cout << "- Used Lambert-W branch jumping to escape local minima.\n";
    } else {
        std::cout << "- Low fracture density → switched to ZetaBreatherOptimizer.\n";
        std::cout << "- Used gradient descent with breathing beta schedule.\n";
    }

    return 0;
}
