/*
 * Spin Glass Comparison Benchmark
 * 
 * Compares three optimization methods on SK spin glass:
 * 1. Vanilla Adafactor (no fracture detection)
 * 2. Simulated Annealing (equivalent compute budget)
 * 3. BAHA-aware Adafactor (fracture-aware scheduling)
 * 
 * Metrics: continuous energy, discrete energy, rounding gap, time, convergence
 * Sizes: N = 64, 128, 256
 * 
 * Author: Sethurathienam Iyer, ShunyaBar Labs
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "baha/baha.hpp"

// Sherrington-Kirkpatrick spin glass
class SpinGlass {
public:
    int N;
    std::vector<std::vector<double>> J;
    std::vector<double> s;
    unsigned seed;

    SpinGlass(int n, unsigned seed_) : N(n), s(n, 0.0), seed(seed_) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(static_cast<double>(N)));

        J.resize(N, std::vector<double>(N, 0.0));
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                J[i][j] = dist(rng);
                J[j][i] = J[i][j];
            }
        }

        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        for (int i = 0; i < N; ++i) {
            s[i] = uniform(rng);
        }
    }

    void reset() {
        std::mt19937 rng(seed + 999); // Different init for fairness
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        for (int i = 0; i < N; ++i) {
            s[i] = uniform(rng);
        }
    }

    double energy() const {
        double E = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                E -= J[i][j] * s[i] * s[j];
            }
        }
        return E;
    }

    std::vector<double> gradient() const {
        std::vector<double> grad(N, 0.0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    grad[i] -= J[i][j] * s[j];
                }
            }
        }
        return grad;
    }

    void clamp() {
        for (int i = 0; i < N; ++i) {
            s[i] = std::max(-1.0, std::min(1.0, s[i]));
        }
    }

    double discrete_energy() const {
        double E = 0.0;
        for (int i = 0; i < N; ++i) {
            int si = (s[i] >= 0.0) ? 1 : -1;
            for (int j = i + 1; j < N; ++j) {
                int sj = (s[j] >= 0.0) ? 1 : -1;
                E -= J[i][j] * si * sj;
            }
        }
        return E;
    }

    double magnetization() const {
        double M = 0.0;
        for (int i = 0; i < N; ++i) {
            M += (s[i] >= 0.0) ? 1.0 : -1.0;
        }
        return M / N;
    }

    bool is_saturated(double tol = 0.01) const {
        for (int i = 0; i < N; ++i) {
            if (std::abs(std::abs(s[i]) - 1.0) > tol) return false;
        }
        return true;
    }
};

struct BenchmarkResult {
    std::string method;
    int N;
    double continuous_energy;
    double discrete_energy;
    double rounding_gap;
    double magnetization;
    bool saturated;
    int fractures;
    double time_ms;
    bool converged;
    bool diverged;
    int oscillations; // Sign changes in energy delta
    double best_energy;
};

// Vanilla Adafactor (no fracture detection)
// Use aggressive LR to expose instability
BenchmarkResult run_vanilla_adafactor(SpinGlass& glass, int steps, double lr = 0.35) {
    glass.reset();
    
    std::vector<double> v(glass.N, 0.0);
    double decay = 0.85, eps = 1e-8;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    double prev_energy = glass.energy();
    double best_energy = prev_energy;
    double prev_delta = 0.0;
    int stall_count = 0;
    int oscillations = 0;
    bool diverged = false;
    
    for (int t = 0; t < steps; ++t) {
        auto grad = glass.gradient();
        
        for (int i = 0; i < glass.N; ++i) {
            v[i] = decay * v[i] + (1.0 - decay) * grad[i] * grad[i];
            glass.s[i] -= lr * grad[i] / (std::sqrt(v[i]) + eps);
        }
        glass.clamp();
        
        double E = glass.energy();
        if (std::isnan(E) || std::isinf(E)) {
            diverged = true;
            break;
        }
        
        if (E < best_energy) best_energy = E;
        
        double delta = E - prev_energy;
        if (t > 0 && delta * prev_delta < 0) oscillations++;
        
        if (std::abs(delta) < 1e-8) stall_count++;
        else stall_count = 0;
        
        prev_delta = delta;
        prev_energy = E;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    double cont_E = glass.energy();
    double disc_E = glass.discrete_energy();
    
    BenchmarkResult r;
    r.method = "Vanilla Adafactor";
    r.N = glass.N;
    r.continuous_energy = cont_E;
    r.discrete_energy = disc_E;
    r.rounding_gap = std::abs(cont_E - disc_E);
    r.magnetization = glass.magnetization();
    r.saturated = glass.is_saturated();
    r.fractures = 0;
    r.time_ms = time_ms;
    r.converged = !diverged;
    r.diverged = diverged;
    r.oscillations = oscillations;
    r.best_energy = best_energy;
    return r;
}

// BAHA-aware Adafactor
// Same aggressive LR, but with fracture-aware adaptation
BenchmarkResult run_baha_adafactor(SpinGlass& glass, int steps, double base_lr = 0.35) {
    glass.reset();
    
    std::vector<double> v(glass.N, 0.0);
    double base_decay = 0.85;
    double lr = base_lr, decay = base_decay, eps = 1e-8;
    int cooldown = 0, fracture_count = 0;
    
    navokoj::FractureDetectorOptimized detector(0.8);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    double prev_energy = glass.energy();
    double best_energy = prev_energy;
    double prev_delta = 0.0;
    int stall_count = 0;
    int oscillations = 0;
    bool diverged = false;
    
    for (int t = 0; t < steps; ++t) {
        double E = glass.energy();
        auto grad = glass.gradient();
        
        // Fracture detection
        detector.record(static_cast<double>(t), -E);
        bool fractured = detector.is_fracture();
        
        // Meta-scheduler
        if (fractured) {
            lr = std::max(0.005, lr * 0.6);
            decay = std::min(0.99, decay * 1.08);
            cooldown = 15;
            fracture_count++;
        } else if (cooldown > 0) {
            cooldown--;
        } else {
            lr = lr + 0.05 * (base_lr - lr);
            decay = decay + 0.05 * (base_decay - decay);
        }
        
        for (int i = 0; i < glass.N; ++i) {
            v[i] = decay * v[i] + (1.0 - decay) * grad[i] * grad[i];
            glass.s[i] -= lr * grad[i] / (std::sqrt(v[i]) + eps);
        }
        glass.clamp();
        
        if (std::isnan(E) || std::isinf(E)) {
            diverged = true;
            break;
        }
        
        if (E < best_energy) best_energy = E;
        
        double delta = E - prev_energy;
        if (t > 0 && delta * prev_delta < 0) oscillations++;
        
        if (std::abs(delta) < 1e-8) stall_count++;
        else stall_count = 0;
        
        prev_delta = delta;
        prev_energy = E;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    double cont_E = glass.energy();
    double disc_E = glass.discrete_energy();
    
    BenchmarkResult r;
    r.method = "BAHA Adafactor";
    r.N = glass.N;
    r.continuous_energy = cont_E;
    r.discrete_energy = disc_E;
    r.rounding_gap = std::abs(cont_E - disc_E);
    r.magnetization = glass.magnetization();
    r.saturated = glass.is_saturated();
    r.fractures = fracture_count;
    r.time_ms = time_ms;
    r.converged = !diverged;
    r.diverged = diverged;
    r.oscillations = oscillations;
    r.best_energy = best_energy;
    return r;
}

// Simulated Annealing (discrete, equivalent compute budget)
BenchmarkResult run_simulated_annealing(SpinGlass& glass, int energy_evals) {
    // Reset to discrete random
    std::mt19937 rng(glass.seed + 12345);
    std::vector<int> spins(glass.N);
    for (int i = 0; i < glass.N; ++i) {
        spins[i] = (rng() % 2 == 0) ? 1 : -1;
    }
    
    // Compute initial energy
    auto compute_energy = [&]() {
        double E = 0.0;
        for (int i = 0; i < glass.N; ++i) {
            for (int j = i + 1; j < glass.N; ++j) {
                E -= glass.J[i][j] * spins[i] * spins[j];
            }
        }
        return E;
    };
    
    // Delta energy for single spin flip
    auto delta_energy = [&](int k) {
        double dE = 0.0;
        for (int j = 0; j < glass.N; ++j) {
            if (j != k) {
                dE += 2.0 * glass.J[k][j] * spins[k] * spins[j];
            }
        }
        return dE;
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    
    double E = compute_energy();
    double best_E = E;
    std::vector<int> best_spins = spins;
    
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::uniform_int_distribution<int> spin_dist(0, glass.N - 1);
    
    // Annealing schedule
    double T_start = 2.0, T_end = 0.01;
    int total_steps = energy_evals / 2; // Each step ~2 energy-related ops
    
    for (int t = 0; t < total_steps; ++t) {
        double T = T_start * std::pow(T_end / T_start, static_cast<double>(t) / total_steps);
        
        int k = spin_dist(rng);
        double dE = delta_energy(k);
        
        if (dE < 0 || uniform(rng) < std::exp(-dE / T)) {
            spins[k] = -spins[k];
            E += dE;
            
            if (E < best_E) {
                best_E = E;
                best_spins = spins;
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Compute magnetization
    double M = 0.0;
    for (int i = 0; i < glass.N; ++i) {
        M += best_spins[i];
    }
    M /= glass.N;
    
    BenchmarkResult r;
    r.method = "Simulated Annealing";
    r.N = glass.N;
    r.continuous_energy = best_E; // SA is discrete, so continuous = discrete
    r.discrete_energy = best_E;
    r.rounding_gap = 0.0; // No rounding needed
    r.magnetization = M;
    r.saturated = true; // Always discrete
    r.fractures = 0;
    r.time_ms = time_ms;
    r.converged = true;
    r.diverged = false;
    r.oscillations = 0;
    r.best_energy = best_E;
    return r;
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::left << std::setw(20) << r.method
              << " | N=" << std::setw(4) << r.N
              << " | E_best=" << std::setw(10) << std::fixed << std::setprecision(4) << r.best_energy
              << " | E_disc=" << std::setw(10) << r.discrete_energy
              << " | gap=" << std::setw(6) << std::setprecision(2) << r.rounding_gap
              << " | osc=" << std::setw(4) << r.oscillations
              << " | frac=" << std::setw(3) << r.fractures
              << " | " << std::setw(7) << std::setprecision(1) << r.time_ms << "ms"
              << " | " << (r.diverged ? "DIVERGED" : "OK") << "\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "SPIN GLASS COMPARISON BENCHMARK\n";
    std::cout << "============================================================\n\n";

    std::vector<int> sizes = {64, 128, 256};
    std::vector<BenchmarkResult> all_results;

    // CSV output
    std::ofstream csv("spin_glass_benchmark.csv");
    csv << "method,N,best_energy,continuous_energy,discrete_energy,rounding_gap,magnetization,saturated,fractures,oscillations,time_ms,converged,diverged\n";

    for (int N : sizes) {
        std::cout << "=== N = " << N << " (search space: 2^" << N << ") ===\n";
        
        SpinGlass glass(N, 42 + N);
        
        // Compute budget: scale with N^2 (energy computation is O(N^2))
        int steps = 300 + N * 2;
        int energy_evals = steps * N * 2; // Approximate energy evaluations for Adafactor
        
        // Run all three methods
        auto r1 = run_vanilla_adafactor(glass, steps);
        print_result(r1);
        all_results.push_back(r1);
        
        auto r2 = run_baha_adafactor(glass, steps);
        print_result(r2);
        all_results.push_back(r2);
        
        auto r3 = run_simulated_annealing(glass, energy_evals);
        print_result(r3);
        all_results.push_back(r3);
        
        std::cout << "\n";
    }

    // Write CSV
    for (const auto& r : all_results) {
        csv << r.method << "," << r.N << ","
            << r.best_energy << "," << r.continuous_energy << "," << r.discrete_energy << ","
            << r.rounding_gap << "," << r.magnetization << ","
            << (r.saturated ? 1 : 0) << "," << r.fractures << "," << r.oscillations << ","
            << r.time_ms << "," << (r.converged ? 1 : 0) << "," << (r.diverged ? 1 : 0) << "\n";
    }
    csv.close();

    std::cout << "============================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "============================================================\n\n";

    // Summary by method
    for (const std::string& method : {"Vanilla Adafactor", "BAHA Adafactor", "Simulated Annealing"}) {
        std::cout << method << ":\n";
        for (int N : sizes) {
            for (const auto& r : all_results) {
                if (r.method == method && r.N == N) {
                    std::cout << "  N=" << N 
                              << ": E_best=" << std::fixed << std::setprecision(4) << r.best_energy
                              << ", osc=" << r.oscillations
                              << ", frac=" << r.fractures
                              << ", time=" << std::setprecision(1) << r.time_ms << "ms"
                              << (r.diverged ? " [DIVERGED]" : "") << "\n";
                }
            }
        }
        std::cout << "\n";
    }

    std::cout << "Results written to spin_glass_benchmark.csv\n\n";

    std::cout << "KEY METRICS TO COMPARE:\n";
    std::cout << "- Oscillations: High = unstable optimization (vanilla should be higher)\n";
    std::cout << "- Fractures: BAHA detects and adapts to phase transitions\n";
    std::cout << "- Best energy: Lower is better (SA may win on discrete)\n";
    std::cout << "- Divergence: Vanilla may diverge with aggressive LR\n";
    std::cout << "- Scaling: Watch oscillations grow with N for vanilla\n";

    return 0;
}
