/*
 * Author: Sethurathienam Iyer
 */
#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>

// =============================================================================
// PARTITION PROBLEM (for testing spectral analysis optimization)
// =============================================================================

struct PartitionState {
    std::vector<int> s;
    long long current_sum = 0;
};

class PartitionProblem {
public:
    PartitionProblem(int n, int seed) : n_(n), rng_(seed) {
        std::uniform_int_distribution<long long> dist(1, 1000000000000LL);
        numbers_.resize(n);
        for(int i=0; i<n; ++i) numbers_[i] = dist(rng_);
    }

    double energy(const PartitionState& state) const {
        return static_cast<double>(std::abs(state.current_sum));
    }

    PartitionState random_state() {
        PartitionState state;
        state.s.resize(n_);
        state.current_sum = 0;
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n_; ++i) {
            state.s[i] = dist(rng_) ? 1 : -1;
            state.current_sum += state.s[i] * numbers_[i];
        }
        return state;
    }

    std::vector<PartitionState> neighbors(const PartitionState& state) {
        std::vector<PartitionState> nbrs;
        std::uniform_int_distribution<> dist(0, n_ - 1);
        for (int i = 0; i < 32; ++i) {
            int idx = dist(rng_);
            PartitionState nbr = state;
            nbr.current_sum -= 2LL * nbr.s[idx] * numbers_[idx];
            nbr.s[idx] *= -1;
            nbrs.push_back(nbr);
        }
        return nbrs;
    }

    double analytic_log_z(double beta) const {
        double sum = 0;
        for (long long a : numbers_) {
            sum += std::log(std::cosh(beta * static_cast<double>(a)));
        }
        return sum;
    }

    double analytic_rho(double beta) const {
        double sum = 0;
        for (long long a : numbers_) {
            double x = beta * static_cast<double>(a);
            if (x > 20.0) continue;
            double s = 1.0 / std::cosh(x);
            sum += (double)a * a * s * s;
        }
        return beta * beta * sum;
    }

private:
    int n_;
    std::vector<long long> numbers_;
    mutable std::mt19937_64 rng_;
};

// =============================================================================
// BENCHMARK RUNNER
// =============================================================================

template<typename Optimizer>
void run_benchmark(const std::string& name, 
                   PartitionProblem& prob,
                   typename Optimizer::Config& config,
                   int n_trials = 5) {
    
    auto energy = [&](const PartitionState& s) { return prob.energy(s); };
    auto sampler = [&]() { return prob.random_state(); };
    auto neighbors = [&](const PartitionState& s) { return prob.neighbors(s); };
    
    std::vector<double> times;
    std::vector<double> energies;
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Benchmark: " << name << "\n";
    std::cout << std::string(60, '=') << "\n";
    
    for (int trial = 0; trial < n_trials; ++trial) {
        Optimizer optimizer(energy, sampler, neighbors);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = optimizer.optimize(config);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        times.push_back(time_ms);
        energies.push_back(result.best_energy);
        
        std::cout << "Trial " << (trial + 1) << "/" << n_trials 
                  << " - Time: " << std::fixed << std::setprecision(2) << time_ms << " ms"
                  << " - Energy: " << std::scientific << std::setprecision(3) << result.best_energy
                  << " - Fractures: " << result.fractures_detected
                  << " - Jumps: " << result.branch_jumps << "\n";
    }
    
    // Compute statistics
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double min_time = *std::min_element(times.begin(), times.end());
    double avg_energy = std::accumulate(energies.begin(), energies.end(), 0.0) / energies.size();
    double min_energy = *std::min_element(energies.begin(), energies.end());
    
    std::cout << "\n📊 RESULTS:\n";
    std::cout << "  Avg Time: " << std::fixed << std::setprecision(2) << avg_time << " ms\n";
    std::cout << "  Min Time: " << min_time << " ms\n";
    std::cout << "  Avg Energy: " << std::scientific << std::setprecision(3) << avg_energy << "\n";
    std::cout << "  Min Energy: " << min_energy << "\n";
}

int main() {
    std::cout << "🚀 BAHA OPTIMIZATION BENCHMARK\n";
    std::cout << "================================\n\n";
    std::cout << "Testing: Number Partitioning (N=10,000)\n";
    std::cout << "Optimizations:\n";
    std::cout << "  - SIMD vectorization (AVX2)\n";
    std::cout << "  - Fast exp approximation\n";
    std::cout << "  - Cache-optimized memory layout\n";
    std::cout << "  - Branch prediction hints\n";
    std::cout << "  - Move semantics\n";
    std::cout << "  - Const correctness\n\n";
    
    const int N = 10000;
    const int TRIALS = 5;
    
    PartitionProblem prob(N, 42);
    
    // Configure BAHA
    navokoj::BranchAwareOptimizer<PartitionState>::Config config;
    config.beta_start = 1e-18;
    config.beta_end = 1e-12;
    config.beta_steps = 500;
    config.fracture_threshold = 1.25;
    config.samples_per_beta = 20;
    config.schedule_type = navokoj::BranchAwareOptimizer<PartitionState>::ScheduleType::GEOMETRIC;
    config.analytic_log_z = [&](double b) { return prob.analytic_log_z(b); };
    config.analytic_rho = [&](double b) { return prob.analytic_rho(b); };
    
    // Run benchmark
    run_benchmark<navokoj::BranchAwareOptimizer<PartitionState>>(
        "BAHA", prob, config, TRIALS);
    
    return 0;
}
