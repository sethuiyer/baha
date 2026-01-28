/*
 * Author: Sethurathienam Iyer
 */
#include "baha.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

// ============================================================================
// LOW-AUTOCORRELATION BINARY SEQUENCES (LABS)
// ============================================================================
// Problem: Find a binary sequence S of length N (x_i in {-1, 1})
// that minimizes the energy E(S) = Sum_{k=1}^{N-1} (C_k)^2
// where C_k is the aperiodic autocorrelation at lag k:
// C_k = Sum_{i=1}^{N-k} x_i * x_{i+k}

// This problem is known for having a "glassy" landscape with many local minima.
// Finding the ground state is computationally very hard.

class LABSProblem {
    int N;
public:
    LABSProblem(int n) : N(n) {}

    // Initial State: Random sequence of {-1, 1}
    std::vector<int> initial_state() {
        std::vector<int> s(N);
        std::mt19937 rng(42); // Seed for reproducibility of initial state? Or random? 
        // Let's use device random for diversity if called multiple times, 
        // but for benchmark stability let's stick to fixed if we want identical starts.
        // Actually, for BAHA vs SA, we might want random starts.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, 1);
        for(int i=0; i<N; ++i) {
            s[i] = dist(gen) ? 1 : -1;
        }
        return s;
    }

    // Energy Calculation O(N^2)
    double energy(const std::vector<int>& s) {
        double e = 0;
        for (int k = 1; k < N; ++k) {
            int ck = 0;
            for (int i = 0; i < N - k; ++i) {
                ck += s[i] * s[i+k];
            }
            e += ck * ck;
        }
        return e;
    }

    // Neighbors: Single Spin Flip
    std::vector<std::vector<int>> neighbors(const std::vector<int>& s) {
        std::vector<std::vector<int>> nbrs;
        // Flip each bit
        for(int i=0; i<N; ++i) {
            std::vector<int> next = s;
            next[i] = -next[i];
            nbrs.push_back(next);
        }
        return nbrs;
    }
};

int main() {
    std::cout << "🧪 THE PHYSICS GLASS TEST (LABS) 🧪" << std::endl;
    std::cout << "Minimizing Autocorrelation Sidelobes." << std::endl;
    std::cout << "Hypothesis: BAHA tunnels through glassy barriers." << std::endl;
    
    int N = 60; // Hard instance size
    std::cout << "Sequence Length N=" << N << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    LABSProblem labs(N);

    // BAHA Setup
    auto energy = [&](const std::vector<int>& s) { return labs.energy(s); };
    auto sampler = [&]() { return labs.initial_state(); };
    auto neighbors = [&](const std::vector<int>& s) { return labs.neighbors(s); };

    navokoj::BranchAwareOptimizer<std::vector<int>> baha(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<std::vector<int>>::Config config;
    
    // LABS landscape is rugged.
    // Optimal Energy for N=60 is typically around E ~ N^2 / something? 
    // Known best for N=60 is roughly E=??? (Merit Factor F = N^2 / (2*E)). 
    // Best known merit factors are ~6-9. So E ~ 3600 / 12 ~ 300.
    // Random energy is much higher.
    
    config.beta_steps = 1000;
    config.beta_end = 5.0; // Lower beta might be enough?
    config.samples_per_beta = 100;
    config.fracture_threshold = 1.3; // Sensitive
    config.max_branches = 5;
    config.verbose = true;
    config.schedule_type = navokoj::BranchAwareOptimizer<std::vector<int>>::ScheduleType::GEOMETRIC;

    std::cout << "Running BAHA..." << std::endl;
    auto result_ba = baha.optimize(config);

    std::cout << "\nBAHA RESULT:" << std::endl;
    std::cout << "Energy: " << result_ba.best_energy << std::endl;
    double merit_ba = (double)(N*N) / (2.0 * result_ba.best_energy);
    std::cout << "Merit Factor: " << merit_ba << std::endl;
    std::cout << "Fractures: " << result_ba.fractures_detected << std::endl;
    std::cout << "Jumps: " << result_ba.branch_jumps << std::endl;

    // SA Baseline
    std::cout << "\nRunning SA Baseline..." << std::endl;
    navokoj::SimulatedAnnealing<std::vector<int>> sa(energy, sampler, neighbors);
    navokoj::SimulatedAnnealing<std::vector<int>>::Config sa_config;
    sa_config.beta_steps = 1000;
    sa_config.steps_per_beta = 10; // Total 10,000 evals (roughly comparable to BAHA ops)
    // Actually BAHA does samples_per_beta * beta_steps samples... that's 100 * 1000 = 100,000 samples.
    // To be fair, give SA 100,000 steps.
    sa_config.steps_per_beta = 100; 

    auto result_sa = sa.optimize(sa_config);

    std::cout << "\nSA RESULT:" << std::endl;
    std::cout << "Energy: " << result_sa.best_energy << std::endl;
    double merit_sa = (double)(N*N) / (2.0 * result_sa.best_energy);
    std::cout << "Merit Factor: " << merit_sa << std::endl;

    // Conclusion
    std::cout << "\n------------------------------------------------------------" << std::endl;
    if (result_ba.best_energy < result_sa.best_energy) {
        std::cout << "🏆 BAHA WINS (Advantage: " << (result_sa.best_energy - result_ba.best_energy) << ")" << std::endl;
    } else if (result_ba.best_energy > result_sa.best_energy) {
        std::cout << "❌ SA WINS (Advantage: " << (result_ba.best_energy - result_sa.best_energy) << ")" << std::endl;
    } else {
        std::cout << "🤝 TIE" << std::endl;
    }

    return 0;
}
