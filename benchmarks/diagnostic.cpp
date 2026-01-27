#include "baha.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <string>

// =============================================================================
// REUSED PROBLEM CLASSES
// =============================================================================

// 1. HARD NUMBER PARTITIONING (N=50 for diagnostic)
struct HardPartitionState { std::vector<int> signs; };
class HardNumberPartitioning {
public:
    HardNumberPartitioning(int n, int seed) : n_(n), rng_(seed) {
        std::uniform_int_distribution<long long> dist(1, 1000000000000LL); // 10^12
        numbers_.resize(n);
        for (int i = 0; i < n; ++i) numbers_[i] = dist(rng_);
    }
    double energy(const HardPartitionState& s) const {
        long long sum = 0;
        for (size_t i = 0; i < numbers_.size(); ++i) sum += s.signs[i] * numbers_[i];
        return static_cast<double>(std::abs(sum));
    }
    HardPartitionState random_state() {
        HardPartitionState s; s.signs.resize(n_);
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n_; ++i) s.signs[i] = dist(rng_) ? 1 : -1;
        return s;
    }
    std::vector<HardPartitionState> neighbors(const HardPartitionState& s) {
        std::vector<HardPartitionState> nbrs;
        for (int i = 0; i < n_; ++i) {
            HardPartitionState nbr = s; nbr.signs[i] *= -1; nbrs.push_back(nbr);
        }
        return nbrs;
    }
private:
    int n_; std::vector<long long> numbers_; mutable std::mt19937_64 rng_;
};

// 2. TSP (N=20)
struct TSPState { std::vector<int> tour; };
class TSP {
public:
    TSP(int n, int seed) : n_cities_(n), rng_(seed) {
        std::uniform_real_distribution<> dist(0, 100);
        coords_.resize(n);
        for (int i = 0; i < n; ++i) coords_[i] = {dist(rng_), dist(rng_)};
    }
    double energy(const TSPState& s) const {
        double dist = 0;
        for (int i = 0; i < n_cities_; ++i) {
            int j = (i + 1) % n_cities_;
            int u = s.tour[i]; int v = s.tour[j];
            dist += std::hypot(coords_[u].first - coords_[v].first, coords_[u].second - coords_[v].second);
        }
        return dist;
    }
    TSPState random_state() {
        TSPState s; s.tour.resize(n_cities_);
        std::iota(s.tour.begin(), s.tour.end(), 0);
        std::shuffle(s.tour.begin(), s.tour.end(), rng_);
        return s;
    }
    std::vector<TSPState> neighbors(const TSPState& s) {
        std::vector<TSPState> nbrs;
        for (int i = 0; i < n_cities_ - 1; ++i) {
            for (int j = i + 1; j < n_cities_; ++j) {
                TSPState nbr = s; std::reverse(nbr.tour.begin() + i, nbr.tour.begin() + j + 1); nbrs.push_back(nbr);
            }
        }
        return nbrs;
    }
private:
    int n_cities_; std::vector<std::pair<double, double>> coords_; mutable std::mt19937 rng_;
};

// 3. HARD RANDOM 3-SAT (N=50, M=213)
struct SatState { std::vector<bool> assignment; };
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
        for (int i = 0; i < n_; ++i) {
            SatState nbr = s; nbr.assignment[i] = !nbr.assignment[i]; nbrs.push_back(nbr);
        }
        return nbrs;
    }
private:
    struct Clause { int lits[3]; bool neg[3]; };
    int n_; std::vector<Clause> clauses_; mutable std::mt19937 rng_;
};

// =============================================================================
// RUNNER WITH CSV LOGGING
// =============================================================================

template<typename Problem, typename State>
void run_diagnostic(const std::string& name, const std::string& filename, std::function<Problem(int)> factory) {
    std::cout << "Running Diagnostic: " << name << " -> " << filename << " ... ";
    std::cout.flush();

    std::ofstream csv(filename);
    csv << "step,beta,energy,rho,event\n";

    auto prob = factory(42); // Fixed seed
    std::function<double(const State&)> energy = [&](const State& s) { return prob.energy(s); };
    std::function<State()> sampler = [&]() { return prob.random_state(); };
    std::function<std::vector<State>(const State&)> neighbors = [&](const State& s) { return prob.neighbors(s); };

    typename navokoj::BranchAwareOptimizer<State>::Config config;
    config.beta_steps = 400; // Granular trace
    config.beta_end = 10.0;
    config.fracture_threshold = 1.0; 
    
    // THE LOGGER
    config.logger = [&](int step, double beta, double e, double rho, const char* event) {
        csv << step << "," << beta << "," << e << "," << rho << "," << event << "\n";
    };

    navokoj::BranchAwareOptimizer<State> ba(energy, sampler, neighbors);
    auto res = ba.optimize(config);

    std::cout << "Done. Found E=" << res.best_energy << ", Jumps=" << res.branch_jumps << "\n";
}

int main() {
    std::cout << "BAHA DIAGNOSTIC SUITE\n";
    std::cout << "=====================\n";

    // Trace 1: Partitioning (Expect Clean Fracture)
    run_diagnostic<HardNumberPartitioning, HardPartitionState>(
        "Partitioning (N=50)", "trace_partition.csv", 
        [](int seed) { return HardNumberPartitioning(50, seed); }
    );

    // Trace 2: TSP (Expect Smooth/No Jumps)
    run_diagnostic<TSP, TSPState>(
        "TSP (N=20)", "trace_tsp.csv", 
        [](int seed) { return TSP(20, seed); }
    );

    // Trace 3: SAT (Expect Noise/Chaos)
    run_diagnostic<HardRandom3Sat, SatState>(
        "3-SAT (N=50, M=213)", "trace_sat.csv", 
        [](int seed) { return HardRandom3Sat(50, 213, seed); }
    );

    return 0;
}
