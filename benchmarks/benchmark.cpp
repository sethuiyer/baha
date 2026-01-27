#include "baha.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <string>

// =============================================================================
// 1. NUMBER PARTITIONING (The "Killer" Example)
// =============================================================================
struct PartitionState {
    std::vector<int> signs; // +1 or -1
};

class NumberPartitioning {
public:
    NumberPartitioning(int n, int seed) : n_(n), rng_(seed) {
        std::uniform_int_distribution<> dist(1, 1000);
        numbers_.resize(n);
        for (int i = 0; i < n; ++i) numbers_[i] = dist(rng_);
    }

    double energy(const PartitionState& s) const {
        long sum = 0;
        for (size_t i = 0; i < numbers_.size(); ++i) {
            sum += s.signs[i] * numbers_[i];
        }
        return std::abs(sum);
    }

    PartitionState random_state() {
        PartitionState s;
        s.signs.resize(n_);
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n_; ++i) s.signs[i] = dist(rng_) ? 1 : -1;
        return s;
    }

    std::vector<PartitionState> neighbors(const PartitionState& s) {
        std::vector<PartitionState> nbrs;
        // Flip one sign
        for (int i = 0; i < n_; ++i) {
            PartitionState nbr = s;
            nbr.signs[i] *= -1;
            nbrs.push_back(nbr);
        }
        return nbrs;
    }

private:
    int n_;
    std::vector<long> numbers_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// 2. MAX-SAT
// =============================================================================
struct MaxSatState {
    std::vector<bool> assignment;
};

class MaxSat {
public:
    MaxSat(int vars, int clauses, int seed) : n_vars_(vars), rng_(seed) {
        std::uniform_int_distribution<> var_dist(0, vars - 1);
        std::uniform_int_distribution<> bool_dist(0, 1);
        
        for (int i = 0; i < clauses; ++i) {
            Clause c;
            for (int j = 0; j < 3; ++j) c.lits[j] = var_dist(rng_);
            for (int j = 0; j < 3; ++j) c.neg[j] = bool_dist(rng_);
            clauses_.push_back(c);
        }
    }

    double energy(const MaxSatState& s) const {
        int unsatisfied = 0;
        for (const auto& c : clauses_) {
            bool sat = false;
            for (int k = 0; k < 3; ++k) {
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

    MaxSatState random_state() {
        MaxSatState s;
        s.assignment.resize(n_vars_);
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n_vars_; ++i) s.assignment[i] = dist(rng_);
        return s;
    }

    std::vector<MaxSatState> neighbors(const MaxSatState& s) {
        std::vector<MaxSatState> nbrs;
        for (int i = 0; i < n_vars_; ++i) {
            MaxSatState nbr = s;
            nbr.assignment[i] = !nbr.assignment[i];
            nbrs.push_back(nbr);
        }
        return nbrs;
    }

private:
    struct Clause {
        int lits[3];
        bool neg[3];
    };
    int n_vars_;
    std::vector<Clause> clauses_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// 3. GRAPH COLORING (Re-implemented for compatibility)
// =============================================================================
struct GraphColoringState {
    std::vector<int> coloring;
};

class GraphColoring {
public:
    GraphColoring(int n, int edges_count, int colors, int seed) : n_nodes_(n), n_colors_(colors), rng_(seed) {
        std::vector<std::pair<int,int>> all_possible;
        for(int i=0; i<n; ++i)
            for(int j=i+1; j<n; ++j) all_possible.push_back({i,j});
        
        std::shuffle(all_possible.begin(), all_possible.end(), rng_);
        int limit = std::min((int)all_possible.size(), edges_count);
        for(int i=0; i<limit; ++i) edges_.push_back(all_possible[i]);
    }

    double energy(const GraphColoringState& state) const {
        int conflicts = 0;
        for (const auto& [i, j] : edges_) {
            if (state.coloring[i] == state.coloring[j]) conflicts++;
        }
        return static_cast<double>(conflicts);
    }

    GraphColoringState random_state() {
        GraphColoringState s;
        s.coloring.resize(n_nodes_);
        std::uniform_int_distribution<> dist(0, n_colors_ - 1);
        for (int i = 0; i < n_nodes_; ++i) s.coloring[i] = dist(rng_);
        return s;
    }

    std::vector<GraphColoringState> neighbors(const GraphColoringState& s) {
        std::vector<GraphColoringState> nbrs;
        for (int i = 0; i < n_nodes_; ++i) {
            for (int c = 0; c < n_colors_; ++c) {
                if (c != s.coloring[i]) {
                    GraphColoringState nbr = s;
                    nbr.coloring[i] = c;
                    nbrs.push_back(nbr);
                }
            }
        }
        return nbrs;
    }

private:
    int n_nodes_;
    int n_colors_;
    std::vector<std::pair<int,int>> edges_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// 4. TSP
// =============================================================================
struct TSPState {
    std::vector<int> tour;
};

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
            int u = s.tour[i];
            int v = s.tour[j];
            dist += std::hypot(coords_[u].first - coords_[v].first, coords_[u].second - coords_[v].second);
        }
        return dist;
    }

    TSPState random_state() {
        TSPState s;
        s.tour.resize(n_cities_);
        std::iota(s.tour.begin(), s.tour.end(), 0);
        std::shuffle(s.tour.begin(), s.tour.end(), rng_);
        return s;
    }

    std::vector<TSPState> neighbors(const TSPState& s) {
        // 2-opt neighborhood (simplified to avoid explosion)
        std::vector<TSPState> nbrs;
        for (int i = 0; i < n_cities_ - 1; ++i) {
            for (int j = i + 1; j < n_cities_; ++j) {
                TSPState nbr = s;
                std::reverse(nbr.tour.begin() + i, nbr.tour.begin() + j + 1);
                nbrs.push_back(nbr);
            }
        }
        return nbrs;
    }

private:
    int n_cities_;
    std::vector<std::pair<double, double>> coords_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// 5. SPIN GLASS (2D Ising)
// =============================================================================
struct SpinGlassState {
    std::vector<int> spins; // +1, -1
};

class SpinGlass {
public:
    SpinGlass(int w, int h, int seed) : w_(w), h_(h), rng_(seed) {
        // Random J couplings +/- 1
        int n_edges = w * h * 2; // Rough approx (periodic)
        std::uniform_int_distribution<> dist(0, 1);
        J_ver_.resize(w*h);
        J_hor_.resize(w*h);
        for(size_t i=0; i<J_ver_.size(); ++i) J_ver_[i] = dist(rng_) ? 1 : -1;
        for(size_t i=0; i<J_hor_.size(); ++i) J_hor_[i] = dist(rng_) ? 1 : -1;
    }

    double energy(const SpinGlassState& s) const {
        double E = 0;
        for (int y = 0; y < h_; ++y) {
            for (int x = 0; x < w_; ++x) {
                int idx = y * w_ + x;
                int right = y * w_ + ((x + 1) % w_);
                int down = ((y + 1) % h_) * w_ + x;
                
                E += -J_hor_[idx] * s.spins[idx] * s.spins[right];
                E += -J_ver_[idx] * s.spins[idx] * s.spins[down];
            }
        }
        return E;
    }

    SpinGlassState random_state() {
        SpinGlassState s;
        s.spins.resize(w_ * h_);
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < w_ * h_; ++i) s.spins[i] = dist(rng_) ? 1 : -1;
        return s;
    }

    std::vector<SpinGlassState> neighbors(const SpinGlassState& s) {
        std::vector<SpinGlassState> nbrs;
        for (int i = 0; i < w_ * h_; ++i) {
            SpinGlassState nbr = s;
            nbr.spins[i] *= -1;
            nbrs.push_back(nbr);
        }
        return nbrs;
    }

private:
    int w_, h_;
    std::vector<int> J_ver_, J_hor_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// MAIN BENCHMARK RUNNER
// =============================================================================

template<typename Problem, typename State>
void run_test(const std::string& name, int n_trials, std::function<Problem(int)> factory) {
    std::cout << "\nTesting " << name << " (" << n_trials << " trials)...\n";
    
    int ba_better = 0, sa_better = 0;
    double ba_avg_e = 0, sa_avg_e = 0;

    for (int t = 0; t < n_trials; ++t) {
        auto prob = factory(t);
        
        auto energy = [&](const State& s) { return prob.energy(s); };
        auto sampler = [&]() mutable { return prob.random_state(); };
        auto neighbors = [&](const State& s) { return prob.neighbors(s); };

        // BA
        navokoj::BranchAwareOptimizer<State> ba(energy, sampler, neighbors);
        auto ba_res = ba.optimize({0.01, 10.0, 200});

        // SA
        navokoj::SimulatedAnnealing<State> sa(energy, sampler, neighbors);
        auto sa_res = sa.optimize({0.01, 10.0, 200});

        ba_avg_e += ba_res.best_energy;
        sa_avg_e += sa_res.best_energy;

        if (ba_res.best_energy < sa_res.best_energy) ba_better++;
        else if (sa_res.best_energy < ba_res.best_energy) sa_better++;
    }

    std::cout << "Results for " << name << ":\n";
    std::cout << "  BA Win Rate: " << ba_better << "/" << n_trials << "\n";
    std::cout << "  SA Win Rate: " << sa_better << "/" << n_trials << "\n";
    std::cout << "  BA Avg Energy: " << ba_avg_e / n_trials << "\n";
    std::cout << "  SA Avg Energy: " << sa_avg_e / n_trials << "\n";
    
    if ((ba_avg_e < sa_avg_e) && (ba_better >= sa_better)) {
         std::cout << "-> 🏆 BA Wins (Structure detected)\n";
    } else if ((std::abs(ba_avg_e - sa_avg_e) < 0.1 * std::abs(sa_avg_e)) || (ba_better == sa_better)) {
         std::cout << "-> 🤝 Tie (Graceful degradation)\n";
    } else {
         std::cout << "-> 🏆 SA Wins (Smooth landscape)\n";
    }
    std::cout << "--------------------------------------------------\n";
}

int main() {
    std::cout << "Running Comprehensive BAHA Benchmark Suite\n";
    std::cout << "==========================================\n";

    // 1. Number Partitioning (Target: BA Wins)
    run_test<NumberPartitioning, PartitionState>("Number Partitioning (N=20)", 10, [](int seed) {
        return NumberPartitioning(20, seed);
    });

    // 2. MAX-SAT (Target: BA Wins or Tie near transition)
    run_test<MaxSat, MaxSatState>("MAX-3-SAT (N=20, M=85, alpha=4.25)", 10, [](int seed) {
        return MaxSat(20, 85, seed);
    });

    // 3. Graph Coloring (Target: BA Wins or Tie)
    run_test<GraphColoring, GraphColoringState>("Graph Coloring (N=15, 3-Color)", 10, [](int seed) {
        return GraphColoring(15, 40, 3, seed);
    });

    // 4. TSP (Target: Tie / SA, Smooth landscape)
    run_test<TSP, TSPState>("TSP (N=15)", 10, [](int seed) {
        return TSP(15, seed);
    });

    // 5. Spin Glass (Target: Tie / SA, Smooth landscape)
    run_test<SpinGlass, SpinGlassState>("Spin Glass (6x6)", 10, [](int seed) {
        return SpinGlass(6, 6, seed);
    });

    return 0;
}
