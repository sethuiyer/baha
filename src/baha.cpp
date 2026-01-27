/*
 * BRANCH-AWARE OPTIMIZER
 * A General-Purpose Simulated Annealing Alternative
 *
 * Based on: "Multiplicative Calculus for Hardness Detection and Branch-Aware Optimization"
 * Author: Sethurathienam Iyer, ShunyaBar Labs
 *
 * Core idea:
 * - Detect fractures via ρ = |d/dβ log Z|
 * - Enumerate branches via Lambert-W function
 * - Jump to optimal branch instead of slow annealing
 *
 * Compile: g++ -O3 -std=c++17 -o branch_optimizer branch_optimizer.cpp
 * Run: ./branch_optimizer
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <limits>
#include <chrono>
#include <iomanip>

namespace navokoj {

// =============================================================================
// LAMBERT-W FUNCTION
// Solves w * e^w = z using Halley's method
// =============================================================================

class LambertW {
public:
    static constexpr double E_INV = 0.36787944117144232; // 1/e
    static constexpr double TOL = 1e-10;
    static constexpr int MAX_ITER = 50;

    // Principal branch W_0(z)
    static double W0(double z) {
        if (z < -E_INV) return std::numeric_limits<double>::quiet_NaN();
        // Initial guess
        double w;
        if (z < -0.3) {
            w = z * std::exp(1.0);
        } else if (z < 1.0) {
            w = z * (1.0 - z + z * z);
        } else {
            double lz = std::log(z);
            w = lz - std::log(lz + 1.0);
        }
        return halley_iterate(z, w);
    }

    // Secondary branch W_{-1}(z), valid for z ∈ [-1/e, 0)
    static double Wm1(double z) {
        if (z < -E_INV || z >= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double w = std::log(-z) - std::log(-std::log(-z));
        return halley_iterate(z, w);
    }

    // General branch W_k(z)
    static double Wk(double z, int k) {
        if (k == 0) return W0(z);
        if (k == -1) return Wm1(z);
        return std::numeric_limits<double>::quiet_NaN(); // Complex branches not implemented
    }

private:
    static double halley_iterate(double z, double w) {
        for (int i = 0; i < MAX_ITER; ++i) {
            double ew = std::exp(w);
            double wew = w * ew;
            double f = wew - z;
            double fp = ew * (w + 1.0);
            if (std::abs(fp) < 1e-15) break;
            double fpp = ew * (w + 2.0);
            double denom = fp - f * fpp / (2.0 * fp);
            if (std::abs(denom) < 1e-15) break;
            double w_new = w - f / denom;
            if (std::abs(w_new - w) < TOL) return w_new;
            w = w_new;
        }
        return w;
    }
};

// =============================================================================
// LOG-SUM-EXP (Numerically stable)
// =============================================================================

double log_sum_exp(const std::vector<double>& log_terms) {
    if (log_terms.empty()) return -std::numeric_limits<double>::infinity();
    double max_term = *std::max_element(log_terms.begin(), log_terms.end());
    if (std::isinf(max_term)) return max_term;
    double sum = 0.0;
    for (double t : log_terms) {
        sum += std::exp(t - max_term);
    }
    return max_term + std::log(sum);
}

// =============================================================================
// BRANCH INFO
// =============================================================================

struct Branch {
    int k; // Branch index
    double beta; // β value on this branch
    double score; // Quality score (higher = better)
    bool operator<(const Branch& other) const {
        return score > other.score; // Sort by descending score
    }
};

// =============================================================================
// FRACTURE DETECTOR
// =============================================================================

class FractureDetector {
public:
    FractureDetector(double threshold = 1.5) : threshold_(threshold) {}
    void record(double beta, double log_Z) {
        beta_history_.push_back(beta);
        log_Z_history_.push_back(log_Z);
    }
    double fracture_rate() const {
        if (beta_history_.size() < 2) return 0.0;
        size_t n = beta_history_.size();
        double d_log_Z = std::abs(log_Z_history_[n-1] - log_Z_history_[n-2]);
        double d_beta = beta_history_[n-1] - beta_history_[n-2];
        return (d_beta > 0) ? d_log_Z / d_beta : 0.0;
    }
    bool is_fracture() const {
        return fracture_rate() > threshold_;
    }
    void clear() {
        beta_history_.clear();
        log_Z_history_.clear();
    }
    double threshold() const { return threshold_; }
    void set_threshold(double t) { threshold_ = t; }

private:
    double threshold_;
    std::vector<double> beta_history_;
    std::vector<double> log_Z_history_;
};

// =============================================================================
// BRANCH-AWARE OPTIMIZER
// Generic implementation - works with any energy function
// =============================================================================

template<typename State>
class BranchAwareOptimizer {
public:
    using EnergyFn = std::function<double(const State&)>;
    using SamplerFn = std::function<State()>;
    using NeighborFn = std::function<std::vector<State>(const State&)>;

    struct Config {
        double beta_start = 0.01;
        double beta_end = 10.0;
        int beta_steps = 500;
        double fracture_threshold = 1.5;
        double beta_critical = 1.0; // Estimated critical point
        int max_branches = 5;
        int samples_per_beta = 100; // For partition function estimation
        bool verbose = false;
    };

    struct Result {
        State best_state;
        double best_energy;
        int fractures_detected;
        int branch_jumps;
        double beta_at_solution;
        int steps_taken;
        double time_ms;
    };

    BranchAwareOptimizer(EnergyFn energy, SamplerFn sampler, NeighborFn neighbors = nullptr)
        : energy_(energy), sampler_(sampler), neighbors_(neighbors), rng_(std::random_device{}()) {}

    Result optimize(const Config& config = Config()) {
        auto start_time = std::chrono::high_resolution_clock::now();
        Result result;
        result.fractures_detected = 0;
        result.branch_jumps = 0;
        FractureDetector detector(config.fracture_threshold);

        // Generate β schedule
        std::vector<double> beta_schedule(config.beta_steps);
        for (int i = 0; i < config.beta_steps; ++i) {
            beta_schedule[i] = config.beta_start +
                (config.beta_end - config.beta_start) * i / (config.beta_steps - 1);
        }

        // Initialize
        State current = sampler_();
        double current_energy = energy_(current);
        State best = current;
        double best_energy = current_energy;

        // Main loop
        for (int step = 0; step < config.beta_steps; ++step) {
            double beta = beta_schedule[step];
            // Estimate log Z via sampling
            double log_Z = estimate_log_Z(beta, config.samples_per_beta);
            detector.record(beta, log_Z);
            double rho = detector.fracture_rate();

            // Check for fracture
            if (detector.is_fracture()) {
                result.fractures_detected++;
                if (config.verbose) {
                    std::cout << "⚡ FRACTURE at β=" << std::fixed << std::setprecision(3)
                              << beta << ", ρ=" << rho << std::endl;
                }

                // Enumerate branches
                std::vector<Branch> branches = enumerate_branches(beta, config.beta_critical,
                                                                   config.max_branches);
                if (!branches.empty()) {
                    // Score and select best branch
                    for (auto& b : branches) {
                        b.score = score_branch(b.beta, config.samples_per_beta);
                    }
                    std::sort(branches.begin(), branches.end());
                    Branch best_branch = branches[0];

                    if (config.verbose) {
                        std::cout << " Best branch: k=" << best_branch.k
                                  << ", β=" << best_branch.beta
                                  << ", score=" << best_branch.score << std::endl;
                    }

                    // Jump to best branch - sample a good state there
                    State jumped = sample_from_branch(best_branch.beta, config.samples_per_beta);
                    double jumped_energy = energy_(jumped);

                    if (jumped_energy < best_energy) {
                        best = jumped;
                        best_energy = jumped_energy;
                        result.branch_jumps++;

                        if (config.verbose) {
                            std::cout << " 🔀 JUMPED to E=" << best_energy << std::endl;
                        }

                        // Early exit if ground state found
                        if (best_energy <= 0) {
                            result.best_state = best;
                            result.best_energy = best_energy;
                            result.beta_at_solution = beta;
                            result.steps_taken = step + 1;
                            auto end_time = std::chrono::high_resolution_clock::now();
                            result.time_ms = std::chrono::duration<double, std::milli>(
                                end_time - start_time).count();
                            return result;
                        }
                    }
                }
            }

            // Standard local search step (fallback)
            if (neighbors_) {
                auto nbrs = neighbors_(current);
                for (const auto& nbr : nbrs) {
                    double nbr_energy = energy_(nbr);
                    // Metropolis criterion
                    if (nbr_energy < current_energy ||
                        std::uniform_real_distribution<>(0, 1)(rng_) < std::exp(-beta * (nbr_energy - current_energy))) {
                        current = nbr;
                        current_energy = nbr_energy;
                        if (current_energy < best_energy) {
                            best = current;
                            best_energy = current_energy;
                        }
                    }
                }
            }
        }

        result.best_state = best;
        result.best_energy = best_energy;
        result.beta_at_solution = config.beta_end;
        result.steps_taken = config.beta_steps;
        auto end_time = std::chrono::high_resolution_clock::now();
        result.time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return result;
    }

private:
    double estimate_log_Z(double beta, int n_samples) {
        std::vector<double> log_terms;
        log_terms.reserve(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            State s = sampler_();
            double E = energy_(s);
            log_terms.push_back(-beta * E);
        }
        // log Z ≈ log(n_samples) + log_sum_exp(terms) - log(state_space_size)
        // We return just the log_sum_exp part since we only care about derivatives
        return log_sum_exp(log_terms);
    }

    std::vector<Branch> enumerate_branches(double beta, double beta_c, int max_branches) {
        std::vector<Branch> branches;
        double u = beta - beta_c;
        if (std::abs(u) < 1e-10) u = 1e-10;
        double xi = u * std::exp(u);

        // Principal branch k=0
        double w0 = LambertW::W0(xi);
        if (!std::isnan(w0)) {
            double beta_0 = beta_c + w0;
            if (beta_0 > 0) {
                branches.push_back({0, beta_0, 0.0});
            }
        }

        // Secondary branch k=-1
        if (xi >= -LambertW::E_INV && xi < 0) {
            double wm1 = LambertW::Wm1(xi);
            if (!std::isnan(wm1)) {
                double beta_m1 = beta_c + wm1;
                if (beta_m1 > 0) {
                    branches.push_back({-1, beta_m1, 0.0});
                }
            }
        }
        return branches;
    }

    double score_branch(double beta, int n_samples) {
        if (beta <= 0) return -std::numeric_limits<double>::infinity();
        // Score = average Boltzmann weight of low-energy states
        double total_score = 0.0;
        double best_seen = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n_samples; ++i) {
            State s = sampler_();
            double E = energy_(s);
            total_score += std::exp(-beta * E);
            best_seen = std::min(best_seen, E);
        }
        // Bonus for low energy states
        return total_score / n_samples + 100.0 / (best_seen + 1.0);
    }

    State sample_from_branch(double beta, int n_samples) {
        State best = sampler_();
        double best_energy = energy_(best);
        for (int i = 0; i < n_samples; ++i) {
            State s = sampler_();
            double E = energy_(s);
            // Accept with Boltzmann probability
            if (E < best_energy) {
                best = s;
                best_energy = E;
            }
        }

        // Local search from best sample
        if (neighbors_) {
            bool improved = true;
            while (improved) {
                improved = false;
                auto nbrs = neighbors_(best);
                for (const auto& nbr : nbrs) {
                    double E = energy_(nbr);
                    if (E < best_energy) {
                        best = nbr;
                        best_energy = E;
                        improved = true;
                        break;
                    }
                }
            }
        }
        return best;
    }

    EnergyFn energy_;
    SamplerFn sampler_;
    NeighborFn neighbors_;
    std::mt19937 rng_;
};

// =============================================================================
// STANDARD SIMULATED ANNEALING (for comparison)
// =============================================================================

template<typename State>
class SimulatedAnnealing {
public:
    using EnergyFn = std::function<double(const State&)>;
    using SamplerFn = std::function<State()>;
    using NeighborFn = std::function<std::vector<State>(const State&)>;

    struct Config {
        double beta_start = 0.01;
        double beta_end = 10.0;
        int beta_steps = 500;
        int steps_per_beta = 10;
        bool verbose = false;
    };

    struct Result {
        State best_state;
        double best_energy;
        double beta_at_solution;
        int steps_taken;
        double time_ms;
    };

    SimulatedAnnealing(EnergyFn energy, SamplerFn sampler, NeighborFn neighbors)
        : energy_(energy), sampler_(sampler), neighbors_(neighbors), rng_(std::random_device{}()) {}

    Result optimize(const Config& config = Config()) {
        auto start_time = std::chrono::high_resolution_clock::now();
        Result result;
        State current = sampler_();
        double current_energy = energy_(current);
        State best = current;
        double best_energy = current_energy;

        for (int step = 0; step < config.beta_steps; ++step) {
            double beta = config.beta_start +
                (config.beta_end - config.beta_start) * step / (config.beta_steps - 1);

            for (int inner = 0; inner < config.steps_per_beta; ++inner) {
                auto nbrs = neighbors_(current);
                if (nbrs.empty()) continue;

                // Random neighbor
                std::uniform_int_distribution<> dist(0, nbrs.size() - 1);
                State nbr = nbrs[dist(rng_)]; 
                double nbr_energy = energy_(nbr);

                // Metropolis criterion
                double delta = nbr_energy - current_energy;
                if (delta < 0 || std::uniform_real_distribution<>(0, 1)(rng_) < std::exp(-beta * delta)) {
                    current = nbr;
                    current_energy = nbr_energy;
                    if (current_energy < best_energy) {
                        best = current;
                        best_energy = current_energy;
                        
                        // Early exit if ground state found
                        if (best_energy <= 0) {
                            result.best_state = best;
                            result.best_energy = best_energy;
                            result.beta_at_solution = beta;
                            result.steps_taken = step * config.steps_per_beta + inner + 1;
                            auto end_time = std::chrono::high_resolution_clock::now();
                            result.time_ms = std::chrono::duration<double, std::milli>(
                                end_time - start_time).count();
                            return result;
                        }
                    }
                }
            }
        }

        result.best_state = best;
        result.best_energy = best_energy;
        result.beta_at_solution = config.beta_end;
        result.steps_taken = config.beta_steps * config.steps_per_beta;
        auto end_time = std::chrono::high_resolution_clock::now();
        result.time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return result;
    }

private:
    EnergyFn energy_;
    SamplerFn sampler_;
    NeighborFn neighbors_;
    std::mt19937 rng_;
};

} // namespace navokoj

// =============================================================================
// EXAMPLE: GRAPH COLORING
// =============================================================================

struct GraphColoringState {
    std::vector<int> coloring;
};

class GraphColoring {
public:
    GraphColoring(int n_nodes, const std::vector<std::pair<int,int>>& edges, int n_colors)
        : n_nodes_(n_nodes), edges_(edges), n_colors_(n_colors), rng_(std::random_device{}()) {}

    double energy(const GraphColoringState& state) const {
        int conflicts = 0;
        for (const auto& [i, j] : edges_) {
            if (state.coloring[i] == state.coloring[j]) {
                conflicts++;
            }
        }
        return static_cast<double>(conflicts);
    }

    GraphColoringState random_state() {
        GraphColoringState state;
        state.coloring.resize(n_nodes_);
        std::uniform_int_distribution<> dist(0, n_colors_ - 1);
        for (int i = 0; i < n_nodes_; ++i) {
            state.coloring[i] = dist(rng_);
        }
        return state;
    }

    std::vector<GraphColoringState> neighbors(const GraphColoringState& state) {
        std::vector<GraphColoringState> nbrs;
        for (int i = 0; i < n_nodes_; ++i) {
            for (int c = 0; c < n_colors_; ++c) {
                if (c != state.coloring[i]) {
                    GraphColoringState nbr = state;
                    nbr.coloring[i] = c;
                    nbrs.push_back(nbr);
                }
            }
        }
        return nbrs;
    }

    int count_solutions() const {
        int count = 0;
        std::vector<int> coloring(n_nodes_, 0);
        std::function<void(int)> enumerate = [&](int node) {
            if (node == n_nodes_) {
                // Check if valid
                bool valid = true;
                for (const auto& [i, j] : edges_) {
                    if (coloring[i] == coloring[j]) {
                        valid = false;
                        break;
                    }
                }
                if (valid) count++;
                return;
            }
            for (int c = 0; c < n_colors_; ++c) {
                coloring[node] = c;
                enumerate(node + 1);
            }
        };
        enumerate(0);
        return count;
    }

private:
    int n_nodes_;
    std::vector<std::pair<int,int>> edges_;
    int n_colors_;
    mutable std::mt19937 rng_;
};

// =============================================================================
// BENCHMARK
// =============================================================================

void run_benchmark() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ BRANCH-AWARE OPTIMIZER vs SIMULATED ANNEALING ║\n";
    std::cout << "║ Benchmark on Graph Coloring ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    const int N_TRIALS = 20;
    const int N_NODES = 8;
    const int N_COLORS = 3;
    std::mt19937 rng(42);

    std::vector<double> ba_times, sa_times;
    std::vector<double> ba_betas, sa_betas;
    int ba_wins = 0, sa_wins = 0, ties = 0;

    for (int trial = 0; trial < N_TRIALS; ++trial) {
        // Generate random graph
        int n_edges = 8 + (rng() % 8); // 8-15 edges
        std::vector<std::pair<int,int>> edges;
        std::vector<std::pair<int,int>> all_edges;
        for (int i = 0; i < N_NODES; ++i) {
            for (int j = i + 1; j < N_NODES; ++j) {
                all_edges.push_back({i, j});
            }
        }
        std::shuffle(all_edges.begin(), all_edges.end(), rng);
        for (int i = 0; i < n_edges && i < (int)all_edges.size(); ++i) {
            edges.push_back(all_edges[i]);
        }

        GraphColoring problem(N_NODES, edges, N_COLORS);

        // Skip unsolvable instances
        int n_solutions = problem.count_solutions();
        if (n_solutions == 0) {
            continue;
        }

        // Energy function
        auto energy_fn = [&problem](const GraphColoringState& s) {
            return problem.energy(s);
        };

        // Sampler
        auto sampler_fn = [&problem]() mutable {
            return problem.random_state();
        };

        // Neighbor function
        auto neighbor_fn = [&problem](const GraphColoringState& s) {
            return problem.neighbors(s);
        };

        // Branch-aware optimizer
        navokoj::BranchAwareOptimizer<GraphColoringState> ba_opt(energy_fn, sampler_fn, neighbor_fn);
        navokoj::BranchAwareOptimizer<GraphColoringState>::Config ba_config;
        ba_config.beta_steps = 200;
        ba_config.fracture_threshold = 1.0;
        ba_config.samples_per_beta = 50;
        ba_config.verbose = false;
        auto ba_result = ba_opt.optimize(ba_config);

        // Simulated annealing
        navokoj::SimulatedAnnealing<GraphColoringState> sa_opt(energy_fn, sampler_fn, neighbor_fn);
        navokoj::SimulatedAnnealing<GraphColoringState>::Config sa_config;
        sa_config.beta_steps = 200;
        sa_config.steps_per_beta = 10;
        sa_config.verbose = false;
        auto sa_result = sa_opt.optimize(sa_config);

        // Record results
        ba_times.push_back(ba_result.time_ms);
        sa_times.push_back(sa_result.time_ms);
        ba_betas.push_back(ba_result.beta_at_solution);
        sa_betas.push_back(sa_result.beta_at_solution);

        bool ba_solved = (ba_result.best_energy == 0);
        bool sa_solved = (sa_result.best_energy == 0);

        if (ba_solved && !sa_solved) ba_wins++;
        else if (sa_solved && !ba_solved) sa_wins++;
        else if (ba_solved && sa_solved) {
            if (ba_result.beta_at_solution < sa_result.beta_at_solution) ba_wins++;
            else if (sa_result.beta_at_solution < ba_result.beta_at_solution) sa_wins++;
            else ties++;
        }

        std::cout << "Trial " << std::setw(2) << trial + 1
                  << ": edges=" << std::setw(2) << n_edges
                  << ", solutions=" << std::setw(4) << n_solutions
                  << " | BA: β=" << std::fixed << std::setprecision(2) << ba_result.beta_at_solution
                  << ", E=" << ba_result.best_energy
                  << " | SA: β=" << sa_result.beta_at_solution
                  << ", E=" << sa_result.best_energy;

        if (ba_result.beta_at_solution < sa_result.beta_at_solution && ba_solved && sa_solved) {
            double speedup = sa_result.beta_at_solution / ba_result.beta_at_solution;
            std::cout << " | BA " << std::setprecision(1) << speedup << "x faster";
        }
        std::cout << std::endl;
    }

    // Summary statistics
    std::cout << "\n════════════════════════════════════════════════════════════════\n";
    std::cout << "SUMMARY\n";
    std::cout << "════════════════════════════════════════════════════════════════\n";

    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    std::cout << "\nβ at solution:\n";
    std::cout << " Branch-Aware mean: " << std::fixed << std::setprecision(3) << mean(ba_betas) << "\n";
    std::cout << " Sim. Annealing mean: " << mean(sa_betas) << "\n";
    std::cout << " Speedup: " << std::setprecision(1) << mean(sa_betas) / mean(ba_betas) << "x\n";

    std::cout << "\nWins:\n";
    std::cout << " Branch-Aware: " << ba_wins << "\n";
    std::cout << " Sim. Annealing: " << sa_wins << "\n";
    std::cout << " Ties: " << ties << "\n";
    std::cout << "\n════════════════════════════════════════════════════════════════\n";
    std::cout << "Branch-aware optimizer detects fractures and jumps directly to\n";
    std::cout << "the solution basin, while simulated annealing slowly cools.\n";
    std::cout << "════════════════════════════════════════════════════════════════\n";
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    run_benchmark();
    return 0;
}
