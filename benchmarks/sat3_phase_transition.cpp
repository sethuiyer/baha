#include "baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

// ============================================================================
// 3-SAT AT PHASE TRANSITION (α ≈ 4.267)
// Comparing Standard SA vs BAHA with Fracture Detection
// ============================================================================

struct Literal {
    int var;      // Variable index (0-based)
    bool negated; // true if ¬x
};

struct Clause {
    Literal lits[3];
};

class SAT3Instance {
public:
    int n_vars;
    std::vector<Clause> clauses;
    
    // State = assignment vector (0 or 1 for each variable)
    using State = std::vector<int>;
    
    // Generate random 3-SAT at clause/variable ratio alpha
    static SAT3Instance generate(int n, double alpha, std::mt19937& rng) {
        SAT3Instance inst;
        inst.n_vars = n;
        int m = static_cast<int>(alpha * n);
        
        std::uniform_int_distribution<int> var_dist(0, n - 1);
        std::uniform_int_distribution<int> sign_dist(0, 1);
        
        for (int i = 0; i < m; ++i) {
            Clause c;
            // Ensure 3 distinct variables
            std::vector<int> vars;
            while (vars.size() < 3) {
                int v = var_dist(rng);
                if (std::find(vars.begin(), vars.end(), v) == vars.end()) {
                    vars.push_back(v);
                }
            }
            for (int j = 0; j < 3; ++j) {
                c.lits[j].var = vars[j];
                c.lits[j].negated = (sign_dist(rng) == 1);
            }
            inst.clauses.push_back(c);
        }
        return inst;
    }
    
    // Count unsatisfied clauses
    int unsat_count(const State& s) const {
        int count = 0;
        for (const auto& c : clauses) {
            bool sat = false;
            for (int j = 0; j < 3; ++j) {
                int val = s[c.lits[j].var];
                if (c.lits[j].negated) val = 1 - val;
                if (val == 1) { sat = true; break; }
            }
            if (!sat) count++;
        }
        return count;
    }
    
    State random_state(std::mt19937& rng) const {
        State s(n_vars);
        std::uniform_int_distribution<int> bit(0, 1);
        for (int i = 0; i < n_vars; ++i) s[i] = bit(rng);
        return s;
    }
    
    std::vector<State> neighbors(const State& s, std::mt19937& rng) const {
        std::vector<State> nbrs;
        // Flip a random subset of variables (not all - too slow)
        std::uniform_int_distribution<int> var_dist(0, n_vars - 1);
        for (int i = 0; i < 32; ++i) {
            State n = s;
            n[var_dist(rng)] ^= 1;
            nbrs.push_back(n);
        }
        return nbrs;
    }
};

// Standard Simulated Annealing (for comparison)
struct SAResult {
    int final_unsat;
    double time_ms;
    int steps;
    std::vector<std::pair<double, int>> trajectory; // (beta, energy) pairs
};

SAResult run_standard_SA(const SAT3Instance& inst, int max_steps, int seed) {
    std::mt19937 rng(seed);
    auto state = inst.random_state(rng);
    int energy = inst.unsat_count(state);
    int best_energy = energy;
    
    SAResult result;
    result.trajectory.reserve(max_steps / 100);
    
    double beta_start = 0.1, beta_end = 20.0;
    double ratio = std::pow(beta_end / beta_start, 1.0 / max_steps);
    double beta = beta_start;
    
    std::uniform_real_distribution<> unif(0, 1);
    std::uniform_int_distribution<int> var_dist(0, inst.n_vars - 1);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < max_steps; ++step) {
        // Single variable flip
        int flip_var = var_dist(rng);
        state[flip_var] ^= 1;
        int new_energy = inst.unsat_count(state);
        
        int delta = new_energy - energy;
        if (delta <= 0 || unif(rng) < std::exp(-beta * delta)) {
            energy = new_energy;
            if (energy < best_energy) best_energy = energy;
        } else {
            state[flip_var] ^= 1; // Reject
        }
        
        beta *= ratio;
        
        if (step % 100 == 0) {
            result.trajectory.push_back({beta, energy});
        }
        
        if (best_energy == 0) break;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.final_unsat = best_energy;
    result.steps = max_steps;
    return result;
}

// BAHA Result with trajectory
struct BAHAResult {
    int final_unsat;
    double time_ms;
    int fractures;
    int branch_jumps;
    std::vector<std::pair<double, int>> trajectory;
    std::vector<double> rho_values; // Fracture rate at each step
};

BAHAResult run_BAHA(const SAT3Instance& inst, int max_steps, int seed) {
    std::mt19937 rng(seed);
    
    auto energy_fn = [&](const SAT3Instance::State& s) -> double {
        return static_cast<double>(inst.unsat_count(s));
    };
    auto sampler_fn = [&]() { return inst.random_state(rng); };
    auto neighbor_fn = [&](const SAT3Instance::State& s) { return inst.neighbors(s, rng); };
    
    navokoj::BranchAwareOptimizer<SAT3Instance::State> baha(energy_fn, sampler_fn, neighbor_fn);
    
    typename navokoj::BranchAwareOptimizer<SAT3Instance::State>::Config config;
    config.beta_start = 0.1;
    config.beta_end = 20.0;
    config.beta_steps = max_steps;
    config.samples_per_beta = 10;
    config.fracture_threshold = 1.5;
    config.max_branches = 3;
    config.verbose = false;
    
    BAHAResult result;
    result.trajectory.reserve(max_steps);
    result.rho_values.reserve(max_steps);
    
    // Custom logger to capture trajectory
    config.logger = [&](int step, double beta, double energy, double rho, const std::string&) {
        if (step % 10 == 0) {
            result.trajectory.push_back({beta, static_cast<int>(energy)});
            result.rho_values.push_back(rho);
        }
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    auto opt_result = baha.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.final_unsat = static_cast<int>(opt_result.best_energy);
    result.fractures = opt_result.fractures_detected;
    result.branch_jumps = opt_result.branch_jumps;
    
    return result;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  3-SAT PHASE TRANSITION BENCHMARK (α ≈ 4.267)                ║\n";
    std::cout << "║  Standard SA vs BAHA with Fracture Detection                 ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    const double ALPHA = 4.267;
    const int MAX_STEPS = 50000;
    const int N_INSTANCES = 20; // Per variable count
    const std::vector<int> VAR_COUNTS = {100, 200};
    
    std::mt19937 rng(42);
    std::ofstream csv("data/sat3_benchmark.csv");
    csv << "n_vars,instance,method,final_unsat,time_ms,fractures,branch_jumps\n";
    
    for (int n : VAR_COUNTS) {
        std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "Testing n=" << n << " variables, α=" << ALPHA 
                  << " (" << static_cast<int>(ALPHA * n) << " clauses)\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        
        int sa_wins = 0, baha_wins = 0, ties = 0;
        double sa_total_time = 0, baha_total_time = 0;
        int sa_total_unsat = 0, baha_total_unsat = 0;
        int total_fractures = 0, total_jumps = 0;
        
        for (int i = 0; i < N_INSTANCES; ++i) {
            auto inst = SAT3Instance::generate(n, ALPHA, rng);
            int seed = i * 1000;
            
            auto sa_result = run_standard_SA(inst, MAX_STEPS, seed);
            auto baha_result = run_BAHA(inst, MAX_STEPS, seed);
            
            csv << n << "," << i << ",SA," << sa_result.final_unsat << "," 
                << sa_result.time_ms << ",0,0\n";
            csv << n << "," << i << ",BAHA," << baha_result.final_unsat << ","
                << baha_result.time_ms << "," << baha_result.fractures << ","
                << baha_result.branch_jumps << "\n";
            
            if (baha_result.final_unsat < sa_result.final_unsat) baha_wins++;
            else if (sa_result.final_unsat < baha_result.final_unsat) sa_wins++;
            else ties++;
            
            sa_total_time += sa_result.time_ms;
            baha_total_time += baha_result.time_ms;
            sa_total_unsat += sa_result.final_unsat;
            baha_total_unsat += baha_result.final_unsat;
            total_fractures += baha_result.fractures;
            total_jumps += baha_result.branch_jumps;
            
            std::cout << "  [" << std::setw(2) << i << "] SA: " << std::setw(3) << sa_result.final_unsat 
                      << " unsat | BAHA: " << std::setw(3) << baha_result.final_unsat 
                      << " unsat (fractures=" << baha_result.fractures 
                      << ", jumps=" << baha_result.branch_jumps << ")\n";
        }
        
        std::cout << "\n📊 SUMMARY for n=" << n << ":\n";
        std::cout << "   BAHA wins: " << baha_wins << ", SA wins: " << sa_wins << ", Ties: " << ties << "\n";
        std::cout << "   Avg Unsat - SA: " << std::fixed << std::setprecision(1) 
                  << (double)sa_total_unsat / N_INSTANCES 
                  << ", BAHA: " << (double)baha_total_unsat / N_INSTANCES << "\n";
        std::cout << "   Avg Time - SA: " << sa_total_time / N_INSTANCES 
                  << "ms, BAHA: " << baha_total_time / N_INSTANCES << "ms\n";
        std::cout << "   Avg Fractures: " << (double)total_fractures / N_INSTANCES 
                  << ", Avg Jumps: " << (double)total_jumps / N_INSTANCES << "\n";
    }
    
    csv.close();
    std::cout << "\n✅ Results written to data/sat3_benchmark.csv\n";
    
    return 0;
}
