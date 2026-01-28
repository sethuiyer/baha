#include "baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

struct Literal { int var; bool negated; };
struct Clause { Literal lits[3]; };

class SAT3Instance {
public:
    int n_vars;
    std::vector<Clause> clauses;
    using State = std::vector<int>;
    
    static SAT3Instance generate(int n, double alpha, std::mt19937& rng) {
        SAT3Instance inst;
        inst.n_vars = n;
        int m = static_cast<int>(alpha * n);
        std::uniform_int_distribution<int> var_dist(0, n - 1);
        std::uniform_int_distribution<int> sign_dist(0, 1);
        for (int i = 0; i < m; ++i) {
            Clause c;
            std::vector<int> vars;
            while (vars.size() < 3) {
                int v = var_dist(rng);
                if (std::find(vars.begin(), vars.end(), v) == vars.end()) vars.push_back(v);
            }
            for (int j = 0; j < 3; ++j) {
                c.lits[j].var = vars[j];
                c.lits[j].negated = (sign_dist(rng) == 1);
            }
            inst.clauses.push_back(c);
        }
        return inst;
    }
    
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
};

int main() {
    int n = 500;
    double alpha = 4.267;
    int max_steps = 10000;
    std::mt19937 rng(123);
    
    auto inst = SAT3Instance::generate(n, alpha, rng);
    auto energy_fn = [&](const std::vector<int>& s) -> double {
        return static_cast<double>(inst.unsat_count(s));
    };
    auto sampler_fn = [&]() { return inst.random_state(rng); };
    
    // 1. Run SA Trace
    {
        std::ofstream sa_csv("data/sa_trace.csv");
        sa_csv << "step,beta,energy\n";
        auto state = inst.random_state(rng);
        int energy = inst.unsat_count(state);
        double beta_start = 0.1, beta_end = 20.0;
        double ratio = std::pow(beta_end / beta_start, 1.0 / max_steps);
        double beta = beta_start;
        std::uniform_real_distribution<> unif(0, 1);
        std::uniform_int_distribution<int> var_dist(0, n - 1);
        
        for (int step = 0; step < max_steps; ++step) {
            int flip_var = var_dist(rng);
            state[flip_var] ^= 1;
            int new_energy = inst.unsat_count(state);
            int delta = new_energy - energy;
            if (delta <= 0 || unif(rng) < std::exp(-beta * delta)) {
                energy = new_energy;
            } else {
                state[flip_var] ^= 1;
            }
            beta *= ratio;
            if (step % 10 == 0) sa_csv << step << "," << beta << "," << energy << "\n";
        }
        sa_csv.close();
    }
    
    // 2. Run BAHA Trace
    {
        std::ofstream baha_csv("data/baha_trace.csv");
        baha_csv << "step,beta,energy,rho,type\n";
        navokoj::BranchAwareOptimizer<std::vector<int>> baha(energy_fn, sampler_fn);
        navokoj::BranchAwareOptimizer<std::vector<int>>::Config config;
        config.beta_start = 0.1;
        config.beta_end = 20.0;
        config.beta_steps = max_steps;
        config.samples_per_beta = 5;
        config.fracture_threshold = 2.0; // Higher threshold
        config.verbose = false;
        config.logger = [&](int step, double beta, double energy, double rho, const std::string& type) {
            baha_csv << step << "," << beta << "," << energy << "," << rho << "," << type << "\n";
        };
        baha.optimize(config);
        baha_csv.close();
    }
    
    std::cout << "Trace complete. Files: data/sa_trace.csv, data/baha_trace.csv\n";
    return 0;
}
