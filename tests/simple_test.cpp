#include <iostream>
#include "baha/baha.hpp"

// Simple test problem: minimize sum of squares
struct QuadraticState {
    std::vector<double> vars;
};

int main() {
    std::cout << "BAHA Framework Test" << std::endl;
    
    // Simple quadratic minimization problem
    auto energy = [](const QuadraticState& s) -> double {
        double sum = 0.0;
        for (double x : s.vars) {
            sum += x * x;
        }
        return sum;
    };
    
    auto sampler = []() -> QuadraticState {
        QuadraticState s;
        s.vars = {1.0, 1.0, 1.0}; // Start with [1, 1, 1]
        return s;
    };
    
    auto neighbors = [](const QuadraticState& s) -> std::vector<QuadraticState> {
        std::vector<QuadraticState> nbrs;
        for (size_t i = 0; i < s.vars.size(); ++i) {
            for (double delta : {-0.1, 0.1}) {
                QuadraticState nbr = s;
                nbr.vars[i] += delta;
                nbrs.push_back(nbr);
            }
        }
        return nbrs;
    };
    
    navokoj::BranchAwareOptimizer<QuadraticState> opt(energy, sampler, neighbors);
    
    typename navokoj::BranchAwareOptimizer<QuadraticState>::Config config;
    config.beta_steps = 100;
    config.beta_end = 5.0;
    config.verbose = false;
    
    auto result = opt.optimize(config);
    
    std::cout << "Optimization completed!" << std::endl;
    std::cout << "Best energy: " << result.best_energy << std::endl;
    std::cout << "Fractures detected: " << result.fractures_detected << std::endl;
    std::cout << "Branch jumps: " << result.branch_jumps << std::endl;
    
    return 0;
}