#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <iomanip>
#include <string>
#include <random>
#include <map>

#include "baha/baha.hpp"

// =============================================================================
// ADVERSARIAL LANDSCAPES: THE HALL OF PAIN
// =============================================================================

const double PI = 3.14159265358979323846;

struct Landscape {
    std::string name;
    int dim;
    std::function<double(const std::vector<double>&)> f;
    double min_bound;
    double max_bound;
    double known_global_min;
};

// 1. Rosenbrock (The "Banana" Valley)
// Global min at (1,1,...,1), f(x) = 0
double rosenbrock(const std::vector<double>& x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size() - 1; ++i) {
        double diff1 = x[i+1] - x[i] * x[i];
        double diff2 = 1.0 - x[i];
        sum += 100.0 * diff1 * diff1 + diff2 * diff2;
    }
    return sum;
}

// 2. Rastrigin (Local Minima Hell)
// Global min at (0,0,...,0), f(x) = 0
double rastrigin(const std::vector<double>& x) {
    double sum = 10.0 * x.size();
    for (double val : x) {
        sum += val * val - 10.0 * std::cos(2.0 * PI * val);
    }
    return sum;
}

// 3. Ackley (Flat Plateau + Deep Hole)
// Global min at (0,0,...,0), f(x) = 0
double ackley(const std::vector<double>& x) {
    double sum_sq = 0.0;
    double sum_cos = 0.0;
    for (double val : x) {
        sum_sq += val * val;
        sum_cos += std::cos(2.0 * PI * val);
    }
    double n = (double)x.size();
    return -20.0 * std::exp(-0.2 * std::sqrt(sum_sq / n)) 
           - std::exp(sum_cos / n) + 20.0 + std::exp(1.0);
}

// 4. Schwefel (Deceptive Boundary Trap)
// Global min at (420.9687, ..., 420.9687), f(x) = 0
// Range: [-500, 500]
double schwefel(const std::vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += val * std::sin(std::sqrt(std::abs(val)));
    }
    return 418.9829 * x.size() - sum;
}

// 5. Griewank (Oscillatory Product)
// Global min at (0,0,...,0), f(x) = 0
double griewank(const std::vector<double>& x) {
    double sum_sq = 0.0;
    double prod_cos = 1.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum_sq += x[i] * x[i];
        prod_cos *= std::cos(x[i] / std::sqrt(i + 1.0));
    }
    return 1.0 + sum_sq / 4000.0 - prod_cos;
}

// 6. Easom (Needle in a Haystack) - 2D only
// Global min at (PI, PI), f(x) = -1
double easom(const std::vector<double>& x) {
    double x1 = x[0];
    double x2 = x[1];
    return -std::cos(x1) * std::cos(x2) * std::exp(-(x1 - PI)*(x1 - PI) - (x2 - PI)*(x2 - PI));
}

// 7. Michalewicz (Steep Valleys)
// Global min depends on dimension.
double michalewicz(const std::vector<double>& x) {
    double sum = 0.0;
    int m = 10;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += std::sin(x[i]) * std::pow(std::sin((i + 1) * x[i] * x[i] / PI), 2 * m);
    }
    return -sum;
}

// =============================================================================
// CONTINUOUS STATE WRAPPER
// =============================================================================

struct ContinuousState {
    std::vector<double> vars;
};

int main() {
    std::cout << "🌋 BAHA vs THE HALL OF PAIN 🌋" << std::endl;
    std::cout << "Benchmarking against adversarial optimization landscapes..." << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    std::vector<Landscape> landscapes = {
        {"Rosenbrock (2D)", 2, rosenbrock, -5.0, 10.0, 0.0},
        {"Rosenbrock (10D)", 10, rosenbrock, -5.0, 10.0, 0.0},
        {"Rastrigin (10D)", 10, rastrigin, -5.12, 5.12, 0.0},
        {"Ackley (10D)", 10, ackley, -32.768, 32.768, 0.0},
        {"Schwefel (10D)", 10, schwefel, -500.0, 500.0, 0.0},
        {"Griewank (10D)", 10, griewank, -600.0, 600.0, 0.0},
        {"Easom (2D Needle)", 2, easom, -100.0, 100.0, -1.0},
        {"Michalewicz (10D)", 10, michalewicz, 0.0, PI, -9.66015} // Approx for 10D
    };

    std::cout << "| Landscape          | Dim | Target | Best Found | Error      | Frac | Jumps | Time(ms) |" << std::endl;
    std::cout << "|--------------------|-----|--------|------------|------------|------|-------|----------|" << std::endl;

    for (const auto& land : landscapes) {
        // Setup Problem
        auto energy = [&](const ContinuousState& s) { return land.f(s.vars); };
        
        auto sampler = [&]() {
            ContinuousState s;
            s.vars.resize(land.dim);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(land.min_bound, land.max_bound);
            for(int i=0; i<land.dim; ++i) s.vars[i] = dis(gen);
            return s;
        };

        // Gaussian mutation neighbor
        auto neighbors = [&](const ContinuousState& s) {
            std::vector<ContinuousState> nbrs;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(0, (land.max_bound - land.min_bound) * 0.05); // 5% scale mutation

            // Generate 5 random neighbors
            for(int k=0; k<5; ++k) {
                ContinuousState n = s;
                // Mutate all dims slightly
                for(int i=0; i<land.dim; ++i) {
                    n.vars[i] += d(gen);
                    // Bounce checks (simple clamp)
                    if(n.vars[i] < land.min_bound) n.vars[i] = land.min_bound + std::abs(d(gen));
                    if(n.vars[i] > land.max_bound) n.vars[i] = land.max_bound - std::abs(d(gen));
                }
                nbrs.push_back(n);
            }
            return nbrs;
        };

        // Run BAHA
        navokoj::BranchAwareOptimizer<ContinuousState> baha(energy, sampler, neighbors);
        navokoj::BranchAwareOptimizer<ContinuousState>::Config config;
        
        config.beta_steps = 3000;
        config.beta_end = 50.0;
        config.samples_per_beta = 100;
        config.fracture_threshold = 1.0; 
        config.max_branches = 10;
        config.verbose = false; 
        config.schedule_type = navokoj::BranchAwareOptimizer<ContinuousState>::ScheduleType::GEOMETRIC;

        auto result = baha.optimize(config);
        
        double error = std::abs(result.best_energy - land.known_global_min);
        
        std::cout << "| " << std::left << std::setw(18) << land.name 
                  << " | " << std::setw(3) << land.dim 
                  << " | " << std::setw(6) << std::fixed << std::setprecision(2) << land.known_global_min 
                  << " | " << std::setw(10) << std::setprecision(6) << result.best_energy 
                  << " | " << std::setw(10) << std::scientific << std::setprecision(2) << error 
                  << " | " << std::setw(4) << result.fractures_detected 
                  << " | " << std::setw(5) << result.branch_jumps 
                  << " | " << std::setw(8) << std::fixed << std::setprecision(1) << result.time_ms 
                  << " |" << std::endl;
    }

    std::cout << "----------------------------------------------------------------" << std::endl;
    return 0;
}
