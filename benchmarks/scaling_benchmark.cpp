#include "baha.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>

// =============================================================================
// SCALING NUMBER PARTITIONING
// Problem: Split N numbers into two sets with equal sum.
// Objective: Minimize |Sum(S1) - Sum(S2)|
// =============================================================================

struct PartitionState {
    std::vector<int> s; // +1 or -1
};

class NumberPartitioningProblem {
public:
    NumberPartitioningProblem(int n, int seed) : n_(n), rng_(seed) {
        // Generate big numbers. 
        // We use int64_t to be safe, but keep range somewhat limited to avoid immediate overflow
        // Range 0 to 10^12. Sum can handle N*10^12 ~ 10^15 (fits in 10^18).
        std::uniform_int_distribution<long long> dist(1, 1000000000000LL);
        numbers_.resize(n);
        for(int i=0; i<n; ++i) numbers_[i] = dist(rng_);
    }

    double energy(const PartitionState& state) const {
        long long sum = 0;
        for (int i = 0; i < n_; ++i) {
            sum += state.s[i] * numbers_[i];
        }
        return static_cast<double>(std::abs(sum));
    }

    PartitionState random_state() {
        PartitionState state;
        state.s.resize(n_);
        std::uniform_int_distribution<> dist(0, 1);
        for (int i = 0; i < n_; ++i) {
            state.s[i] = dist(rng_) ? 1 : -1;
        }
        return state;
    }

    std::vector<PartitionState> neighbors(const PartitionState& state) {
        // Large neighborhood is expensive at N=1000.
        // We will sample a random subset of neighbors if N is large?
        // Actually BAHA needs full neighborhood for local optimality check?
        // No, standard BAHA uses a generator.
        // Let's iterate all N neighbors. N=1000 is small enough for modern CPU.
        std::vector<PartitionState> nbrs;
        nbrs.reserve(n_);
        for (int i = 0; i < n_; ++i) {
            PartitionState nbr = state;
            nbr.s[i] *= -1;
            nbrs.push_back(nbr);
        }
        return nbrs;
    }

    int get_n() const { return n_; }

private:
    int n_;
    std::vector<long long> numbers_;
    mutable std::mt19937_64 rng_;
};

// =============================================================================
// MAIN RUNNER
// =============================================================================

template<typename Problem, typename State>
void run_test(int n, int trials) {
    std::cout << "\nRunning Number Partitioning (N=" << n << ")\n";
    std::cout << "------------------------------------------------------------\n";
    
    double ba_total_e = 0;
    double sa_total_e = 0;
    int ba_better = 0, sa_better = 0;

    for(int t=0; t<trials; ++t) {
        std::cout << "Trial " << (t+1) << "... ";
        std::cout.flush();

        Problem prob(n, t + 4242);

        std::function<double(const State&)> energy = [&](const State& s) { return prob.energy(s); };
        std::function<State()> sampler = [&]() { return prob.random_state(); };
        std::function<std::vector<State>(const State&)> neighbors = [&](const State& s) { return prob.neighbors(s); };

        typename navokoj::BranchAwareOptimizer<State>::Config ba_config;
        ba_config.beta_steps = 2000; 
        ba_config.beta_end = 20.0;
        ba_config.max_branches = 10;
        ba_config.schedule_type = navokoj::BranchAwareOptimizer<State>::ScheduleType::GEOMETRIC;

        navokoj::BranchAwareOptimizer<State> ba(energy, sampler, neighbors);
        auto ba_res = ba.optimize(ba_config);

        typename navokoj::SimulatedAnnealing<State>::Config sa_config;
        sa_config.beta_steps = 2000; // Same iteration budget
        sa_config.beta_end = 20.0;
        sa_config.steps_per_beta = 50; 

        navokoj::SimulatedAnnealing<State> sa(energy, sampler, neighbors);
        auto sa_res = sa.optimize(sa_config);

        ba_total_e += ba_res.best_energy;
        sa_total_e += sa_res.best_energy;
        
        if (ba_res.best_energy < sa_res.best_energy) {
            ba_better++;
            std::cout << "BA Wins (E=" << ba_res.best_energy << " vs " << sa_res.best_energy << ")\n";
        } else if (sa_res.best_energy < ba_res.best_energy) {
            sa_better++;
            std::cout << "SA Wins (E=" << ba_res.best_energy << " vs " << sa_res.best_energy << ")\n";
        } else {
            std::cout << "Tie (E=" << ba_res.best_energy << ")\n";
        }
    }
    
    std::cout << "Summary N=" << n << ":\n";
    std::cout << "  BA Win Rate: " << ba_better << "/" << trials << "\n";
    std::cout << "  BA Avg Residue: " << (ba_total_e / trials) << "\n";
    std::cout << "  SA Avg Residue: " << (sa_total_e / trials) << "\n";
    std::cout << "  Advantage factor: " << ((sa_total_e + 1.0) / (ba_total_e + 1.0)) << "x\n";
}

int main() {
    std::cout << "🔨 THE HAMMER: SCALING BENCHMARK 🔨\n";
    std::cout << "Testing if BAHA's fracture detection survives at scale.\n";
    
    // N=200
    run_test<NumberPartitioningProblem, PartitionState>(200, 3);
    
    // N=500
    run_test<NumberPartitioningProblem, PartitionState>(500, 3);
    
    // N=1000
    run_test<NumberPartitioningProblem, PartitionState>(1000, 3);

    return 0;
}
