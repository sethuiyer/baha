/*
 * Author: Sethurathienam Iyer
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include "baha.hpp"

// ============================================================================
// THE QUANTUM MINER (Hash Reversal)
// ============================================================================
// Goal: Find a nonce 'x' such that Hash(x) is close to a Target.
// This simulates Bitcoin mining or Pre-image attacks.
// The landscape should be perfectly "rugged" (uncorrelated).
// Theories predict BAHA should FAIL here (perform no better than Random Walk).
// If BAHA wins, our Hash is weak or BAHA is magic.
// ============================================================================

// A toy "Avalanche" hash function to simulate cryptographic hardness
// It mixes bits thoroughly so flipping one bit changes ~50% of output bits.
uint64_t avalanche(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Problem State
// We look for a 64-bit integer 'nonce'.
// Cost = Number of bit differences (Hamming Distance) between Hash(nonce) and Target.
class MiningProblem {
    uint64_t target;
public:
    MiningProblem(uint64_t t) : target(t) {}

    // Initial state: random 64-bit integer
    std::vector<int> initial_state() {
        // We represent the 64-bit integer as a vector of 64 bits?
        // BAHA works on discrete vectors. Let's map {0,1}^64.
        std::vector<int> state(64);
        std::random_device rd;
        std::mt19937_64 gen(rd());
        uint64_t r = gen();
        for(int i=0; i<64; ++i) state[i] = (r >> i) & 1;
        return state;
    }

    // Cost Function: Hamming Distance to Target
    double cost(const std::vector<int>& state) {
        uint64_t nonce = 0;
        for(int i=0; i<64; ++i) {
            if(state[i]) nonce |= (1ULL << i);
        }
        uint64_t hash = avalanche(nonce);
        
        // Hamming distance
        uint64_t diff = hash ^ target;
        // Count set bits
        int dist = 0;
        while(diff) {
            dist += (diff & 1);
            diff >>= 1;
        }
        return (double)dist;
    }
    
    int size() const { return 64; }
};

// Simulated Annealing for Comparison
double run_sa(MiningProblem& problem, int max_steps) {
    auto current = problem.initial_state();
    double current_energy = problem.cost(current);
    double best_energy = current_energy;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> bit_dist(0, 63);

    double T = 10.0;
    double alpha = 0.9995;

    for(int k=0; k<max_steps; ++k) {
        // Flip one bit
        int idx = bit_dist(gen);
        current[idx] = 1 - current[idx];
        double new_energy = problem.cost(current);

        if (new_energy < current_energy || dis(gen) < exp((current_energy - new_energy) / T)) {
            current_energy = new_energy;
            if (new_energy < best_energy) best_energy = new_energy;
        } else {
            // Revert
            current[idx] = 1 - current[idx];
        }
        T *= alpha;
    }
    return best_energy; // Return best distance found
}

int main() {
    std::cout << "⛏️  MINING BENCHMARK (Hash Reversal) ⛏️" << std::endl;
    std::cout << "Testing Unstructured Search (No Correlations)" << std::endl;
    std::cout << "Problem: Find preimage x such that Hash(x) matches Target (64 bits)." << std::endl;
    std::cout << "State Space: 2^64 (~1.8e19). Optimum is E=0 (statistically impossible to find)." << std::endl;
    std::cout << "We look for 'closeness' (Hamming Distance)." << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // Target is arbitrary
    uint64_t target = 0xDEADBEEFCAFEBABEULL;
    MiningProblem problem(target);

    // Run 5 Trials
    for(int i=1; i<=5; ++i) {
        std::cout << "Trial " << i << "..." << std::flush;
        
        // BAHA
        // Define neighbor function for BAHA
        auto neighbors = [](const std::vector<int>& s) {
            std::vector<std::vector<int>> nbrs;
            // Generate a few random bit flips as neighbors
            for(int k=0; k<10; ++k) {
                std::vector<int> n = s;
                int idx = rand() % 64;
                n[idx] = 1 - n[idx];
                nbrs.push_back(n);
            }
            return nbrs;
        };

        // Sampler
        auto sampler = [&problem]() { return problem.initial_state(); };
        
        // Energy
        auto energy = [&problem](const std::vector<int>& s) { return problem.cost(s); };

        navokoj::BranchAwareOptimizer<std::vector<int>> baha(energy, sampler, neighbors);
        navokoj::BranchAwareOptimizer<std::vector<int>>::Config config;
        config.beta_steps = 200; // Keep it short
        config.beta_end = 20.0;
        config.schedule_type = navokoj::BranchAwareOptimizer<std::vector<int>>::ScheduleType::GEOMETRIC;

        auto result = baha.optimize(config);
        double baha_score = result.best_energy;

        // SA
        double sa_score = run_sa(problem, 20000); 

        std::cout << " Best Dist: BA=" << baha_score << " vs SA=" << sa_score;
        if (baha_score < sa_score) std::cout << " -> BA Wins";
        else if (baha_score > sa_score) std::cout << " -> SA Wins";
        else std::cout << " -> Tie";
        std::cout << std::endl;
    }

    return 0;
}
