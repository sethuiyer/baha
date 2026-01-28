/*
 * Author: Sethurathienam Iyer
 */
#include "baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <bitset>

// ============================================================================
// CHACHA20 STATE RECOVERY BENCHMARK (2 Rounds)
// ============================================================================
// Objective: Given Output State (512 bits), recover Input State (512 bits).
// We attack a reduced version (2 rounds) to see if structure is exploitable.
// Full ChaCha has 20 rounds. 2 rounds provides basic diffusion/confusion:
// 1 Column Round + 1 Diagonal Round.

// ChaCha20 Constants and Operations
#define ROTL(a,b) (((a) << (b)) | ((a) >> (32 - (b))))
#define QR(a, b, c, d) \
    a += b; d ^= a; d = ROTL(d,16); \
    c += d; b ^= c; b = ROTL(b,12); \
    a += b; d ^= a; d = ROTL(d, 8); \
    c += d; b ^= c; b = ROTL(b, 7);

class ChaCha20Problem {
    std::vector<uint32_t> target_output; // 16 x 32-bit state

public:
    ChaCha20Problem(const std::vector<uint32_t>& target) : target_output(target) {}

    static void chacha_2rounds(std::vector<uint32_t>& x) {
        // Round 1: Column Round (Quarter Rounds on columns)
        QR(x[0], x[4], x[8],  x[12]);
        QR(x[1], x[5], x[9],  x[13]);
        QR(x[2], x[6], x[10], x[14]);
        QR(x[3], x[7], x[11], x[15]);

        // Round 2: Diagonal Round (Quarter Rounds on diagonals)
        QR(x[0], x[5], x[10], x[15]);
        QR(x[1], x[6], x[11], x[12]);
        QR(x[2], x[7], x[8],  x[13]);
        QR(x[3], x[4], x[9],  x[14]);
    }

    // Generate a random 512-bit state
    static std::vector<uint32_t> random_state() {
        std::vector<uint32_t> state(16);
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<uint32_t> dist;
        for(int i=0; i<16; ++i) state[i] = dist(rng);
        return state;
    }

    // Energy: Hamming Distance between Permutation(Candidate) and TargetOutput
    // Lower energy = closer to matching output.
    double energy(const std::vector<uint32_t>& candidate_input) {
        std::vector<uint32_t> x = candidate_input;
        chacha_2rounds(x);

        int hamming_dist = 0;
        for(int i=0; i<16; ++i) {
            hamming_dist += __builtin_popcount(x[i] ^ target_output[i]);
        }
        return (double)hamming_dist;
    }

    // Neighbors: Single Bit Flips
    // We want to verify if single flips in input propagate to large changes in output 
    // but possibly constrained by "fractures" (basins of attraction).
    std::vector<std::vector<uint32_t>> neighbors(const std::vector<uint32_t>& s) {
        std::vector<std::vector<uint32_t>> nbrs;
        
        // Strategy: Flip 1 bit in random words to explore local neighborhoods.
        // Full set of single bit flips = 512 neighbors. That's fine for BAHA.
        // Let's sample a subset to keep it fast, or do all? 512 is okay.
        
        // Let's do a targeted subset: Flip bits in just 4 random words per step to save compute?
        // No, let's look at all 16 words, maybe 1 bit per word to keep size manageable (16 neighbors)?
        // Too small. Let's do 1 bit in *each* position for *one* random word + some others.
        // Actually, let's just do random 32 bit flips across the state.
        
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> word_dist(0, 15);
        std::uniform_int_distribution<int> bit_dist(0, 31);

        for(int i=0; i<50; ++i) { // 50 Random single-bit neighbors
            std::vector<uint32_t> next = s;
            int w = word_dist(rng);
            int b = bit_dist(rng);
            next[w] ^= (1 << b);
            nbrs.push_back(next);
        }
        
        return nbrs;
    }
};

int main() {
    std::cout << "🪞 THE BOSS LEVEL: CHACHA20 'HOUSE OF MIRRORS' 🪞" << std::endl;
    std::cout << "Problem: Invert 2-Round ChaCha20 (Recover 512-bit Input)." << std::endl;
    std::cout << "Testing 'Fracture Field' Hypothesis on ARX Structure." << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // 1. Create a known Target
    std::vector<uint32_t> TARGET_INPUT = ChaCha20Problem::random_state();
    std::vector<uint32_t> TARGET_OUTPUT = TARGET_INPUT;
    ChaCha20Problem::chacha_2rounds(TARGET_OUTPUT);

    std::cout << "Target Input (Hash): " << std::hex << TARGET_INPUT[0] << "..." << std::dec << std::endl;

    ChaCha20Problem problem(TARGET_OUTPUT);

    // BAHA Setup
    auto energy = [&](const std::vector<uint32_t>& s) { return problem.energy(s); };
    auto sampler = [&]() { return ChaCha20Problem::random_state(); };
    auto neighbors = [&](const std::vector<uint32_t>& s) { return problem.neighbors(s); };

    navokoj::BranchAwareOptimizer<std::vector<uint32_t>> baha(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<std::vector<uint32_t>>::Config config;
    
    // Landscape: Extremely Chaotic/Glassy due to ARX.
    // Max Energy = 512 (all bits wrong). Random state ~ 256. 
    // We want Energy -> 0.
    config.beta_steps = 2000;
    config.beta_end = 8.0; 
    config.samples_per_beta = 50; 
    config.fracture_threshold = 1.25; // High sensitivity
    config.max_branches = 10;
    config.verbose = true;
    config.schedule_type = navokoj::BranchAwareOptimizer<std::vector<uint32_t>>::ScheduleType::GEOMETRIC;

    std::cout << "Running BAHA..." << std::endl;
    auto result_ba = baha.optimize(config);

    std::cout << "\nBAHA RESULT:" << std::endl;
    std::cout << "Energy (Hamming Dist): " << result_ba.best_energy << " / 512" << std::endl;
    std::cout << "Fractures: " << result_ba.fractures_detected << std::endl;
    std::cout << "Jumps: " << result_ba.branch_jumps << std::endl;
    
    // Check bit accuracy of recovered input vs target input
    int input_bit_errors = 0;
    for(int i=0; i<16; ++i) {
        input_bit_errors += __builtin_popcount(result_ba.best_state[i] ^ TARGET_INPUT[i]);
    }
    std::cout << "Input Recovery Error: " << input_bit_errors << " bits / 512" << std::endl;


    // SA Baseline
    std::cout << "\nRunning SA Baseline..." << std::endl;
    navokoj::SimulatedAnnealing<std::vector<uint32_t>> sa(energy, sampler, neighbors);
    navokoj::SimulatedAnnealing<std::vector<uint32_t>>::Config sa_config;
    sa_config.beta_steps = 2000;
    sa_config.steps_per_beta = 50; 
    auto result_sa = sa.optimize(sa_config);

    std::cout << "\nSA RESULT:" << std::endl;
    std::cout << "Energy (Hamming Dist): " << result_sa.best_energy << std::endl;
    int sa_input_errors = 0;
    for(int i=0; i<16; ++i) {
        sa_input_errors += __builtin_popcount(result_sa.best_state[i] ^ TARGET_INPUT[i]);
    }
    std::cout << "Input Recovery Error: " << sa_input_errors << " bits / 512" << std::endl;

    return 0;
}
