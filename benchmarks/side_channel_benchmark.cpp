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
// SIDE-CHANNEL ATTACK SIMULATION (Toy Cipher)
// ============================================================================
// Objective: Recover a 32-bit Secret Key K
// Information: "Leaky" Hamming Weights of intermediate states during encryption.
// Attack Model: We try to find a key K' that produces the same leakage profile 
// as the target key K for a set of known plaintexts.

// Toy SPN Cipher Constants
const int NUM_ROUNDS = 4;
const int BLOCK_SIZE = 32;
const int SBOX[16] = {
    0x6, 0x4, 0xC, 0x5, 0x0, 0x7, 0x2, 0xE,
    0x1, 0xF, 0x3, 0xD, 0x8, 0xA, 0x9, 0xB
};
const int PBOX[32] = {
    0, 4, 8, 12, 16, 20, 24, 28,
    1, 5, 9, 13, 17, 21, 25, 29,
    2, 6, 10, 14, 18, 22, 26, 30,
    3, 7, 11, 15, 19, 23, 27, 31
};

class ToyCipher {
public:
    static uint32_t encrypt(uint32_t plaintext, uint32_t key, std::vector<int>& leakage) {
        uint32_t state = plaintext ^ key; // Initial Key Add
        leakage.push_back(__builtin_popcount(state)); // Leakage: Hamming Weight

        for (int r = 0; r < NUM_ROUNDS; ++r) {
            // Substitution Layer (4-bit S-boxes)
            uint32_t next_state = 0;
            for (int i = 0; i < 8; ++i) {
                int shift = i * 4;
                int nibble = (state >> shift) & 0xF;
                next_state |= (SBOX[nibble] << shift);
            }
            state = next_state;
            
            // Permutation Layer
            next_state = 0;
            for(int i=0; i<32; ++i) {
                if ((state >> i) & 1) {
                    next_state |= (1 << PBOX[i]);
                }
            }
            state = next_state;

            // Key Mixing (Simple XOR with same key for this toy example)
            state ^= key;
            
            // Record Leakage
            leakage.push_back(__builtin_popcount(state));
        }
        return state;
    }
};

class SideChannelAttack {
    uint32_t target_key;
    std::vector<uint32_t> plaintexts;
    std::vector<std::vector<int>> target_leakages;

public:
    SideChannelAttack(uint32_t key, int num_traces) : target_key(key) {
        std::mt19937 rng(12345); // Fixed seed for reproducibility
        std::uniform_int_distribution<uint32_t> dist;

        for(int i=0; i<num_traces; ++i) {
            uint32_t pt = dist(rng);
            plaintexts.push_back(pt);
            
            std::vector<int> trace;
            ToyCipher::encrypt(pt, target_key, trace);
            target_leakages.push_back(trace);
        }
    }

    double energy(uint32_t candidate_key) {
        double sse = 0;
        for(size_t i=0; i<plaintexts.size(); ++i) {
            std::vector<int> candidate_trace;
            ToyCipher::encrypt(plaintexts[i], candidate_key, candidate_trace);
            
            // Sum of Squared Errors between Hamming Weights
            for(size_t j=0; j<candidate_trace.size(); ++j) {
                double diff = (double)(candidate_trace[j] - target_leakages[i][j]);
                sse += diff * diff;
            }
        }
        return sse;
    }

    uint32_t random_key() {
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_int_distribution<uint32_t> dist;
        return dist(rng);
    }

    // Neighbors: Flip 1 bit, 2 bits, or swap bits
    std::vector<uint32_t> neighbors(uint32_t k) {
        std::vector<uint32_t> nbrs;
        // Single Bit Flips
        for(int i=0; i<32; ++i) {
            nbrs.push_back(k ^ (1 << i));
        }
        // Random 2-bit flips (to help jump out of local minima)
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, 31);
        for(int i=0; i<5; ++i) {
            int b1 = dist(rng);
            int b2 = dist(rng);
            if(b1 != b2) nbrs.push_back(k ^ (1 << b1) ^ (1 << b2));
        }
        return nbrs;
    }
};

int main() {
    std::cout << "🔐 SECURITY RISK ASSESSMENT (SIDE-CHANNEL) 🔐" << std::endl;
    std::cout << "Recovering 32-bit Key from Hamming Weight Leakage." << std::endl;
    std::cout << "Target: Exploit 'Leakage Fractures' to reconstruct the key." << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    uint32_t HIDDEN_KEY = 0xDEADBEEF; // The secret
    int NUM_TRACES = 10; // Few traces makes it harder/ambiguous? Or easier?
                         // Too few = many keys fit. Too many = precise landscape.
                         // 10 traces * 5 rounds = 50 constraints. Should be enough for 32 bits.
    
    SideChannelAttack attack(HIDDEN_KEY, NUM_TRACES);

    // BAHA Setup
    auto energy = [&](const uint32_t& k) { return attack.energy(k); };
    auto sampler = [&]() { return attack.random_key(); };
    auto neighbors = [&](const uint32_t& k) { return attack.neighbors(k); };

    navokoj::BranchAwareOptimizer<uint32_t> baha(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<uint32_t>::Config config;
    
    // Landscape is "Correlation Landscape". Very rugged.
    // SSE can be large.
    config.beta_steps = 1000;
    config.beta_end = 5.0; 
    config.samples_per_beta = 100;
    config.fracture_threshold = 1.3;
    config.max_branches = 8; // Need high branching for this type of problem
    config.verbose = true;
    config.schedule_type = navokoj::BranchAwareOptimizer<uint32_t>::ScheduleType::GEOMETRIC;

    std::cout << "Target Key: " << std::hex << HIDDEN_KEY << std::dec << std::endl;
    std::cout << "Running BAHA..." << std::endl;
    auto result_ba = baha.optimize(config);

    std::cout << "\nBAHA RESULT:" << std::endl;
    std::cout << "Best Key Found: " << std::hex << result_ba.best_state << std::dec << std::endl;
    std::cout << "Energy (SSE): " << result_ba.best_energy << std::endl;
    std::cout << "Fractures: " << result_ba.fractures_detected << std::endl;
    std::cout << "Jumps: " << result_ba.branch_jumps << std::endl;
    
    int bit_errors = __builtin_popcount(result_ba.best_state ^ HIDDEN_KEY);
    std::cout << "Bit Errors: " << bit_errors << "/32" << std::endl;

    if (result_ba.best_state == HIDDEN_KEY) {
        std::cout << "🚨 CRITICAL: EXACT KEY RECOVERY! 🚨" << std::endl;
    } else if (bit_errors <= 2) {
         std::cout << "⚠️ WARNING: NEAR-KEY RECOVERY (" << bit_errors << " bits off) ⚠️" << std::endl;
    } else {
        std::cout << "✅ SAFE: Key not recovered." << std::endl;
    }

    // SA Baseline
    std::cout << "\nRunning SA Baseline..." << std::endl;
    navokoj::SimulatedAnnealing<uint32_t> sa(energy, sampler, neighbors);
    navokoj::SimulatedAnnealing<uint32_t>::Config sa_config;
    sa_config.beta_steps = 1000;
    sa_config.steps_per_beta = 50; 
    auto result_sa = sa.optimize(sa_config);

    std::cout << "\nSA RESULT:" << std::endl;
    std::cout << "Best Key Found: " << std::hex << result_sa.best_state << std::dec << std::endl;
    std::cout << "Energy (SSE): " << result_sa.best_energy << std::endl;
    std::cout << "Bit Errors: " << __builtin_popcount(result_sa.best_state ^ HIDDEN_KEY) << "/32" << std::endl;

    return 0;
}
