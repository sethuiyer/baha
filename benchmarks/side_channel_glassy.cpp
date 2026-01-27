#include "baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <bitset>

// ============================================================================
// GLASSY SIDE-CHANNEL ATTACK (Frustrated Encoding)
// ============================================================================
// Objective: Recover a 32-bit Secret Key K
// twist: We don't optimize K directly. We optimize a Spin Chain S.
// Mapping: K_i = S_i XOR S_{i+1} (Periodic Boundary)
// This introduces "Ferromagnetic/Antiferromagnetic" domain wall dynamics.
// Flipping one spin S_i changes TWO bits in K (at i and i-1).
// This makes individual bit flips impossible; we can only do "correlated" flips.
// This constraint creates a much more rugged, "fractured" landscape.

// Toy SPN Cipher (Same as before)
const int NUM_ROUNDS = 4;
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
        uint32_t state = plaintext ^ key;
        leakage.push_back(__builtin_popcount(state));

        for (int r = 0; r < NUM_ROUNDS; ++r) {
            uint32_t next_state = 0;
            for (int i = 0; i < 8; ++i) {
                int shift = i * 4;
                int nibble = (state >> shift) & 0xF;
                next_state |= (SBOX[nibble] << shift);
            }
            state = next_state;
            
            next_state = 0;
            for(int i=0; i<32; ++i) {
                if ((state >> i) & 1) next_state |= (1 << PBOX[i]);
            }
            state = next_state;
            state ^= key;
            leakage.push_back(__builtin_popcount(state));
        }
        return state;
    }
};

class GlassyAttack {
    uint32_t target_key;
    std::vector<uint32_t> plaintexts;
    std::vector<std::vector<int>> target_leakages;

public:
    GlassyAttack(uint32_t key, int num_traces) : target_key(key) {
        std::mt19937 rng(12345);
        std::uniform_int_distribution<uint32_t> dist;

        for(int i=0; i<num_traces; ++i) {
            uint32_t pt = dist(rng);
            plaintexts.push_back(pt);
            std::vector<int> trace;
            ToyCipher::encrypt(pt, target_key, trace);
            target_leakages.push_back(trace);
        }
    }

    // Encoding: spins vector -> uint32_t key
    // S_i in {0, 1}
    uint32_t decode_key(const std::vector<int>& spins) {
        uint32_t k = 0;
        int N = spins.size();
        for(int i=0; i<N; ++i) {
            int val = spins[i] ^ spins[(i+1)%N]; // Interaction term
            if (val) k |= (1 << i);
        }
        return k;
    }

    double energy(const std::vector<int>& spins) {
        uint32_t candidate_key = decode_key(spins);
        double sse = 0;
        for(size_t i=0; i<plaintexts.size(); ++i) {
            std::vector<int> candidate_trace;
            ToyCipher::encrypt(plaintexts[i], candidate_key, candidate_trace);
            for(size_t j=0; j<candidate_trace.size(); ++j) {
                double diff = (double)(candidate_trace[j] - target_leakages[i][j]);
                sse += diff * diff;
            }
        }
        return sse;
    }

    std::vector<int> random_state() {
        std::vector<int> s(32);
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, 1);
        for(int i=0; i<32; ++i) s[i] = dist(rng);
        return s;
    }

    // Neighbors: Flip spin in the domain representation
    std::vector<std::vector<int>> neighbors(const std::vector<int>& s) {
        std::vector<std::vector<int>> nbrs;
        // Single Spin Flip (Changes 2 bits in Key) -> Domain Wall creation/annihilation
        for(int i=0; i<32; ++i) {
            std::vector<int> n = s;
            n[i] = 1 - n[i];
            nbrs.push_back(n);
        }
        
        // Multi-spin flips for good measure?
        // Let's stick to single spin flips to emphasize the structure.
        return nbrs;
    }
};

int main() {
    std::cout << "🪞 GLASSY PARAMETRIZATION TEST 🪞" << std::endl;
    std::cout << "Recovering Key using Spin Chain Encoding (K_i = S_i XOR S_i+1)." << std::endl;
    std::cout << "Hypothesis: Induced fractures allow BAHA to find the key." << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    uint32_t HIDDEN_KEY = 0xDEADBEEF; // Parity is Even (24 bits set). Representable.
    int NUM_TRACES = 10;
    
    GlassyAttack attack(HIDDEN_KEY, NUM_TRACES);

    // BAHA Setup
    auto energy = [&](const std::vector<int>& s) { return attack.energy(s); };
    auto sampler = [&]() { return attack.random_state(); };
    auto neighbors = [&](const std::vector<int>& s) { return attack.neighbors(s); };

    navokoj::BranchAwareOptimizer<std::vector<int>> baha(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<std::vector<int>>::Config config;
    
    config.beta_steps = 1000;
    config.beta_end = 5.0; 
    config.samples_per_beta = 100;
    config.fracture_threshold = 1.3;
    config.max_branches = 8;
    config.verbose = true;
    config.schedule_type = navokoj::BranchAwareOptimizer<std::vector<int>>::ScheduleType::GEOMETRIC;

    std::cout << "Target Key: " << std::hex << HIDDEN_KEY << std::dec << std::endl;
    std::cout << "Running BAHA (Glassy Encoding)..." << std::endl;
    auto result_ba = baha.optimize(config);

    uint32_t final_key = attack.decode_key(result_ba.best_state);

    std::cout << "\nBAHA RESULT:" << std::endl;
    std::cout << "Best Key Found: " << std::hex << final_key << std::dec << std::endl;
    std::cout << "Energy (SSE): " << result_ba.best_energy << std::endl;
    std::cout << "Fractures: " << result_ba.fractures_detected << std::endl;
    std::cout << "Jumps: " << result_ba.branch_jumps << std::endl;
    
    int bit_errors = __builtin_popcount(final_key ^ HIDDEN_KEY);
    std::cout << "Bit Errors: " << bit_errors << "/32" << std::endl;

    if (final_key == HIDDEN_KEY) {
        std::cout << "🚨 CRITICAL: EXACT KEY RECOVERY! 🚨" << std::endl;
    } else if (bit_errors <= 2) {
         std::cout << "⚠️ WARNING: NEAR-KEY RECOVERY (" << bit_errors << " bits off) ⚠️" << std::endl;
    } else {
        std::cout << "✅ SAFE: Key not recovered." << std::endl;
    }

    // SA Baseline
    std::cout << "\nRunning SA Baseline (Glassy Encoding)..." << std::endl;
    navokoj::SimulatedAnnealing<std::vector<int>> sa(energy, sampler, neighbors);
    navokoj::SimulatedAnnealing<std::vector<int>>::Config sa_config;
    sa_config.beta_steps = 1000;
    sa_config.steps_per_beta = 50; 
    auto result_sa = sa.optimize(sa_config);
    uint32_t sa_key = attack.decode_key(result_sa.best_state);

    std::cout << "\nSA RESULT:" << std::endl;
    std::cout << "Best Key Found: " << std::hex << sa_key << std::dec << std::endl;
    std::cout << "Energy (SSE): " << result_sa.best_energy << std::endl;
    std::cout << "Bit Errors: " << __builtin_popcount(sa_key ^ HIDDEN_KEY) << "/32" << std::endl;

    return 0;
}
