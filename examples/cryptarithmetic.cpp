/*
 * Author: Sethurathienam Iyer
 * 
 * CRYPTARITHMETIC PUZZLE SOLVER
 * Solve: SEND + MORE = MONEY
 * Each letter is a unique digit 0-9
 */

#include "baha.hpp"
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <set>

// State: Assignment of digits to letters {S, E, N, D, M, O, R, Y}
struct CryptoState {
    std::array<int, 8> digits;  // S=0, E=1, N=2, D=3, M=4, O=5, R=6, Y=7
};

class SendMoreMoney {
public:
    SendMoreMoney(int seed) : rng_(seed) {}

    // Energy = constraint violations
    // 1. All digits must be unique
    // 2. SEND + MORE == MONEY
    // 3. S and M cannot be 0 (leading digits)
    double energy(const CryptoState& s) const {
        int violations = 0;
        
        // Check uniqueness - count duplicate pairs
        for (int i = 0; i < 8; ++i) {
            for (int j = i + 1; j < 8; ++j) {
                if (s.digits[i] == s.digits[j]) {
                    violations += 10;  // Heavy penalty for duplicates
                }
            }
        }
        
        // Check leading zeros - HEAVY penalty
        if (s.digits[0] == 0) violations += 10000;  // S != 0
        if (s.digits[4] == 0) violations += 10000;  // M != 0
        
        // Compute SEND + MORE - MONEY
        int S = s.digits[0], E = s.digits[1], N = s.digits[2], D = s.digits[3];
        int M = s.digits[4], O = s.digits[5], R = s.digits[6], Y = s.digits[7];
        
        int SEND = 1000*S + 100*E + 10*N + D;
        int MORE = 1000*M + 100*O + 10*R + E;
        int MONEY = 10000*M + 1000*O + 100*N + 10*E + Y;
        
        int diff = std::abs(SEND + MORE - MONEY);
        
        return violations + diff;
    }

    CryptoState random_state() {
        CryptoState s;
        // Generate a random permutation, ensuring S and M are not 0
        std::vector<int> digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::shuffle(digits.begin(), digits.end(), rng_);
        
        // Place digits, but ensure S (index 0) and M (index 4) are not 0
        int j = 0;
        for (int i = 0; i < 8; ++i) {
            if ((i == 0 || i == 4) && digits[j] == 0) {
                // Find a non-zero digit to swap with
                for (int k = j + 1; k < 10; ++k) {
                    if (digits[k] != 0) {
                        std::swap(digits[j], digits[k]);
                        break;
                    }
                }
            }
            s.digits[i] = digits[j++];
        }
        return s;
    }

    std::vector<CryptoState> neighbors(const CryptoState& s) {
        std::vector<CryptoState> nbrs;
        
        // Swap any two letter assignments
        for (int i = 0; i < 8; ++i) {
            for (int j = i + 1; j < 8; ++j) {
                CryptoState n = s;
                std::swap(n.digits[i], n.digits[j]);
                nbrs.push_back(n);
            }
        }
        
        // Change one letter to a different unused digit
        std::set<int> used(s.digits.begin(), s.digits.end());
        for (int i = 0; i < 8; ++i) {
            for (int d = 0; d <= 9; ++d) {
                if (used.find(d) == used.end()) {
                    CryptoState n = s;
                    n.digits[i] = d;
                    nbrs.push_back(n);
                }
            }
        }
        
        return nbrs;
    }

    void print_solution(const CryptoState& s) const {
        int S = s.digits[0], E = s.digits[1], N = s.digits[2], D = s.digits[3];
        int M = s.digits[4], O = s.digits[5], R = s.digits[6], Y = s.digits[7];
        
        std::cout << "\n  Solution found!\n";
        std::cout << "  ---------------\n";
        std::cout << "  S=" << S << " E=" << E << " N=" << N << " D=" << D << "\n";
        std::cout << "  M=" << M << " O=" << O << " R=" << R << " Y=" << Y << "\n\n";
        
        int SEND = 1000*S + 100*E + 10*N + D;
        int MORE = 1000*M + 100*O + 10*R + E;
        int MONEY = 10000*M + 1000*O + 100*N + 10*E + Y;
        
        std::cout << "    " << SEND << "\n";
        std::cout << "  + " << MORE << "\n";
        std::cout << "  -------\n";
        std::cout << "   " << MONEY << "\n\n";
        
        if (SEND + MORE == MONEY) {
            std::cout << "  VERIFIED: " << SEND << " + " << MORE << " = " << MONEY << "\n";
        }
    }

private:
    mutable std::mt19937 rng_;
};

int main() {
    std::cout << "CRYPTARITHMETIC PUZZLE\n";
    std::cout << "======================\n\n";
    std::cout << "Solving:   SEND\n";
    std::cout << "         + MORE\n";
    std::cout << "         ------\n";
    std::cout << "          MONEY\n\n";
    std::cout << "Using BAHA to find the unique solution...\n\n";

    SendMoreMoney puzzle(42);

    auto energy = [&](const CryptoState& s) { return puzzle.energy(s); };
    auto sampler = [&]() { return puzzle.random_state(); };
    auto neighbors = [&](const CryptoState& s) { return puzzle.neighbors(s); };

    navokoj::BranchAwareOptimizer<CryptoState> baha(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<CryptoState>::Config config;

    config.beta_steps = 2000;
    config.beta_end = 100.0;
    config.fracture_threshold = 2.0;
    config.samples_per_beta = 100;
    config.max_branches = 10;
    config.verbose = false;  // Less noise
    config.schedule_type = navokoj::BranchAwareOptimizer<CryptoState>::ScheduleType::GEOMETRIC;

    auto start = std::chrono::high_resolution_clock::now();
    auto result = baha.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\n==========================================\n";
    std::cout << "BAHA RESULTS\n";
    std::cout << "==========================================\n";
    std::cout << "Time: " << time_ms << " ms\n";
    std::cout << "Final Energy: " << result.best_energy << "\n";
    std::cout << "Fractures: " << result.fractures_detected << "\n";
    std::cout << "Branch Jumps: " << result.branch_jumps << "\n";

    if (result.best_energy == 0) {
        puzzle.print_solution(result.best_state);
    } else {
        std::cout << "\nDid not find exact solution. Best energy: " << result.best_energy << "\n";
        puzzle.print_solution(result.best_state);
    }

    return 0;
}
