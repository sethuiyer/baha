/*
 * N-Queens Problem using BAHA
 * Place N queens on an N×N chessboard such that no two queens attack each other
 */
#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

struct QueensState {
    std::vector<int> queens;  // queens[i] = column of queen in row i
    int n;
    
    QueensState() : n(0) {}
    QueensState(int size) : n(size), queens(size, -1) {}
};

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::stoi(argv[1]) : 8;
    
    std::cout << "============================================================\n";
    std::cout << "N-QUEENS PROBLEM: Place " << n << " queens on " << n << "×" << n << " board\n";
    std::cout << "============================================================\n";
    
    // Energy: count of attacking pairs
    auto energy = [](const QueensState& s) -> double {
        int conflicts = 0;
        for (int i = 0; i < s.n; ++i) {
            if (s.queens[i] == -1) continue;
            for (int j = i + 1; j < s.n; ++j) {
                if (s.queens[j] == -1) continue;
                // Same column
                if (s.queens[i] == s.queens[j]) conflicts++;
                // Same diagonal
                if (std::abs(s.queens[i] - s.queens[j]) == std::abs(i - j)) conflicts++;
            }
        }
        return static_cast<double>(conflicts);
    };
    
    // Random initial placement
    auto sampler = [n]() -> QueensState {
        QueensState s(n);
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, n - 1);
        for (int i = 0; i < n; ++i) {
            s.queens[i] = dist(rng);
        }
        return s;
    };
    
    // Neighbors: move one queen to a different column in same row
    auto neighbors = [n](const QueensState& s) -> std::vector<QueensState> {
        std::vector<QueensState> nbrs;
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> row_dist(0, n - 1);
        std::uniform_int_distribution<int> col_dist(0, n - 1);
        
        for (int k = 0; k < 20; ++k) {
            QueensState nbr = s;
            int row = row_dist(rng);
            int new_col = col_dist(rng);
            while (new_col == nbr.queens[row]) {
                new_col = col_dist(rng);
            }
            nbr.queens[row] = new_col;
            nbrs.push_back(nbr);
        }
        return nbrs;
    };
    
    navokoj::BranchAwareOptimizer<QueensState> opt(energy, sampler, neighbors);
    
    typename navokoj::BranchAwareOptimizer<QueensState>::Config config;
    config.beta_steps = 500;
    config.beta_end = 15.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.5;
    config.max_branches = 5;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<QueensState>::ScheduleType::GEOMETRIC;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = opt.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    std::cout << "\nResult:\n";
    std::cout << "Final energy (conflicts): " << result.best_energy << "\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed << "s\n";
    
    if (result.best_energy == 0) {
        std::cout << "\n✅ SOLUTION FOUND!\n\n";
        // Print board
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << (result.best_state.queens[i] == j ? "Q " : ". ");
            }
            std::cout << "\n";
        }
    } else {
        std::cout << "\n⚠️  No perfect solution found (conflicts remaining: " 
                  << (int)result.best_energy << ")\n";
    }
    
    return 0;
}
