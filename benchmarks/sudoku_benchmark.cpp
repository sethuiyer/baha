/*
 * Author: Sethurathienam Iyer
 */
#include "baha.hpp"
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <set>
#include <iomanip>

// =============================================================================
// SUDOKU SOLVER USING BAHA
// =============================================================================

struct SudokuState {
    std::array<std::array<int, 9>, 9> grid;
};

class SudokuProblem {
public:
    SudokuProblem(const std::array<std::array<int, 9>, 9>& initial, int seed)
        : initial_(initial), rng_(seed) {
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                fixed_[i][j] = (initial[i][j] != 0);
            }
        }
    }

    double energy(const SudokuState& state) const {
        int violations = 0;
        
        for (int i = 0; i < 9; ++i) {
            std::set<int> seen;
            for (int j = 0; j < 9; ++j) {
                int v = state.grid[i][j];
                if (v != 0) {
                    if (seen.count(v)) violations++;
                    seen.insert(v);
                }
            }
        }
        
        for (int j = 0; j < 9; ++j) {
            std::set<int> seen;
            for (int i = 0; i < 9; ++i) {
                int v = state.grid[i][j];
                if (v != 0) {
                    if (seen.count(v)) violations++;
                    seen.insert(v);
                }
            }
        }
        
        for (int bi = 0; bi < 3; ++bi) {
            for (int bj = 0; bj < 3; ++bj) {
                std::set<int> seen;
                for (int di = 0; di < 3; ++di) {
                    for (int dj = 0; dj < 3; ++dj) {
                        int i = bi * 3 + di;
                        int j = bj * 3 + dj;
                        int v = state.grid[i][j];
                        if (v != 0) {
                            if (seen.count(v)) violations++;
                            seen.insert(v);
                        }
                    }
                }
            }
        }
        
        int empty = 0;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (state.grid[i][j] == 0) empty++;
            }
        }
        
        return violations * 10.0 + empty * 0.1;
    }

    SudokuState random_state() {
        SudokuState state;
        state.grid = initial_;
        
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (!fixed_[i][j]) {
                    std::vector<int> candidates = get_candidates(state, i, j);
                    if (!candidates.empty()) {
                        std::uniform_int_distribution<int> dist(0, candidates.size() - 1);
                        state.grid[i][j] = candidates[dist(rng_)];
                    } else {
                        std::uniform_int_distribution<int> digit_dist(1, 9);
                        state.grid[i][j] = digit_dist(rng_);
                    }
                }
            }
        }
        return state;
    }

    std::vector<int> get_candidates(const SudokuState& state, int row, int col) const {
        std::set<int> used;
        for (int j = 0; j < 9; ++j) {
            if (state.grid[row][j] != 0) used.insert(state.grid[row][j]);
        }
        for (int i = 0; i < 9; ++i) {
            if (state.grid[i][col] != 0) used.insert(state.grid[i][col]);
        }
        int bi = (row / 3) * 3, bj = (col / 3) * 3;
        for (int di = 0; di < 3; ++di) {
            for (int dj = 0; dj < 3; ++dj) {
                if (state.grid[bi + di][bj + dj] != 0) {
                    used.insert(state.grid[bi + di][bj + dj]);
                }
            }
        }
        std::vector<int> candidates;
        for (int d = 1; d <= 9; ++d) {
            if (used.find(d) == used.end()) candidates.push_back(d);
        }
        return candidates;
    }

    std::vector<SudokuState> neighbors(const SudokuState& state) {
        std::vector<SudokuState> nbrs;
        std::vector<std::pair<int, int>> violated_cells;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (!fixed_[i][j] && get_conflicts(state, i, j) > 0) {
                    violated_cells.push_back({i, j});
                }
            }
        }
        
        if (violated_cells.empty()) {
            return generate_random_neighbors(state);
        }
        
        std::uniform_int_distribution<int> cell_dist(0, violated_cells.size() - 1);
        
        for (int k = 0; k < 10; ++k) {
            auto [r, c] = violated_cells[cell_dist(rng_)];
            int current_conflicts = get_conflicts(state, r, c);
            
            for (int v = 1; v <= 9; ++v) {
                if (v == state.grid[r][c]) continue;
                SudokuState trial = state;
                trial.grid[r][c] = v;
                int new_conflicts = get_conflicts(trial, r, c);
                if (new_conflicts < current_conflicts) {
                    nbrs.push_back(trial);
                }
            }
        }
        
        if (nbrs.empty()) {
            return generate_random_neighbors(state);
        }
        return nbrs;
    }

    std::vector<SudokuState> generate_random_neighbors(const SudokuState& state) {
        std::vector<SudokuState> nbrs;
        std::uniform_int_distribution<int> pos_dist(0, 8);
        std::uniform_int_distribution<int> digit_dist(1, 9);
        
        for (int k = 0; k < 10; ++k) {
            int i = pos_dist(rng_);
            int j = pos_dist(rng_);
            if (!fixed_[i][j]) {
                SudokuState nbr = state;
                nbr.grid[i][j] = digit_dist(rng_);
                nbrs.push_back(nbr);
            }
        }
        return nbrs;
    }

    int get_conflicts(const SudokuState& state, int r, int c) const {
        int val = state.grid[r][c];
        if (val == 0) return 0;
        int conflicts = 0;
        for (int j = 0; j < 9; ++j) if (j != c && state.grid[r][j] == val) conflicts++;
        for (int i = 0; i < 9; ++i) if (i != r && state.grid[i][c] == val) conflicts++;
        int bi = (r / 3) * 3, bj = (c / 3) * 3;
        for (int di = 0; di < 3; ++di) {
            for (int dj = 0; dj < 3; ++dj) {
                int ni = bi + di, nj = bj + dj;
                if ((ni != r || nj != c) && state.grid[ni][nj] == val) conflicts++;
            }
        }
        return conflicts;
    }

    void print_grid(const SudokuState& state) const {
        std::cout << "+-------+-------+-------+\n";
        for (int i = 0; i < 9; ++i) {
            std::cout << "| ";
            for (int j = 0; j < 9; ++j) {
                if (state.grid[i][j] == 0) {
                    std::cout << ". ";
                } else {
                    std::cout << state.grid[i][j] << " ";
                }
                if (j % 3 == 2) std::cout << "| ";
            }
            std::cout << "\n";
            if (i % 3 == 2 && i < 8) {
                std::cout << "+-------+-------+-------+\n";
            }
        }
        std::cout << "+-------+-------+-------+\n";
    }

    bool verify(const SudokuState& state) const {
        return energy(state) == 0.0;
    }

private:
    std::array<std::array<int, 9>, 9> initial_;
    std::array<std::array<bool, 9>, 9> fixed_;
    mutable std::mt19937 rng_;
};

int main() {
    std::cout << "AI ESCARGOT - SUDOKU BENCHMARK\n";
    std::cout << "Solving with BAHA Fracture Detection\n";
    std::cout << "==========================================\n\n";

    std::array<std::array<int, 9>, 9> ai_escargot = {{
        {1, 0, 0, 0, 0, 7, 0, 9, 0},
        {0, 3, 0, 0, 2, 0, 0, 0, 8},
        {0, 0, 9, 6, 0, 0, 5, 0, 0},
        {0, 0, 5, 3, 0, 0, 9, 0, 0},
        {0, 1, 0, 0, 8, 0, 0, 0, 2},
        {6, 0, 0, 0, 0, 4, 0, 0, 0},
        {3, 0, 0, 0, 0, 0, 0, 1, 0},
        {0, 4, 0, 0, 0, 0, 0, 0, 7},
        {0, 0, 7, 0, 0, 0, 3, 0, 0}
    }};

    SudokuProblem problem(ai_escargot, 42);

    std::cout << "Input Puzzle:\n";
    SudokuState initial;
    initial.grid = ai_escargot;
    problem.print_grid(initial);
    std::cout << "\n";

    auto energy = [&](const SudokuState& s) { return problem.energy(s); };
    auto sampler = [&]() { return problem.random_state(); };
    auto neighbors = [&](const SudokuState& s) { return problem.neighbors(s); };

    navokoj::BranchAwareOptimizer<SudokuState> baha(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<SudokuState>::Config config;

    config.beta_steps = 5000;
    config.beta_end = 50.0;
    config.beta_critical = 5.0;
    config.samples_per_beta = 100;
    config.fracture_threshold = 20.0;
    config.max_branches = 12;
    config.verbose = true;
    config.schedule_type = navokoj::BranchAwareOptimizer<SudokuState>::ScheduleType::GEOMETRIC;

    std::cout << "Starting BAHA optimization...\n\n";
    auto start = std::chrono::high_resolution_clock::now();
    auto result = baha.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\n==========================================\n";
    std::cout << "OPTIMIZATION COMPLETE\n";
    std::cout << "==========================================\n";
    std::cout << "Time: " << std::fixed << std::setprecision(2) << time_ms / 1000.0 << " seconds\n";
    std::cout << "Final Energy: " << result.best_energy << "\n";
    std::cout << "Fractures Detected: " << result.fractures_detected << "\n";
    std::cout << "Branch Jumps: " << result.branch_jumps << "\n\n";

    std::cout << "Solution:\n";
    problem.print_grid(result.best_state);

    if (problem.verify(result.best_state)) {
        std::cout << "\nSOLVED! All constraints satisfied.\n";
    } else {
        std::cout << "\nNot fully solved. Violations remaining: " << result.best_energy << "\n";
    }

    return 0;
}
