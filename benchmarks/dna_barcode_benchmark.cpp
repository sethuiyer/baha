#include "baha.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <set>
#include <iomanip>
#include <fstream>

// =============================================================================
// DNA BARCODE OPTIMIZATION
// Novel Application: Design N barcodes with multi-constraint satisfaction
// =============================================================================

const char BASES[] = {'A', 'C', 'G', 'T'};

struct BarcodeState {
    std::vector<std::string> barcodes;  // N barcodes of length L each
};

class DNABarcodeOptimizer {
public:
    DNABarcodeOptimizer(int n_barcodes, int barcode_length, int min_hamming_dist, int seed)
        : N_(n_barcodes), L_(barcode_length), D_(min_hamming_dist), rng_(seed) {}

    double energy(const BarcodeState& state) const {
        double total_energy = 0.0;
        
        // 1. Hamming Distance Constraint (pairwise)
        for (int i = 0; i < N_; ++i) {
            for (int j = i + 1; j < N_; ++j) {
                int dist = hamming_distance(state.barcodes[i], state.barcodes[j]);
                if (dist < D_) {
                    total_energy += (D_ - dist) * 100.0;  // Heavy penalty for collision
                }
            }
        }
        
        // 2. GC Content Constraint (40-60% per barcode)
        for (const auto& bc : state.barcodes) {
            double gc = gc_content(bc);
            if (gc < 0.40) total_energy += (0.40 - gc) * 50.0;
            if (gc > 0.60) total_energy += (gc - 0.60) * 50.0;
        }
        
        // 3. Homopolymer Run Constraint (no runs > 3)
        for (const auto& bc : state.barcodes) {
            int max_run = max_homopolymer_run(bc);
            if (max_run > 3) {
                total_energy += (max_run - 3) * 30.0;
            }
        }
        
        // 4. Self-complementarity Penalty (avoid hairpins)
        for (const auto& bc : state.barcodes) {
            total_energy += hairpin_score(bc) * 20.0;
        }
        
        return total_energy;
    }

    BarcodeState random_state() {
        BarcodeState state;
        state.barcodes.resize(N_);
        std::uniform_int_distribution<int> base_dist(0, 3);
        
        for (int i = 0; i < N_; ++i) {
            std::string bc(L_, 'A');
            for (int j = 0; j < L_; ++j) {
                bc[j] = BASES[base_dist(rng_)];
            }
            state.barcodes[i] = bc;
        }
        return state;
    }

    std::vector<BarcodeState> neighbors(const BarcodeState& state) {
        std::vector<BarcodeState> nbrs;
        std::uniform_int_distribution<int> bc_dist(0, N_ - 1);
        std::uniform_int_distribution<int> pos_dist(0, L_ - 1);
        std::uniform_int_distribution<int> base_dist(0, 3);
        
        // Generate 32 random single-base mutations
        for (int k = 0; k < 32; ++k) {
            BarcodeState nbr = state;
            int bc_idx = bc_dist(rng_);
            int pos = pos_dist(rng_);
            char new_base = BASES[base_dist(rng_)];
            nbr.barcodes[bc_idx][pos] = new_base;
            nbrs.push_back(nbr);
        }
        
        return nbrs;
    }

    void print_solution(const BarcodeState& state) const {
        std::cout << "\n🧬 OPTIMAL BARCODE SET 🧬\n";
        std::cout << "========================\n";
        
        int min_dist = L_;
        int violations = 0;
        
        for (int i = 0; i < N_; ++i) {
            double gc = gc_content(state.barcodes[i]);
            int max_run = max_homopolymer_run(state.barcodes[i]);
            std::cout << "BC" << std::setw(2) << i << ": " << state.barcodes[i]
                      << " | GC=" << std::fixed << std::setprecision(0) << gc * 100 << "%"
                      << " | MaxRun=" << max_run;
            
            if (gc < 0.40 || gc > 0.60) { std::cout << " ⚠️GC"; violations++; }
            if (max_run > 3) { std::cout << " ⚠️HomoPoly"; violations++; }
            std::cout << "\n";
        }
        
        // Compute pairwise distances
        std::cout << "\nPairwise Hamming Distances:\n";
        for (int i = 0; i < N_; ++i) {
            for (int j = i + 1; j < N_; ++j) {
                int dist = hamming_distance(state.barcodes[i], state.barcodes[j]);
                if (dist < min_dist) min_dist = dist;
                if (dist < D_) violations++;
            }
        }
        
        std::cout << "Minimum pairwise distance: " << min_dist << " (required: " << D_ << ")\n";
        std::cout << "Total violations: " << violations << "\n";
        std::cout << "Final energy: " << energy(state) << "\n";
    }

    void export_barcodes(const BarcodeState& state, const std::string& filename) const {
        std::ofstream out(filename);
        for (int i = 0; i < N_; ++i) {
            out << ">barcode_" << i << "\n" << state.barcodes[i] << "\n";
        }
        out.close();
        std::cout << "Barcodes exported to: " << filename << "\n";
    }

private:
    int hamming_distance(const std::string& a, const std::string& b) const {
        int dist = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) dist++;
        }
        return dist;
    }

    double gc_content(const std::string& seq) const {
        int gc = 0;
        for (char c : seq) {
            if (c == 'G' || c == 'C') gc++;
        }
        return (double)gc / seq.size();
    }

    int max_homopolymer_run(const std::string& seq) const {
        int max_run = 1, current_run = 1;
        for (size_t i = 1; i < seq.size(); ++i) {
            if (seq[i] == seq[i-1]) {
                current_run++;
                max_run = std::max(max_run, current_run);
            } else {
                current_run = 1;
            }
        }
        return max_run;
    }

    double hairpin_score(const std::string& seq) const {
        // Simple self-complementarity check
        int score = 0;
        int n = seq.size();
        for (int i = 0; i < n/2; ++i) {
            char a = seq[i];
            char b = seq[n - 1 - i];
            // Check Watson-Crick pairing
            if ((a == 'A' && b == 'T') || (a == 'T' && b == 'A') ||
                (a == 'G' && b == 'C') || (a == 'C' && b == 'G')) {
                score++;
            }
        }
        return (double)score / (n/2);
    }

    int N_, L_, D_;
    mutable std::mt19937 rng_;
};

int main() {
    std::cout << "🧬 DNA BARCODE OPTIMIZATION USING BAHA 🧬\n";
    std::cout << "First Application of Fracture Detection to Molecular Design\n";
    std::cout << "============================================================\n\n";

    // Configure: 48 barcodes, 12 bases each, minimum Hamming distance 4
    int N_BARCODES = 48;
    int BARCODE_LENGTH = 12;
    int MIN_HAMMING = 4;

    std::cout << "Parameters:\n";
    std::cout << "  - Barcodes: " << N_BARCODES << "\n";
    std::cout << "  - Length: " << BARCODE_LENGTH << " bp\n";
    std::cout << "  - Min Hamming Distance: " << MIN_HAMMING << "\n";
    std::cout << "  - Search Space: 4^" << BARCODE_LENGTH << " = " 
              << std::scientific << std::pow(4, BARCODE_LENGTH) << " per barcode\n\n";

    DNABarcodeOptimizer problem(N_BARCODES, BARCODE_LENGTH, MIN_HAMMING, 42);

    auto energy = [&](const BarcodeState& s) { return problem.energy(s); };
    auto sampler = [&]() { return problem.random_state(); };
    auto neighbors = [&](const BarcodeState& s) { return problem.neighbors(s); };

    navokoj::BranchAwareOptimizer<BarcodeState> baha(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<BarcodeState>::Config config;

    config.beta_steps = 2000;
    config.beta_end = 15.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.5;
    config.max_branches = 8;
    config.verbose = true;
    config.schedule_type = navokoj::BranchAwareOptimizer<BarcodeState>::ScheduleType::GEOMETRIC;

    std::cout << "Starting BAHA optimization...\n\n";
    auto start = std::chrono::high_resolution_clock::now();
    auto result = baha.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\n============================================================\n";
    std::cout << "OPTIMIZATION COMPLETE\n";
    std::cout << "============================================================\n";
    std::cout << "Final Energy: " << result.best_energy << "\n";
    std::cout << "Fractures Detected: " << result.fractures_detected << "\n";
    std::cout << "Branch Jumps: " << result.branch_jumps << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(1) << time_ms / 1000.0 << " seconds\n";

    problem.print_solution(result.best_state);
    problem.export_barcodes(result.best_state, "data/barcodes.fasta");

    // Compare with random baseline
    std::cout << "\n📊 COMPARISON WITH RANDOM GENERATION 📊\n";
    double best_random_energy = 1e9;
    for (int i = 0; i < 100; ++i) {
        auto random_state = problem.random_state();
        double e = problem.energy(random_state);
        if (e < best_random_energy) best_random_energy = e;
    }
    std::cout << "Best Random Energy (100 trials): " << best_random_energy << "\n";
    std::cout << "BAHA Improvement: " << (best_random_energy - result.best_energy) / best_random_energy * 100.0 << "%\n";

    return 0;
}
