/*
 * Bin Packing Problem using BAHA
 * Pack items into minimum number of bins without exceeding capacity
 */
#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <numeric>

struct BinPackingState {
    std::vector<std::vector<int>> bins;  // bins[i] = items in bin i
    int n_items;
    int max_bins;
    
    BinPackingState() : n_items(0), max_bins(0) {}
    BinPackingState(int items, int max_b) : n_items(items), max_bins(max_b) {
        bins.resize(max_b);
    }
};

int main(int argc, char** argv) {
    int n_items = (argc > 1) ? std::stoi(argv[1]) : 20;
    int bin_capacity = (argc > 2) ? std::stoi(argv[2]) : 100;
    int max_bins = (argc > 3) ? std::stoi(argv[3]) : 10;
    
    // Generate random items
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> size_dist(10, 40);
    
    std::vector<int> items(n_items);
    int total_size = 0;
    for (int i = 0; i < n_items; ++i) {
        items[i] = size_dist(rng);
        total_size += items[i];
    }
    int min_bins = (total_size + bin_capacity - 1) / bin_capacity;
    
    std::cout << "============================================================\n";
    std::cout << "BIN PACKING PROBLEM: " << n_items << " items, capacity " 
              << bin_capacity << ", min bins=" << min_bins << "\n";
    std::cout << "Total size: " << total_size << "\n";
    std::cout << "============================================================\n";
    
    // Energy: number of bins used + penalty for capacity violations
    auto energy = [bin_capacity](const BinPackingState& s) -> double {
        int bins_used = 0;
        int violations = 0;
        
        for (const auto& bin : s.bins) {
            if (bin.empty()) continue;
            bins_used++;
            
            int bin_size = 0;
            for (int item : bin) {
                bin_size += item;
            }
            
            if (bin_size > bin_capacity) {
                violations += (bin_size - bin_capacity);
            }
        }
        
        // Penalty for unassigned items
        std::vector<bool> assigned(s.n_items, false);
        for (const auto& bin : s.bins) {
            for (int item_idx : bin) {
                if (item_idx >= 0 && item_idx < s.n_items) {
                    assigned[item_idx] = true;
                }
            }
        }
        int unassigned = std::count(assigned.begin(), assigned.end(), false);
        
        return bins_used * 1.0 + violations * 100.0 + unassigned * 1000.0;
    };
    
    // Random initial assignment
    auto sampler = [n_items, max_bins, &items, bin_capacity]() -> BinPackingState {
        BinPackingState s(n_items, max_bins);
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> bin_dist(0, max_bins - 1);
        
        std::vector<int> item_list(n_items);
        std::iota(item_list.begin(), item_list.end(), 0);
        std::shuffle(item_list.begin(), item_list.end(), rng);
        
        for (int item_idx : item_list) {
            int bin = bin_dist(rng);
            s.bins[bin].push_back(item_idx);
        }
        
        return s;
    };
    
    // Neighbors: move item between bins, swap items, or empty a bin
    auto neighbors = [n_items, max_bins](const BinPackingState& s) -> std::vector<BinPackingState> {
        std::vector<BinPackingState> nbrs;
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> bin_dist(0, max_bins - 1);
        std::uniform_int_distribution<int> item_dist(0, n_items - 1);
        
        for (int k = 0; k < 25; ++k) {
            BinPackingState nbr = s;
            int op = rng() % 3;
            
            if (op == 0) {
                // Move item from one bin to another
                int from_bin = bin_dist(rng);
                if (nbr.bins[from_bin].empty()) continue;
                int to_bin = bin_dist(rng);
                int idx = rng() % nbr.bins[from_bin].size();
                int item = nbr.bins[from_bin][idx];
                nbr.bins[from_bin].erase(nbr.bins[from_bin].begin() + idx);
                nbr.bins[to_bin].push_back(item);
            } else if (op == 1) {
                // Swap two items between bins
                int bin1 = bin_dist(rng);
                int bin2 = bin_dist(rng);
                if (nbr.bins[bin1].empty() || nbr.bins[bin2].empty()) continue;
                int idx1 = rng() % nbr.bins[bin1].size();
                int idx2 = rng() % nbr.bins[bin2].size();
                std::swap(nbr.bins[bin1][idx1], nbr.bins[bin2][idx2]);
            } else {
                // Move all items from one bin to another (consolidate)
                int from_bin = bin_dist(rng);
                int to_bin = bin_dist(rng);
                if (nbr.bins[from_bin].empty() || from_bin == to_bin) continue;
                nbr.bins[to_bin].insert(nbr.bins[to_bin].end(), 
                                        nbr.bins[from_bin].begin(), 
                                        nbr.bins[from_bin].end());
                nbr.bins[from_bin].clear();
            }
            
            nbrs.push_back(nbr);
        }
        
        return nbrs;
    };
    
    navokoj::BranchAwareOptimizer<BinPackingState> opt(energy, sampler, neighbors);
    
    typename navokoj::BranchAwareOptimizer<BinPackingState>::Config config;
    config.beta_steps = 500;
    config.beta_end = 12.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.7;
    config.max_branches = 5;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<BinPackingState>::ScheduleType::GEOMETRIC;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = opt.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    // Calculate final stats
    int bins_used = 0;
    for (const auto& bin : result.best_state.bins) {
        if (!bin.empty()) bins_used++;
    }
    
    std::cout << "\nResult:\n";
    std::cout << "Bins used: " << bins_used << " (minimum possible: " << min_bins << ")\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed << "s\n";
    
    std::cout << "\nBin assignments:\n";
    for (size_t b = 0; b < result.best_state.bins.size(); ++b) {
        if (result.best_state.bins[b].empty()) continue;
        int bin_size = 0;
        for (int item_idx : result.best_state.bins[b]) {
            bin_size += items[item_idx];
        }
        std::cout << "Bin " << b << " (size=" << bin_size << "/" << bin_capacity << "): ";
        for (int item_idx : result.best_state.bins[b]) {
            std::cout << item_idx << "(" << items[item_idx] << ") ";
        }
        std::cout << "\n";
    }
    
    return 0;
}
