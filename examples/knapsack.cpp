/*
 * 0/1 Knapsack Problem using BAHA
 * Maximize value while staying under weight limit
 */
#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>

struct KnapsackState {
    std::vector<bool> items;  // true = included, false = excluded
    int n_items;
    
    KnapsackState() : n_items(0) {}
    KnapsackState(int n) : n_items(n), items(n, false) {}
};

struct Item {
    int weight;
    int value;
};

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::stoi(argv[1]) : 20;
    int capacity = (argc > 2) ? std::stoi(argv[2]) : 50;
    
    // Generate random items
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> weight_dist(1, 10);
    std::uniform_int_distribution<int> value_dist(5, 30);
    
    std::vector<Item> items(n);
    int total_value = 0;
    for (int i = 0; i < n; ++i) {
        items[i].weight = weight_dist(rng);
        items[i].value = value_dist(rng);
        total_value += items[i].value;
    }
    
    std::cout << "============================================================\n";
    std::cout << "0/1 KNAPSACK PROBLEM: " << n << " items, capacity " << capacity << "\n";
    std::cout << "Total possible value: " << total_value << "\n";
    std::cout << "============================================================\n";
    
    // Energy: negative value + penalty for exceeding capacity
    auto energy = [&items, capacity](const KnapsackState& s) -> double {
        int total_weight = 0;
        int total_value = 0;
        
        for (size_t i = 0; i < s.items.size(); ++i) {
            if (s.items[i]) {
                total_weight += items[i].weight;
                total_value += items[i].value;
            }
        }
        
        // Penalty for exceeding capacity
        if (total_weight > capacity) {
            return 1e6 + (total_weight - capacity) * 1000.0 - total_value;
        }
        
        return -static_cast<double>(total_value);  // Negative because we minimize
    };
    
    // Random selection
    auto sampler = [n]() -> KnapsackState {
        KnapsackState s(n);
        std::mt19937 rng(std::random_device{}());
        std::bernoulli_distribution dist(0.3);  // 30% chance to include
        for (int i = 0; i < n; ++i) {
            s.items[i] = dist(rng);
        }
        return s;
    };
    
    // Neighbors: flip one item
    auto neighbors = [n](const KnapsackState& s) -> std::vector<KnapsackState> {
        std::vector<KnapsackState> nbrs;
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, n - 1);
        
        for (int k = 0; k < 30; ++k) {
            KnapsackState nbr = s;
            int flip = dist(rng);
            nbr.items[flip] = !nbr.items[flip];
            nbrs.push_back(nbr);
        }
        return nbrs;
    };
    
    navokoj::BranchAwareOptimizer<KnapsackState> opt(energy, sampler, neighbors);
    
    typename navokoj::BranchAwareOptimizer<KnapsackState>::Config config;
    config.beta_steps = 400;
    config.beta_end = 12.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.7;
    config.max_branches = 5;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<KnapsackState>::ScheduleType::GEOMETRIC;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = opt.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    // Calculate final stats
    int final_weight = 0;
    int final_value = 0;
    for (size_t i = 0; i < result.best_state.items.size(); ++i) {
        if (result.best_state.items[i]) {
            final_weight += items[i].weight;
            final_value += items[i].value;
        }
    }
    
    std::cout << "\nResult:\n";
    std::cout << "Best value: " << final_value << "\n";
    std::cout << "Total weight: " << final_weight << " / " << capacity << "\n";
    std::cout << "Weight utilization: " << std::fixed << std::setprecision(1)
              << (100.0 * final_weight / capacity) << "%\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed << "s\n";
    
    std::cout << "\nSelected items:\n";
    for (size_t i = 0; i < result.best_state.items.size(); ++i) {
        if (result.best_state.items[i]) {
            std::cout << "  Item " << i << ": weight=" << items[i].weight 
                      << ", value=" << items[i].value << "\n";
        }
    }
    
    return 0;
}
