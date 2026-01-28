/*
 * Traveling Salesman Problem (TSP) using BAHA
 * Find shortest route visiting all cities exactly once
 */
#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <chrono>

struct TSPState {
    std::vector<int> tour;  // Permutation of city indices
    int n_cities;
    
    TSPState() : n_cities(0) {}
    TSPState(int n) : n_cities(n), tour(n) {
        std::iota(tour.begin(), tour.end(), 0);
    }
};

struct City {
    double x, y;
    
    double distance(const City& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
};

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::stoi(argv[1]) : 15;
    
    // Generate random cities
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> coord_dist(0.0, 100.0);
    
    std::vector<City> cities(n);
    for (int i = 0; i < n; ++i) {
        cities[i].x = coord_dist(rng);
        cities[i].y = coord_dist(rng);
    }
    
    std::cout << "============================================================\n";
    std::cout << "TRAVELING SALESMAN PROBLEM: " << n << " cities\n";
    std::cout << "============================================================\n";
    
    // Energy: total tour distance
    auto energy = [&cities](const TSPState& s) -> double {
        double total_distance = 0.0;
        for (size_t i = 0; i < s.tour.size(); ++i) {
            int from = s.tour[i];
            int to = s.tour[(i + 1) % s.tour.size()];
            total_distance += cities[from].distance(cities[to]);
        }
        return total_distance;
    };
    
    // Random tour
    auto sampler = [n]() -> TSPState {
        TSPState s(n);
        std::mt19937 rng(std::random_device{}());
        std::shuffle(s.tour.begin(), s.tour.end(), rng);
        return s;
    };
    
    // Neighbors: 2-opt swaps
    auto neighbors = [n](const TSPState& s) -> std::vector<TSPState> {
        std::vector<TSPState> nbrs;
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, n - 1);
        
        for (int k = 0; k < 20; ++k) {
            TSPState nbr = s;
            int i = dist(rng);
            int j = dist(rng);
            if (i > j) std::swap(i, j);
            // Reverse segment between i and j (2-opt)
            std::reverse(nbr.tour.begin() + i, nbr.tour.begin() + j + 1);
            nbrs.push_back(nbr);
        }
        return nbrs;
    };
    
    navokoj::BranchAwareOptimizer<TSPState> opt(energy, sampler, neighbors);
    
    typename navokoj::BranchAwareOptimizer<TSPState>::Config config;
    config.beta_steps = 500;
    config.beta_end = 15.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.8;
    config.max_branches = 6;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<TSPState>::ScheduleType::GEOMETRIC;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = opt.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    std::cout << "\nResult:\n";
    std::cout << "Shortest tour distance: " << std::fixed << std::setprecision(2) 
              << result.best_energy << "\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed << "s\n";
    
    std::cout << "\nTour: ";
    for (int city : result.best_state.tour) {
        std::cout << city << " → ";
    }
    std::cout << result.best_state.tour[0] << "\n";
    
    return 0;
}
