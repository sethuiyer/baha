/*
 * Vehicle Routing Problem (VRP) using BAHA
 * Minimize total distance for vehicles to serve all customers
 */
#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <numeric>

struct VRPState {
    std::vector<std::vector<int>> routes;  // routes[i] = customer sequence for vehicle i
    int n_customers;
    int n_vehicles;
    
    VRPState() : n_customers(0), n_vehicles(0) {}
    VRPState(int customers, int vehicles) : n_customers(customers), n_vehicles(vehicles) {
        routes.resize(vehicles);
    }
};

struct Customer {
    double x, y;
    int demand;
    
    double distance(const Customer& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
};

int main(int argc, char** argv) {
    int n_customers = (argc > 1) ? std::stoi(argv[1]) : 15;
    int n_vehicles = (argc > 2) ? std::stoi(argv[2]) : 3;
    int capacity = (argc > 3) ? std::stoi(argv[3]) : 100;
    
    // Generate depot and customers
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> coord_dist(0.0, 100.0);
    std::uniform_int_distribution<int> demand_dist(5, 25);
    
    Customer depot{50.0, 50.0, 0};
    std::vector<Customer> customers(n_customers);
    for (int i = 0; i < n_customers; ++i) {
        customers[i].x = coord_dist(rng);
        customers[i].y = coord_dist(rng);
        customers[i].demand = demand_dist(rng);
    }
    
    std::cout << "============================================================\n";
    std::cout << "VEHICLE ROUTING PROBLEM: " << n_customers << " customers, " 
              << n_vehicles << " vehicles, capacity " << capacity << "\n";
    std::cout << "============================================================\n";
    
    // Energy: total distance + penalty for capacity violations
    auto energy = [&](const VRPState& s) -> double {
        double total_distance = 0.0;
        int capacity_violations = 0;
        
        for (const auto& route : s.routes) {
            if (route.empty()) continue;
            
            // Distance from depot to first customer
            total_distance += depot.distance(customers[route[0]]);
            
            // Distance between customers
            for (size_t i = 0; i < route.size() - 1; ++i) {
                total_distance += customers[route[i]].distance(customers[route[i+1]]);
            }
            
            // Distance from last customer back to depot
            total_distance += customers[route.back()].distance(depot);
            
            // Check capacity
            int route_demand = 0;
            for (int c : route) {
                route_demand += customers[c].demand;
            }
            if (route_demand > capacity) {
                capacity_violations += (route_demand - capacity);
            }
        }
        
        // Penalty for unserved customers
        std::vector<bool> served(n_customers, false);
        for (const auto& route : s.routes) {
            for (int c : route) {
                if (c >= 0 && c < n_customers) served[c] = true;
            }
        }
        int unserved = std::count(served.begin(), served.end(), false);
        
        return total_distance + capacity_violations * 1000.0 + unserved * 5000.0;
    };
    
    // Random initial assignment
    auto sampler = [n_customers, n_vehicles]() -> VRPState {
        VRPState s(n_customers, n_vehicles);
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> vehicle_dist(0, n_vehicles - 1);
        
        std::vector<int> customer_list(n_customers);
        std::iota(customer_list.begin(), customer_list.end(), 0);
        std::shuffle(customer_list.begin(), customer_list.end(), rng);
        
        for (int c : customer_list) {
            int v = vehicle_dist(rng);
            s.routes[v].push_back(c);
        }
        
        return s;
    };
    
    // Neighbors: move customer between routes, swap customers, or reverse segment
    auto neighbors = [n_customers, n_vehicles](const VRPState& s) -> std::vector<VRPState> {
        std::vector<VRPState> nbrs;
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> vehicle_dist(0, n_vehicles - 1);
        std::uniform_int_distribution<int> customer_dist(0, n_customers - 1);
        
        for (int k = 0; k < 20; ++k) {
            VRPState nbr = s;
            int op = rng() % 3;
            
            if (op == 0) {
                // Move customer from one route to another
                int from_v = vehicle_dist(rng);
                if (nbr.routes[from_v].empty()) continue;
                int to_v = vehicle_dist(rng);
                int idx = rng() % nbr.routes[from_v].size();
                int customer = nbr.routes[from_v][idx];
                nbr.routes[from_v].erase(nbr.routes[from_v].begin() + idx);
                nbr.routes[to_v].push_back(customer);
            } else if (op == 1) {
                // Swap two customers between routes
                int v1 = vehicle_dist(rng);
                int v2 = vehicle_dist(rng);
                if (nbr.routes[v1].empty() || nbr.routes[v2].empty()) continue;
                int idx1 = rng() % nbr.routes[v1].size();
                int idx2 = rng() % nbr.routes[v2].size();
                std::swap(nbr.routes[v1][idx1], nbr.routes[v2][idx2]);
            } else {
                // Reverse a segment in a route
                int v = vehicle_dist(rng);
                if (nbr.routes[v].size() < 2) continue;
                int start = rng() % nbr.routes[v].size();
                int end = rng() % nbr.routes[v].size();
                if (start > end) std::swap(start, end);
                std::reverse(nbr.routes[v].begin() + start, nbr.routes[v].begin() + end + 1);
            }
            
            nbrs.push_back(nbr);
        }
        
        return nbrs;
    };
    
    navokoj::BranchAwareOptimizer<VRPState> opt(energy, sampler, neighbors);
    
    typename navokoj::BranchAwareOptimizer<VRPState>::Config config;
    config.beta_steps = 600;
    config.beta_end = 15.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.8;
    config.max_branches = 6;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<VRPState>::ScheduleType::GEOMETRIC;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = opt.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    // Calculate final stats
    double final_distance = 0.0;
    for (const auto& route : result.best_state.routes) {
        if (route.empty()) continue;
        final_distance += depot.distance(customers[route[0]]);
        for (size_t i = 0; i < route.size() - 1; ++i) {
            final_distance += customers[route[i]].distance(customers[route[i+1]]);
        }
        final_distance += customers[route.back()].distance(depot);
    }
    
    std::cout << "\nResult:\n";
    std::cout << "Total distance: " << std::fixed << std::setprecision(2) << final_distance << "\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed << "s\n";
    
    std::cout << "\nRoutes:\n";
    for (size_t v = 0; v < result.best_state.routes.size(); ++v) {
        if (result.best_state.routes[v].empty()) continue;
        int demand = 0;
        for (int c : result.best_state.routes[v]) {
            demand += customers[c].demand;
        }
        std::cout << "Vehicle " << v << " (demand=" << demand << "/" << capacity << "): ";
        std::cout << "Depot → ";
        for (int c : result.best_state.routes[v]) {
            std::cout << c << " → ";
        }
        std::cout << "Depot\n";
    }
    
    return 0;
}
