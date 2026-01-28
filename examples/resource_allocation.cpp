/*
 * Resource Allocation Problem using BAHA
 * Allocate limited resources to tasks maximizing total value
 */
#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <chrono>

struct AllocationState {
    std::vector<std::vector<double>> allocation;  // allocation[i][j] = resource j allocated to task i
    int n_tasks;
    int n_resources;
    
    AllocationState() : n_tasks(0), n_resources(0) {}
    AllocationState(int tasks, int resources) : n_tasks(tasks), n_resources(resources),
                                                 allocation(tasks, std::vector<double>(resources, 0.0)) {}
};

struct Task {
    std::vector<double> resource_needs;  // Minimum needed of each resource
    std::vector<double> resource_limits;  // Maximum can use of each resource
    double base_value;
    std::vector<double> value_per_resource;  // Value per unit of resource
};

int main(int argc, char** argv) {
    int n_tasks = (argc > 1) ? std::stoi(argv[1]) : 10;
    int n_resources = (argc > 2) ? std::stoi(argv[2]) : 5;
    
    // Generate tasks and resource constraints
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> need_dist(0.5, 2.0);
    std::uniform_real_distribution<double> limit_dist(2.0, 5.0);
    std::uniform_real_distribution<double> value_dist(10.0, 50.0);
    std::uniform_real_distribution<double> resource_value_dist(1.0, 5.0);
    
    std::vector<Task> tasks(n_tasks);
    std::vector<double> total_resources(n_resources);
    
    for (int i = 0; i < n_tasks; ++i) {
        tasks[i].resource_needs.resize(n_resources);
        tasks[i].resource_limits.resize(n_resources);
        tasks[i].value_per_resource.resize(n_resources);
        
        for (int j = 0; j < n_resources; ++j) {
            tasks[i].resource_needs[j] = need_dist(rng);
            tasks[i].resource_limits[j] = limit_dist(rng);
            tasks[i].value_per_resource[j] = resource_value_dist(rng);
            total_resources[j] += tasks[i].resource_limits[j] * 0.5;  // Total available
        }
        tasks[i].base_value = value_dist(rng);
    }
    
    std::cout << "============================================================\n";
    std::cout << "RESOURCE ALLOCATION PROBLEM: " << n_tasks << " tasks, " 
              << n_resources << " resource types\n";
    std::cout << "============================================================\n";
    
    // Energy: negative total value + penalty for constraint violations
    auto energy = [&tasks, &total_resources](const AllocationState& s) -> double {
        double total_value = 0.0;
        double violations = 0.0;
        
        // Check resource constraints
        std::vector<double> resource_used(s.n_resources, 0.0);
        
        for (int i = 0; i < s.n_tasks; ++i) {
            double task_value = tasks[i].base_value;
            
            for (int j = 0; j < s.n_resources; ++j) {
                double allocated = s.allocation[i][j];
                resource_used[j] += allocated;
                
                // Check if allocation is within limits
                if (allocated < tasks[i].resource_needs[j]) {
                    violations += (tasks[i].resource_needs[j] - allocated) * 100.0;
                }
                if (allocated > tasks[i].resource_limits[j]) {
                    violations += (allocated - tasks[i].resource_limits[j]) * 100.0;
                }
                
                // Value from resource allocation
                task_value += allocated * tasks[i].value_per_resource[j];
            }
            
            total_value += task_value;
        }
        
        // Check total resource availability
        for (int j = 0; j < s.n_resources; ++j) {
            if (resource_used[j] > total_resources[j]) {
                violations += (resource_used[j] - total_resources[j]) * 1000.0;
            }
        }
        
        return -total_value + violations;  // Negative because we minimize
    };
    
    // Random initial allocation
    auto sampler = [n_tasks, n_resources, &tasks, &total_resources]() -> AllocationState {
        AllocationState s(n_tasks, n_resources);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> alloc_dist(0.0, 1.0);
        
        for (int i = 0; i < n_tasks; ++i) {
            for (int j = 0; j < n_resources; ++j) {
                // Random allocation between need and limit
                double range = tasks[i].resource_limits[j] - tasks[i].resource_needs[j];
                s.allocation[i][j] = tasks[i].resource_needs[j] + alloc_dist(rng) * range;
            }
        }
        
        return s;
    };
    
    // Neighbors: adjust allocation amounts
    auto neighbors = [n_tasks, n_resources, &tasks](const AllocationState& s) -> std::vector<AllocationState> {
        std::vector<AllocationState> nbrs;
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> task_dist(0, n_tasks - 1);
        std::uniform_int_distribution<int> resource_dist(0, n_resources - 1);
        std::uniform_real_distribution<double> delta_dist(-0.5, 0.5);
        
        for (int k = 0; k < 25; ++k) {
            AllocationState nbr = s;
            int task = task_dist(rng);
            int resource = resource_dist(rng);
            
            double delta = delta_dist(rng);
            double new_val = nbr.allocation[task][resource] + delta;
            // Clamp to valid range
            new_val = std::max(tasks[task].resource_needs[resource], 
                              std::min(tasks[task].resource_limits[resource], new_val));
            nbr.allocation[task][resource] = new_val;
            
            nbrs.push_back(nbr);
        }
        
        return nbrs;
    };
    
    navokoj::BranchAwareOptimizer<AllocationState> opt(energy, sampler, neighbors);
    
    typename navokoj::BranchAwareOptimizer<AllocationState>::Config config;
    config.beta_steps = 500;
    config.beta_end = 12.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.7;
    config.max_branches = 5;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<AllocationState>::ScheduleType::GEOMETRIC;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = opt.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    // Calculate final stats
    double final_value = 0.0;
    std::vector<double> resource_used(n_resources, 0.0);
    
    for (int i = 0; i < n_tasks; ++i) {
        double task_value = tasks[i].base_value;
        for (int j = 0; j < n_resources; ++j) {
            resource_used[j] += result.best_state.allocation[i][j];
            task_value += result.best_state.allocation[i][j] * tasks[i].value_per_resource[j];
        }
        final_value += task_value;
    }
    
    std::cout << "\nResult:\n";
    std::cout << "Total value: " << std::fixed << std::setprecision(2) << final_value << "\n";
    std::cout << "Resource utilization: ";
    for (int j = 0; j < n_resources; ++j) {
        std::cout << std::fixed << std::setprecision(1) << resource_used[j] 
                  << "/" << total_resources[j];
        if (j < n_resources - 1) std::cout << ", ";
    }
    std::cout << "\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed << "s\n";
    
    return 0;
}
