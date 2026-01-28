#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include "baha/baha.hpp"

// =============================================================================
// MIXED-INTEGER BENCHMARK: CAPACITATED FACILITY LOCATION PROBLEM (CFLP)
// =============================================================================
// Variables:
// 1. y[j] in {0, 1}: Open facility j? (Discrete)
// 2. x[i][j] in [0, 1]: Fraction of customer i demand met by facility j (Continuous)
//
// Minimize: sum(FixedCost[j] * y[j]) + sum(Cost[i][j] * x[i][j] * Demand[i])
// Subject to:
// - sum(x[i][j] over j) = 1 for all i (All demand met)
// - sum(x[i][j] * Demand[i] over i) <= Capacity[j] * y[j] (Capacity limit)

struct ProblemInstance {
    int num_facilities;
    int num_customers;
    std::vector<double> fixed_costs;
    std::vector<double> capacities;
    std::vector<double> demands;
    std::vector<std::vector<double>> transport_costs; // [customer][facility] cost per unit
};

struct MilpState {
    std::vector<int> open_facilities; // Binary
    std::vector<std::vector<double>> flows; // Continuous fractions
};

class FacilityLocationProblem {
    ProblemInstance problem;
    double penalty_weight = 10000.0;

public:
    FacilityLocationProblem(int M, int N, int seed) {
        problem.num_facilities = M;
        problem.num_customers = N;
        
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> cost_dist(100.0, 500.0);
        std::uniform_real_distribution<> cap_dist(50.0, 150.0);
        std::uniform_real_distribution<> dem_dist(5.0, 20.0);
        std::uniform_real_distribution<> dist_dist(1.0, 50.0);

        // Init facilities
        for(int j=0; j<M; ++j) {
            problem.fixed_costs.push_back(cost_dist(gen) * 5.0); // Expensive to open
            problem.capacities.push_back(cap_dist(gen));
        }

        // Init customers
        for(int i=0; i<N; ++i) {
            problem.demands.push_back(dem_dist(gen));
            std::vector<double> costs;
            for(int j=0; j<M; ++j) {
                costs.push_back(dist_dist(gen));
            }
            problem.transport_costs.push_back(costs);
        }
    }

    double energy(const MilpState& s) const {
        double cost = 0.0;
        double violations = 0.0;

        // 1. Fixed Costs
        for(int j=0; j<problem.num_facilities; ++j) {
            if(s.open_facilities[j]) cost += problem.fixed_costs[j];
        }

        std::vector<double> used_capacity(problem.num_facilities, 0.0);

        // 2. Transport Costs & Demand Constraints
        for(int i=0; i<problem.num_customers; ++i) {
            double met_demand = 0.0;
            for(int j=0; j<problem.num_facilities; ++j) {
                if(s.flows[i][j] > 1e-4) {
                    double amount = s.flows[i][j] * problem.demands[i];
                    cost += problem.transport_costs[i][j] * amount;
                    met_demand += s.flows[i][j];
                    used_capacity[j] += amount;
                }
            }
            // Penalty for unmet demand
            violations += std::abs(1.0 - met_demand); 
        }

        // 3. Capacity Constraints
        for(int j=0; j<problem.num_facilities; ++j) {
            double cap = s.open_facilities[j] ? problem.capacities[j] : 0.0;
            if(used_capacity[j] > cap) {
                violations += (used_capacity[j] - cap);
            }
        }

        return cost + violations * penalty_weight;
    }

    // Heuristic Sampler: Randomly open facilities, greedily assign demand
    MilpState random_state() const {
        MilpState s;
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> bin(0, 1);
        
        s.open_facilities.resize(problem.num_facilities);
        for(int j=0; j<problem.num_facilities; ++j) s.open_facilities[j] = bin(gen);
        
        // Ensure at least one is open to avoid infinite penalty
        if (std::all_of(s.open_facilities.begin(), s.open_facilities.end(), [](int i){return i==0;})) {
            s.open_facilities[gen() % problem.num_facilities] = 1;
        }

        s.flows.resize(problem.num_customers, std::vector<double>(problem.num_facilities, 0.0));
        
        // Naive flow assignment: distribute equally among open facilities
        for(int i=0; i<problem.num_customers; ++i) {
            std::vector<int> open_indices;
            for(int j=0; j<problem.num_facilities; ++j) if(s.open_facilities[j]) open_indices.push_back(j);
            
            if(!open_indices.empty()) {
                double share = 1.0 / open_indices.size();
                for(int idx : open_indices) s.flows[i][idx] = share;
            }
        }
        return s;
    }

    std::vector<MilpState> neighbors(const MilpState& s) const {
        std::vector<MilpState> nbrs;
        std::mt19937 gen(std::random_device{}());

        // NEIGHBOR TYPE 1: DISCRETE (Flip open/close)
        for(int k=0; k<3; ++k) { // Generate a few discrete neighbors
            MilpState n = s;
            int idx = gen() % problem.num_facilities;
            n.open_facilities[idx] = !n.open_facilities[idx];
            
            // Heuristic cleanup: if closed, zero out flow
            if(n.open_facilities[idx] == 0) {
                for(int i=0; i<problem.num_customers; ++i) n.flows[i][idx] = 0.0;
            }
            nbrs.push_back(n);
        }

        // NEIGHBOR TYPE 2: CONTINUOUS (Rebalance flows)
        // Pick a customer, move flow from facility A to facility B
        for(int k=0; k<5; ++k) {
            MilpState n = s;
            int cust = gen() % problem.num_customers;
            int f1 = gen() % problem.num_facilities;
            int f2 = gen() % problem.num_facilities;
            
            if (f1 != f2) {
                double amount = std::uniform_real_distribution<>(0.0, 0.5)(gen); // Move up to 50%
                if (n.flows[cust][f1] >= amount) {
                    n.flows[cust][f1] -= amount;
                    n.flows[cust][f2] += amount;
                    nbrs.push_back(n);
                }
            }
        }

        return nbrs;
    }

    void print_summary(const MilpState& s) {
        double total_fixed = 0;
        int open_count = 0;
        for(int j=0; j<problem.num_facilities; ++j) {
            if(s.open_facilities[j]) {
                total_fixed += problem.fixed_costs[j];
                open_count++;
            }
        }

        // Recalculate violations to show user
        double cost = 0.0;
        double unmet = 0.0;
        double overcap = 0.0;
        std::vector<double> used_capacity(problem.num_facilities, 0.0);

        for(int i=0; i<problem.num_customers; ++i) {
            double met = 0.0;
            for(int j=0; j<problem.num_facilities; ++j) {
                if(s.flows[i][j] > 1e-4) {
                    double amt = s.flows[i][j] * problem.demands[i];
                    cost += problem.transport_costs[i][j] * amt;
                    met += s.flows[i][j];
                    used_capacity[j] += amt;
                }
            }
            if(std::abs(met - 1.0) > 1e-3) unmet += std::abs(1.0 - met);
        }

        for(int j=0; j<problem.num_facilities; ++j) {
            double cap = s.open_facilities[j] ? problem.capacities[j] : 0.0;
            if(used_capacity[j] > cap) overcap += (used_capacity[j] - cap);
        }

        std::cout << "Facilities Open: " << open_count << "/" << problem.num_facilities << "\n";
        std::cout << "Fixed Cost: " << total_fixed << "\n";
        std::cout << "Variable Cost: " << cost << "\n";
        std::cout << "Total Cost: " << (total_fixed + cost) << "\n";
        std::cout << "Violations: Unmet Demand=" << unmet << ", OverCapacity=" << overcap << "\n";
    }
};

int main() {
    std::cout << "🏭 BAHA MIXED-INTEGER BENCHMARK: FACILITY LOCATION 🏭" << std::endl;
    // 10 Facilities, 50 Customers
    FacilityLocationProblem problem(10, 50, 12345);

    auto energy = [&](const MilpState& s) { return problem.energy(s); };
    auto sampler = [&]() { return problem.random_state(); };
    auto neighbors = [&](const MilpState& s) { return problem.neighbors(s); };

    navokoj::BranchAwareOptimizer<MilpState> baha(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<MilpState>::Config config;
    
    config.beta_steps = 2000;
    config.beta_end = 15.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.5;
    config.max_branches = 5;
    config.verbose = true;
    config.schedule_type = navokoj::BranchAwareOptimizer<MilpState>::ScheduleType::GEOMETRIC;

    auto result = baha.optimize(config);

    std::cout << "\n✅ Optimization Complete.\n";
    std::cout << "Best Energy (Weighted): " << result.best_energy << "\n";
    problem.print_summary(result.best_state);

    return 0;
}
