/*
 * Author: Sethurathienam Iyer
 */
#include "baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <set>
#include <algorithm>
#include <iomanip>

// =============================================================================
// COMBINATORIAL SPECTRUM AUCTION OPTIMIZATION
// =============================================================================
struct SpectrumAllocationState {
    std::vector<int> package_assignments; // package_assignments[i] = company_id that gets package i (-1 if unassigned)
};

class SpectrumAuctionProblem {
public:
    struct Package {
        int id;
        std::vector<int> frequency_bands;  // Which frequency bands are in this package
        double value;                      // How much the package is worth to whoever gets it
    };
    
    struct Company {
        int id;
        std::map<int, double> package_values;  // What each package is worth to this company
        std::set<int> interested_packages;     // Which packages this company is interested in
    };

    SpectrumAuctionProblem(int n_companies, int n_packages, int n_bands, int seed) 
        : n_companies_(n_companies), n_packages_(n_packages), n_bands_(n_bands), rng_(seed) {
        
        // Generate frequency bands for each package
        std::uniform_int_distribution<> band_dist(0, n_bands - 1);
        std::uniform_real_distribution<> value_dist(1000.0, 100000.0);  // Values in range [1000, 100000]
        
        // Create packages with random frequency bands
        for (int i = 0; i < n_packages; ++i) {
            Package pkg;
            pkg.id = i;
            
            // Each package contains 1-5 frequency bands
            int bands_in_pkg = 1 + (rng_() % 5);
            std::set<int> selected_bands;
            while (selected_bands.size() < bands_in_pkg) {
                selected_bands.insert(band_dist(rng_));
            }
            
            for (int band : selected_bands) {
                pkg.frequency_bands.push_back(band);
            }
            pkg.value = value_dist(rng_);
            packages_.push_back(pkg);
        }
        
        // Create companies and assign interest in packages
        std::uniform_int_distribution<> interest_dist(3, 10);  // Each company interested in 3-10 packages
        
        for (int c = 0; c < n_companies; ++c) {
            Company comp;
            comp.id = c;
            
            // Determine which packages this company is interested in
            int n_interested = interest_dist(rng_);
            std::set<int> interested_set;
            while (interested_set.size() < n_interested && interested_set.size() < n_packages_) {
                interested_set.insert(rng_() % n_packages_);
            }
            
            for (int pkg_id : interested_set) {
                // Company values this package with some random variation
                double base_value = packages_[pkg_id].value;
                double company_specific_value = base_value * (0.5 + (rng_() / (double)rng_.max()) * 0.8); // 0.5 to 1.3 times base value
                comp.package_values[pkg_id] = company_specific_value;
                comp.interested_packages.insert(pkg_id);
            }
            
            companies_.push_back(comp);
        }
        
        // Build interference constraints: which packages interfere with each other
        build_interference_graph();
    }

    double energy(const SpectrumAllocationState& state) const {
        double total_penalty = 0.0;
        double total_revenue = 0.0;
        
        // Calculate revenue from assignments
        for (int i = 0; i < n_packages_; ++i) {
            int company_id = state.package_assignments[i];
            if (company_id != -1 && company_id < n_companies_) {
                // Check if this company values this package
                auto it = companies_[company_id].package_values.find(i);
                if (it != companies_[company_id].package_values.end()) {
                    total_revenue += it->second;
                }
            }
        }
        
        // Apply penalties for constraint violations
        // 1. Interference penalties: if interfering packages assigned to different companies
        for (const auto& constraint : interference_constraints_) {
            int pkg1 = constraint.first;
            int pkg2 = constraint.second;
            
            int comp1 = state.package_assignments[pkg1];
            int comp2 = state.package_assignments[pkg2];
            
            // If both packages assigned to different companies, apply penalty
            if (comp1 != -1 && comp2 != -1 && comp1 != comp2) {
                total_penalty += 50000;  // Large penalty for interference violation
            }
        }
        
        // 2. Capacity penalties: if a company gets too many packages (simplified)
        std::vector<int> packages_per_company(n_companies_, 0);
        for (int i = 0; i < n_packages_; ++i) {
            int company_id = state.package_assignments[i];
            if (company_id != -1) {
                packages_per_company[company_id]++;
            }
        }
        
        for (int c = 0; c < n_companies_; ++c) {
            if (packages_per_company[c] > 10) {  // Simplified capacity constraint
                total_penalty += (packages_per_company[c] - 10) * 10000;  // Penalty for exceeding capacity
            }
        }
        
        // Return energy as negative revenue plus penalties
        // (Lower energy = higher revenue - we want to minimize energy)
        return -total_revenue + total_penalty;
    }

    SpectrumAllocationState random_state() const {
        SpectrumAllocationState state;
        state.package_assignments.resize(n_packages_);
        
        std::uniform_int_distribution<> company_dist(-1, n_companies_ - 1);  // -1 means unassigned
        
        // Assign each package randomly to a company or leave unassigned
        for (int i = 0; i < n_packages_; ++i) {
            state.package_assignments[i] = company_dist(rng_);
        }
        
        return state;
    }

    std::vector<SpectrumAllocationState> neighbors(const SpectrumAllocationState& state) const {
        std::vector<SpectrumAllocationState> nbrs;
        
        // Generate neighbors by changing assignment of one package
        for (int i = 0; i < n_packages_; ++i) {
            for (int c = -1; c < n_companies_; ++c) {  // -1 = unassigned
                if (c != state.package_assignments[i]) {
                    SpectrumAllocationState nbr = state;
                    nbr.package_assignments[i] = c;
                    nbrs.push_back(nbr);
                }
            }
        }
        
        return nbrs;
    }

    void print_solution(const SpectrumAllocationState& state) const {
        std::cout << "\nSOLUTION SUMMARY:\n";
        std::cout << "================\n";
        
        double total_revenue = 0.0;
        for (int i = 0; i < n_packages_; ++i) {
            int company_id = state.package_assignments[i];
            if (company_id != -1) {
                auto it = companies_[company_id].package_values.find(i);
                if (it != companies_[company_id].package_values.end()) {
                    std::cout << "Package " << i << " (bands: ";
                    for (int band : packages_[i].frequency_bands) {
                        std::cout << band << " ";
                    }
                    std::cout << ") -> Company " << company_id 
                              << " (value: $" << std::fixed << std::setprecision(2) << it->second << ")\n";
                    total_revenue += it->second;
                }
            }
        }
        
        std::cout << "\nTotal Revenue: $" << std::fixed << std::setprecision(2) << total_revenue << "\n";
        
        // Count constraint violations
        int violations = 0;
        for (const auto& constraint : interference_constraints_) {
            int pkg1 = constraint.first;
            int pkg2 = constraint.second;
            
            int comp1 = state.package_assignments[pkg1];
            int comp2 = state.package_assignments[pkg2];
            
            if (comp1 != -1 && comp2 != -1 && comp1 != comp2) {
                violations++;
            }
        }
        std::cout << "Interference Violations: " << violations << "\n";
    }

private:
    void build_interference_graph() {
        // Simplified interference model: packages that share frequency bands interfere
        for (int i = 0; i < n_packages_; ++i) {
            for (int j = i + 1; j < n_packages_; ++j) {
                // Check if packages i and j share any frequency bands
                std::set<int> bands_i(packages_[i].frequency_bands.begin(), packages_[i].frequency_bands.end());
                
                bool shares_band = false;
                for (int band : packages_[j].frequency_bands) {
                    if (bands_i.count(band) > 0) {
                        shares_band = true;
                        break;
                    }
                }
                
                if (shares_band) {
                    interference_constraints_.emplace_back(i, j);
                }
            }
        }
    }

    int n_companies_;
    int n_packages_;
    int n_bands_;
    std::vector<Package> packages_;
    std::vector<Company> companies_;
    std::vector<std::pair<int, int>> interference_constraints_;  // Pairs of packages that interfere
    mutable std::mt19937 rng_;
};

// =============================================================================
// MAIN RUNNER
// =============================================================================
int main() {
    std::cout << "📡 SPECTRUM AUCTION OPTIMIZATION USING BAHA 📡\n";
    std::cout << "Hunting for optimal allocation fractures...\n";
    std::cout << "==========================================\n\n";

    // Create a challenging auction: 10 companies, 30 packages, 50 frequency bands
    SpectrumAuctionProblem auction(10, 30, 50, 42);

    // Wrap in BAHA-compatible functions
    std::function<double(const SpectrumAllocationState&)> energy = 
        [&](const SpectrumAllocationState& s) { return auction.energy(s); };
    
    std::function<SpectrumAllocationState()> sampler = 
        [&]() { return auction.random_state(); };
    
    std::function<std::vector<SpectrumAllocationState>(const SpectrumAllocationState&)> neighbors = 
        [&](const SpectrumAllocationState& s) { return auction.neighbors(s); };

    // Configure BAHA
    typename navokoj::BranchAwareOptimizer<SpectrumAllocationState>::Config config;
    config.beta_steps = 2000;           // More steps for complex problem
    config.beta_end = 25.0;             // Higher beta for precision
    config.samples_per_beta = 75;       // More samples for accuracy
    config.fracture_threshold = 2.0;    // Adjusted for this problem type
    config.max_branches = 8;            // More branches for complex landscape
    config.verbose = true;              // Show progress
    config.schedule_type = navokoj::BranchAwareOptimizer<SpectrumAllocationState>::ScheduleType::GEOMETRIC;

    // Run optimization
    navokoj::BranchAwareOptimizer<SpectrumAllocationState> ba(energy, sampler, neighbors);
    std::cout << "Starting BAHA optimization...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = ba.optimize(config);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    std::cout << "\n🎯 OPTIMIZATION COMPLETE 🎯\n";
    std::cout << "Best energy found: " << result.best_energy << "\n";
    std::cout << "Time taken: " << duration << " ms\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Beta at solution: " << result.beta_at_solution << "\n\n";
    
    // Print detailed solution
    auction.print_solution(result.best_state);
    
    // Compare with random baseline
    std::cout << "\n📊 COMPARISON WITH RANDOM SOLUTIONS 📊\n";
    std::cout << "=====================================\n";
    
    double best_random_energy = 1e9;
    SpectrumAllocationState best_random_state;
    
    for (int i = 0; i < 100; ++i) {  // Try 100 random solutions
        auto random_state = auction.random_state();
        double random_energy = auction.energy(random_state);
        if (random_energy < best_random_energy) {
            best_random_energy = random_energy;
            best_random_state = random_state;
        }
    }
    
    std::cout << "Best random solution energy: " << best_random_energy << "\n";
    std::cout << "BAHA improvement: " << (best_random_energy - result.best_energy) << " energy units\n";
    std::cout << "Relative improvement: " << ((best_random_energy - result.best_energy) / std::abs(best_random_energy)) * 100.0 << "%\n\n";
    
    std::cout << "Random solution summary:\n";
    auction.print_solution(best_random_state);

    return 0;
}