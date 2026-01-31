#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "baha/baha.hpp"

struct GridState {
    std::vector<int> shed; // 0 = keep, 1 = shed
};

enum LoadType { INDUSTRIAL, COMMERCIAL, RESIDENTIAL, CRITICAL };

struct Load {
    std::string name;
    double mw;          // Megawatts
    LoadType type;
    double priority;    // Higher = more critical (1-10)
    std::vector<int> neighbors; // Adjacent loads (cascade risk)
};

struct Scenario {
    double capacity_deficit_mw; // How much to shed
    int stress_level;           // 1=low, 2=medium, 3=high
    int hour;                   // 0-23 (affects load profiles)
};

struct ConstraintCheck {
    std::string name;
    bool satisfied;
    bool hard;
    std::string note;
};

struct ConstraintResult {
    int hard_violations = 0;
    double soft_penalty = 0.0;
    std::vector<ConstraintCheck> checks;
};

static const std::vector<Load> kLoads = {
    {"Hospital complex", 8.5, CRITICAL, 10.0, {1, 3}},
    {"Water treatment", 5.2, CRITICAL, 10.0, {0, 2}},
    {"Steel mill A", 45.0, INDUSTRIAL, 3.0, {1, 4, 5}},
    {"Data center", 12.0, COMMERCIAL, 7.0, {0, 5}},
    {"Steel mill B", 48.0, INDUSTRIAL, 3.0, {2, 6}},
    {"Shopping mall", 6.5, COMMERCIAL, 4.0, {2, 3, 7}},
    {"Manufacturing plant", 32.0, INDUSTRIAL, 3.5, {4, 8}},
    {"Residential zone A", 18.0, RESIDENTIAL, 5.0, {5, 9}},
    {"Chemical plant", 38.0, INDUSTRIAL, 4.0, {6, 10}},
    {"Residential zone B", 22.0, RESIDENTIAL, 5.0, {7, 11}},
    {"Warehouse district", 15.0, COMMERCIAL, 3.0, {8, 12}},
    {"Residential zone C", 20.0, RESIDENTIAL, 5.0, {9, 13}},
    {"Auto assembly", 42.0, INDUSTRIAL, 3.5, {10, 14}},
    {"Office complex", 9.0, COMMERCIAL, 4.5, {11}},
    {"Residential zone D", 16.0, RESIDENTIAL, 5.0, {12}},
};

static Scenario gScenario{80.0, 2, 14}; // 80 MW deficit, medium stress, 2pm

static ConstraintResult evaluate_constraints(const GridState& s) {
    ConstraintResult result;
    
    double total_shed = 0.0;
    for (size_t i = 0; i < kLoads.size(); ++i) {
        if (s.shed[i]) total_shed += kLoads[i].mw;
    }
    
    // Hard constraints
    {
        bool ok = true;
        for (size_t i = 0; i < kLoads.size(); ++i) {
            if (kLoads[i].type == CRITICAL && s.shed[i]) {
                ok = false;
                break;
            }
        }
        if (!ok) result.hard_violations++;
        result.checks.push_back({"Never shed critical loads (hospital, water)", ok, true,
                                 "Life-safety infrastructure must remain operational."});
    }
    {
        const double target = gScenario.capacity_deficit_mw;
        const bool ok = (total_shed >= target * 0.95 && total_shed <= target * 1.15);
        if (!ok) result.hard_violations++;
        result.checks.push_back({"Meet capacity deficit target (±15%)", ok, true,
                                 "Must shed enough to stabilize grid."});
    }
    
    // Soft constraints
    {
        int industrial_count = 0;
        for (size_t i = 0; i < kLoads.size(); ++i) {
            if (s.shed[i] && kLoads[i].type == INDUSTRIAL) industrial_count++;
        }
        const bool ok = industrial_count <= 2;
        if (!ok) result.soft_penalty += 15.0;
        result.checks.push_back({"Limit industrial shedding (≤2 plants)", ok, false,
                                 "Minimize economic impact of industrial shutdowns."});
    }
    {
        int cascade_risk = 0;
        for (size_t i = 0; i < kLoads.size(); ++i) {
            if (!s.shed[i]) continue;
            for (int nbr : kLoads[i].neighbors) {
                if (s.shed[nbr]) cascade_risk++;
            }
        }
        const bool ok = cascade_risk <= 2;
        if (!ok) result.soft_penalty += 20.0;
        result.checks.push_back({"Avoid shedding adjacent loads (cascade risk)", ok, false,
                                 "Neighboring shed loads create voltage instability."});
    }
    {
        int residential_count = 0;
        for (size_t i = 0; i < kLoads.size(); ++i) {
            if (s.shed[i] && kLoads[i].type == RESIDENTIAL) residential_count++;
        }
        const bool ok = residential_count <= 2;
        if (!ok) result.soft_penalty += 12.0;
        result.checks.push_back({"Limit residential shedding (≤2 zones)", ok, false,
                                 "Minimize public impact during peak hours."});
    }
    {
        bool commercial_during_business = false;
        for (size_t i = 0; i < kLoads.size(); ++i) {
            if (s.shed[i] && kLoads[i].type == COMMERCIAL && 
                gScenario.hour >= 8 && gScenario.hour <= 18) {
                commercial_during_business = true;
                break;
            }
        }
        const bool ok = !commercial_during_business;
        if (!ok) result.soft_penalty += 10.0;
        result.checks.push_back({"Avoid commercial shedding during business hours", ok, false,
                                 "Business operations heavily impacted during daytime."});
    }
    
    return result;
}

static double energy(const GridState& s) {
    double total_unserved = 0.0;
    double priority_weighted_loss = 0.0;
    double cascade_penalty = 0.0;
    
    for (size_t i = 0; i < kLoads.size(); ++i) {
        if (s.shed[i]) {
            total_unserved += kLoads[i].mw;
            priority_weighted_loss += kLoads[i].mw * kLoads[i].priority;
        }
    }
    
    // Cascade interaction model: shedding adjacent loads creates instability
    for (size_t i = 0; i < kLoads.size(); ++i) {
        if (!s.shed[i]) continue;
        int adjacent_shed = 0;
        for (int nbr : kLoads[i].neighbors) {
            if (s.shed[nbr]) adjacent_shed++;
        }
        // Nonlinear penalty for cascade risk
        if (adjacent_shed > 0) {
            cascade_penalty += 15.0 * adjacent_shed * adjacent_shed;
        }
    }
    
    // Deficit penalty: strongly penalize not meeting the target
    double total_shed = 0.0;
    for (size_t i = 0; i < kLoads.size(); ++i) {
        if (s.shed[i]) total_shed += kLoads[i].mw;
    }
    const double target = gScenario.capacity_deficit_mw;
    const double deficit_error = std::abs(total_shed - target);
    const double deficit_penalty = (deficit_error > target * 0.15) ? 
                                   deficit_error * 50.0 : 0.0;
    
    // Grid stress modifier (higher stress = higher penalty for each MW shed)
    const double stress_multiplier = 0.8 + 0.2 * gScenario.stress_level;
    
    const ConstraintResult constraints = evaluate_constraints(s);
    
    // Weighted energy: priority loss + cascade risk + deficit penalty + constraint violations
    const double hard_penalty = constraints.hard_violations * 500.0;
    
    return stress_multiplier * priority_weighted_loss + 
           cascade_penalty + 
           deficit_penalty +
           hard_penalty + 
           constraints.soft_penalty;
}

static GridState sampler() {
    static std::mt19937 rng(42);
    std::bernoulli_distribution pick(0.3);
    GridState s;
    s.shed.resize(kLoads.size());
    for (size_t i = 0; i < kLoads.size(); ++i) {
        s.shed[i] = pick(rng) ? 1 : 0;
    }
    return s;
}

static std::vector<GridState> neighbors(const GridState& s) {
    std::vector<GridState> nbrs;
    nbrs.reserve(s.shed.size());
    for (size_t i = 0; i < s.shed.size(); ++i) {
        GridState n = s;
        n.shed[i] = 1 - n.shed[i];
        nbrs.push_back(std::move(n));
    }
    return nbrs;
}

static void print_state(const GridState& s) {
    double total_shed = 0.0;
    std::cout << "Load shedding plan:\n";
    for (size_t i = 0; i < kLoads.size(); ++i) {
        if (s.shed[i]) {
            std::cout << "  ❌ SHED: " << kLoads[i].name 
                      << " (" << kLoads[i].mw << " MW, priority=" 
                      << kLoads[i].priority << ")\n";
            total_shed += kLoads[i].mw;
        }
    }
    std::cout << "\nLoads kept online:\n";
    for (size_t i = 0; i < kLoads.size(); ++i) {
        if (!s.shed[i]) {
            std::cout << "  ✅ KEEP: " << kLoads[i].name 
                      << " (" << kLoads[i].mw << " MW, priority=" 
                      << kLoads[i].priority << ")\n";
        }
    }
    std::cout << "\nTotal shed: " << std::fixed << std::setprecision(1) 
              << total_shed << " MW (target: " << gScenario.capacity_deficit_mw << " MW)\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "SMART GRID LOAD SHEDDING OPTIMIZER (BAHA)\n";
    std::cout << "============================================================\n\n";

    std::cout << "Scenario:\n";
    std::cout << "  Capacity deficit: " << gScenario.capacity_deficit_mw << " MW\n";
    std::cout << "  Grid stress level: " << gScenario.stress_level << " (1=low, 2=med, 3=high)\n";
    std::cout << "  Time: " << gScenario.hour << ":00 (hour of day)\n\n";

    navokoj::BranchAwareOptimizer<GridState> opt(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<GridState>::Config config;
    config.beta_start = 0.01;
    config.beta_end = 8.0;
    config.beta_steps = 250;
    config.samples_per_beta = 80;
    config.max_branches = 8;
    config.fracture_threshold = 1.8;
    config.schedule_type = navokoj::BranchAwareOptimizer<GridState>::ScheduleType::GEOMETRIC;

    auto result = opt.optimize(config);

    std::cout << "Result:\n";
    std::cout << "Best energy: " << std::fixed << std::setprecision(3) << result.best_energy << "\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(3) << result.time_ms / 1000.0 << "s\n\n";

    print_state(result.best_state);

    const ConstraintResult constraint_result = evaluate_constraints(result.best_state);

    std::cout << "\nConstraint satisfaction report:\n";
    for (const auto& check : constraint_result.checks) {
        std::cout << "  " << (check.satisfied ? "✅" : "❌") << " "
                  << (check.hard ? "[HARD] " : "[SOFT] ")
                  << check.name << " — " << check.note << "\n";
    }
    std::cout << "\nSummary:\n";
    std::cout << "  Hard violations: " << constraint_result.hard_violations << "\n";
    std::cout << "  Soft penalty: " << std::fixed << std::setprecision(1)
              << constraint_result.soft_penalty << "\n\n";

    std::cout << "Interpretation:\n";
    std::cout << "- Fractures indicate cascade tipping points (voltage collapse boundaries).\n";
    std::cout << "- Branch jumps correspond to switching load shedding regimes.\n";
    std::cout << "- Critical loads (hospital, water) must never be shed.\n";
    std::cout << "- Adjacent loads create nonlinear cascade risk (voltage instability).\n";

    return 0;
}
