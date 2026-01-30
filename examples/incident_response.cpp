#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "baha/baha.hpp"

struct IRState {
    std::vector<int> actions; // 0/1
};

struct Action {
    std::string name;
    double risk_reduction;
    double disruption_cost;
};

struct Scenario {
    int severity;          // 1=low, 2=medium, 3=high
    bool exfil_suspected;  // data exfiltration suspected
    int criticality;       // 1=low, 2=medium, 3=high
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

static const std::vector<Action> kActions = {
    {"Isolate web tier", 18.0, 10.0},
    {"Isolate DB tier", 22.0, 18.0},
    {"Rotate service keys", 14.0, 6.0},
    {"Block SMB (445)", 12.0, 3.0},
    {"Disable VPN access", 16.0, 14.0},
    {"Enforce MFA everywhere", 20.0, 8.0},
    {"Reset admin tokens", 15.0, 7.0},
    {"Block C2 domains", 17.0, 4.0},
    {"Deploy EDR containment", 24.0, 12.0},
    {"Disable lateral movement", 19.0, 9.0},
};

static Scenario gScenario{2, true, 2};

static ConstraintResult evaluate_constraints(const IRState& s) {
    ConstraintResult result;
    const bool isolate_web = s.actions[0] == 1;
    const bool isolate_db = s.actions[1] == 1;
    const bool rotate_keys = s.actions[2] == 1;
    const bool block_smb = s.actions[3] == 1;
    const bool disable_vpn = s.actions[4] == 1;
    const bool enforce_mfa = s.actions[5] == 1;
    const bool block_c2 = s.actions[7] == 1;
    const bool deploy_edr = s.actions[8] == 1;
    const bool disable_lateral = s.actions[9] == 1;

    // Hard constraints
    {
        const bool ok = (enforce_mfa || disable_vpn);
        if (!ok) result.hard_violations++;
        result.checks.push_back({"Auth containment (MFA or VPN disabled)", ok, true,
                                 "Prevents credential reuse during incident."});
    }
    {
        const bool ok = (!gScenario.exfil_suspected) || (block_c2 && disable_lateral);
        if (!ok) result.hard_violations++;
        result.checks.push_back({"Exfil suspected => Block C2 + lateral movement", ok, true,
                                 "Stops data exfiltration paths."});
    }
    {
        const bool ok = (!deploy_edr) || rotate_keys;
        if (!ok) result.hard_violations++;
        result.checks.push_back({"EDR requires key rotation", ok, true,
                                 "Containment without credential reset is brittle."});
    }

    // Soft constraints (penalties)
    {
        const bool ok = !(isolate_db && !isolate_web);
        if (!ok) result.soft_penalty += 15.0;
        result.checks.push_back({"Avoid DB isolation without web isolation", ok, false,
                                 "Prevents partial outages and cascading failures."});
    }
    {
        const bool ok = !(isolate_web && isolate_db && gScenario.severity < 3);
        if (!ok) result.soft_penalty += 12.0;
        result.checks.push_back({"Avoid full isolation unless severity is high", ok, false,
                                 "Limits business impact for medium incidents."});
    }
    {
        const bool ok = !(disable_vpn && gScenario.criticality == 3 && !enforce_mfa);
        if (!ok) result.soft_penalty += 10.0;
        result.checks.push_back({"High criticality => don't disable VPN without MFA", ok, false,
                                 "Keeps critical access available with safeguards."});
    }
    {
        const bool ok = !(gScenario.severity >= 2 && !block_smb);
        if (!ok) result.soft_penalty += 6.0;
        result.checks.push_back({"Severity ≥ medium => block SMB", ok, false,
                                 "Reduces ransomware propagation risk."});
    }
    {
        const bool ok = !(gScenario.severity >= 2 && !rotate_keys);
        if (!ok) result.soft_penalty += 6.0;
        result.checks.push_back({"Severity ≥ medium => rotate keys", ok, false,
                                 "Limits credential replay after compromise."});
    }

    return result;
}

static double energy(const IRState& s) {
    const double base_risk = 110.0 + 5.0 * gScenario.severity;
    double risk = base_risk;
    double disruption = 0.0;

    for (size_t i = 0; i < kActions.size(); ++i) {
        if (s.actions[i]) {
            risk -= kActions[i].risk_reduction;
            disruption += kActions[i].disruption_cost;
        }
    }

    // Interaction penalties (tipping points / infeasibility boundaries)
    const bool isolate_web = s.actions[0] == 1;
    const bool isolate_db = s.actions[1] == 1;
    const bool disable_vpn = s.actions[4] == 1;
    const bool enforce_mfa = s.actions[5] == 1;
    const bool deploy_edr = s.actions[8] == 1;

    // If DB is isolated but web is not, service disruption spikes.
    if (isolate_db && !isolate_web) {
        disruption += 20.0;
    }

    // Disabling VPN without MFA increases residual risk.
    if (disable_vpn && !enforce_mfa) {
        risk += 12.0;
    }

    // EDR is most effective when lateral movement is disabled.
    if (deploy_edr && s.actions[9] == 0) {
        risk += 8.0;
    }

    // Clamp risk to non-negative.
    risk = std::max(0.0, risk);

    const ConstraintResult constraints = evaluate_constraints(s);

    // Weighted sum: prioritize risk containment, but penalize outages.
    const double risk_weight = 1.0 + 0.25 * gScenario.severity;
    const double disruption_weight = 0.7 + 0.25 * gScenario.criticality;
    const double hard_penalty = constraints.hard_violations * 50.0;

    return risk_weight * risk + disruption_weight * disruption +
           hard_penalty + constraints.soft_penalty;
}

static IRState sampler() {
    static std::mt19937 rng(1337);
    std::bernoulli_distribution pick(0.4);
    IRState s;
    s.actions.resize(kActions.size());
    for (size_t i = 0; i < kActions.size(); ++i) {
        s.actions[i] = pick(rng) ? 1 : 0;
    }
    return s;
}

static std::vector<IRState> neighbors(const IRState& s) {
    std::vector<IRState> nbrs;
    nbrs.reserve(s.actions.size());
    for (size_t i = 0; i < s.actions.size(); ++i) {
        IRState n = s;
        n.actions[i] = 1 - n.actions[i];
        nbrs.push_back(std::move(n));
    }
    return nbrs;
}

static void print_state(const IRState& s) {
    std::cout << "Selected actions:\n";
    for (size_t i = 0; i < kActions.size(); ++i) {
        if (s.actions[i]) {
            std::cout << "  - " << kActions[i].name << "\n";
        }
    }
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "AI INCIDENT RESPONSE PLAYBOOK OPTIMIZATION (BAHA)\n";
    std::cout << "============================================================\n\n";

    std::cout << "Scenario:\n";
    std::cout << "  Severity: " << gScenario.severity << " (1=low, 2=med, 3=high)\n";
    std::cout << "  Exfil suspected: " << (gScenario.exfil_suspected ? "yes" : "no") << "\n";
    std::cout << "  Service criticality: " << gScenario.criticality
              << " (1=low, 2=med, 3=high)\n\n";

    navokoj::BranchAwareOptimizer<IRState> opt(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<IRState>::Config config;
    config.beta_start = 0.01;
    config.beta_end = 6.0;
    config.beta_steps = 200;
    config.samples_per_beta = 60;
    config.max_branches = 6;
    config.fracture_threshold = 1.5;
    config.schedule_type = navokoj::BranchAwareOptimizer<IRState>::ScheduleType::GEOMETRIC;

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
    std::cout << "- Risk is minimized by strong containment actions.\n";
    std::cout << "- Disruption penalties discourage over-isolation.\n";
    std::cout << "- Fractures indicate containment vs outage tipping points.\n";

    return 0;
}
