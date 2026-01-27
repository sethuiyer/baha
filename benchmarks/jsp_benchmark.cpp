#include "baha.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <map>
#include <iomanip>

// ============================================================================
// JOB SHOP SCHEDULING PROBLEM (JSP)
// ============================================================================
// Objective: Minimize Makespan (Total time to complete all jobs)
// Input: 
//   - N jobs, M machines.
//   - Each job has M operations, to be performed in a specific order.
//   - Each operation requires a specific machine and has a fixed duration.
// Constraint:
//   - A machine can process only one operation at a time.
//   - Operations of a job must be processed in the given sequence.

struct JobOperation {
    int job_id;
    int op_id; // 0 to M-1
    int machine_id;
    int duration;
};

// Problem Instance Data
class JSPInstance {
public:
    int n_jobs;
    int n_machines;
    // jobs[j][k] is the k-th operation of job j
    std::vector<std::vector<JobOperation>> jobs;

    JSPInstance(int nj, int nm, int seed) : n_jobs(nj), n_machines(nm) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<> duration_dist(10, 100);
        
        jobs.resize(n_jobs);
        for(int j=0; j<n_jobs; ++j) {
            // Generate a random machine sequence for this job
            std::vector<int> machines(n_machines);
            for(int m=0; m<n_machines; ++m) machines[m] = m;
            std::shuffle(machines.begin(), machines.end(), rng);

            for(int k=0; k<n_machines; ++k) {
                JobOperation op;
                op.job_id = j;
                op.op_id = k;
                op.machine_id = machines[k];
                op.duration = duration_dist(rng);
                jobs[j].push_back(op);
            }
        }
    }
};

// State Representation: 
// "Operation-based Representation"
// A permutation of size N_JOBS * N_MACHINES.
// The sequence contains each job ID exactly N_MACHINES times.
// The k-th occurrence of job J in the sequence refers to the k-th operation of job J.
// This ensures all precedence constraints are naturally handled (we just schedule them in order of appearance).
using JSPState = std::vector<int>;

class JSPSolver {
    const JSPInstance& instance;
    
public:
    JSPSolver(const JSPInstance& inst) : instance(inst) {}

    JSPState initial_state() {
        JSPState state;
        for(int j=0; j<instance.n_jobs; ++j) {
            for(int m=0; m<instance.n_machines; ++m) {
                state.push_back(j);
            }
        }
        std::mt19937 rng(42);
        std::shuffle(state.begin(), state.end(), rng);
        return state;
    }

    double calculate_makespan(const JSPState& state) {
        // Track when each machine becomes free
        std::vector<int> machine_free_time(instance.n_machines, 0);
        // Track when each job's last operation finished
        std::vector<int> job_next_available_time(instance.n_jobs, 0);
        // Track which operation index (0..M-1) we are on for each job
        std::vector<int> job_op_index(instance.n_jobs, 0);

        for (int job_id : state) {
            int op_idx = job_op_index[job_id];
            const auto& op = instance.jobs[job_id][op_idx];
            
            // Start time is max of:
            // 1. When the machine is free
            // 2. When the job's previous operation finished
            int start_time = std::max(machine_free_time[op.machine_id], job_next_available_time[job_id]);
            int end_time = start_time + op.duration;

            // Update trackers
            machine_free_time[op.machine_id] = end_time;
            job_next_available_time[job_id] = end_time;
            job_op_index[job_id]++;
        }

        // Makespan is the max completion time across all machines (or jobs)
        int makespan = 0;
        for(int t : machine_free_time) makespan = std::max(makespan, t);
        return (double)makespan;
    }

    // Neighbors: Swap adjacent elements in the permutation
    std::vector<JSPState> get_neighbors(const JSPState& state) {
        std::vector<JSPState> nbrs;
        // Generate a random subset of swaps to keep branching manageable
        // In full local search you'd check Critical Path logic (swapping non-critical is useless).
        // Since we want BAHA to DISCOVER this, we'll just use random adjacent swaps.
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<> dist(0, state.size() - 2);
        
        for(int k=0; k<15; ++k) { // Generate 15 neighbors
            JSPState n = state;
            int i = dist(rng);
            // Only swap if they are different jobs (swapping same job id does nothing in this representation)
             if (n[i] != n[i+1]) {
                std::swap(n[i], n[i+1]);
                nbrs.push_back(n);
            }
        }
        return nbrs;
    }
};

int main() {
    std::cout << "🏭 JOB SHOP SCHEDULING (JSP) BENCHMARK 🏭" << std::endl;
    std::cout << "Minimizing Makespan (Time) using BAHA." << std::endl;
    std::cout << "Hypothesis: Fractures occur when the Critical Path shifts." << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // Hard Instance: 15 Jobs, 15 Machines
    // This is moderately large (225 operations).
    JSPInstance problem(15, 15, 12345);
    JSPSolver solver(problem);

    auto energy = [&](const JSPState& s) { return solver.calculate_makespan(s); };
    auto sampler = [&]() { return solver.initial_state(); };
    auto neighbors = [&](const JSPState& s) { return solver.get_neighbors(s); };

    // BAHA Configuration
    navokoj::BranchAwareOptimizer<JSPState> baha(energy, sampler, neighbors);
    navokoj::BranchAwareOptimizer<JSPState>::Config config;
    config.beta_steps = 1000;
    config.beta_end = 10.0; // Makespan values are large, but differences are small (integers).
    // Actually, Energy is ~2000. Delta E is ~10-50. Beta=10 might be too high? 
    // If Delta E = 10, exp(-Beta*Delta) = exp(-100) -> 0.
    // We need Beta * Delta ~ 1-5 for transitions. So Beta should be ~ 0.1 to 0.5.
    // Let's set Beta End to 1.0.
    config.beta_end = 1.0; 
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.5;
    config.max_branches = 5;
    config.verbose = true;
    config.schedule_type = navokoj::BranchAwareOptimizer<JSPState>::ScheduleType::GEOMETRIC;
    config.logger = [](int step, double beta, double energy, double rho, const char* event) {
        // Minimal logging to stdout if needed, or rely on verbose
    };

    std::cout << "Running BAHA..." << std::endl;
    auto result = baha.optimize(config);

    std::cout << "\nRESULT:" << std::endl;
    std::cout << "Best Makespan: " << result.best_energy << std::endl;
    std::cout << "Fractures: " << result.fractures_detected << std::endl;
    std::cout << "Jumps: " << result.branch_jumps << std::endl;

    // Simple Comparison with Random Sampling
    std::cout << "\nRunning Random Sampling Baseline (10,000 samples)..." << std::endl;
    double best_random = 1e9;
    for(int i=0; i<10000; ++i) {
        auto s = solver.initial_state();
        double e = solver.calculate_makespan(s);
        if(e < best_random) best_random = e;
    }
    std::cout << "Best Random Makespan: " << best_random << std::endl;
    std::cout << "Improvement: " << (best_random - result.best_energy) << " units (" 
              << ((best_random - result.best_energy)/best_random * 100.0) << "%)" << std::endl;

    return 0;
}
