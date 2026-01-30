/**
 * SPECTRAL MONSTER: C++ Benchmark for BAHA
 * 
 * Goal: Solve the "Monster" Graph (N=1300, p=0.314) using Log-Prime Weights.
 * Architecture: High-Performance C++ with Vectorized Energy.
 */

#include "baha/baha.hpp" // Assumes this header exists in include path
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <set>

// =============================================================================
// PARAMETERS
// =============================================================================
const int N_NODES = 1300;
const double EDGE_PROB = 0.314;
const int N_COLORS = 4;
const int DOMAIN_SIZE = 4; // Simplified: Fixed Domain Size 4
const int SEED = 999;

// =============================================================================
// HELPERS
// =============================================================================

// Sieve of Eratosthenes
std::vector<int> generate_primes(int n) {
    std::vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (int p = 2; p * p <= n; p++) {
        if (is_prime[p]) {
            for (int i = p * p; i <= n; i += p)
                is_prime[i] = false;
        }
    }
    std::vector<int> primes;
    for (int p = 2; p <= n; p++) {
        if (is_prime[p]) primes.push_back(p);
    }
    return primes;
}

// Graph Representation
struct Graph {
    int n;
    std::vector<std::pair<int, int>> edges; // Flat edge list for fast iteration
    std::vector<int> degrees;
    std::vector<double> node_weights;
};

// =============================================================================
// STATE & OPERATORS
// =============================================================================

// State is just a vector of colors
using State = std::vector<int>;

// Energy Function: This is the HOT LOOP
// E = Sum of (W[u] + W[v]) for all conflicting edges
double energy_spectral(const State& s, const Graph& G) {
    double total_penalty = 0.0;
    
    // Manual unrolling / vectorization opportunity for compiler
    const size_t m = G.edges.size();
    const auto* edge_data = G.edges.data();
    
    for (size_t i = 0; i < m; ++i) {
        int u = edge_data[i].first;
        int v = edge_data[i].second;
        
        // Branchless update if possible (though check is cheap)
        if (s[u] == s[v]) {
            total_penalty += (G.node_weights[u] + G.node_weights[v]);
        }
    }
    return total_penalty;
}

// Basic Sampler
State sampler() {
    State s(N_NODES);
    std::mt19937 rng(std::random_device{}()); // Use local rng to avoid static connection issues
    std::uniform_int_distribution<int> dist(0, N_COLORS - 1);
    for(int i=0; i<N_NODES; ++i) s[i] = dist(rng);
    return s;
}

// Neighbors: 1-flip neighborhood
std::vector<State> neighbors(const State& s) {
    std::vector<State> nbrs;
    nbrs.reserve(20); // Sample 20 neighbors like Python version
    
    // We need a deterministic RNG here for "fair" comparison or just fast local one
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> node_dist(0, N_NODES - 1);
    
    for(int k=0; k<20; ++k) {
        int node = node_dist(rng);
        int old_c = s[node];
        
        // Pick new color
        int new_c = (old_c + 1 + (rng() % (N_COLORS - 1))) % N_COLORS;
        
        State new_s = s;
        new_s[node] = new_c;
        nbrs.push_back(std::move(new_s));
    }
    return nbrs;
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::cout << "==========================================\n";
    std::cout << " SPECTRAL MONSTER (C++ Benchmark)\n";
    std::cout << "==========================================\n";
    
    // 1. GENERATE GRAPH
    std::cout << " Generating Graph N=" << N_NODES << ", p=" << EDGE_PROB << "...\n";
    Graph G;
    G.n = N_NODES;
    G.degrees.resize(N_NODES, 0);
    
    std::mt19937 rng(SEED);
    std::uniform_real_distribution<double> p_dist(0.0, 1.0);
    
    int edge_count = 0;
    for (int i = 0; i < N_NODES; ++i) {
        for (int j = i + 1; j < N_NODES; ++j) {
            if (p_dist(rng) < EDGE_PROB) {
                G.edges.push_back({i, j});
                G.degrees[i]++;
                G.degrees[j]++;
                edge_count++;
            }
        }
    }
    std::cout << " Edges: " << edge_count << " (Target ~265k)\n";
    
    // 2. ASSIGN SPECTRAL WEIGHTS
    std::cout << " Assigning Log-Prime Weights...\n";
    auto primes = generate_primes(12000); // Plenty
    
    // Sort nodes by degree descending
    std::vector<std::pair<int, int>> nodes_by_deg;
    for(int i=0; i<N_NODES; ++i) nodes_by_deg.push_back({i, G.degrees[i]});
    std::sort(nodes_by_deg.begin(), nodes_by_deg.end(), [](auto& a, auto& b){
        return a.second > b.second; // Descending
    });
    
    G.node_weights.resize(N_NODES);
    for(int rank=0; rank<N_NODES; ++rank) {
        int node = nodes_by_deg[rank].first;
        int p = primes[rank];
        // KEY INNOVATION: 1.0 / log(p)
        G.node_weights[node] = 1.0 / std::log((double)p);
    }
    
    std::cout << " Top Weight (Deg " << nodes_by_deg[0].second << "): " 
              << G.node_weights[nodes_by_deg[0].first] << "\n";
    std::cout << " Bot Weight (Deg " << nodes_by_deg.back().second << "): " 
              << G.node_weights[nodes_by_deg.back().first] << "\n";
              
    // 3. CONFIGURE BAHA
    using namespace navokoj;
    
    // Bind energy function
    auto energy_fn = [&](const State& s) { return energy_spectral(s, G); };
    
    // Instantiate Optimizer
    BranchAwareOptimizer<State> optimizer(energy_fn, sampler, neighbors);
    
    BranchAwareOptimizer<State>::Config config;
    config.beta_start = 0.01;
    config.beta_end = 10.0;
    config.beta_steps = 1000; // More steps because C++ is fast!
    config.samples_per_beta = 50; // Proper sampling now possible
    config.timeout_ms = 15000.0; // 15s Wall Clock Limit (Same as Python)
    config.verbose = false; // Keep it clean
    
    std::cout << " Initializing BAHA (C++ Native)...\n";
    auto result = optimizer.optimize(config);
    
    // 4. REPORT
    int raw_conflicts = 0;
    for(auto& e : G.edges) {
        if (result.best_state[e.first] == result.best_state[e.second])
            raw_conflicts++;
    }
    
    std::cout << "\n==========================================\n";
    std::cout << " RESULTS\n";
    std::cout << "==========================================\n";
    std::cout << " Time Taken: " << result.time_ms / 1000.0 << " s\n";
    std::cout << " Steps Taken: " << result.steps_taken << "\n";
    std::cout << " Final Energy: " << result.best_energy << "\n";
    std::cout << " RAW Conflicts: " << raw_conflicts << "\n";
    std::cout << " Branch Jumps: " << result.branch_jumps << "\n";
    std::cout << " Fractures: " << result.fractures_detected << "\n";
    
    if (raw_conflicts == 0) {
        std::cout << "\n✅ SUCCESS: C++ Spectral BAHA Tamed the Monster!\n";
        return 0;
    } else {
        std::cout << "\n❌ FAILURE: " << raw_conflicts << " conflicts remain.\n";
        return 1;
    }
}
