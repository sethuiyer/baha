/**
 * HYBRID MONSTER: C++ Benchmark for BAHA (MCMC + Adafactor)
 * 
 * Goal: Solve the "Monster" Graph using a Hybrid Strategy.
 * 1. MCMC (Spectral SA) finds a structural skeleton (Satisfied Subgraph).
 * 2. Adafactor optimizes unsatisfied regions, treating skeleton as Manifold Constraints.
 */

#include "baha/baha.hpp"
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
const int SEED = 42;

// =============================================================================
// GLOBAL GRAPH & WEIGHTS
// =============================================================================
struct Graph {
    int n;
    std::vector<std::pair<int, int>> edges;
    std::vector<double> node_weights;
    std::vector<int> degrees;
};

Graph G;

// MCMC Helper Logic
std::vector<int> generate_primes(int n) {
    std::vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (int p = 2; p * p <= n; p++) if (is_prime[p]) for (int i = p * p; i <= n; i += p) is_prime[i] = false;
    std::vector<int> primes;
    for (int p = 2; p <= n; p++) if (is_prime[p]) primes.push_back(p);
    return primes;
}

// =============================================================================
// PHASE 1: MCMC (SPECTRAL SA)
// =============================================================================
using MCMCState = std::vector<int>;

double mcmc_energy(const MCMCState& s) {
    double E = 0.0;
    for(const auto& edge : G.edges) {
        if(s[edge.first] == s[edge.second]) {
            E += (G.node_weights[edge.first] + G.node_weights[edge.second]);
        }
    }
    return E;
}

MCMCState mcmc_sampler() {
    MCMCState s(N_NODES);
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, N_COLORS-1);
    for(int i=0; i<N_NODES; ++i) s[i] = dist(rng);
    return s;
}

std::vector<MCMCState> mcmc_neighbors(const MCMCState& s) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::vector<MCMCState> nbrs;
    for(int k=0; k<10; ++k) { // Small neighborhood for speed
        int node = rng() % N_NODES;
        int new_c = (s[node] + 1 + (rng() % (N_COLORS - 1))) % N_COLORS;
        MCMCState next = s;
        next[node] = new_c;
        nbrs.push_back(next);
    }
    return nbrs;
}

// =============================================================================
// PHASE 2: ADAFACTOR (CONTINUOUS)
// =============================================================================
using ContinuousState = std::vector<double>;

// Helper: Convert Discrete -> Continuous One-Hot
ContinuousState encode(const MCMCState& s, double noise_scale=0.01) {
    ContinuousState x(N_NODES * N_COLORS, 0.0);
    static thread_local std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> noise(0.0, noise_scale);
    
    for(int i=0; i<N_NODES; ++i) {
        for(int c=0; c<N_COLORS; ++c) {
            double val = (s[i] == c) ? 0.9 : 0.1 / (N_COLORS-1);
            val += noise(rng);
            if(val < 0.001) val = 0.001;
            if(val > 0.999) val = 0.999;
            x[i*N_COLORS + c] = val;
        }
    }
    return x;
}

// Helpr: Convert Continuous -> Discrete
MCMCState decode(const ContinuousState& x) {
    MCMCState s(N_NODES);
    for(int i=0; i<N_NODES; ++i) {
        int best_c = 0;
        double max_p = -1.0;
        for(int c=0; c<N_COLORS; ++c) {
            if(x[i*N_COLORS + c] > max_p) {
                max_p = x[i*N_COLORS + c];
                best_c = c;
            }
        }
        s[i] = best_c;
    }
    return s;
}

// Adafactor Energy & Gradient
// KEY INNOVATION: Constraint Mask
// satisfied_mask[i] = true if node i is "Stable" from MCMC
std::vector<bool> stable_mask; 

double adafactor_energy(const ContinuousState& x) {
    double E = 0.0;
    // Conflict Energy on Unsatisfied Edges
    // Penalty Energy on deviation from Stable Skeleton
    
    // 1. Edge Conflicts
    for(const auto& edge : G.edges) {
        int u = edge.first;
        int v = edge.second;
        double w = G.node_weights[u] + G.node_weights[v];
        
        double dot = 0.0;
        for(int c=0; c<N_COLORS; ++c) 
            dot += x[u*N_COLORS+c] * x[v*N_COLORS+c];
            
        E += w * dot; // Minimize overlap
    }
    
    // 2. Stability Anchor ( prevent null collapse )
    // If node i is stable, force x[i] to stay near 1.0 (implied)
    // Actually simpler: Enforce Simplex Constraint strictly via penalty
    double simplex_penalty = 0.0;
    for(int i=0; i<N_NODES; ++i) {
        double sum = 0.0;
        for(int c=0; c<N_COLORS; ++c) sum += x[i*N_COLORS+c];
        simplex_penalty += (sum - 1.0)*(sum - 1.0);
    }
    
    return E + 10.0 * simplex_penalty;
}

ContinuousState adafactor_gradient(const ContinuousState& x) {
    ContinuousState g(x.size(), 0.0);
    
    // 1. Edge Gradients
    for(const auto& edge : G.edges) {
        int u = edge.first;
        int v = edge.second;
        double w = G.node_weights[u] + G.node_weights[v];
        
        for(int c=0; c<N_COLORS; ++c) {
            // d(UV)/dU = V
            g[u*N_COLORS+c] += w * x[v*N_COLORS+c]; 
            g[v*N_COLORS+c] += w * x[u*N_COLORS+c];
        }
    }
    
    // 2. Simplex Constraint Gradient
    // d/dx (sum - 1)^2 = 2(sum-1) * 1
    for(int i=0; i<N_NODES; ++i) {
        double sum = 0.0;
        for(int c=0; c<N_COLORS; ++c) sum += x[i*N_COLORS+c];
        double term = 2.0 * 10.0 * (sum - 1.0);
        
        // Apply gradient to ALL nodes
        for(int c=0; c<N_COLORS; ++c) g[i*N_COLORS+c] += term;
    }
    
    // 3. STABILITY ANCHOR (The Plot Twist)
    // If node is stable, massively penalize changing it? 
    // Actually, simply by initializing from MCMC result (encode), 
    // Adafactor starts in a valley. We just need to ensure it doesn't leave the valley for the null state.
    // The Simplex Constraint above prevents the null state.
    
    return g;
}

// =============================================================================
// MAIN HYBRID LOOP
// =============================================================================

int main() {
    std::cout << "==========================================\n";
    std::cout << " HYBRID MONSTER (MCMC + Adafactor)\n";
    std::cout << "==========================================\n";
    
    // 1. Setup Graph & Weights
    G.n = N_NODES;
    G.degrees.resize(N_NODES, 0);
    std::mt19937 rng(SEED);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for(int i=0; i<N_NODES; ++i) for(int j=i+1; j<N_NODES; ++j) 
        if(dist(rng) < EDGE_PROB) {
            G.edges.push_back({i,j});
            G.degrees[i]++; G.degrees[j]++;
        }
    
    auto primes = generate_primes(12000);
    std::vector<std::pair<int,int>> nodes_sorted;
    for(int i=0; i<N_NODES; ++i) nodes_sorted.push_back({i, G.degrees[i]});
    std::sort(nodes_sorted.begin(), nodes_sorted.end(), [](auto& a, auto& b){return a.second > b.second;});
    G.node_weights.resize(N_NODES);
    for(int r=0; r<N_NODES; ++r) G.node_weights[nodes_sorted[r].first] = 1.0 / std::log((double)primes[r]);
    
    // 2. Hybrid Init
    MCMCState current_disc = mcmc_sampler();
    
    // 3. Loop
    for(int cycle=0; cycle<3; ++cycle) {
        std::cout << "\n--- CYCLE " << cycle << " ---\n";
        
        // PHASE A: MCMC
        std::cout << "[MCMC] exploring structure...\n";
        using namespace navokoj;
        auto mcmc_e = [&](const MCMCState& s) { return mcmc_energy(s); };
        BranchAwareOptimizer<MCMCState> mcmc_opt(mcmc_e, [&](){return current_disc;}, mcmc_neighbors); // Init from current
        
        BranchAwareOptimizer<MCMCState>::Config m_conf;
        m_conf.beta_start = 0.1; m_conf.beta_end = 5.0; m_conf.beta_steps = 100;
        m_conf.samples_per_beta = 10;
        m_conf.verbose = false;
        
        auto m_res = mcmc_opt.optimize(m_conf);
        current_disc = m_res.best_state;
        
        // Count Conflicts
        int conflicts = 0;
        for(auto& e : G.edges) if(current_disc[e.first] == current_disc[e.second]) conflicts++;
        std::cout << "[MCMC] Conflicts: " << conflicts << "\n";
        
        if(conflicts == 0) break;
        
        // PHASE B: ADAFACTOR
        std::cout << "[Adafactor] relaxing manifold...\n";
        ContinuousState x_init = encode(current_disc);
        
        auto ada_e = [&](const ContinuousState& x) { return adafactor_energy(x); };
        auto ada_g = [&](const ContinuousState& x) { return adafactor_gradient(x); };
        
        navokoj::AdafactorOptimizer ada_opt(ada_e, ada_g, x_init);
        navokoj::AdafactorOptimizer::Config a_conf;
        a_conf.steps = 500;
        a_conf.learning_rate = 0.05;
        
        auto a_res = ada_opt.optimize(a_conf);
        
        // Collapse
        MCMCState polished = decode(a_res.best_state);
         int p_conflicts = 0;
        for(auto& e : G.edges) if(polished[e.first] == polished[e.second]) p_conflicts++;
        std::cout << "[Adafactor] Collapsed Conflicts: " << p_conflicts << "\n";
        
        current_disc = polished;
        if(p_conflicts == 0) break;
    }
    
    std::cout << "\nDONE.\n";
    return 0;
}
