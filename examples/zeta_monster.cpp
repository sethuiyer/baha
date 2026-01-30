/**
 * ZETA MONSTER: C++ Benchmark for BAHA (Zeta Breather)
 * 
 * Goal: Solve the "Monster" Graph using Oscillating Continuous Relaxation.
 * Strategy:
 * 1. Adafactor: Oscillate Beta (0.5 <-> 1.5).
 * 2. Handover: When Beta peaks, extract state and Polish with MCMC.
 */

#include "baha/baha.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>

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
};
Graph G;

std::vector<int> generate_primes(int n) {
    std::vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (int p = 2; p * p <= n; p++) if (is_prime[p]) for (int i = p * p; i <= n; i += p) is_prime[i] = false;
    std::vector<int> primes;
    for (int p = 2; p <= n; p++) if (is_prime[p]) primes.push_back(p);
    return primes;
}

// =============================================================================
// HELPER: DISCRETE -> CONTINUOUS -> DISCRETE
// =============================================================================
using ContinuousState = std::vector<double>;
using DiscreteState = std::vector<int>;

DiscreteState decode(const ContinuousState& x) {
    DiscreteState s(N_NODES);
    for(int i=0; i<N_NODES; ++i) {
        int best_c = 0;
        double max_p = -1.0;
        for(int c=0; c<N_COLORS; ++c) {
            double val = x[i*N_COLORS+c];
            if(val > max_p) { max_p = val; best_c = c; }
        }
        s[i] = best_c;
    }
    return s;
}

// =============================================================================
// BREATHER ENGINE
// =============================================================================
// Adafactor needs Gradients. We define them dynamically based on Beta.

// Global "Beta" control for the breather
double GLOBAL_BETA = 1.0;

double adafactor_energy(const ContinuousState& x) {
    double E = 0.0;
    // Conflict Energy on Unsatisfied Edges
    for(const auto& edge : G.edges) {
        int u = edge.first;
        int v = edge.second;
        double w = G.node_weights[u] + G.node_weights[v];
        
        double dot = 0.0;
        for(int c=0; c<N_COLORS; ++c) 
            dot += x[u*N_COLORS+c] * x[v*N_COLORS+c];
            
        // Scale by Beta: Higher Beta = Sharper landscape (Hypothesis)
        // Or simply: E is potential.
        E += w * dot;
    }
    // Simplex Penalty (Constant, not scaled by Beta usually, to maintain manifold)
    double pen = 0.0;
    for(int i=0; i<N_NODES; ++i) {
        double sum = 0.0;
        for(int c=0; c<N_COLORS; ++c) sum += x[i*N_COLORS+c];
        pen += (sum - 1.0)*(sum - 1.0);
    }
    return GLOBAL_BETA * E + 10.0 * pen;
}

ContinuousState adafactor_gradient(const ContinuousState& x) {
    ContinuousState g(x.size(), 0.0);
    // 1. Edge Gradients
    for(const auto& edge : G.edges) {
        int u = edge.first;
        int v = edge.second;
        double w = G.node_weights[u] + G.node_weights[v];
        for(int c=0; c<N_COLORS; ++c) {
            // Symmetry handling: d(uv)/du -> v * w
            double term_u = GLOBAL_BETA * w * x[v*N_COLORS+c];
            double term_v = GLOBAL_BETA * w * x[u*N_COLORS+c];
            
            g[u*N_COLORS+c] += term_u;
            g[v*N_COLORS+c] += term_v;
        }
    }
    // 2. Simplex
    for(int i=0; i<N_NODES; ++i) {
        double sum = 0.0;
        for(int c=0; c<N_COLORS; ++c) sum += x[i*N_COLORS+c];
        double term = 2.0 * 10.0 * (sum - 1.0);
        for(int c=0; c<N_COLORS; ++c) g[i*N_COLORS+c] += term;
    }
    return g;
}

// =============================================================================
// MCMC POLISH
// =============================================================================
double mcmc_energy(const DiscreteState& s) {
    double E = 0.0;
    for(const auto& edge : G.edges) 
        if(s[edge.first] == s[edge.second]) 
            E += (G.node_weights[edge.first] + G.node_weights[edge.second]);
    return E;
}
DiscreteState mcmc_sampler_dummy() { return DiscreteState(N_NODES, 0); }
std::vector<DiscreteState> mcmc_neighbors(const DiscreteState& s) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::vector<DiscreteState> nbrs;
    for(int k=0; k<10; ++k) {
        int node = rng() % N_NODES;
        int new_c = (s[node] + 1 + (rng() % (N_COLORS - 1))) % N_COLORS;
        DiscreteState next = s;
        next[node] = new_c;
        nbrs.push_back(next);
    }
    return nbrs;
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::cout << "==========================================\n";
    std::cout << " ZETA MONSTER (Breather)\n";
    std::cout << "==========================================\n";
    
    // 1. Setup Graph
    G.n = N_NODES;
    std::vector<int> degrees(N_NODES, 0);
    std::mt19937 rng(SEED);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for(int i=0; i<N_NODES; ++i) for(int j=i+1; j<N_NODES; ++j) 
        if(dist(rng) < EDGE_PROB) {
            G.edges.push_back({i,j});
            degrees[i]++; degrees[j]++;
        }
    
    // 2. Weights
    auto primes = generate_primes(12000);
    std::vector<std::pair<int,int>> nodes_sorted;
    for(int i=0; i<N_NODES; ++i) nodes_sorted.push_back({i, degrees[i]});
    std::sort(nodes_sorted.begin(), nodes_sorted.end(), [](auto& a, auto& b){return a.second > b.second;});
    G.node_weights.resize(N_NODES);
    for(int r=0; r<N_NODES; ++r) G.node_weights[nodes_sorted[r].first] = 1.0 / std::log((double)primes[r]);
    
    // 3. Initialize Continuous
    ContinuousState x(N_NODES * N_COLORS);
    std::normal_distribution<double> noise(0.0, 0.01);
    for(auto& val : x) val = 0.25 + noise(rng);
    
    // 4. Zeta Loop
    using namespace navokoj;
    auto ae = [&](const ContinuousState& s) { return adafactor_energy(s); };
    auto ag = [&](const ContinuousState& s) { return adafactor_gradient(s); };
    
    AdafactorOptimizer ada(ae, ag, x);
    
    int total_steps = 10000; 
    int chunk_size = 100;
    int best_conflicts = 999999;
    
    std::cout << "Cycling Beta 0.5 <-> 1.5 ...\n";
    
    for(int t=0; t<total_steps; t+=chunk_size) {
        // OSCILLATE BETA
        // Period = 2000 steps
        // Sin goes -1 to 1. So 1.0 + 0.5*sin goes 0.5 to 1.5. Correct.
        double phase = (double)(t % 2000) / 2000.0 * 6.28318;
        GLOBAL_BETA = 1.0 + 0.5 * std::sin(phase);
        
        AdafactorOptimizer::Config conf;
        conf.steps = chunk_size;
        conf.learning_rate = 0.01;
        
        auto res = ada.optimize(conf);
        
        // Check Peak (Beta > 1.45)
        if (GLOBAL_BETA > 1.45) {
            // HANDOVER
            DiscreteState s = decode(res.best_state);
            
            // Quick MCMC Polish
            BranchAwareOptimizer<DiscreteState> mcmc(mcmc_energy, [&](){return s;}, mcmc_neighbors);
            BranchAwareOptimizer<DiscreteState>::Config mconf;
            mconf.beta_start = 5.0; mconf.beta_end = 5.0; // Frozen Polish
            mconf.beta_steps = 20; 
            mconf.samples_per_beta = 5;
            mconf.verbose = false;
            
            auto mres = mcmc.optimize(mconf);
            s = mres.best_state;
            
            int c = 0;
            for(auto& e : G.edges) if(s[e.first] == s[e.second]) c++;
            
            if(c < best_conflicts) {
                best_conflicts = c;
                std::cout << " Step " << t << " | Beta " << std::fixed << std::setprecision(2) << GLOBAL_BETA 
                          << " | New Best: " << c << "\n";
            }
        }
    }
    
    std::cout << "\n-----------------\n";
    std::cout << " BEST CONFLICTS: " << best_conflicts << "\n";
    
    return 0;
}
