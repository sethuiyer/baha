/**
 * ADAFACTOR MONSTER: C++ Benchmark for BAHA (Continuous Relaxation)
 * 
 * Goal: Solve the "Monster" Graph (N=1300, p=0.314) using Adafactor + Prime Weights.
 * Architecture: High-Performance C++ Gradient Descent.
 */

#include "baha/baha.hpp" // Assumes AdafactorOptimizer is here
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
const int SEED = 999;

// =============================================================================
// HELPERS
// =============================================================================

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

struct Graph {
    int n_nodes;
    std::vector<std::pair<int, int>> edges;
    std::vector<double> node_weights;
};

// =============================================================================
// CONTINUOUS STATE & GRADIENTS
// =============================================================================

// State: Flattened vector of size N_NODES * N_COLORS.
// x_{i,c} is the probability node i has color c.
using State = std::vector<double>;

// Helper to index flattened state
inline int idx(int node, int color) {
    return node * N_COLORS + color;
}

// Energy: E = Sum_{(u,v) in E} Sum_{c} x_{u,c} * x_{v,c} * (W[u] + W[v])
// This represents the expected penalty if x represents probabilities.
double energy_fn(const State& s, const Graph& G) {
    double total_E = 0.0;
    const size_t m = G.edges.size();
    const auto* edge_data = G.edges.data();
    
    // O(M * C)
    for (size_t i = 0; i < m; ++i) {
        int u = edge_data[i].first;
        int v = edge_data[i].second;
        double w_uv = G.node_weights[u] + G.node_weights[v];
        
        double dot_prod = 0.0;
        for (int c = 0; c < N_COLORS; ++c) {
            dot_prod += s[idx(u,c)] * s[idx(v,c)];
        }
        total_E += w_uv * dot_prod;
    }
    
    // Soft constraint: Penalty for sum(x_{i,c}) != 1
    // E_constraint = lambda * Sum_i (Sum_c x_{i,c} - 1)^2
    // We enforce physically by gradient, but energy needs to reflect it to track progress?
    // Actually, let's keep energy pure conflict for metric, and handle constraint in gradient.
    return total_E;
}

// Gradient: Analytical Derivative
// dE/dx_{i,c} = Sum_{j in N(i)} x_{j,c} * (W[i] + W[j])
// Plus constraint force: 2 * lambda * (Sum_c x_{i,c} - 1)
State gradient_fn(const State& s, const Graph& G) {
    State g(s.size(), 0.0);
    const double lambda = 5.0; // Constraint strength
    
    // 1. Conflict Gradient
    const size_t m = G.edges.size();
    const auto* edge_data = G.edges.data();
    
    for (size_t i = 0; i < m; ++i) {
        int u = edge_data[i].first;
        int v = edge_data[i].second;
        double w_uv = G.node_weights[u] + G.node_weights[v];
        
        for (int c = 0; c < N_COLORS; ++c) {
            double prob_u = s[idx(u,c)];
            double prob_v = s[idx(v,c)];
            
            // dE/dx_u = w * x_v
            g[idx(u,c)] += w_uv * prob_v;
            // dE/dx_v = w * x_u
            g[idx(v,c)] += w_uv * prob_u;
        }
    }
    
    // 2. Constraint Gradient
    for (int i = 0; i < G.n_nodes; ++i) {
        double sum_p = 0.0;
        for (int c = 0; c < N_COLORS; ++c) sum_p += s[idx(i,c)];
        
        double diff = sum_p - 1.0;
        double force = 2.0 * lambda * diff;
        
        for (int c = 0; c < N_COLORS; ++c) {
            g[idx(i,c)] += force;
        }
    }
    return g;
}

// Discrete Collapse
std::vector<int> collapse(const State& s) {
    std::vector<int> colors(N_NODES);
    for (int i = 0; i < N_NODES; ++i) {
        int best_c = 0;
        double max_p = -1.0;
        for (int c = 0; c < N_COLORS; ++c) {
            if (s[idx(i,c)] > max_p) {
                max_p = s[idx(i,c)];
                best_c = c;
            }
        }
        colors[i] = best_c;
    }
    return colors;
}

int main() {
    std::cout << "==========================================\n";
    std::cout << " ADAFACTOR MONSTER (C++ Continuous)\n";
    std::cout << "==========================================\n";
    
    // 1. Setup Graph
    std::cout << " Generating Graph N=" << N_NODES << "...\n";
    Graph G;
    G.n_nodes = N_NODES;
    std::vector<int> degrees(N_NODES, 0);
    
    std::mt19937 rng(SEED);
    std::uniform_real_distribution<double> p_dist(0.0, 1.0);
    
    for (int i = 0; i < N_NODES; ++i) {
        for (int j = i + 1; j < N_NODES; ++j) {
            if (p_dist(rng) < EDGE_PROB) {
                G.edges.push_back({i, j});
                degrees[i]++;
                degrees[j]++;
            }
        }
    }
    
    // 2. Log-Prime Weights
    auto primes = generate_primes(12000);
    std::vector<std::pair<int, int>> nodes_by_deg;
    for(int i=0; i<N_NODES; ++i) nodes_by_deg.push_back({i, degrees[i]});
    std::sort(nodes_by_deg.begin(), nodes_by_deg.end(), [](auto& a, auto& b){ return a.second > b.second; });
    
    G.node_weights.resize(N_NODES);
    for(int rank=0; rank<N_NODES; ++rank) {
        int node = nodes_by_deg[rank].first;
        G.node_weights[node] = 1.0 / std::log((double)primes[rank]);
    }
    
    // 3. Initialize State (Noisy 1/k)
    State s0(N_NODES * N_COLORS);
    std::normal_distribution<double> noise(0.0, 0.01);
    double initial_p = 1.0 / N_COLORS; // 0.25
    for(auto& val : s0) val = initial_p + noise(rng);
    
    // 4. Optimize
    // using namespace navokoj; // Don't pull in everything
    
    auto e_fn = [&](const navokoj::AdafactorOptimizer::State& s) { return energy_fn(s, G); };
    auto g_fn = [&](const navokoj::AdafactorOptimizer::State& s) { return gradient_fn(s, G); };
    
    navokoj::AdafactorOptimizer optimizer(e_fn, g_fn, s0);
    
    navokoj::AdafactorOptimizer::Config config;
    config.learning_rate = 0.05;
    config.steps = 5000;          // Increase steps
    config.timeout_ms = 15000.0;  // Fair comparison
    
    std::cout << " Running Step 1: Adafactor Optimization...\n";
    auto result = optimizer.optimize(config);
    
    // 5. Verify
    auto final_colors = collapse(result.best_state);
    int conflicts = 0;
    for(auto& e : G.edges) {
        if (final_colors[e.first] == final_colors[e.second])
            conflicts++;
    }
    
    std::cout << "\n RESULTS:\n";
    std::cout << " Time: " << result.time_ms / 1000.0 << " s\n";
    std::cout << " Final Continuous Energy: " << result.best_energy << "\n";
    std::cout << " Discrete Conflicts: " << conflicts << "\n";
    
    return 0;
}
