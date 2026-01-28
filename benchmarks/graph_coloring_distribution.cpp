/*
 * Graph Coloring Distribution Test
 * Run BAHA on multiple independent G(n, p) instances to analyze color distribution
 * 
 * Purpose: Determine if BAHA's low color count is reproducible or statistical variance
 */

#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <set>
#include <numeric>
#include <cmath>

struct GraphColoringState {
    std::vector<int> colors;
    int n_vertices;
    
    GraphColoringState() : n_vertices(0) {}
    GraphColoringState(int n, int initial_colors, int seed) : n_vertices(n), colors(n, 0) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, initial_colors - 1);
        for (int i = 0; i < n_vertices; ++i) {
            colors[i] = dist(rng);
        }
    }
};

struct Graph {
    std::vector<std::vector<bool>> adj;
    int n_vertices;
    int edge_count;
    
    Graph(int n, double edge_prob, int seed) : n_vertices(n), edge_count(0) {
        adj.assign(n, std::vector<bool>(n, false));
        std::mt19937 rng(seed);
        std::bernoulli_distribution dist(edge_prob);
        
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (dist(rng)) {
                    adj[i][j] = adj[j][i] = true;
                    edge_count++;
                }
            }
        }
    }
    
    // Calculate max degree (useful for chromatic number bounds)
    int max_degree() const {
        int max_deg = 0;
        for (int i = 0; i < n_vertices; ++i) {
            int deg = 0;
            for (int j = 0; j < n_vertices; ++j) {
                if (adj[i][j]) deg++;
            }
            max_deg = std::max(max_deg, deg);
        }
        return max_deg;
    }
};

struct TrialResult {
    int instance_seed;
    int colors_used;
    int conflicts;
    int fractures;
    int jumps;
    double time_ms;
    int edge_count;
    int max_degree;
};

TrialResult run_trial(int n_vertices, double edge_prob, int instance_seed) {
    Graph graph(n_vertices, edge_prob, instance_seed);
    const int initial_max_colors = n_vertices;
    
    // Energy function
    auto energy = [&](const GraphColoringState& state) -> double {
        double conflicts = 0;
        std::set<int> unique_colors;
        
        for (int i = 0; i < graph.n_vertices; ++i) {
            unique_colors.insert(state.colors[i]);
            for (int j = i + 1; j < graph.n_vertices; ++j) {
                if (graph.adj[i][j] && state.colors[i] == state.colors[j]) {
                    conflicts += 1;
                }
            }
        }
        return conflicts * 1000.0 + unique_colors.size();
    };
    
    // Sampler
    int sampler_seed = instance_seed + 10000;
    auto sampler = [&, sampler_seed]() mutable {
        GraphColoringState state(n_vertices, initial_max_colors, sampler_seed++);
        return state;
    };
    
    // Neighbors
    auto neighbors = [&](const GraphColoringState& s) {
        std::vector<GraphColoringState> nbrs;
        std::mt19937 rng(std::random_device{}());
        
        int max_color = 0;
        for (int c : s.colors) {
            max_color = std::max(max_color, c);
        }
        
        for (int i = 0; i < 10; ++i) {
            GraphColoringState nbr = s;
            int vertex_to_change = rng() % n_vertices;
            
            if (max_color > 0) {
                nbr.colors[vertex_to_change] = rng() % (max_color + 1);
                nbrs.push_back(nbr);
            }
            
            if (max_color + 1 < initial_max_colors) {
                nbr = s;
                nbr.colors[vertex_to_change] = max_color + 1;
                nbrs.push_back(nbr);
            }
        }
        
        return nbrs;
    };
    
    // Create optimizer
    navokoj::BranchAwareOptimizer<GraphColoringState> optimizer(energy, sampler, neighbors);
    
    // Configure
    navokoj::BranchAwareOptimizer<GraphColoringState>::Config config;
    config.beta_start = 0.01;
    config.beta_end = 100.0;
    config.beta_steps = 3000;
    config.fracture_threshold = 2.0;
    config.samples_per_beta = 100;
    config.schedule_type = navokoj::BranchAwareOptimizer<GraphColoringState>::ScheduleType::GEOMETRIC;
    
    // Run
    auto start = std::chrono::high_resolution_clock::now();
    auto result = optimizer.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Verify and count colors
    double conflicts = 0;
    std::set<int> unique_colors;
    for (int i = 0; i < graph.n_vertices; ++i) {
        unique_colors.insert(result.best_state.colors[i]);
        for (int j = i + 1; j < graph.n_vertices; ++j) {
            if (graph.adj[i][j] && result.best_state.colors[i] == result.best_state.colors[j]) {
                conflicts += 1;
            }
        }
    }
    
    TrialResult tr;
    tr.instance_seed = instance_seed;
    tr.colors_used = unique_colors.size();
    tr.conflicts = static_cast<int>(conflicts);
    tr.fractures = result.fractures_detected;
    tr.jumps = result.branch_jumps;
    tr.time_ms = duration.count();
    tr.edge_count = graph.edge_count;
    tr.max_degree = graph.max_degree();
    
    return tr;
}

int main() {
    std::cout << "Graph Coloring Distribution Test\n";
    std::cout << "=================================\n\n";
    
    const int n_vertices = 200;
    const double edge_prob = 0.5;
    const int n_trials = 10;
    
    std::cout << "Parameters:\n";
    std::cout << "  Vertices: " << n_vertices << "\n";
    std::cout << "  Edge probability: " << edge_prob << "\n";
    std::cout << "  Trials: " << n_trials << "\n\n";
    
    // Theoretical expected chromatic number for G(n, p)
    // chi(G) ~ n * log(1/(1-p)) / (2 * log(n))
    double expected_chi = n_vertices * std::log(1.0 / (1.0 - edge_prob)) / (2.0 * std::log(n_vertices));
    std::cout << "Theoretical expected chi(G): ~" << std::fixed << std::setprecision(1) << expected_chi << "\n\n";
    
    std::vector<TrialResult> results;
    
    std::cout << "Running " << n_trials << " independent trials...\n\n";
    std::cout << "| Trial | Seed | Edges | MaxDeg | Colors | Conflicts | Fractures | Jumps | Time (s) |\n";
    std::cout << "|-------|------|-------|--------|--------|-----------|-----------|-------|----------|\n";
    
    for (int trial = 0; trial < n_trials; ++trial) {
        int seed = 1000 + trial * 137;  // Different seeds for each instance
        
        TrialResult tr = run_trial(n_vertices, edge_prob, seed);
        results.push_back(tr);
        
        std::cout << "| " << std::setw(5) << trial + 1;
        std::cout << " | " << std::setw(4) << seed;
        std::cout << " | " << std::setw(5) << tr.edge_count;
        std::cout << " | " << std::setw(6) << tr.max_degree;
        std::cout << " | " << std::setw(6) << tr.colors_used;
        std::cout << " | " << std::setw(9) << tr.conflicts;
        std::cout << " | " << std::setw(9) << tr.fractures;
        std::cout << " | " << std::setw(5) << tr.jumps;
        std::cout << " | " << std::setw(8) << std::fixed << std::setprecision(1) << tr.time_ms / 1000.0;
        std::cout << " |\n";
    }
    
    // Calculate statistics
    std::vector<int> colors;
    std::vector<int> conflicts;
    std::vector<int> fractures;
    std::vector<int> jumps;
    std::vector<double> times;
    
    for (const auto& tr : results) {
        colors.push_back(tr.colors_used);
        conflicts.push_back(tr.conflicts);
        fractures.push_back(tr.fractures);
        jumps.push_back(tr.jumps);
        times.push_back(tr.time_ms);
    }
    
    auto mean = [](const std::vector<int>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    
    auto stddev = [&mean](const std::vector<int>& v) {
        double m = mean(v);
        double sum = 0;
        for (int x : v) sum += (x - m) * (x - m);
        return std::sqrt(sum / v.size());
    };
    
    int min_colors = *std::min_element(colors.begin(), colors.end());
    int max_colors = *std::max_element(colors.begin(), colors.end());
    double mean_colors = mean(colors);
    double std_colors = stddev(colors);
    
    int valid_count = std::count(conflicts.begin(), conflicts.end(), 0);
    
    double mean_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    
    std::cout << "\n=== SUMMARY STATISTICS ===\n\n";
    std::cout << "Colors Used:\n";
    std::cout << "  Min:    " << min_colors << "\n";
    std::cout << "  Max:    " << max_colors << "\n";
    std::cout << "  Mean:   " << std::fixed << std::setprecision(2) << mean_colors << "\n";
    std::cout << "  StdDev: " << std::fixed << std::setprecision(2) << std_colors << "\n";
    std::cout << "  Expected (theory): ~" << std::fixed << std::setprecision(1) << expected_chi << "\n\n";
    
    std::cout << "Validity:\n";
    std::cout << "  Valid colorings: " << valid_count << "/" << n_trials << " (" 
              << std::fixed << std::setprecision(0) << (100.0 * valid_count / n_trials) << "%)\n\n";
    
    std::cout << "Performance:\n";
    std::cout << "  Mean time: " << std::fixed << std::setprecision(1) << mean_time / 1000.0 << " s\n";
    std::cout << "  Mean fractures: " << std::fixed << std::setprecision(0) << mean(fractures) << "\n";
    std::cout << "  Mean jumps: " << std::fixed << std::setprecision(1) << mean(jumps) << "\n";
    std::cout << "  Jump rate: " << std::fixed << std::setprecision(2) 
              << (100.0 * mean(jumps) / mean(fractures)) << "%\n\n";
    
    // Interpretation
    std::cout << "=== INTERPRETATION ===\n\n";
    
    if (mean_colors < expected_chi * 0.75) {
        std::cout << "RESULT: BAHA consistently finds colorings with significantly fewer colors\n";
        std::cout << "than theoretical expectation (" << std::fixed << std::setprecision(1) 
                  << mean_colors << " vs " << expected_chi << ").\n\n";
        std::cout << "This suggests BAHA exploits thermodynamic structure in dense random graphs\n";
        std::cout << "that sequential heuristics miss.\n";
    } else if (mean_colors < expected_chi) {
        std::cout << "RESULT: BAHA finds colorings slightly below theoretical expectation\n";
        std::cout << "(" << std::fixed << std::setprecision(1) << mean_colors << " vs " << expected_chi << ").\n\n";
        std::cout << "This is consistent with BAHA exploiting instance-specific structure.\n";
    } else {
        std::cout << "RESULT: BAHA finds colorings near theoretical expectation\n";
        std::cout << "(" << std::fixed << std::setprecision(1) << mean_colors << " vs " << expected_chi << ").\n\n";
        std::cout << "Earlier low-color results were likely statistical outliers.\n";
    }
    
    if (std_colors < 5) {
        std::cout << "\nLow variance (StdDev=" << std::fixed << std::setprecision(2) << std_colors 
                  << ") indicates consistent behavior across instances.\n";
    } else {
        std::cout << "\nHigh variance (StdDev=" << std::fixed << std::setprecision(2) << std_colors 
                  << ") indicates instance-dependent behavior.\n";
    }
    
    return 0;
}
