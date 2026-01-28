/*
 * Network Design Problem using BAHA
 * Design network topology minimizing cost while ensuring connectivity
 */
#include "baha/baha.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <queue>
#include <numeric>

struct NetworkState {
    std::vector<std::vector<bool>> edges;  // edges[i][j] = true if link exists
    int n_nodes;
    
    NetworkState() : n_nodes(0) {}
    NetworkState(int n) : n_nodes(n), edges(n, std::vector<bool>(n, false)) {}
};

struct Node {
    double x, y;
    int demand;
    
    double distance(const Node& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
};

// Check if graph is connected using BFS
bool is_connected(const NetworkState& s) {
    if (s.n_nodes == 0) return false;
    
    std::vector<bool> visited(s.n_nodes, false);
    std::queue<int> q;
    q.push(0);
    visited[0] = true;
    int visited_count = 1;
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (int v = 0; v < s.n_nodes; ++v) {
            if (s.edges[u][v] && !visited[v]) {
                visited[v] = true;
                q.push(v);
                visited_count++;
            }
        }
    }
    
    return visited_count == s.n_nodes;
}

int main(int argc, char** argv) {
    int n_nodes = (argc > 1) ? std::stoi(argv[1]) : 15;
    
    // Generate nodes
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> coord_dist(0.0, 100.0);
    std::uniform_int_distribution<int> demand_dist(1, 10);
    
    std::vector<Node> nodes(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        nodes[i].x = coord_dist(rng);
        nodes[i].y = coord_dist(rng);
        nodes[i].demand = demand_dist(rng);
    }
    
    std::cout << "============================================================\n";
    std::cout << "NETWORK DESIGN PROBLEM: " << n_nodes << " nodes\n";
    std::cout << "============================================================\n";
    
    // Energy: total link cost + penalty for disconnected network
    auto energy = [&nodes](const NetworkState& s) -> double {
        double total_cost = 0.0;
        int link_count = 0;
        
        for (int i = 0; i < s.n_nodes; ++i) {
            for (int j = i + 1; j < s.n_nodes; ++j) {
                if (s.edges[i][j]) {
                    double dist = nodes[i].distance(nodes[j]);
                    // Cost = distance * (base cost + demand factor)
                    total_cost += dist * (1.0 + 0.1 * (nodes[i].demand + nodes[j].demand));
                    link_count++;
                }
            }
        }
        
        // Penalty for disconnected network
        if (!is_connected(s)) {
            return total_cost + 10000.0;
        }
        
        return total_cost;
    };
    
    // Random initial network
    auto sampler = [n_nodes]() -> NetworkState {
        NetworkState s(n_nodes);
        std::mt19937 rng(std::random_device{}());
        std::bernoulli_distribution dist(0.3);  // 30% chance of link
        
        for (int i = 0; i < n_nodes; ++i) {
            for (int j = i + 1; j < n_nodes; ++j) {
                s.edges[i][j] = s.edges[j][i] = dist(rng);
            }
        }
        
        return s;
    };
    
    // Neighbors: add/remove edge, or swap edges
    auto neighbors = [n_nodes](const NetworkState& s) -> std::vector<NetworkState> {
        std::vector<NetworkState> nbrs;
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> node_dist(0, n_nodes - 1);
        
        for (int k = 0; k < 25; ++k) {
            NetworkState nbr = s;
            int op = rng() % 2;
            
            if (op == 0) {
                // Toggle an edge
                int u = node_dist(rng);
                int v = node_dist(rng);
                if (u != v) {
                    nbr.edges[u][v] = !nbr.edges[u][v];
                    nbr.edges[v][u] = nbr.edges[u][v];
                }
            } else {
                // Remove one edge, add another (swap)
                std::vector<std::pair<int, int>> existing, missing;
                for (int i = 0; i < n_nodes; ++i) {
                    for (int j = i + 1; j < n_nodes; ++j) {
                        if (nbr.edges[i][j]) existing.push_back({i, j});
                        else missing.push_back({i, j});
                    }
                }
                if (!existing.empty() && !missing.empty()) {
                    auto remove = existing[rng() % existing.size()];
                    auto add = missing[rng() % missing.size()];
                    nbr.edges[remove.first][remove.second] = false;
                    nbr.edges[remove.second][remove.first] = false;
                    nbr.edges[add.first][add.second] = true;
                    nbr.edges[add.second][add.first] = true;
                }
            }
            
            nbrs.push_back(nbr);
        }
        
        return nbrs;
    };
    
    navokoj::BranchAwareOptimizer<NetworkState> opt(energy, sampler, neighbors);
    
    typename navokoj::BranchAwareOptimizer<NetworkState>::Config config;
    config.beta_steps = 500;
    config.beta_end = 15.0;
    config.samples_per_beta = 50;
    config.fracture_threshold = 1.8;
    config.max_branches = 6;
    config.verbose = false;
    config.schedule_type = navokoj::BranchAwareOptimizer<NetworkState>::ScheduleType::GEOMETRIC;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = opt.optimize(config);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    // Calculate final stats
    double final_cost = 0.0;
    int final_links = 0;
    for (int i = 0; i < n_nodes; ++i) {
        for (int j = i + 1; j < n_nodes; ++j) {
            if (result.best_state.edges[i][j]) {
                double dist = nodes[i].distance(nodes[j]);
                final_cost += dist * (1.0 + 0.1 * (nodes[i].demand + nodes[j].demand));
                final_links++;
            }
        }
    }
    
    bool connected = is_connected(result.best_state);
    
    std::cout << "\nResult:\n";
    std::cout << "Total cost: " << std::fixed << std::setprecision(2) << final_cost << "\n";
    std::cout << "Number of links: " << final_links << "\n";
    std::cout << "Network connected: " << (connected ? "Yes" : "No") << "\n";
    std::cout << "Fractures detected: " << result.fractures_detected << "\n";
    std::cout << "Branch jumps: " << result.branch_jumps << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed << "s\n";
    
    if (connected) {
        std::cout << "\nNetwork links:\n";
        for (int i = 0; i < n_nodes; ++i) {
            for (int j = i + 1; j < n_nodes; ++j) {
                if (result.best_state.edges[i][j]) {
                    double dist = nodes[i].distance(nodes[j]);
                    std::cout << "  " << i << " ↔ " << j << " (distance=" 
                              << std::fixed << std::setprecision(1) << dist << ")\n";
                }
            }
        }
    }
    
    return 0;
}
