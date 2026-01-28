#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

struct RamseyCliqueData {
    int num_edges = 0;
    int k_edges = 0;
    int num_cliques = 0;
    std::vector<int> clique_edges;
};

inline int ramsey_edge_idx(int u, int v, int n) {
    if (u > v) std::swap(u, v);
    return (2 * n - 1 - u) * u / 2 + v - u - 1;
}

inline RamseyCliqueData ramsey_build_cliques(int n, int k) {
    RamseyCliqueData data;
    data.num_edges = n * (n - 1) / 2;
    data.k_edges = k * (k - 1) / 2;

    std::vector<int> selector(n, 0);
    std::fill(selector.end() - k, selector.end(), 1);

    std::vector<int> clique_edges;
    clique_edges.reserve(1024);

    int num_cliques = 0;
    do {
        std::vector<int> verts;
        verts.reserve(k);
        for (int i = 0; i < n; ++i) {
            if (selector[i]) verts.push_back(i);
        }

        for (int i = 0; i < k; ++i) {
            for (int j = i + 1; j < k; ++j) {
                clique_edges.push_back(ramsey_edge_idx(verts[i], verts[j], n));
            }
        }
        num_cliques++;
    } while (std::next_permutation(selector.begin(), selector.end()));

    data.num_cliques = num_cliques;
    data.clique_edges = std::move(clique_edges);
    return data;
}
