/*
 * Author: Sethurathienam Iyer
 */
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>

// STANDALONE ZERO-TRUST RAMSEY VERIFIER
// Exhaustively checks an edge coloring for monochromatic cliques.

class Verifier {
    int N, K, Colors;
    std::vector<int> edges;

public:
    Verifier(int n, int k, int c, const std::string& filename) : N(n), K(k), Colors(c) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open solution file!" << std::endl;
            exit(1);
        }
        int val;
        while (file >> val) edges.push_back(val);
        
        int expected_edges = N * (N - 1) / 2;
        if (edges.size() != (size_t)expected_edges) {
            std::cerr << "Edge count mismatch! Expected " << expected_edges << ", got " << edges.size() << std::endl;
            exit(1);
        }
    }

    int edge_idx(int u, int v) const {
        if (u > v) std::swap(u, v);
        return (2 * N - 1 - u) * u / 2 + v - u - 1;
    }

    void verify() {
        std::cout << "--- Exhaustive CPU Verification ---" << std::endl;
        std::cout << "Checking K_" << N << " for monochromatic K_" << K << "..." << std::endl;
        
        long long cliques_checked = 0;
        long long mono_found = 0;

        std::vector<int> selector(N);
        std::fill(selector.end() - K, selector.end(), 1);

        do {
            cliques_checked++;
            std::vector<int> verts;
            for (int i = 0; i < N; ++i) if (selector[i]) verts.push_back(i);
            
            int c0 = edges[edge_idx(verts[0], verts[1])];
            bool mono = true;
            for (int i = 0; i < K; ++i) {
                for (int j = i + 1; j < K; ++j) {
                    if (edges[edge_idx(verts[i], verts[j])] != c0) {
                        mono = false;
                        goto next;
                    }
                }
            }
            if (mono) mono_found++;
            next:;
        } while (std::next_permutation(selector.begin(), selector.end()));

        std::cout << "Cliques Checked: " << cliques_checked << std::endl;
        if (mono_found == 0) {
            std::cout << "✅ VERIFIED: No monochromatic K_" << K << " found." << std::endl;
        } else {
            std::cout << "❌ FAILED: " << mono_found << " monochromatic cliques detected." << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: ./verifier <N> <K> <Colors> <solution_file>" << std::endl;
        return 1;
    }
    int N = std::stoi(argv[1]);
    int K = std::stoi(argv[2]);
    int C = std::stoi(argv[3]);
    Verifier v(N, K, C, argv[4]);
    v.verify();
    return 0;
}
