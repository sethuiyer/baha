#include "ramsey_backend.h"
#include "ramsey_common.h"

#include <iostream>
#include <random>
#include <vector>

namespace {

std::vector<int> make_state(int num_edges, int colors) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, colors - 1);
    std::vector<int> edges(num_edges);
    for (int i = 0; i < num_edges; ++i) edges[i] = dist(rng);
    return edges;
}

bool check_backend(RamseyBackend& backend, const std::vector<int>& edges, double expected) {
    const double value = backend.energy(edges);
    if (value != expected) {
        std::cerr << "Mismatch on " << backend.name() << ": " << value << " vs " << expected << "\n";
        return false;
    }
    std::cout << backend.name() << " OK (" << value << ")\n";
    return true;
}

} // namespace

int main() {
    const int n = 8;
    const int k = 3;
    const int colors = 2;

    RamseyCliqueData cliques = ramsey_build_cliques(n, k);
    auto edges = make_state(cliques.num_edges, colors);

    auto cpu = create_cpu_backend(cliques.clique_edges, cliques.num_cliques, cliques.k_edges, cliques.num_edges);
    const double cpu_energy = cpu->energy(edges);
    if (!check_backend(*cpu, edges, cpu_energy)) return 1;

#ifdef BAHA_HAVE_CUDA_BACKEND
    auto cuda = create_cuda_backend(cliques.clique_edges, cliques.num_cliques, cliques.k_edges, cliques.num_edges);
    if (!check_backend(*cuda, edges, cpu_energy)) return 1;
#endif

#ifdef BAHA_HAVE_MPS_BACKEND
    auto mps = create_mps_backend(cliques.clique_edges, cliques.num_cliques, cliques.k_edges, cliques.num_edges);
    if (!check_backend(*mps, edges, cpu_energy)) return 1;
#endif

    std::cout << "Ramsey backend test passed.\n";
    return 0;
}
