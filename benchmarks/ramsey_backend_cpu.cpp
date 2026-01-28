#include "ramsey_backend.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace {

class CpuBackend final : public RamseyBackend {
public:
    CpuBackend(std::vector<int> clique_edges, int num_cliques, int k_edges, int num_edges)
        : clique_edges_(std::move(clique_edges)),
          num_cliques_(num_cliques),
          k_edges_(k_edges),
          num_edges_(num_edges) {}

    double energy(const std::vector<int>& edges) override {
        // Assumes edges.size() == num_edges_; returns large penalty otherwise.
        if (edges.size() != static_cast<size_t>(num_edges_)) return 1e18;

        int mono_count = 0;
        const int* edge_colors = edges.data();
        const int* clique_ptr = clique_edges_.data();

        for (int c = 0; c < num_cliques_; ++c) {
            const int offset = c * k_edges_;
            const int first_color = edge_colors[clique_ptr[offset]];
            bool mono = true;

            for (int i = 1; i < k_edges_; ++i) {
                if (edge_colors[clique_ptr[offset + i]] != first_color) {
                    mono = false;
                    break;
                }
            }

            mono_count += mono ? 1 : 0;
        }

        return static_cast<double>(mono_count);
    }

    const char* name() const noexcept override { return "cpu"; }

private:
    std::vector<int> clique_edges_;
    int num_cliques_ = 0;
    int k_edges_ = 0;
    int num_edges_ = 0;
};

} // namespace

std::unique_ptr<RamseyBackend> create_cpu_backend(
    const std::vector<int>& clique_edges,
    int num_cliques,
    int k_edges,
    int num_edges) {
    return std::make_unique<CpuBackend>(clique_edges, num_cliques, k_edges, num_edges);
}
