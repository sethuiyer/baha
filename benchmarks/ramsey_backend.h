#pragma once

#include <memory>
#include <vector>

class RamseyBackend {
public:
    virtual ~RamseyBackend() = default;
    virtual double energy(const std::vector<int>& edges) = 0;
    virtual const char* name() const noexcept = 0;
};

std::unique_ptr<RamseyBackend> create_cpu_backend(
    const std::vector<int>& clique_edges,
    int num_cliques,
    int k_edges,
    int num_edges);

#ifdef BAHA_HAVE_CUDA_BACKEND
std::unique_ptr<RamseyBackend> create_cuda_backend(
    const std::vector<int>& clique_edges,
    int num_cliques,
    int k_edges,
    int num_edges);
#endif

#ifdef BAHA_HAVE_MPS_BACKEND
std::unique_ptr<RamseyBackend> create_mps_backend(
    const std::vector<int>& clique_edges,
    int num_cliques,
    int k_edges,
    int num_edges);
#endif
