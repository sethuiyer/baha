#include "ramsey_backend.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <vector>

namespace {

__global__ void count_mono_cliques_kernel(
    const int* edges,
    const int* clique_edges,
    int* mono_count,
    int num_cliques,
    int k_edges) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cliques) return;

    const int offset = idx * k_edges;
    const int first_color = edges[clique_edges[offset]];
    bool is_mono = true;

    for (int i = 1; i < k_edges; ++i) {
        if (edges[clique_edges[offset + i]] != first_color) {
            is_mono = false;
            break;
        }
    }

    if (is_mono) atomicAdd(mono_count, 1);
}

inline void cuda_check(cudaError_t err, const char* context) {
    if (err == cudaSuccess) return;
    std::fprintf(stderr, "CUDA error (%s): %s\n", context, cudaGetErrorString(err));
}

class CudaBackend final : public RamseyBackend {
public:
    CudaBackend(std::vector<int> clique_edges, int num_cliques, int k_edges, int num_edges)
        : clique_edges_host_(std::move(clique_edges)),
          num_cliques_(num_cliques),
          k_edges_(k_edges),
          num_edges_(num_edges) {
        const size_t clique_bytes = clique_edges_host_.size() * sizeof(int);
        cuda_check(cudaMalloc(&d_clique_edges_, clique_bytes), "cudaMalloc(clique_edges)");
        cuda_check(cudaMemcpy(d_clique_edges_, clique_edges_host_.data(), clique_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(clique_edges)");

        cuda_check(cudaMalloc(&d_edges_, num_edges_ * sizeof(int)), "cudaMalloc(edges)");
        cuda_check(cudaMalloc(&d_count_, sizeof(int)), "cudaMalloc(count)");
    }

    ~CudaBackend() override {
        cudaFree(d_clique_edges_);
        cudaFree(d_edges_);
        cudaFree(d_count_);
    }

    double energy(const std::vector<int>& edges) override {
        // Assumes edges.size() == num_edges_; returns large penalty otherwise.
        if (edges.size() != static_cast<size_t>(num_edges_)) return 1e18;

        cuda_check(cudaMemcpy(d_edges_, edges.data(), num_edges_ * sizeof(int), cudaMemcpyHostToDevice),
                   "cudaMemcpy(edges)");
        cuda_check(cudaMemset(d_count_, 0, sizeof(int)), "cudaMemset(count)");

        const int threads = 256;
        const int blocks = (num_cliques_ + threads - 1) / threads;
        count_mono_cliques_kernel<<<blocks, threads>>>(d_edges_, d_clique_edges_, d_count_, num_cliques_, k_edges_);
        cuda_check(cudaDeviceSynchronize(), "kernel sync");

        int count = 0;
        cuda_check(cudaMemcpy(&count, d_count_, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(count)");
        return static_cast<double>(count);
    }

    const char* name() const noexcept override { return "cuda"; }

private:
    std::vector<int> clique_edges_host_;
    int num_cliques_ = 0;
    int k_edges_ = 0;
    int num_edges_ = 0;
    int* d_clique_edges_ = nullptr;
    int* d_edges_ = nullptr;
    int* d_count_ = nullptr;
};

} // namespace

std::unique_ptr<RamseyBackend> create_cuda_backend(
    const std::vector<int>& clique_edges,
    int num_cliques,
    int k_edges,
    int num_edges) {
    return std::make_unique<CudaBackend>(clique_edges, num_cliques, k_edges, num_edges);
}
