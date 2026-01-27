#include <cuda_runtime.h>
#include <stdio.h>

// Global variable for monotonic count
__device__ int g_mono_count = 0;

__global__ void count_mono_cliques_kernel(const int* d_edges, const int* d_clique_edges, int num_cliques, int k_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cliques) {
        // Offset into d_clique_edges
        const int* c_edges = &d_clique_edges[idx * k_edges];
        
        int first_color = d_edges[c_edges[0]];
        bool is_mono = true;
        for (int i = 1; i < k_edges; ++i) {
            if (d_edges[c_edges[i]] != first_color) {
                is_mono = false;
                break;
            }
        }
        
        if (is_mono) {
            atomicAdd(&g_mono_count, 1);
        }
    }
}

extern "C" {

void cuda_malloc_int(int** ptr, size_t size) {
    cudaMalloc(ptr, size);
}

void cuda_free(void* ptr) {
    cudaFree(ptr);
}

void cuda_memcpy_h2d(int* dest, const int* src, size_t size) {
    cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

int launch_ramsey_kernel(int* d_edges, int* d_clique_edges, int num_cliques, int k_edges) {
    int zero = 0;
    cudaMemcpyToSymbol(g_mono_count, &zero, sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (num_cliques + threadsPerBlock - 1) / threadsPerBlock;
    
    count_mono_cliques_kernel<<<blocks, threadsPerBlock>>>(d_edges, d_clique_edges, num_cliques, k_edges);
    cudaDeviceSynchronize();

    int result;
    cudaMemcpyFromSymbol(&result, g_mono_count, sizeof(int));
    return result;
}

}
