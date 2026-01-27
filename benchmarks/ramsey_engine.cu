#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ramsey_engine.h"

__device__ int g_mono_count = 0;

__global__ void count_mono_cliques_kernel(const int* d_edges, const uint16_t* d_clique_edges, int num_cliques, int k_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cliques) {
        const uint16_t* c_edges = &d_clique_edges[idx * k_edges];
        int first_color = d_edges[c_edges[0]];
        bool is_mono = true;
        for (int i = 1; i < k_edges; ++i) {
            if (d_edges[c_edges[i]] != first_color) {
                is_mono = false;
                break;
            }
        }
        if (is_mono) atomicAdd(&g_mono_count, 1);
    }
}

typedef struct {
    uint16_t* d_clique_edges;
    int* d_edges_state;
    int num_cliques;
    int k_edges;
} RamseyCudaHandle;

extern "C" void* ramsey_cuda_init(const uint16_t* clique_edges, int num_cliques, int k_edges, int num_edges) {
    RamseyCudaHandle* h = (RamseyCudaHandle*)malloc(sizeof(RamseyCudaHandle));
    h->num_cliques = num_cliques;
    h->k_edges = k_edges;
    
    size_t clique_size = (size_t)num_cliques * k_edges * sizeof(uint16_t);
    printf("Allocating %.2f MB for %d cliques...\n", (double)clique_size / (1024*1024), num_cliques);

    if (cudaMalloc(&h->d_clique_edges, clique_size) != cudaSuccess) return NULL;
    cudaMemcpy(h->d_clique_edges, clique_edges, clique_size, cudaMemcpyHostToDevice);
    
    if (cudaMalloc(&h->d_edges_state, (size_t)num_edges * sizeof(int)) != cudaSuccess) return NULL;
    return (void*)h;
}

extern "C" double ramsey_cuda_evaluate(void* handle, const int* edges, int num_edges) {
    RamseyCudaHandle* h = (RamseyCudaHandle*)handle;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(h->d_edges_state, edges, (size_t)num_edges * sizeof(int), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpyToSymbol(g_mono_count, &zero, sizeof(int));
    
    int tpb = 256;
    int blocks = (h->num_cliques + tpb - 1) / tpb;
    count_mono_cliques_kernel<<<blocks, tpb>>>(h->d_edges_state, h->d_clique_edges, h->num_cliques, h->k_edges);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    static int calls = 0;
    if (calls++ % 1000 == 0) {
        printf("  [GPU Timer] N=102 evaluation: %.3f ms\n", milliseconds);
    }

    int count;
    cudaMemcpyFromSymbol(&count, g_mono_count, sizeof(int));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return (double)count;
}

extern "C" void ramsey_cuda_free(void* handle) {
    RamseyCudaHandle* h = (RamseyCudaHandle*)handle;
    if (h) {
        cudaFree(h->d_clique_edges);
        cudaFree(h->d_edges_state);
        free(h);
    }
}
