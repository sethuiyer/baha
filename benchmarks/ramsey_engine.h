/*
 * Author: Sethurathienam Iyer
 */
#ifndef RAMSEY_ENGINE_H
#define RAMSEY_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for GPU data
// Using uint16_t for clique edges to save 50% memory (needed for N=102)
void* ramsey_cuda_init(const uint16_t* clique_edges, int num_cliques, int k_edges, int num_edges);
double ramsey_cuda_evaluate(void* handle, const int* edges, int num_edges);
void ramsey_cuda_free(void* handle);

#ifdef __cplusplus
}
#endif

#endif
