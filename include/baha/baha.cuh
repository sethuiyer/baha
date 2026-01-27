#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace baha {
namespace device {

// ============================================================================
// CONFIG STRUCTURES
// ============================================================================

struct SwarmConfig {
    float beta_start;
    float beta_end;
    int beta_steps;
    int samples_per_beta;
    float fracture_threshold;
    unsigned long long seed;
};

// ============================================================================
// COOPERATIVE BLOCK ENERGY REDUCTION
// For problems like Ramsey where energy = count of constraints violated
// Each thread checks a subset of constraints, then warp/block reduction
// ============================================================================

// Warp-level reduction (no sync needed within warp)
__device__ __forceinline__ int warp_reduce_sum(int val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level reduction using shared memory
template<int BLOCK_SIZE>
__device__ int block_reduce_sum(int val, int* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    // Intra-warp reduction
    val = warp_reduce_sum(val);
    
    // Write warp results to shared
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // First warp reduces the warp results
    if (wid == 0) {
        val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : 0;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// ============================================================================
// RAMSEY-SPECIFIC COOPERATIVE KERNEL
// One Block = One BAHA Optimizer
// Threads cooperatively compute energy via parallel clique checking
// ============================================================================

template<int N_EDGES, int K_EDGES, int BLOCK_SIZE = 256>
__global__ void ramsey_baha_kernel(
    const int* __restrict__ cliques,   // Flattened clique edge indices
    int num_cliques,
    SwarmConfig config,
    int* best_energies,                // Output: best energy per block
    unsigned int* best_states,         // Output: best state per block (packed)
    int words_per_state                // How many 32-bit words per state
) {
    __shared__ int s_reduce[32];       // For block reduction
    __shared__ unsigned int s_state[64]; // Current state (max ~1000 edges = 63 words)
    __shared__ unsigned int s_best_state[64];
    __shared__ int s_current_energy;
    __shared__ int s_best_energy;
    __shared__ curandState s_rng;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Thread 0 initializes shared state
    if (tid == 0) {
        curand_init(config.seed + bid, 0, 0, &s_rng);
        s_best_energy = 999999;
    }
    __syncthreads();
    
    // Initialize random state (all threads cooperate)
    if (tid < words_per_state) {
        s_state[tid] = curand(&s_rng) ^ (tid * 12345);
        s_best_state[tid] = s_state[tid];
    }
    __syncthreads();
    
    // Compute initial energy (parallel clique check)
    int my_conflicts = 0;
    for (int c = tid; c < num_cliques; c += BLOCK_SIZE) {
        const int* clique = cliques + c * K_EDGES;
        int e0 = clique[0];
        int w0 = e0 / 16, b0 = (e0 % 16) * 2;
        int color = (s_state[w0] >> b0) & 3;
        
        bool mono = true;
        for (int k = 1; k < K_EDGES; ++k) {
            int e = clique[k];
            int w = e / 16, b = (e % 16) * 2;
            if (((s_state[w] >> b) & 3) != color) {
                mono = false;
                break;
            }
        }
        if (mono) my_conflicts++;
    }
    
    int total = block_reduce_sum<BLOCK_SIZE>(my_conflicts, s_reduce);
    if (tid == 0) {
        s_current_energy = total;
        s_best_energy = total;
    }
    __syncthreads();
    
    // Annealing loop
    float beta = config.beta_start;
    float rate = powf(config.beta_end / config.beta_start, 1.0f / config.beta_steps);
    
    for (int step = 0; step < config.beta_steps; ++step) {
        for (int sample = 0; sample < config.samples_per_beta; ++sample) {
            // Thread 0 proposes a mutation
            int flip_edge, flip_word, flip_bit;
            unsigned int old_val, new_val;
            
            if (tid == 0) {
                flip_edge = curand(&s_rng) % N_EDGES;
                flip_word = flip_edge / 16;
                flip_bit = (flip_edge % 16) * 2;
                old_val = s_state[flip_word];
                int old_color = (old_val >> flip_bit) & 3;
                int new_color = (old_color + 1 + (curand(&s_rng) % 2)) % 3;
                new_val = (old_val & ~(3 << flip_bit)) | (new_color << flip_bit);
                s_state[flip_word] = new_val; // Apply mutation
            }
            __syncthreads();
            
            // Recompute energy with mutation
            my_conflicts = 0;
            for (int c = tid; c < num_cliques; c += BLOCK_SIZE) {
                const int* clique = cliques + c * K_EDGES;
                int e0 = clique[0];
                int w0 = e0 / 16, b0 = (e0 % 16) * 2;
                int color = (s_state[w0] >> b0) & 3;
                
                bool mono = true;
                for (int k = 1; k < K_EDGES; ++k) {
                    int e = clique[k];
                    int w = e / 16, b = (e % 16) * 2;
                    if (((s_state[w] >> b) & 3) != color) {
                        mono = false;
                        break;
                    }
                }
                if (mono) my_conflicts++;
            }
            
            total = block_reduce_sum<BLOCK_SIZE>(my_conflicts, s_reduce);
            
            // Thread 0 does Metropolis accept/reject
            if (tid == 0) {
                int new_energy = total;
                int delta = new_energy - s_current_energy;
                
                bool accept = (delta < 0) || 
                              (curand_uniform(&s_rng) < expf(-beta * delta));
                
                if (accept) {
                    s_current_energy = new_energy;
                    if (new_energy < s_best_energy) {
                        s_best_energy = new_energy;
                        for (int i = 0; i < words_per_state; ++i)
                            s_best_state[i] = s_state[i];
                    }
                } else {
                    // Reject - revert mutation
                    s_state[flip_word] = old_val;
                }
            }
            __syncthreads();
            
            // Early exit if solved
            if (s_best_energy == 0) goto done;
        }
        
        beta *= rate;
    }
    
done:
    // Write results
    if (tid == 0) {
        best_energies[bid] = s_best_energy;
    }
    if (tid < words_per_state) {
        best_states[bid * words_per_state + tid] = s_best_state[tid];
    }
}

} // namespace device
} // namespace baha
