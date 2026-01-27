#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iomanip>

// =============================================================================
// HP PROTEIN FOLDING (2D Lattice) w/ Visualization
// =============================================================================
#define MAX_N 64
#define BLOCK_SIZE 256 

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// -----------------------------------------------------------------------------
// KERNEL: Energy Calc
// -----------------------------------------------------------------------------
__device__ int calculate_energy_dev(const int* sequence, const int* moves, int n) {
    int x[MAX_N];
    int y[MAX_N];
    
    x[0] = 0; y[0] = 0;
    
    for(int i=0; i<n-1; ++i) {
        int dir = moves[i]; 
        int dx = 0, dy = 0;
        if(dir==0) dy=1;
        else if(dir==1) dy=-1;
        else if(dir==2) dx=-1;
        else if(dir==3) dx=1;
        
        x[i+1] = x[i] + dx;
        y[i+1] = y[i] + dy;
    }
    
    // Self-avoidance (Soft Penalty)
    int collisions = 0;
    for(int i=0; i<n; ++i) {
        for(int j=i+1; j<n; ++j) {
            if (x[i] == x[j] && y[i] == y[j]) collisions++;
        }
    }
    
    // H-H contacts
    int contacts = 0;
    for(int i=0; i<n; ++i) {
        if(sequence[i] != 1) continue; 
        for(int j=i+2; j<n; ++j) { 
             if(sequence[j] != 1) continue;
             int dist = abs(x[i] - x[j]) + abs(y[i] - y[j]);
             if (dist == 1) contacts++; 
        }
    }
    // Energy = -Contacts + Penalty * Collisions
    return -contacts + (collisions * 50);
}

// -----------------------------------------------------------------------------
// KERNEL: Setup
// -----------------------------------------------------------------------------
__global__ void setup_kernel(int n, curandState* states, int* global_population_moves, int* global_population_energies, const int* d_sequence, unsigned long long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Init curand
    curand_init(seed, tid, 0, &states[tid]);
    
    // Init random moves
    int* my_moves = &global_population_moves[tid * MAX_N];
    for(int i=0; i<n-1; ++i) {
        my_moves[i] = curand(&states[tid]) % 4;
    }
    
    global_population_energies[tid] = calculate_energy_dev(d_sequence, my_moves, n);
}

// -----------------------------------------------------------------------------
// KERNEL: Fold Step
// -----------------------------------------------------------------------------
__global__ void fold_step_kernel(
    int n, 
    curandState* states, 
    int* global_population_moves, 
    int* global_population_energies, 
    const int* d_sequence, 
    float beta, 
    int loops_per_launch,
    int* global_best_moves,
    int* global_best_energy
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState local_state = states[tid];
    
    int my_moves[MAX_N];
    int* global_ptr = &global_population_moves[tid * MAX_N];
    
    // Load state
    for(int i=0; i<n-1; ++i) my_moves[i] = global_ptr[i];
    int current_E = global_population_energies[tid];
    
    // Shared memory for fracture communication
    __shared__ int block_best_E;
    __shared__ int block_best_moves[MAX_N];
    
    if (threadIdx.x == 0) block_best_E = 999999;
    __syncthreads();
    
    // Metropolis Loop
    for(int step=0; step<loops_per_launch; ++step) {
        int idx = curand(&local_state) % (n-1);
        int old_move = my_moves[idx];
        int new_move = curand(&local_state) % 4;
        
        my_moves[idx] = new_move;
        int new_E = calculate_energy_dev(d_sequence, my_moves, n);
        
        float delta = (float)(new_E - current_E);
        if (delta <= 0 || curand_uniform(&local_state) < expf(-delta * beta)) {
            current_E = new_E;
        } else {
            my_moves[idx] = old_move; // Revert
        }
    }
    
    // --- BAHA FRACTURE JUMP ---
    // 1. Share best within block
    atomicMin(&block_best_E, current_E);
    __syncthreads();
    
    if (current_E == block_best_E) {
        // Winner writes moves to shared (potential race condition but ok for demo)
        for(int k=0; k<n-1; ++k) block_best_moves[k] = my_moves[k];
        
        // Winner also updates global best (approx)
        atomicMin(global_best_energy, current_E);
        if (*global_best_energy == current_E) {
             for(int k=0; k<n-1; ++k) global_best_moves[k] = my_moves[k];
        }
    }
    __syncthreads();
    
    // 2. Jump if valid fracture detected
    // If block best is much better than me, jump
    if (block_best_E < current_E - 2) {
        for(int k=0; k<n-1; ++k) my_moves[k] = block_best_moves[k];
        current_E = block_best_E;
    }
    
    // Save state back to global
    for(int i=0; i<n-1; ++i) global_ptr[i] = my_moves[i];
    global_population_energies[tid] = current_E;
    states[tid] = local_state;
}


// -----------------------------------------------------------------------------
// CPU HOST
// -----------------------------------------------------------------------------
int main() {
    std::cout << "🧬 PROTIEN FOLDING ANIMATOR 🧬\n";
    
    // Bench: Slightly longer, hard sequence
    int n = 50;
    std::vector<int> h_sequence(n);
    // HPH pattern
    for(int i=0; i<n; ++i) h_sequence[i] = (i % 2 == 0 || i % 5 == 0) ? 1 : 0;
    
    // Device allocs
    int num_blocks = 128;
    int threads_per_block = 256;
    int total_threads = num_blocks * threads_per_block;
    
    int* d_sequence; cudaMalloc(&d_sequence, n * sizeof(int));
    cudaMemcpy(d_sequence, h_sequence.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    
    curandState* d_states; cudaMalloc(&d_states, total_threads * sizeof(curandState));
    
    int* d_pop_moves; cudaMalloc(&d_pop_moves, total_threads * MAX_N * sizeof(int));
    int* d_pop_enes; cudaMalloc(&d_pop_enes, total_threads * sizeof(int));
    
    int* d_best_moves; cudaMalloc(&d_best_moves, MAX_N * sizeof(int));
    int* d_best_energy; cudaMalloc(&d_best_energy, sizeof(int));
    
    int h_init_best = 999999;
    cudaMemcpy(d_best_energy, &h_init_best, sizeof(int), cudaMemcpyHostToDevice);
    
    // Log setup
    std::ofstream log("protein_log.csv");
    log << "frame,energy,moves\n";
    
    // 1. Setup
    setup_kernel<<<num_blocks, threads_per_block>>>(n, d_states, d_pop_moves, d_pop_enes, d_sequence, time(NULL));
    cudaCheckError(cudaDeviceSynchronize());
    
    // 2. Loop
    float beta = 0.1f;
    float beta_end_target = 10.0f;
    int total_frames = 100;
    float beta_step = pow(beta_end_target / beta, 1.0f / total_frames);
    
    std::vector<int> h_best_moves(MAX_N);
    int h_best_e = 999999;
    
    for(int frame=0; frame<total_frames; ++frame) {
        // Run burst
        fold_step_kernel<<<num_blocks, threads_per_block>>>(
            n, d_states, d_pop_moves, d_pop_enes, d_sequence, beta, 100, // 100 metropolis steps per frame
            d_best_moves, d_best_energy
        );
        cudaCheckError(cudaDeviceSynchronize());
        
        // Read best
        int current_global_best_e;
        cudaMemcpy(&current_global_best_e, d_best_energy, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Only verify moves if energy improved or periodically
        cudaMemcpy(h_best_moves.data(), d_best_moves, (n-1)*sizeof(int), cudaMemcpyDeviceToHost);
        
        std::cout << "Frame " << frame << " (Beta=" << std::fixed << std::setprecision(2) << beta << "): Best E=" << current_global_best_e << "\r";
        std::cout.flush();
        
        // Write to log
        log << frame << "," << current_global_best_e << ",";
        for(int i=0; i<n-1; ++i) log << h_best_moves[i];
        log << "\n";
        
        beta *= beta_step;
    }
    
    log.close();
    std::cout << "\nDone! Log saved to protein_log.csv\n";
    
    cudaFree(d_sequence);
    cudaFree(d_states);
    cudaFree(d_pop_moves);
    cudaFree(d_pop_enes);
    cudaFree(d_best_moves);
    cudaFree(d_best_energy);
    
    return 0;
}
