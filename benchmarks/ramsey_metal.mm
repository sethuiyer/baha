/*
 * Author: Sethurathienam Iyer
 * 
 * METAL-ACCELERATED RAMSEY SOLVER
 * Uses Apple GPU to parallelize clique checking
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "baha.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <fstream>
#include <cstring>

class RamseyMetal {
    int N, K, Colors;
    int num_edges, num_cliques, k_edges;
    
    // Metal objects
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> cliqueEdgesBuffer;
    id<MTLBuffer> edgesBuffer;
    id<MTLBuffer> countBuffer;
    id<MTLBuffer> numCliquesBuffer;
    id<MTLBuffer> kEdgesBuffer;

public:
    RamseyMetal(int n, int k, int colors) : N(n), K(k), Colors(colors) {
        num_edges = N * (N - 1) / 2;
        k_edges = K * (K - 1) / 2;
        
        // Initialize Metal
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal not available!" << std::endl;
            exit(1);
        }
        std::cout << "Using GPU: " << [device.name UTF8String] << std::endl;
        
        commandQueue = [device newCommandQueue];
        
        // Load shader
        NSError* error = nil;
        NSString* shaderPath = @"benchmarks/ramsey_metal.metal";
        NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath 
                                                           encoding:NSUTF8StringEncoding 
                                                              error:&error];
        if (error) {
            // Try metallib
            NSString* libPath = @"benchmarks/ramsey_metal.metallib";
            id<MTLLibrary> library = [device newLibraryWithFile:libPath error:&error];
            if (error) {
                std::cerr << "Failed to load shader: " << [[error localizedDescription] UTF8String] << std::endl;
                exit(1);
            }
            id<MTLFunction> function = [library newFunctionWithName:@"count_mono_cliques"];
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        } else {
            id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
            if (error) {
                std::cerr << "Shader compile error: " << [[error localizedDescription] UTF8String] << std::endl;
                exit(1);
            }
            id<MTLFunction> function = [library newFunctionWithName:@"count_mono_cliques"];
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        }
        
        // Generate cliques
        std::cout << "Precomputing cliques for N=" << N << "..." << std::endl;
        std::vector<int> cliqueIndices;
        std::vector<int> selector(N);
        std::fill(selector.end() - K, selector.end(), 1);
        
        num_cliques = 0;
        do {
            std::vector<int> verts;
            for (int i = 0; i < N; ++i) if (selector[i]) verts.push_back(i);
            
            for (int i = 0; i < K; ++i) {
                for (int j = i + 1; j < K; ++j) {
                    cliqueIndices.push_back(edge_idx(verts[i], verts[j]));
                }
            }
            num_cliques++;
        } while (std::next_permutation(selector.begin(), selector.end()));
        
        std::cout << "Cliques: " << num_cliques << " (" 
                  << (num_cliques * k_edges * 4) / (1024*1024) << " MB)" << std::endl;
        
        // Allocate GPU buffers
        cliqueEdgesBuffer = [device newBufferWithBytes:cliqueIndices.data()
                                                length:cliqueIndices.size() * sizeof(int)
                                               options:MTLResourceStorageModeShared];
        
        edgesBuffer = [device newBufferWithLength:num_edges * sizeof(int)
                                          options:MTLResourceStorageModeShared];
        
        countBuffer = [device newBufferWithLength:sizeof(int)
                                          options:MTLResourceStorageModeShared];
        
        numCliquesBuffer = [device newBufferWithBytes:&num_cliques
                                               length:sizeof(int)
                                              options:MTLResourceStorageModeShared];
        
        kEdgesBuffer = [device newBufferWithBytes:&k_edges
                                           length:sizeof(int)
                                          options:MTLResourceStorageModeShared];
    }
    
    int edge_idx(int u, int v) const {
        if (u > v) std::swap(u, v);
        return (2 * N - 1 - u) * u / 2 + v - u - 1;
    }
    
    double energy(const std::vector<int>& host_edges) {
        // Copy edges to GPU
        memcpy([edgesBuffer contents], host_edges.data(), num_edges * sizeof(int));
        
        // Reset counter
        int zero = 0;
        memcpy([countBuffer contents], &zero, sizeof(int));
        
        // Execute kernel
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:edgesBuffer offset:0 atIndex:0];
        [encoder setBuffer:cliqueEdgesBuffer offset:0 atIndex:1];
        [encoder setBuffer:countBuffer offset:0 atIndex:2];
        [encoder setBuffer:numCliquesBuffer offset:0 atIndex:3];
        [encoder setBuffer:kEdgesBuffer offset:0 atIndex:4];
        
        MTLSize gridSize = MTLSizeMake(num_cliques, 1, 1);
        NSUInteger threadGroupSize = std::min((int)pipeline.maxTotalThreadsPerThreadgroup, 256);
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Read result
        int result;
        memcpy(&result, [countBuffer contents], sizeof(int));
        return (double)result;
    }
    
    std::vector<int> random_state() {
        std::vector<int> s(num_edges);
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, Colors - 1);
        for (int i = 0; i < num_edges; ++i) s[i] = dist(rng);
        return s;
    }
    
    std::vector<std::vector<int>> neighbors(const std::vector<int>& s) {
        std::vector<std::vector<int>> nbrs;
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> edge_dist(0, num_edges - 1);
        
        for (int i = 0; i < 100; ++i) {
            std::vector<int> n = s;
            int idx = edge_dist(rng);
            n[idx] = (n[idx] + 1) % Colors;
            nbrs.push_back(n);
        }
        return nbrs;
    }
    
    void export_witness(const std::vector<int>& edges, const std::string& filename) {
        std::ofstream out(filename);
        out << "u,v,color\n";
        int idx = 0;
        for (int u = 0; u < N; ++u) {
            for (int v = u + 1; v < N; ++v) {
                out << u << "," << v << "," << edges[idx++] << "\n";
            }
        }
        std::cout << "Witness exported to " << filename << std::endl;
    }
    
    int get_N() const { return N; }
    int get_num_cliques() const { return num_cliques; }
};

int main(int argc, char** argv) {
    @autoreleasepool {
        int N = (argc > 1) ? std::stoi(argv[1]) : 102;
        
        std::cout << "============================================================" << std::endl;
        std::cout << "METAL-ACCELERATED RAMSEY SOLVER: R(5,5,5) @ N=" << N << std::endl;
        std::cout << "============================================================" << std::endl;
        
        RamseyMetal problem(N, 5, 3);
        
        auto energy = [&](const std::vector<int>& s) { return problem.energy(s); };
        auto sampler = [&]() { return problem.random_state(); };
        auto neighbors = [&](const std::vector<int>& s) { return problem.neighbors(s); };
        
        navokoj::BranchAwareOptimizer<std::vector<int>> baha(energy, sampler, neighbors);
        navokoj::BranchAwareOptimizer<std::vector<int>>::Config config;
        
        config.beta_steps = 500;
        config.beta_end = 15.0;
        config.samples_per_beta = 40;
        config.fracture_threshold = 1.8;
        config.max_branches = 8;
        config.verbose = false;
        config.schedule_type = navokoj::BranchAwareOptimizer<std::vector<int>>::ScheduleType::GEOMETRIC;
        
        // Progress logging with mutable state
        struct ProgressState {
            int last_energy = INT_MAX;
            double best_seen = 1e9;
            std::chrono::high_resolution_clock::time_point start;
        };
        auto progress = std::make_shared<ProgressState>();
        progress->start = std::chrono::high_resolution_clock::now();
        
        config.logger = [progress, &config](int step, double beta, double current_energy, double rho, const char* event) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - progress->start).count();
            
            if (current_energy < progress->best_seen) {
                progress->best_seen = current_energy;
            }
            
            int e = (int)progress->best_seen;
            bool improved = (e < progress->last_energy);
            
            if (improved || step % 50 == 0) {
                std::cout << "\r[" << std::fixed << std::setprecision(1) << elapsed << "s] "
                          << "Step " << step << "/" << config.beta_steps 
                          << " | b=" << std::setprecision(2) << beta
                          << " | E=" << e;
                
                if (strcmp(event, "branch_jump") == 0) {
                    std::cout << " [JUMP]";
                }
                if (rho > 10.0) {
                    std::cout << " | rho=" << std::setprecision(0) << rho;
                }
                
                if (improved) {
                    std::cout << " <-- NEW BEST" << std::endl;
                    progress->last_energy = e;
                } else {
                    std::cout << std::flush;
                }
            }
        };
        
        std::cout << "\nStarting BAHA with Metal GPU acceleration..." << std::endl;
        std::cout << "Checking " << problem.get_num_cliques() << " cliques per energy evaluation\n" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = baha.optimize(config);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << std::endl;
        
        std::cout << "\n============================================================" << std::endl;
        std::cout << "RESULT: R(5,5,5) @ N=" << N << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "Final Energy: " << result.best_energy << std::endl;
        std::cout << "Time: " << elapsed << " seconds" << std::endl;
        std::cout << "Fractures: " << result.fractures_detected << std::endl;
        std::cout << "Branch Jumps: " << result.branch_jumps << std::endl;
        
        if (result.best_energy == 0) {
            std::cout << "VALID! R(5,5,5) > " << N << " PROVEN!" << std::endl;
            problem.export_witness(result.best_state, "data/ramsey_" + std::to_string(N) + "_witness.csv");
        } else {
            std::cout << "Violations remaining: " << (int)result.best_energy << std::endl;
        }
        
        return 0;
    }
}
