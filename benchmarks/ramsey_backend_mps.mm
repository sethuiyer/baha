#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ramsey_backend.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace {

class MpsBackend final : public RamseyBackend {
public:
    MpsBackend(std::vector<int> clique_edges, int num_cliques, int k_edges, int num_edges)
        : clique_edges_(std::move(clique_edges)),
          num_cliques_(num_cliques),
          k_edges_(k_edges),
          num_edges_(num_edges) {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            std::cerr << "Metal not available.\n";
            return;
        }

        command_queue_ = [device_ newCommandQueue];
        pipeline_ = load_pipeline();

        clique_edges_buffer_ = [device_ newBufferWithBytes:clique_edges_.data()
                                                   length:clique_edges_.size() * sizeof(int)
                                                  options:MTLResourceStorageModeShared];

        edges_buffer_ = [device_ newBufferWithLength:num_edges_ * sizeof(int)
                                             options:MTLResourceStorageModeShared];

        count_buffer_ = [device_ newBufferWithLength:sizeof(int)
                                             options:MTLResourceStorageModeShared];

        num_cliques_buffer_ = [device_ newBufferWithBytes:&num_cliques_
                                                  length:sizeof(int)
                                                 options:MTLResourceStorageModeShared];

        k_edges_buffer_ = [device_ newBufferWithBytes:&k_edges_
                                              length:sizeof(int)
                                             options:MTLResourceStorageModeShared];
    }

    double energy(const std::vector<int>& edges) override {
        // Assumes edges.size() == num_edges_; returns large penalty otherwise.
        if (!pipeline_ || edges.size() != static_cast<size_t>(num_edges_)) return 1e18;

        std::memcpy([edges_buffer_ contents], edges.data(), num_edges_ * sizeof(int));

        int zero = 0;
        std::memcpy([count_buffer_ contents], &zero, sizeof(int));

        id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:edges_buffer_ offset:0 atIndex:0];
        [encoder setBuffer:clique_edges_buffer_ offset:0 atIndex:1];
        [encoder setBuffer:count_buffer_ offset:0 atIndex:2];
        [encoder setBuffer:num_cliques_buffer_ offset:0 atIndex:3];
        [encoder setBuffer:k_edges_buffer_ offset:0 atIndex:4];

        MTLSize grid_size = MTLSizeMake(num_cliques_, 1, 1);
        NSUInteger tg_size = std::min((int)pipeline_.maxTotalThreadsPerThreadgroup, 256);
        MTLSize threadgroup_size = MTLSizeMake(tg_size, 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        int result = 0;
        std::memcpy(&result, [count_buffer_ contents], sizeof(int));
        return static_cast<double>(result);
    }

    const char* name() const noexcept override { return "mps"; }

private:
    id<MTLComputePipelineState> load_pipeline() {
        NSError* error = nil;
        // Assumes working directory is repo root; falls back to metallib if needed.
        NSString* shader_path = @"benchmarks/ramsey_metal.metal";
        NSString* shader_source = [NSString stringWithContentsOfFile:shader_path
                                                            encoding:NSUTF8StringEncoding
                                                               error:&error];
        if (error) {
            NSString* lib_path = @"benchmarks/ramsey_metal.metallib";
            id<MTLLibrary> library = [device_ newLibraryWithFile:lib_path error:&error];
            if (error) {
                std::cerr << "Failed to load Metal shader: " << [[error localizedDescription] UTF8String] << '\n';
                return nil;
            }
            id<MTLFunction> function = [library newFunctionWithName:@"count_mono_cliques"];
            return [device_ newComputePipelineStateWithFunction:function error:&error];
        }

        id<MTLLibrary> library = [device_ newLibraryWithSource:shader_source options:nil error:&error];
        if (error) {
            std::cerr << "Shader compile error: " << [[error localizedDescription] UTF8String] << '\n';
            return nil;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"count_mono_cliques"];
        return [device_ newComputePipelineStateWithFunction:function error:&error];
    }

    std::vector<int> clique_edges_;
    int num_cliques_ = 0;
    int k_edges_ = 0;
    int num_edges_ = 0;

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> command_queue_ = nil;
    id<MTLComputePipelineState> pipeline_ = nil;
    id<MTLBuffer> clique_edges_buffer_ = nil;
    id<MTLBuffer> edges_buffer_ = nil;
    id<MTLBuffer> count_buffer_ = nil;
    id<MTLBuffer> num_cliques_buffer_ = nil;
    id<MTLBuffer> k_edges_buffer_ = nil;
};

} // namespace

std::unique_ptr<RamseyBackend> create_mps_backend(
    const std::vector<int>& clique_edges,
    int num_cliques,
    int k_edges,
    int num_edges) {
    return std::make_unique<MpsBackend>(clique_edges, num_cliques, k_edges, num_edges);
}
