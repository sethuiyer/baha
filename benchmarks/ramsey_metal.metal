/*
 * Author: Sethurathienam Iyer
 * Metal Compute Shader for Ramsey Clique Counting
 */

#include <metal_stdlib>
using namespace metal;

// Each thread checks one clique for monochromatic coloring
kernel void count_mono_cliques(
    device const int* edges [[buffer(0)]],           // Edge colorings
    device const int* clique_edges [[buffer(1)]],    // Flattened clique edge indices
    device atomic_int* mono_count [[buffer(2)]],     // Output counter
    constant int& num_cliques [[buffer(3)]],
    constant int& k_edges [[buffer(4)]],             // Edges per clique (10 for K5)
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)num_cliques) return;
    
    // Get this clique's edge indices
    int offset = idx * k_edges;
    
    int first_color = edges[clique_edges[offset]];
    bool is_mono = true;
    
    for (int i = 1; i < k_edges; ++i) {
        if (edges[clique_edges[offset + i]] != first_color) {
            is_mono = false;
            break;
        }
    }
    
    if (is_mono) {
        atomic_fetch_add_explicit(mono_count, 1, memory_order_relaxed);
    }
}
