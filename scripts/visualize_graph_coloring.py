#!/usr/bin/env python3
"""
Visual Graph Coloring Demo with BAHA
Generates a small graph, shows the coloring as an interactive visualization.

This is a toy example for demonstrations - uses Python bindings to BAHA concept.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

def create_petersen_graph():
    """Create the famous Petersen graph - requires exactly 3 colors."""
    return nx.petersen_graph()

def create_random_graph(n=12, p=0.4, seed=42):
    """Create a random graph."""
    return nx.erdos_renyi_graph(n, p, seed=seed)

def baha_style_coloring(G, max_iterations=1000, max_colors=None):
    """
    Simulates BAHA-style graph coloring with fracture detection.
    Returns coloring and stats.
    
    If max_colors is set, constrains the coloring to use at most that many colors.
    """
    n = G.number_of_nodes()
    edges = list(G.edges())
    
    # Determine color bound
    color_bound = max_colors if max_colors else n
    
    # Initialize random coloring within bounds
    colors = {node: np.random.randint(0, color_bound) for node in G.nodes()}
    best_colors = colors.copy()
    
    def count_conflicts(c):
        return sum(1 for u, v in edges if c[u] == c[v])
    
    def count_unique_colors(c):
        return len(set(c.values()))
    
    best_conflicts = count_conflicts(colors)
    best_n_colors = count_unique_colors(colors)
    
    # BAHA-style optimization with fracture simulation
    fractures = 0
    jumps = 0
    beta = 0.01
    beta_end = 10.0
    
    log_z_prev = -beta * best_conflicts
    
    for iteration in range(max_iterations):
        # Anneal
        beta = 0.01 + (beta_end - 0.01) * (iteration / max_iterations)
        
        # Try a move: change one node's color
        node = np.random.choice(list(G.nodes()))
        old_color = colors[node]
        new_color = np.random.randint(0, color_bound)  # Stay within bound
        colors[node] = new_color
        
        new_conflicts = count_conflicts(colors)
        delta_e = new_conflicts - count_conflicts({**colors, node: old_color})
        
        # Metropolis acceptance
        if delta_e < 0 or np.random.random() < np.exp(-beta * delta_e):
            # Accept
            if new_conflicts < best_conflicts:
                best_colors = colors.copy()
                best_conflicts = new_conflicts
                best_n_colors = count_unique_colors(colors)
        else:
            colors[node] = old_color  # Reject
        
        # Simulate fracture detection
        log_z = -beta * count_conflicts(colors)
        rho = abs(log_z - log_z_prev) / (beta_end / max_iterations)
        
        if rho > 50:  # Fracture threshold
            fractures += 1
            # Simulate branch jump occasionally
            if np.random.random() < 0.02:
                jumps += 1
                # Jump: greedy reassignment of conflicting nodes
                for u, v in edges:
                    if colors[u] == colors[v]:
                        colors[u] = (colors[u] + 1) % color_bound
        
        log_z_prev = log_z
        
        # Early exit if perfect
        if best_conflicts == 0:
            break
    
    return best_colors, {
        'conflicts': best_conflicts,
        'colors_used': count_unique_colors(best_colors),
        'fractures': fractures,
        'jumps': jumps,
        'iterations': iteration + 1
    }

def visualize_coloring(G, colors, stats, title="BAHA Graph Coloring"):
    """Create a beautiful visualization of the colored graph."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color palette (pleasing colors)
    palette = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Sky Blue
        '#96CEB4',  # Sage
        '#FFEAA7',  # Yellow
        '#DDA0DD',  # Plum
        '#98D8C8',  # Mint
        '#F7DC6F',  # Gold
        '#BB8FCE',  # Purple
        '#85C1E9',  # Light Blue
    ]
    
    # Map colors to palette
    unique_colors = sorted(set(colors.values()))
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(unique_colors)}
    node_colors = [color_map[colors[node]] for node in G.nodes()]
    
    # Layout
    if G.number_of_nodes() == 10 and G.number_of_edges() == 15:
        # Petersen graph special layout
        pos = nx.shell_layout(G, nlist=[range(5, 10), range(5)])
    else:
        pos = nx.spring_layout(G, seed=42, k=2)
    
    # Draw on ax1
    ax1.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='#444466', width=2, alpha=0.7)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, 
                           node_size=800, edgecolors='white', linewidths=2)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=12, font_weight='bold', 
                            font_color='white')
    
    ax1.set_title(title, fontsize=16, fontweight='bold', color='white', pad=20)
    ax1.axis('off')
    
    # Stats panel on ax2
    ax2.set_facecolor('#1a1a2e')
    ax2.axis('off')
    
    # Create stats text
    stats_text = f"""
    🎨 BAHA Graph Coloring Results
    ══════════════════════════════
    
    📊 Graph:
       • Vertices: {G.number_of_nodes()}
       • Edges: {G.number_of_edges()}
    
    ✨ Solution:
       • Colors Used: {stats['colors_used']}
       • Conflicts: {stats['conflicts']}
       • Status: {'✅ VALID' if stats['conflicts'] == 0 else '⚠️ CONFLICTS'}
    
    ⚡ BAHA Stats:
       • Fractures Detected: {stats['fractures']}
       • Branch Jumps: {stats['jumps']}
       • Iterations: {stats['iterations']}
    
    🎯 Jump Rate: {100*stats['jumps']/max(stats['fractures'],1):.1f}%
    """
    
    ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=14,
             verticalalignment='center', fontfamily='monospace', color='white',
             bbox=dict(boxstyle='round,pad=1', facecolor='#16213e', edgecolor='#4ECDC4'))
    
    # Color legend
    legend_y = 0.15
    for i, (c, hex_color) in enumerate(color_map.items()):
        ax2.add_patch(FancyBboxPatch((0.1 + i*0.08, legend_y), 0.05, 0.05, 
                                     boxstyle="round,pad=0.01", 
                                     facecolor=hex_color, edgecolor='white',
                                     transform=ax2.transAxes))
        ax2.text(0.125 + i*0.08, legend_y - 0.03, f'C{c}', transform=ax2.transAxes,
                 fontsize=10, ha='center', color='white')
    
    plt.tight_layout()
    
    # Save
    output_path = 'data/graph_coloring_demo.png'
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', edgecolor='none',
                bbox_inches='tight')
    print(f"✅ Saved visualization to {output_path}")
    
    plt.show()
    return output_path

def main():
    print("🎨 BAHA Graph Coloring Visual Demo")
    print("=" * 40)
    
    # Create Petersen graph (classic - needs exactly 3 colors)
    print("\n📌 Testing on Petersen Graph (chromatic number = 3) with max 6 colors...")
    G = create_petersen_graph()
    
    # Run BAHA-style coloring with max 6 colors
    colors, stats = baha_style_coloring(G, max_iterations=500, max_colors=6)
    
    print(f"   Vertices: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   Colors used: {stats['colors_used']}")
    print(f"   Conflicts: {stats['conflicts']}")
    print(f"   Fractures: {stats['fractures']}")
    print(f"   Jumps: {stats['jumps']}")
    
    # Visualize
    visualize_coloring(G, colors, stats, "Petersen Graph - 6-Color BAHA")
    
    # Also try a random graph with exactly 6 colors
    print("\n📌 Testing on Random Graph G(20, 0.5) with FORCED 6 colors (harder!)...")
    G2 = create_random_graph(20, 0.5)
    colors2, stats2 = baha_style_coloring(G2, max_iterations=2000, max_colors=6)
    
    print(f"   Vertices: {G2.number_of_nodes()}")
    print(f"   Edges: {G2.number_of_edges()}")
    print(f"   Colors used: {stats2['colors_used']}")
    print(f"   Conflicts: {stats2['conflicts']}")
    print(f"   Fractures: {stats2['fractures']}")
    print(f"   Jumps: {stats2['jumps']}")
    
    visualize_coloring(G2, colors2, stats2, "G(20,0.5) FORCED 6 Colors - BAHA")

if __name__ == "__main__":
    main()
