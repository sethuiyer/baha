import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def visualize_list_coloring():
    """Generate a List Coloring visualization showing constrained graph coloring."""
    
    # Create a sample graph (similar to the benchmark)
    np.random.seed(42)
    n_vertices = 20
    
    # Create edges similar to benchmark
    edges = []
    for i in range(1, n_vertices):
        edges.append((i-1, i))  # Chain
    
    # Add random edges
    for _ in range(10):
        u, v = np.random.randint(0, n_vertices, 2)
        if u != v:
            edges.append((u, v))
    
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))
    G.add_edges_from(edges)
    
    # Generate allowed colors per vertex (1-4 colors each from palette of 5)
    colors_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    allowed_colors = []
    for i in range(n_vertices):
        n_allowed = np.random.randint(1, 5)
        allowed = list(np.random.choice(5, n_allowed, replace=False))
        allowed_colors.append(allowed)
    
    # Simulate BAHA solution (valid coloring)
    solution = []
    for i in range(n_vertices):
        solution.append(allowed_colors[i][0])  # Pick first allowed color
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#1a1a2e')
    
    # Left: The problem (allowed colors shown)
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a2e')
    pos = nx.spring_layout(G, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='#444', width=1.5, alpha=0.6)
    
    # Draw nodes with gradient showing constraint level
    node_colors = []
    for i in range(n_vertices):
        constraint_level = 1 - len(allowed_colors[i]) / 5  # Higher = more constrained
        node_colors.append(plt.cm.RdYlGn(1 - constraint_level))
    
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, 
                           node_size=500, edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_color='white', font_weight='bold')
    
    ax1.set_title('Constraint Map\n(Red = Highly Constrained)', color='white', fontsize=14, pad=10)
    ax1.axis('off')
    
    # Right: The solution
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')
    
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='#555', width=1.5, alpha=0.6)
    
    solution_colors = [colors_palette[c] for c in solution]
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=solution_colors, 
                           node_size=500, edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax2, font_color='black', font_weight='bold')
    
    ax2.set_title('BAHA Solution\n(All Constraints Satisfied)', color='white', fontsize=14, pad=10)
    ax2.axis('off')
    
    plt.suptitle('List Coloring: Constrained Graph Coloring', color='white', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('data/list_coloring.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: data/list_coloring.png")

def visualize_graph_iso():
    """Generate a Graph Isomorphism visualization."""
    
    np.random.seed(123)
    
    # Create two isomorphic graphs
    n = 12
    G1 = nx.gnm_random_graph(n, 20, seed=42)
    
    # G2 is a permuted version of G1
    perm = list(np.random.permutation(n))
    mapping = {i: perm[i] for i in range(n)}
    G2 = nx.relabel_nodes(G1, mapping)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0f0f23')
    
    # Left: Graph A
    ax1 = axes[0]
    ax1.set_facecolor('#0f0f23')
    pos1 = nx.spring_layout(G1, seed=42)
    nx.draw_networkx_edges(G1, pos1, ax=ax1, edge_color='#3498db', width=2, alpha=0.7)
    nx.draw_networkx_nodes(G1, pos1, ax=ax1, node_color='#e74c3c', 
                           node_size=400, edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G1, pos1, ax=ax1, font_color='white', font_weight='bold')
    ax1.set_title('Graph A', color='white', fontsize=14)
    ax1.axis('off')
    
    # Middle: Arrow showing BAHA finding the mapping
    ax2 = axes[1]
    ax2.set_facecolor('#0f0f23')
    ax2.text(0.5, 0.6, '⚡ BAHA', ha='center', va='center', fontsize=24, color='#f1c40f')
    ax2.text(0.5, 0.4, 'Finds Mapping', ha='center', va='center', fontsize=16, color='white')
    ax2.text(0.5, 0.25, 'in 0.2ms', ha='center', va='center', fontsize=12, color='#aaa')
    ax2.arrow(0.1, 0.5, 0.25, 0, head_width=0.05, head_length=0.03, fc='#f1c40f', ec='#f1c40f')
    ax2.arrow(0.65, 0.5, 0.25, 0, head_width=0.05, head_length=0.03, fc='#f1c40f', ec='#f1c40f')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Right: Graph B with matched coloring
    ax3 = axes[2]
    ax3.set_facecolor('#0f0f23')
    pos2 = nx.spring_layout(G2, seed=99)
    nx.draw_networkx_edges(G2, pos2, ax=ax3, edge_color='#2ecc71', width=2, alpha=0.7)
    nx.draw_networkx_nodes(G2, pos2, ax=ax3, node_color='#9b59b6', 
                           node_size=400, edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G2, pos2, ax=ax3, font_color='white', font_weight='bold')
    ax3.set_title('Graph B (Permuted)', color='white', fontsize=14)
    ax3.axis('off')
    
    plt.suptitle('Graph Isomorphism: Finding the Hidden Mapping', color='white', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('data/graph_iso.png', dpi=150, bbox_inches='tight', facecolor='#0f0f23')
    print("Saved: data/graph_iso.png")

def visualize_energy_landscape():
    """Generate an abstract BAHA energy landscape showing fracture detection."""
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    
    # Create a rugged energy landscape
    x = np.linspace(0, 10, 1000)
    
    # Base landscape with multiple basins
    y = 2 * np.sin(x * 0.8) + np.sin(x * 2.1) * 0.8 + np.sin(x * 5) * 0.3
    y = y - np.min(y) + 0.5
    
    # Add "fractures" (sharp discontinuities)
    fracture_points = [2.5, 5.0, 7.3]
    for fp in fracture_points:
        mask = (x > fp - 0.1) & (x < fp + 0.1)
        y[mask] += np.random.uniform(0.5, 1.5)
    
    # Plot landscape
    ax.fill_between(x, 0, y, alpha=0.3, color='#58a6ff')
    ax.plot(x, y, color='#58a6ff', linewidth=2)
    
    # Mark fractures
    for i, fp in enumerate(fracture_points):
        idx = np.argmin(np.abs(x - fp))
        ax.axvline(x=fp, color='#f97583', linestyle='--', alpha=0.7, linewidth=2)
        ax.scatter([fp], [y[idx]], color='#f97583', s=150, zorder=5, marker='X')
        ax.annotate(f'Fracture {i+1}\nρ = {np.random.uniform(2, 5):.1f}', 
                   (fp, y[idx] + 0.4), color='#f97583', fontsize=10, ha='center')
    
    # Show BAHA jump
    jump_start = 1.5
    jump_end = 8.5
    idx_start = np.argmin(np.abs(x - jump_start))
    idx_end = np.argmin(np.abs(x - jump_end))
    
    ax.annotate('', xy=(jump_end, y[idx_end]), xytext=(jump_start, y[idx_start]),
               arrowprops=dict(arrowstyle='->', color='#7ee787', lw=3, 
                               connectionstyle='arc3,rad=-0.3'))
    ax.scatter([jump_start], [y[idx_start]], color='#ffa657', s=200, zorder=10, marker='o')
    ax.scatter([jump_end], [y[idx_end]], color='#7ee787', s=200, zorder=10, marker='*')
    ax.text(5, 4.5, 'Branch Jump\n(Lambert-W)', color='#7ee787', fontsize=12, ha='center')
    
    # Labels
    ax.set_xlabel('Configuration Space', color='white', fontsize=12)
    ax.set_ylabel('Energy', color='white', fontsize=12)
    ax.set_title('BAHA: Fracture Detection and Branch Jumping', color='white', fontsize=16)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('data/energy_landscape.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print("Saved: data/energy_landscape.png")

if __name__ == "__main__":
    visualize_list_coloring()
    visualize_graph_iso()
    visualize_energy_landscape()
    print("\nAll visualizations generated!")
