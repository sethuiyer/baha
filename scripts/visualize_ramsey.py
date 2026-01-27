import numpy as np
import matplotlib.pyplot as plt

def plot_ramsey(adj_file, n_nodes, output_file):
    try:
        with open(adj_file, 'r') as f:
            edges = [int(x) for x in f.read().split()]
    except Exception as e:
        print(f"Error reading solution file: {e}")
        return

    # Create circular layout
    theta = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    plt.figure(figsize=(12, 12), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # Define colors
    colors = ['#FF2244', '#22FF44', '#2244FF'] # Vibrant R, G, B
    
    e_idx = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            color = colors[edges[e_idx]]
            plt.plot([x[i], x[j]], [y[i], y[j]], color=color, alpha=0.1, linewidth=0.5)
            e_idx += 1
            
    # Plot nodes
    plt.scatter(x, y, c='white', s=20, zorder=10)
    
    plt.title(f"Ramsey R(5,5,5) Constructive Proof | N={n_nodes}", color='white', fontsize=16)
    plt.axis('off')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    import sys
    n = 102
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    plot_ramsey("benchmarks/solution_102.adj", n, "ramsey_102.png")
