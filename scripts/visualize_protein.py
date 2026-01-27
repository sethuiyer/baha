import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Load Data
df = pd.read_csv('protein_log.csv')

# Reconstruct Sequence (Must match C++ logic)
N = 50
sequence = []
for i in range(N):
    is_h = (i % 2 == 0 or i % 5 == 0)
    sequence.append(1 if is_h else 0) # 1=H (Red), 0=P (Blue)

# Function to get coordinates from moves string
def get_coords(moves_str):
    x, y = [0], [0]
    curr_x, curr_y = 0, 0
    # Moves string length is N-1
    # moves_str might be int or string in CSV
    moves = str(moves_str)
    
    for char in moves:
        move = int(char)
        if move == 0: curr_y += 1 # U
        elif move == 1: curr_y -= 1 # D
        elif move == 2: curr_x -= 1 # L
        elif move == 3: curr_x += 1 # R
        x.append(curr_x)
        y.append(curr_y)
    return x, y

moves_list = df['moves'].tolist()
energies = df['energy'].tolist()
frames = df['frame'].tolist()

# Setup Plot
fig, ax = plt.subplots(figsize=(8, 8))
plt.title("BAHA Protein Folding (GPU Swarm)", fontsize=16)

def update(frame_idx):
    ax.clear()
    
    moves = moves_list[frame_idx]
    energy = energies[frame_idx]
    frame_num = frames[frame_idx]
    
    x, y = get_coords(moves)
    
    # Plot Backbone
    ax.plot(x, y, color='gray', linestyle='-', linewidth=1, zorder=1)
    
    # Plot Residues
    for i in range(N):
        color = 'red' if sequence[i] == 1 else 'blue'
        label = 'H' if sequence[i] == 1 else 'P'
        ax.scatter(x[i], y[i], color=color, s=100, zorder=2, edgecolors='black')
        # Optional: Label index
        # ax.text(x[i], y[i], str(i), fontsize=8, ha='center', va='center', color='white')
    
    # Highlight start
    ax.scatter(x[0], y[0], color='green', s=150, zorder=3, marker='*', label='Start')
    
    # Formatting
    ax.set_title(f"Frame {frame_num} | Energy: {energy}", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Dynamic limits
    margin = 2
    ax.set_xlim(min(x)-margin, max(x)+margin)
    ax.set_ylim(min(y)-margin, max(y)+margin)
    
    return ax,

ani = animation.FuncAnimation(fig, update, frames=len(df), interval=100)
ani.save('/home/sethuiyer/.gemini/antigravity/brain/5d38eba0-0543-40af-b1b2-c55f3d2ce459/protein_folding.gif', writer='pillow', fps=10)
print("Animation saved!")
