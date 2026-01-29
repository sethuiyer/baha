import sys
import os
# Ensure the current directory is in the path so we can find pybaha
sys.path.append(os.getcwd())
import pybaha
import random

def solve_n_queens(n):
    print(f"👑 Solving {n}-Queens using BAHA (Python Binding) 👑")
    print("==================================================")
    
    # Define Energy in Python
    def energy(state):
        conflicts = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                # Same row or same diagonal
                if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                    conflicts += 1
        return float(conflicts)

    # Define Sampler in Python
    def sampler():
        return [random.randint(0, n - 1) for _ in range(n)]

    # Define Neighbor Function in Python
    def neighbors(state):
        nbrs = []
        for i in range(len(state)):
            for val in range(n):
                if val != state[i]:
                    new_state = list(state)
                    new_state[i] = val
                    nbrs.append(new_state)
        return nbrs

    # Initialize Optimizer
    opt = pybaha.Optimizer(energy, sampler, neighbors)
    
    # Configure
    config = pybaha.Config()
    config.beta_steps = 1000
    config.beta_end = 15.0
    config.verbose = True
    config.schedule_type = pybaha.ScheduleType.GEOMETRIC
    
    # Optimize
    print("Running BAHA core...")
    result = opt.optimize(config)
    
    print("\n" + "="*50)
    print(f"🎯 RESULT: Energy = {result.best_energy}")
    print(f"📍 Best State: {result.best_state}")
    print(f"⏱️  Time: {result.time_ms:.2f} ms ({result.time_ms/1000.0:.3f} s)")
    print(f"⚡ Fractures Detected: {result.fractures_detected}")
    print(f"🔀 Branch Jumps: {result.branch_jumps}")
    print("="*50)

    if result.best_energy == 0:
        print("✅ PERFECT SOLUTION FOUND!")
        # Print a simple board
        board = [["." for _ in range(n)] for _ in range(n)]
        for i, col in enumerate(result.best_state):
            board[col][i] = "Q"
        print("\n".join([" ".join(row) for row in board]))
    else:
        print("❌ CONFLICTS REMAIN")

if __name__ == "__main__":
    solve_n_queens(8)
