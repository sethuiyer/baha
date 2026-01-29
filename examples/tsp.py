import sys
import os
sys.path.append(os.getcwd())
import pybaha
import random

# Fixed seed for reproducibility of coordinates
random.seed(42)
cities = [(random.random(), random.random()) for _ in range(12)]

def dist(c1, c2):
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5

def energy(path):
    total = sum(dist(cities[path[i]], cities[path[i-1]]) for i in range(len(path)))
    return float(total)

def sampler():
    path = list(range(len(cities)))
    random.shuffle(path)
    return path

def neighbors(path):
    # Swap two random cities
    nbrs = []
    # Test a few swaps
    for _ in range(20):
        nbr = list(path)
        i, j = random.sample(range(len(path)), 2)
        nbr[i], nbr[j] = nbr[j], nbr[i]
        nbrs.append(nbr)
    return nbrs

print("🗺️ RUNNING TSP TUTORIAL 🗺️")
opt = pybaha.Optimizer(energy, sampler, neighbors)
config = pybaha.Config()
config.verbose = True
config.beta_steps = 1000
result = opt.optimize(config)

print(f"\nFinal Path Distance: {result.best_energy:.4f}")
print(f"Fractures: {result.fractures_detected}, Jumps: {result.branch_jumps}")
