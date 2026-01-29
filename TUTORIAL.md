# 🐍 BAHA Python Tutorial: Mastering Discrete Optimization

This tutorial guides you through using BAHA (Branch-Aware Holonomy Annealing) within the Python ecosystem. By the end, you'll be able to solve complex combinatorial problems that standard optimizers struggle with.

---

## 🚀 1. Installation

BAHA's core is high-performance C++17. To use it in Python, we use `pybind11` to bridge the gap.

### Prerequisites
- Python 3.7+
- A C++17 compatible compiler (GCC 7+, Clang 5+)
- `pip install pybind11`

### Build from Source
From the repository root:
```bash
python3 bindings/python/setup.py install --user
```

Verify the installation:
```python
import pybaha
print(f"BAHA Python module loaded successfully.")
```

---

## 🧩 2. The Three Pillars of BAHA

To solve a problem with BAHA, you must define three things:

1.  **Energy Function**: A Python function that takes a `state` and returns a `float`. Lower is better (Target is 0).
2.  **Sampler**: A function that generates a random starting `state`.
3.  **Neighbors (Optional but Recommended)**: A function that returns a list of "nearby" states (local moves).

---

## 📝 3. Basic Example: The Target Finder

Let's find the number 42 in a search space of integers.

```python
import pybaha
import random

# Pillar 1: Energy (Distance to 42)
def energy(x):
    return float(abs(x - 42))

# Pillar 2: Sampler (Start anywhere)
def sampler():
    return random.randint(0, 1000)

# Pillar 3: Neighbors (Step up or down)
def neighbors(x):
    return [x - 1, x + 1]

# Setup
opt = pybaha.Optimizer(energy, sampler, neighbors)
config = pybaha.Config()
config.verbose = True

# Execute
result = opt.optimize(config)

print(f"🎯 Solution: {result.best_state} (Energy: {result.best_energy})")
```

---

## 👑 4. Advanced Example: N-Queens

Solving N-Queens is a classic "hard" constraint problem.

```python
import pybaha
import random

N = 8

def energy(state):
    conflicts = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Row check or Diagonal check
            if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                conflicts += 1
    return float(conflicts)

def sampler():
    return [random.randint(0, N-1) for _ in range(N)]

def neighbors(state):
    nbrs = []
    for i in range(N):
        for val in range(N):
            if val != state[i]:
                nbr = list(state)
                nbr[i] = val
                nbrs.append(nbr)
    return nbrs

# BAHA is particularly good here because it detects the "bottlenecks" 
# where changing one queen makes things worse before they get better.
opt = pybaha.Optimizer(energy, sampler, neighbors)
result = opt.optimize()

if result.best_energy == 0:
    print(f"✅ Solved: {result.best_state}")

# Expected Output:
# 🎯 Solution: [3, 1, 6, 2, 5, 7, 4, 0] (Energy: 0.0)
# ⚡ Fractures Detected: 2
# 🔀 Branch Jumps: 2
```

---

## 📦 5. Case Study: Bin Packing (Multi-Constraint)

Bin packing is non-trivial because it involves discrete bins and strict capacity limits.

```python
import pybaha
import random

items = [10, 20, 30, 40, 50, 60, 70, 80] # Weights
BIN_CAPACITY = 100

def energy(state):
    # state[i] is the bin index for item i
    bin_totals = {}
    for i, bin_idx in enumerate(state):
        bin_totals[bin_idx] = bin_totals.get(bin_idx, 0) + items[i]
    
    violations = 0
    for total in bin_totals.values():
        if total > BIN_CAPACITY:
            violations += (total - BIN_CAPACITY)
    
    # We also want to minimize the number of bins used
    num_bins = len(bin_totals)
    return float(violations * 1000 + num_bins)

def sampler():
    return [random.randint(0, len(items)-1) for _ in range(len(items))]

# BAHA will jump branches when a bin "overflows", effectively 
# finding the optimal packing arrangement without getting 
# stuck in "nearly full" local minima.
opt = pybaha.Optimizer(energy, sampler)
result = opt.optimize()

# Expected Output:
# Final Bins Used: 4
# Fractures: 991, Jumps: 2
```

---

## 🗺️ 6. Case Study: Traveling Salesman (TSP)

TSP is a permutation problem. The "fractures" here occur when paths cross or become significantly inefficient.

```python
import pybaha
import random

# Simple 2D coordinates for 10 cities
cities = [(random.random(), random.random()) for _ in range(10)]

def dist(c1, c2):
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5

def energy(path):
    # path is a permutation of range(len(cities))
    total = sum(dist(cities[path[i]], cities[path[i-1]]) for i in range(len(path)))
    return float(total)

def sampler():
    path = list(range(len(cities)))
    random.shuffle(path)
    return path

def neighbors(path):
    # Swap two random cities
    nbrs = []
    for _ in range(20):
        nbr = list(path)
        i, j = random.sample(range(len(path)), 2)
        nbr[i], nbr[j] = nbr[j], nbr[i]
        nbrs.append(nbr)
    return nbrs

opt = pybaha.Optimizer(energy, sampler, neighbors)
result = opt.optimize()

# Expected Output:
# Final Path Distance: 2.7225
# Fractures: 975, Jumps: 7
```

---

## 🔐 7. Case Study: Cryptanalytic Key Recovery

BAHA is effectively a "search weapon" for finding keys in noisy side-channel leakage.

```python
import pybaha
import random

# Target: 32-bit key (hidden)
TARGET_KEY = 0xDEADBEEF

def energy(candidate):
    # Simulate "leakage" comparison (e.g. Hamming weight distance)
    # real cryptanalysis would use actual power traces
    leakage_distance = bin(candidate ^ TARGET_KEY).count('1')
    return float(leakage_distance)

def sampler():
    return random.getrandbits(32)

def neighbors(k):
    # Flip single bits in the key
    return [k ^ (1 << i) for i in range(32)]

# BAHA identifies "Leakage Fractures" and jumps branches to 
# reconstruct the key even with high noise.
opt = pybaha.Optimizer(energy, sampler, neighbors)
result = opt.optimize()

if result.best_state == TARGET_KEY:
    print(f"🚨 ALERT: KEY RECOVERED: {hex(result.best_state)}")

# Expected Output:
# ⚡ FRACTURE at β=0.020, ρ=15.920
#  🔀 JUMPED to E=0.000
# 🚨 ALERT: KEY RECOVERED: 0xdeadbeef
```

---

## ⚙️ 8. Tuning the Engine (`Config`)

BAHA has several parameters to handle different landscape types:

| Parameter | Default | When to Change |
| :--- | :--- | :--- |
| `beta_steps` | 500 | Increase for larger/harder problems. |
| `beta_end` | 10.0 | Increase if you are getting "stuck" near the global minimum. |
| `fracture_threshold` | 1.5 | Decrease (e.g. 1.1) to make the jump logic more aggressive. |
| `max_branches` | 5 | Increase if the landscape has many "shattered" clusters (e.g. SAT). |
| `schedule_type` | `LINEAR` | Use `GEOMETRIC` for problems with exponential energy ranges. |

---

## 📊 6. Interpreting the Results

The `Result` object tells you *how* BAHA solved the problem:

- **`best_energy`**: Final result.
- **`fractures_detected`**: How many times the landscape "broke" (high stress).
- **`branch_jumps`**: How many times BAHA used the Riemann surface to teleport to a new basin.
- **`time_ms`**: Total solve time in milliseconds.

---

## 💡 Pro-Tip: Moving Faster
Python callbacks can be slow for very large N. If you need maximum speed:
1.  Keep your state objects small (e.g. lists, not dictionaries).
2.  Use the `neighbors` function to return only a **subset** of neighbors if the full neighborhood is massive.
3.  For God-Tier speed, use the C++ API directly.
