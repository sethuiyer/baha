# 🚀 BAHA Quickstart (10 Minutes)

BAHA (Branch-Aware Holonomy Annealing) is a next-generation optimizer designed for hard, discrete landscapes. It works by detecting "fractures" in the energy surface and jumping across them using complex-plane branch enumeration.

You don't need to be a C++ expert to use it.

---

## 🐍 Option 1: Python (High Accessibility)

The Python binding allows you to define problems in pure Python while the C++ core handles the heavy lifting.

### 1. Build and Install
```bash
# In the project root
python3 bindings/python/setup.py install --user
```

### 2. Solve a Problem
```python
import pybaha
import random

# 1. Define your energy function (Problem: target the number 42)
def energy(x): 
    return float((x - 42)**2)

# 2. Define how to pick a random starting point
def sampler(): 
    return random.randint(0, 100)

# 3. Create and run the optimizer
opt = pybaha.Optimizer(energy, sampler)
result = opt.optimize()

print(f"🎯 Target Found: {result.best_state}")
print(f"⚡ Jumps Made: {result.branch_jumps}")
```

---

## 💻 Option 2: CLI (Zero-Code)

If you have a standard problem (like Number Partitioning), you can solve it via JSON without writing any code.

### 1. Compile the CLI
```bash
g++ -O3 -std=c++17 -I include src/baha_cli.cpp -o baha-cli
```

### 2. Run a Problem
Create a `problem.json`:
```json
{
  "problem": "number_partitioning",
  "data": [10, 25, 40, 13, 9, 22, 31],
  "config": {
    "beta_steps": 500,
    "verbose": false
  }
}
```

Solve it:
```bash
./baha-cli problem.json
```

---

## 🎨 Visualization

To see the engine solving a graph coloring problem in real-time with fracture detection:
```bash
python3 scripts/visualize_graph_coloring.py
```

Check the results in `data/graph_coloring_demo.png`.

---

## 🏁 Summary of Tools
- **pybind11**: The bridge between Python and C++.
- **nlohmann/json**: The bridge between JSON specifications and C++.
- **BAHA Core**: The pure C++ engine doing the work.
