<p align="center">
  <img src="logo.png" alt="BAHA Logo" width="300"/>
</p>

# BAHA: Branch-Aware Holonomy Annealing

> **Better than Simulated Annealing. Uses phase transitions (fractures) to escape local minima.**

[![Watch the Presentation](https://img.shields.io/badge/YouTube-Watch%20Presentation-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=jVKetFO7SgM)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18373732.svg)](https://doi.org/10.5281/zenodo.18373732)

---

## How It Works

**BAHA is a topology navigator, not just a hill-climber.**

Most optimizers get stuck in local minima because they treat the landscape as one smooth surface. BAHA detects when the solution space **shatters** (a thermodynamic fracture) and uses complex-plane branch enumeration to jump to entirely different solution basins.

![How BAHA Works](howitworks.png)
![How BAHA Works 2](howitworks2.png)
![Problem Type](problemtype.png)

> **Key Concept**: When the problem gets hard, it "cracks". BAHA hears the crack and jumps through it.

[View Example Code: List Coloring with NetworkX](examples/list_coloring_networkx.py)

---

## Python-First Simplicity

Get the raw speed of a C++17 engine with the usability of a Python script.

![Python API](pyapi.png)

### Quick Start

```python
import pybaha
import random

# 1. Define Energy (e.g. N-Queens conflict count)
def energy(state): return count_conflicts(state)

# 2. Define Moves
def neighbors(state): return [swap_two_queens(state) for _ in range(10)]

# 3. Solve
opt = pybaha.Optimizer(energy, random_sampler, neighbors)
result = opt.optimize(pybaha.Config(timeout_ms=5000))

print(f"Result: {result.best_state}")
# ⚡ Fractures Detected: 5 | 🔀 Jumps: 1
```

---

## Proven Results

BAHA isn't just theory. It dominates on structured combinatorial problems.

Some of the noteworthy results are:

| Benchmark | Result | Notes |
|-----------|--------|-------|
| **Ramsey R(5,5,5)** | **N > 52** | Solved 2.6M constraints in <1s. [Verifiable Witness](data/ramsey_52_witness.csv) |
| **Graph Isomorphism** | **100% Success** | vs 20% for Simulated Annealing (N=50) |
| **DNA Barcode** | **Perfect (0 Violations)** | First application of fracture-aware optimization to bioinformatics |

---

## Installation

### Python
```bash
python3 bindings/python/setup.py install --user
```

### C++ (Header-Only)
```bash
mkdir build && cd build && cmake .. && make -j
```

---

## Documentation


For deep technical details, theory, and C++ API reference, see the [docs/](docs/) folder.

### Documentation Index

| File | Category | Content Summary | Target |
| :--- | :--- | :--- | :--- |
| **[README.md](README.md)** | Landing | **Main Entry Point**. Overview, installation, quickstart. | Everyone |
| **[QUICKSTART.md](docs/guides/QUICKSTART.md)** | Guide | **10-Minute Guide**. Python vs CLI usage, zero-code examples. | New Users |
| **[TUTORIAL.md](docs/guides/TUTORIAL.md)** | Tutorial | **Python Tutorial**. Step-by-step code for N-Queens, TSP. | Developers |
| **[complexity.md](docs/technical/complexity.md)** | Technical | **Algorithmic Analysis**. Big-O complexity ($O(B \cdot N \cdot S)$). | Researchers |
| **[CASE_STUDY.md](docs/technical/CASE_STUDY.md)** | Technical | **Deep Dive**. Detailed results for 26+ domains. | Researchers |
| **[PROBLEM_LIST.md](docs/technical/PROBLEM_LIST.md)** | Technical | **Catalog**. List of 26 tested problems w/ fracture types. | Researchers |
| **[OPTIMIZATION_GUIDE.md](docs/technical/OPTIMIZATION_GUIDE.md)** | Technical | **Performance**. Low-level C++/CUDA optimizations. | Contributors |

---

## Citation

If you use BAHA in your research:

```bibtex
@article{iyer2026multiplicative,
  title={Multiplicative Calculus for Hardness Detection and Branch-Aware Optimization},
  author={Iyer, Sethurathienam},
  year={2026},
  doi={10.5281/zenodo.18373732}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE).


Made with <3 at ShunyaBar Labs - https://shunyabar.foo/
