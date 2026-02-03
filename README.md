<p align="center">
  <img src="logo.png" alt="BAHA Logo" width="300"/>
</p>

# BAHA: Branch-Aware Holonomy Annealing

> **Simulated annealing which uses thermodynamics of the landscape to escape local minima.**

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

# 3. Solve with one line!
result = pybaha.optimize(energy, random_sampler, neighbors)

print(f"Result: {result.best_state}, Energy: {result.best_energy}")
# ⚡ Fracture Density: 0.92 | Time: 345ms
```

### 🔮 ZetaOptimizer (High-Performance Hybrid)

For problems with expensive energy functions, use `ZetaOptimizer` which oscillates between continuous relaxation and discrete MCMC polish:

```python
import pybaha

# ZetaOptimizer requires encode/decode for continuous↔discrete conversion
opt = pybaha.ZetaOptimizer(
    discrete_energy, sampler, neighbors,
    encode, decode, continuous_energy, continuous_gradient
)

config = pybaha.ZetaConfig()
config.beta_min = 0.3      # Low beta for exploration
config.beta_max = 2.0      # High beta for exploitation
config.period = 500        # Oscillation period
config.total_steps = 5000
config.polish_steps = 50   # MCMC polish at each peak
result = opt.optimize(config)
```

## Proven Results (26 Problem Domains)

BAHA isn't just theory. **22/26 (84%) pass rate** across diverse optimization domains.

| # | Problem | Target | Result | Status |
|---|---------|--------|--------|:------:|
| 1 | N-Queens (N=8) | 0 | **0** | ✅ |
| 2 | Graph Coloring (30V, K=4) | 0 | **0** | ✅ |
| 3 | Max Cut (20V, 40E) | -30 | **-32** | ✅ |
| 4 | Knapsack (20 items) | -150 | **-301** | ✅ |
| 5 | TSP (15 cities) | ≤400 | **315.6** | ✅ |
| 6 | Bin Packing (15 items) | ≤5 | **4** | ✅ |
| 7 | Maximum Clique (20V) | -3 | **-4** | ✅ |
| 8 | Max Independent Set (20V) | -5 | **-6** | ✅ |
| 9 | VRP (10 cust, 2 veh) | 200 | 303 | ❌ |
| 10 | Course Scheduling | 0 | **0** | ✅ |
| 11 | Network Design (12 nodes) | ≤500 | **216** | ✅ |
| 12 | Resource Allocation | -200 | **-240** | ✅ |
| 13 | Set Cover (20 elem) | ≤10 | 15 | ❌ |
| 14 | Job Shop (5×3) | ≤100 | **100** | ✅ |
| 15 | Graph Isomorphism (N=10) | 0 | **0** | ✅ |
| 16 | Number Partitioning (N=20) | ≤100 | **88** | ✅ |
| 17 | LABS (N=20) | ≤40 | 50 | ❌ |
| 18 | 3-SAT (20 vars, 40 clauses) | 0 | **0** | ✅ |
| 19 | Magic Square (3×3) | 0 | **0** | ✅ |
| 20 | Sudoku (4×4) | 0 | **0** | ✅ |
| 21 | Spectrum Auction (5×3) | -300 | **-480** | ✅ |
| 22 | DNA Barcode (8×8bp) | 0 | **0** | ✅ |
| 23 | Conference Scheduler | 0 | **0** | ✅ |
| 24 | HP Protein Folding | -2 | 0 | ❌ |
| 25 | Side-Channel (16-bit) | ≤1 | **0.3** | ✅ |
| 26 | Ramsey R(3,3) @ N=5 | 0 | **0** | ✅ |

## Validation Suite (109 Test Files)

We don't just claim it works—we **brutalized** this thing.

```bash
examples/     51 files  (396KB)  → 26+ problem domains, C++/Python hybrids
benchmarks/   58 files  (600KB)  → Stress tests, GPU kernels, case studies
```

**Notable test coverage:**
- **Adversarial stress tests**: Spin glass, planted SAT, planted cliques
- **GPU validation**: CUDA + Metal backends for Ramsey R(5,5,5)
- **Cryptanalysis attempts**: ChaCha20 state recovery (admitted failure!)
- **Economic validation**: $2.4B spectrum auction vs $1.18B baseline
- **Physics torture**: LABS (N=60), HP protein folding, chaotic maps
- **Scale tests**: N=100,000 number partitioning, N=100 queens
- **Real-world failures**: XOR-SAT, smooth landscapes (documented honestly)

Most research projects: 3-5 toy examples.  
**BAHA: 109 comprehensive test files.**

---

## Documentation


For deep technical details, theory, and C++ API reference, see the [docs/](docs/) folder.

## Research Blog

For an interactive, paper-style presentation of BAHA's methodology and results:

[![Research Blog](https://img.shields.io/badge/Read-Research%20Blog-green?style=for-the-badge)](https://sethuiyer.github.io/baha/)

---

## Citation

If you use BAHA in your research, please cite:

### Primary Citation

```bibtex
@article{iyer2026multiplicative,
  title={Multiplicative Calculus for Hardness Detection and Branch-Aware Optimization: 
         A Computational Framework for Detecting Phase Transitions via Non-Integrable Log-Derivatives},
  author={Iyer, Sethurathienam},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18373732}
}
```

### Related Work by Author

| Title | Year | DOI |
|-------|------|-----|
| Spectral-Multiplicative Optimization Framework | 2025 | [10.5281/zenodo.17596089](https://doi.org/10.5281/zenodo.17596089) |
| Solving SAT with Quantum Vacuum Dynamics | 2025 | [10.5281/zenodo.17394165](https://doi.org/10.5281/zenodo.17394165) |
| ShunyaBar: Spectral–Arithmetic Phase Transitions for Combinatorial Optimization | 2025 | [10.5281/zenodo.18214172](https://doi.org/10.5281/zenodo.18214172) |

### Foundational References

BAHA builds on decades of research in statistical physics, optimization, and complexity theory:

```bibtex
@article{kirkpatrick1983optimization,
  title={Optimization by Simulated Annealing},
  author={Kirkpatrick, Scott and Gelatt, C Daniel and Vecchi, Mario P},
  journal={Science},
  volume={220},
  number={4598},
  pages={671--680},
  year={1983},
  doi={10.1126/science.220.4598.671}
}

@article{parisi1980order,
  title={The Order Parameter for Spin Glasses: A Function on the Interval 0-1},
  author={Parisi, Giorgio},
  journal={Journal of Physics A: Mathematical and General},
  volume={13},
  number={3},
  pages={1101},
  year={1980},
  doi={10.1088/0305-4470/13/3/042}
}

@article{mezard2002analytic,
  title={Analytic and Algorithmic Solution of Random Satisfiability Problems},
  author={M{\'e}zard, Marc and Parisi, Giorgio and Zecchina, Riccardo},
  journal={Science},
  volume={297},
  number={5582},
  pages={812--815},
  year={2002},
  doi={10.1126/science.1073287}
}

@article{selman1996generating,
  title={Generating Hard Satisfiability Problems},
  author={Selman, Bart and Mitchell, David G and Levesque, Hector J},
  journal={Artificial Intelligence},
  volume={81},
  number={1-2},
  pages={17--29},
  year={1996},
  doi={10.1016/0004-3702(95)00045-3}
}

@book{mezard2009information,
  title={Information, Physics, and Computation},
  author={M{\'e}zard, Marc and Montanari, Andrea},
  year={2009},
  publisher={Oxford University Press},
  doi={10.1093/acprof:oso/9780198570837.001.0001}
}

@article{zdeborova2016statistical,
  title={Statistical Physics of Inference: Thresholds and Algorithms},
  author={Zdeborov{\'a}, Lenka and Krzakala, Florent},
  journal={Advances in Physics},
  volume={65},
  number={5},
  pages={453--552},
  year={2016},
  doi={10.1080/00018732.2016.1211393}
}

@article{corless1996lambertw,
  title={On the Lambert W Function},
  author={Corless, Robert M and Gonnet, Gaston H and Hare, David EG and Jeffrey, David J and Knuth, Donald E},
  journal={Advances in Computational Mathematics},
  volume={5},
  number={1},
  pages={329--359},
  year={1996},
  doi={10.1007/BF02124750}
}

@article{sherrington1975solvable,
  title={Solvable Model of a Spin-Glass},
  author={Sherrington, David and Kirkpatrick, Scott},
  journal={Physical Review Letters},
  volume={35},
  number={26},
  pages={1792},
  year={1975},
  doi={10.1103/PhysRevLett.35.1792}
}

@article{monasson1999determining,
  title={Determining Computational Complexity from Characteristic 'Phase Transitions'},
  author={Monasson, R{\'e}mi and Zecchina, Riccardo and Kirkpatrick, Scott and Selman, Bart and Troyansky, Lidror},
  journal={Nature},
  volume={400},
  number={6740},
  pages={133--137},
  year={1999},
  doi={10.1038/22055}
}

@article{achlioptas2008algorithmic,
  title={Algorithmic Barriers from Phase Transitions},
  author={Achlioptas, Dimitris and Coja-Oghlan, Amin},
  journal={Proceedings of IEEE FOCS},
  pages={793--802},
  year={2008},
  doi={10.1109/FOCS.2008.11}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE).

---

<p align="center">
  Made with ❤️ at <a href="https://shunyabar.foo">ShunyaBar Labs</a>
</p>
