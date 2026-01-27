# BAHA Case Studies

This document showcases BAHA's performance across diverse hard optimization problems, demonstrating the power of fracture detection and branch-aware navigation.

---

## 1. Energy Landscape & Fracture Detection

<p align="center">
  <img src="data/energy_landscape.webp" alt="BAHA Energy Landscape" width="700"/>
</p>

**Core Concept:** Unlike Simulated Annealing which "walks" through the energy landscape, BAHA detects **fractures** (sharp discontinuities in log Z) and performs **Lambert-W branch jumps** to teleport between solution basins.

| Metric | Traditional SA | BAHA |
|--------|---------------|------|
| Navigation | Random walk | Directed jumps |
| Fracture Response | Ignores | Exploits |
| Basin Discovery | Slow diffusion | Instant recognition |

---

## 2. List Coloring (Constrained Graph Coloring)

<p align="center">
  <img src="data/list_coloring.webp" alt="List Coloring" width="650"/>
</p>

**Problem:** Each vertex has its own list of allowed colors. Find a valid coloring where no adjacent vertices share the same color.

**Why It's Hard:** More constrained than standard graph coloring—some vertices may only have 1-2 options, creating bottlenecks.

| Benchmark | BAHA | Random (Best of 100) |
|-----------|------|----------------------|
| Energy | 0 (Perfect) | 150+ |
| Improvement | — | **80%** |
| Time | 1.2 seconds | N/A |

---

## 3. Graph Isomorphism (N=50)

<p align="center">
  <img src="data/graph_iso.webp" alt="Graph Isomorphism" width="650"/>
</p>

**Problem:** Given two graphs, find the vertex mapping that proves they are structurally identical.

**Why It's Hard:** The search space is $N! = 3 \times 10^{64}$ for N=50. Brute force is impossible.

| Benchmark | BAHA | Simulated Annealing |
|-----------|------|---------------------|
| Success Rate | **100%** | 20% |
| Solve Time | 0.2ms | Failed most cases |

---

## 4. Ramsey Theory: R(5,5,5) @ N=52

<p align="center">
  <img src="data/ramsey_102.webp" alt="Ramsey Coloring" width="500"/>
</p>

**Problem:** 3-color the edges of a complete graph $K_N$ such that no monochromatic $K_5$ exists.

**Why It's Hard:** At N=52, there are $3^{1326}$ colorings and 2.6 million clique constraints. The "Phase Transition" makes random search hopeless.

| Scale | Constraints | Result | Time |
|-------|-------------|--------|------|
| N=52 | 2.6M cliques | **Perfect (E=0)** | <30 sec |
| N=102 | 83.2M cliques | Reduced to 150 violations | Scale test |

---

## 5. Number Partitioning (Spectral Mode)

**Problem:** Split N large integers into two sets with equal sum.

**Why It's Hard:** The "Easiest Hard Problem" — random solutions average $O(\sqrt{N}) \cdot \text{max}(a_i)$ residue.

**BAHA Innovation:** Uses **Analytical Specific Heat** instead of sampling, achieving $O(N \log N)$ complexity.

| Scale | Method | Time | Improvement |
|-------|--------|------|-------------|
| N=1,000 | Spectral BAHA | **34 ms** | — |
| N=100,000 | Spectral BAHA | **13.6 sec** | $1.5 \times 10^6 \times$ over random |

---

## 6. Cryptanalysis: ChaCha20 State Recovery

**Problem:** Recover internal state of 2-round ChaCha20 from known outputs.

**Why It's Hard:** ARX ciphers mix addition, rotation, and XOR—creating a "glassy" energy landscape with no clear basins.

| Result | Observation |
|--------|-------------|
| Fracture Rate | ρ ≈ 0 (Flat landscape) |
| Interpretation | **No exploitable structure** — ChaCha20 is secure against BAHA-style attacks |

> This is a **security validation**: BAHA's failure to find fractures proves the cipher's hardness.

---

## Summary: When to Use BAHA

| Problem Type | BAHA Advantage | Best Mode |
|--------------|----------------|-----------|
| Constraint Satisfaction | Fracture exploitation | Standard |
| Partitioning | Analytical moments | Spectral |
| Graph Problems | Basin jumping | GPU-accelerated |
| Cryptanalysis | Hardness detection | Diagnostic |

**BAHA isn't just an optimizer—it's a hardness detector.** If BAHA finds fractures, the problem has exploitable structure. If it doesn't, you've proven the landscape is genuinely random.

---

*For implementation details, see the [examples/](examples/) and [benchmarks/](benchmarks/) directories.*
