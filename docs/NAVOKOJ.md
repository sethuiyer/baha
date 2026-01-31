# Navokoj: Industrial-Grade Constraint Thermodynamics Engine

> **BAHA is the open foundation. Navokoj has one more trick — the one that matters.**

[![Navokoj](https://img.shields.io/badge/Try-Navokoj-blue?style=for-the-badge)](https://navokoj.shunyabar.foo)

---

## What is Navokoj?

**Navokoj** is ShunyaBar Labs' commercial constraint satisfaction engine. It uses BAHA-style fracture detection, QSTATE native encoding, and **one core proprietary technique** that prevents optimizers from getting stuck in glassy landscapes.

Without that technique, you hit the glass ceiling. With it, you get:

- **Graceful degradation** — never just "UNSAT and bye"
- **Anytime solutions** — always returns best-so-far
- **Sub-100ms median latency** for enterprise workloads
- **Millions of clauses** without choking

This document shares empirical benchmarks from January 2026.

---

## Core API Evaluation (Jan 2026)

### Overall Summary

| Metric | Value |
|--------|-------|
| Total Tests Executed | 47 |
| Tests Passed | 45 |
| **Pass Rate** | **95.7%** |
| Known Limitations | 2 |
| Crash Rate | 0% |
| Median Solve Time | 88 ms |
| 95th Percentile Solve Time | 261 ms |
| Minimum Solve Time (Floor) | 35–40 ms |
| Batch Throughput | 30–32 solves/sec |
| Diagnostic Throughput | 130–2,600 vars/ms |

---

## Solve Time Benchmarks

### CNF / Expression / Scheduling

| Problem Class | Variables | Clauses/Ops | Engine | Solve Time | Satisfaction |
|---------------|-----------|-------------|--------|------------|--------------|
| Minimal | 3 | 2 | nano | 36 ms | 100% |
| Small sparse | 50 | 3 | nano | 40–153 ms | 100% |
| Medium dense | 500 | 35 | nano | 88 ms | 100% |
| Large sparse | 1,000 | 2 | nano | 40–112 ms | 100% |
| Extra-large sparse | 10,000 | 5 | nano | 174 ms | 100% |
| Max clause length | 100 | 1 (50 lits) | nano | 261 ms | 100% |
| UNSAT simple | 3 | 2 | nano | 106 ms | 50% |
| UNSAT complex | 5 | 4 | nano | 710 ms | 75% |
| Diagnostic | 50,000 | 5 | nano | 379 ms | 99.9% |
| Schedule (small) | 6 slots | 2 | nano | 97 ms | 100% |
| Schedule (medium) | 105 slots | 4 | nano | 99 ms | 100% |
| Schedule (infeasible) | 6 slots | 2 | mini | 2,874 ms | 88.9% |
| Batch (3 problems) | 60 | 9 | nano | 99 ms | 100% |
| Batch (5 problems) | 1,850 | 15 | nano | 156 ms | 100% |
| Expression simple | 3 | 2 ops | mini | 36 ms | 100% |
| Expression XOR | 6 | 5 ops | mini | 37 ms | 100% |
| Expression nested | 5 | 3 ops | mini | 42 ms | 100% |
| Expression complex | 18 | 7 ops | mini | 45 ms | 100% |

---

## Edge Case Coverage

### Structural & Operator Edge Cases

| Edge Case | Result |
|-----------|--------|
| Empty constraints | SAT |
| Single variable / clause | SAT |
| Max variables (100k) | SAT |
| Sparse structure | SAT |
| Dense structure | SAT |
| Mixed polarity | SAT |
| Monotone positive / negative | SAT |
| Direct contradiction | UNSAT (50%) |
| Deep nesting (≤5) | SAT |
| XOR chains | SAT |
| Mixed operators | SAT |

---

## Anytime / Partial Satisfaction

### Infeasible Instance Outcomes

| Scenario | Satisfaction Rate | Notes |
|----------|-------------------|-------|
| Direct contradiction | 66.7% | Blame split across vars |
| Multi-clause contradiction | 75% | Single variable blamed |
| Scheduling infeasible | 88.9% | Coverage constraint violated |
| Timeout | Best-so-far | Always returns assignment |

**Key insight**: Navokoj never says "UNSAT" and walks away. It always returns the best partial solution with blame attribution.

---

## Industrial Stress Tests

### Kubernetes Placement (Dec 2025)

| Metric | Value |
|--------|-------|
| Services | 100 |
| Nodes | 50 |
| Variables | 5,000 |
| **Total Clauses** | **2,061,600** |
| Satisfaction | 100% |
| Solve Time | ~17 minutes |
| Compute Cost | ~$2 |
| Violations | 0 |

---

## PSPACE-Complete Benchmarks

### Unified SAT Encoding Results

| Problem Class | Tests | Passed | Avg Solve Time |
|---------------|-------|--------|----------------|
| QBF (Nested) | 4 | 4 | 0.040 s |
| Sokoban | 3 | 3 | 0.047 s |
| Geography | 4 | 4 | 0.105 s |
| Reversi | 3 | 3 | 0.208 s |
| Planning (STRIPS) | 3 | 3 | 0.182 s |
| Parity Games | 4 | 4 | 0.052 s |
| **Total** | **21** | **21** | **0.124 s avg** |

**100% pass rate on PSPACE-complete problems.**

---

## Sokoban Planning Benchmarks

| Level | Variables | Clauses | Satisfaction | Solve Time |
|-------|-----------|---------|--------------|------------|
| Micro | 156 | 509 | 99.80% | 10.5 s |
| Easy | 1,738 | 24,418 | 100% | 22.3 s |
| Medium | 2,840 | 54,728 | 100% | 4.3 s |
| Classic Mini | 3,056 | 53,174 | 100% | 3.9 s |

> **"Bigger is faster" paradox**: Higher constraint density helps the solver prune more aggressively. Medium (54K clauses) solves 5x faster than Easy (24K clauses).

---

## Reversible Pebbling (PSPACE, Temporal Constraints)

| Instance | Clauses | Satisfaction | Solve Time |
|----------|---------|--------------|------------|
| Small | 10,235 | 100% | 2.8 s |
| Medium | 25,078 | 100% | 8.7 s |
| Large | 53,123 | 100% | 28.4 s |
| Extra Large | 101,561 | 99.94% | 43.6 s |
| Industrial | 299,780 | 99.39% | 126.5 s |
| Institutional | 476,243 | 98.60% | ~259 s |
| Hyper-Scale | 727,063 | 97.78% | ~340 s |
| **Limit Breaker 2** | **2,441,865** | **97.55%** | **~860 s** |

**2.4 million clauses. 97.55% satisfaction. 14 minutes.**

---

## Ramsey Theory Stress Tests

| Problem | Clauses | Satisfaction | Solve Time |
|---------|---------|--------------|------------|
| R(3,3,3) N=16 | ~2,000 | 100% | 400 s |
| R(3,3,3) N=20 | 4,180 | 99.93% | ~1,008 s |
| R₁₀(3) K100 | 1.84M | 100% | ~614 s |
| R(5,5,5) N=42 | 2.55M | 100% | GPU |
| **R(5,5,5) N=52** | **~8M** | **100%** | **~17 min** |
| R(6,6) K35 | 3.25M | 100% | ~24 min |

---

## SATLIB & Industrial Benchmarks

| Benchmark Type | Satisfaction Rate |
|----------------|-------------------|
| Random 3-SAT | 98% |
| Graph Coloring | 96% |
| XOR + CNF Hybrid | 99.9% |
| 5-way XOR Chains | 100% |
| Weighted Constraints | 100% |
| Boolean Expressions | 100% |

---

## Engine Comparison

| Engine | Satisfaction Rate | Speed | Quality | Use Case |
|--------|-------------------|-------|---------|----------|
| **PRO** | **92.57%** | 7.90/sec | 99.92% | Mission-critical |
| MINI | 31.37% | 10.64/sec | 99.55% | Balanced |
| NANO | 3.24% | Ultra-fast | 96.41% | Real-time |

**PRO engine achieves 92.57% perfect solve rate on SAT Competition 2024 benchmark.**

---

## Headline Results

| Achievement | Details |
|-------------|---------|
| **AI Escargot** | Hardest known Sudoku, solved in **9 seconds** (native QSTATE encoding) |
| **SAT 2024** | **92.57%** perfect solve rate |
| **2.4M clauses** | 97.55% satisfaction in 14 minutes |
| **R(5,5,5) N=52** | 8M clauses, 100% satisfaction, 17 minutes |
| **PSPACE** | 100% pass rate, <200ms average |

---

## TL;DR

- **Sub-100ms median latency** for enterprise SAT
- **Graceful degradation everywhere** — never "UNSAT and bye"
- **Millions of clauses** handled without choking
- **PSPACE problems actually usable**, not just theoretical
- This is less "solver" and more **constraint thermodynamics engine**

---

## Try Navokoj

[![Navokoj](https://img.shields.io/badge/Try-Navokoj-blue?style=for-the-badge)](https://navokoj.shunyabar.foo)

**BAHA** (this repo) is the open-source foundation. Navokoj adds one more technique — the one that breaks through glassy landscapes.

---

<p align="center">
  <a href="https://shunyabar.foo">ShunyaBar Labs</a> · <a href="https://navokoj.shunyabar.foo">Navokoj</a>
</p>
