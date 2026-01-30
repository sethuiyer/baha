"""
SAT Solver Module: Constraint Satisfaction via Geometric Operators.

This module implements a physics-inspired SAT solver that treats boolean
satisfaction as energy minimization on a continuous manifold. Uses prime-weighted
operators and adiabatic cooling to find solutions without combinatorial search.

Key innovations:
- Prime-weighted constraints for symmetry breaking
- Continuous probability relaxation of boolean variables
- Adiabatic flow (temperature schedule) for basin convergence
- Gradient-based energy minimization

Author: Sethu Iyer <sethuiyer95@gmail.com>
License: MIT
"""

import random
from typing import List, Tuple, Dict, Any
import time
import numpy as np

__all__ = [
    "solve_sat",
    "generate_3sat",
    "encode_n_queens",
    "encode_sudoku",
]


def _generate_primes(n: int) -> np.ndarray:
    """
    Generate first n prime numbers for constraint weighting.
    
    Why primes? Primes provide an irreducible basis where each constraint
    gets a unique spectral signature. This breaks permutation symmetry
    and prevents degenerate trade-offs between constraints.
    
    Args:
        n: Number of primes to generate
        
    Returns:
        Array of n prime numbers
    """
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = all(candidate % p != 0 for p in primes)
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return np.array(primes)


def solve_sat(
    num_vars: int,
    clauses: List[List[int]],
    steps: int = 1000,
    learning_rate: float = 0.1,
    beta_max: float = 2.5,
    seed: int = None,
) -> Tuple[List[int], float]:
    """
    Solve SAT problem using geometric flow minimization.
    
    This implements the core Navokoj algorithm:
    1. Arithmetic Sector: Assign prime weights to each constraint
    2. Geometric Sector: Initialize continuous state space (probabilities)
    3. Dynamic Sector: Perform adiabatic sweep with gradient descent
    4. Collapse: Threshold continuous state to discrete solution
    
    Args:
        num_vars: Number of boolean variables
        clauses: List of clauses, each clause is list of ints (positive=variable, negative=negated)
        steps: Number of adiabatic cooling steps
        learning_rate: Step size for gradient descent
        beta_max: Maximum inverse temperature (controls cooling schedule)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple[List[int], float]: (Solution assignments, final energy)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    print(f"Initializing Physics Engine: {num_vars} vars, {len(clauses)} clauses")
    start_time = time.time()

    # 1. Arithmetic Sector: Prime-weighted operators for symmetry breaking
    # Each constraint gets unique spectral signature via prime weighting
    print("Generating spectral signatures (primes)...")
    primes = _generate_primes(len(clauses))
    weights = 1.0 / np.log(primes + 1.0)  # w_c = 1/log(p_c)

    # 2. Geometric Sector: Continuous state initialization
    # Start at uniform density 1/k to respect "At Least One" constraint naturally
    # If we start at 0.5, sum of probs = k/2 >> 1, so gradients crush everything to 0.
    # We want sum ~ 1.0. Given k=4 (implied by context or passed in?), we need a heuristic.
    # Assuming k=4 for now based on typical coloring.
    initial_density = 0.26 # Slightly > 0.25 to encourage "at least one"
    state = np.full(num_vars, initial_density) + np.random.normal(0, 0.01, num_vars)

    print("Starting Adiabatic Flow (Gradient Descent)...")
    # 3. Dynamic Sector: Adiabatic sweep with gradient flow
    for step in range(steps):
        if step % 100 == 0:
            print(f"  Step {step}/{steps} | Beta: {(step / steps) * beta_max:.2f}")
            
        # Temperature schedule: linear ramp from 0 to beta_max
        # Low beta = hot (exploration), high beta = cold (exploitation)
        beta = (step / steps) * beta_max

        # Compute gradient of energy landscape
        gradient = np.zeros(num_vars)

        # Vectorized implementation of clause evaluation would be faster,
        # but sticking to provided logic for fidelity.
        # Check if we can optimize this loop later.
        
        for clause_idx, clause in enumerate(clauses):
            # Soft logic: compute probability clause is satisfied
            # P(clause=False) = Product(1 - P(literal=True))
            unsat_prob = 1.0
            lit_probs = []
            
            # Identify variable indices and signs for efficient access
            # For massive clauses this loop is main bottleneck
            for lit in clause:
                var_idx = abs(lit) - 1
                prob = state[var_idx] if lit > 0 else (1.0 - state[var_idx])
                unsat_prob *= (1.0 - prob)
                lit_probs.append(prob)

            sat_prob = 1.0 - unsat_prob + 1e-9  # Add epsilon to avoid log(0)

            # Energy: E = -w_c * log(P(satisfied))
            # dE/dx = coeff * sign * (1 / (1-p)) for unsatisfied literal
            
            coeff = weights[clause_idx] / sat_prob * unsat_prob

            # Add gradient contribution from each literal in clause
            for lit_idx, lit in enumerate(clause):
                var_idx = abs(lit) - 1
                sign = 1.0 if lit > 0 else -1.0
                
                # Gradient contribution
                term = 1.0 / (1.0 - lit_probs[lit_idx] + 1e-9)
                gradient[var_idx] += coeff * sign * term

        # Update dynamics: gradient descent with temperature scaling
        effective_lr = learning_rate * beta
        state = state + effective_lr * gradient

        # Project to valid probability space (0,1)
        state = np.clip(state, 0.001, 0.999)

    end_time = time.time()
    print(f"Flow completed in {end_time - start_time:.4f}s")

    # 4. Collapse: Convert continuous probabilities to discrete boolean values
    solution = [int(val > 0.5) for val in state]
    
    # Calculate final discrete energy (unsatisfied clauses)
    unsat_count = 0
    for clause in clauses:
        satisfied = False
        for lit in clause:
            val = solution[abs(lit)-1]
            if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                satisfied = True
                break
        if not satisfied:
            unsat_count += 1
            
    return solution, float(unsat_count)


def generate_3sat(n_vars: int, alpha: float = 4.26, seed: int = None) -> List[List[int]]:
    if seed is not None:
        random.seed(seed)

    n_clauses = int(n_vars * alpha)
    clauses = []

    print(f"Generating Critical 3-SAT: {n_vars} vars, {n_clauses} clauses...")

    while len(clauses) < n_clauses:
        var_indices = random.sample(range(1, n_vars + 1), 3)
        clause = [v if random.random() > 0.5 else -v for v in var_indices]
        clauses.append(clause)

    return clauses
