#!/usr/bin/env python3
"""
Ramsey R(5,5,5) @ N=52 Witness Verifier
Verifies that the 3-coloring has NO monochromatic K5 clique.

Usage: python verify_ramsey.py data/ramsey_52_witness.csv
"""
import sys
import csv
from itertools import combinations

def verify_ramsey_witness(filepath, N=52, K=5, C=3):
    # Load edge colorings
    edges = {}
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            u, v, color = int(row['u']), int(row['v']), int(row['color'])
            edges[(u, v)] = color
    
    print(f"Loaded {len(edges)} edges")
    assert len(edges) == N * (N - 1) // 2, f"Expected {N*(N-1)//2} edges"
    
    # Check all K-cliques for mono
    mono_cliques = 0
    for vertices in combinations(range(N), K):
        clique_edges = [(vertices[i], vertices[j]) for i in range(K) for j in range(i+1, K)]
        colors = set(edges[e] for e in clique_edges)
        if len(colors) == 1:
            mono_cliques += 1
            print(f"  Mono clique found: {vertices}, color={colors.pop()}")
    
    print(f"\nResult: {mono_cliques} monochromatic K{K} cliques")
    if mono_cliques == 0:
        print("✅ VALID! R(5,5,5) > 52 PROVEN")
        return True
    else:
        print("❌ INVALID witness")
        return False

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ramsey_52_witness.csv"
    verify_ramsey_witness(path)
