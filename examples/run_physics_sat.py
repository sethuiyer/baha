import networkx as nx
import physics_sat
import random
import sys

def encode_graph_coloring(G, n_colors):
    """
    Encode Graph Coloring as SAT.
    Variables: X_{i,c} for i in Nodes, c in Colors.
    Index mapping: variable (i * n_colors + c + 1)
    """
    num_nodes = G.number_of_nodes()
    clauses = []
    
    def var(node, color):
        # 1-based indexing for SAT
        return (node * n_colors + color) + 1
    
    # Constraint 1: Each node must have at least one color
    for i in range(num_nodes):
        clauses.append([var(i, c) for c in range(n_colors)])
        
    # Constraint 2: Each node must have at most one color (pairwise exclusion)
    # Optional for "at least one" if we strictly minimize conflicts, 
    # but strict coloring requires exactly one.
    # For efficiency, we often skip this if "at least one" + edge penalties drive it to valid state,
    # but for correctness let's add it.
    for i in range(num_nodes):
        for c1 in range(n_colors):
            for c2 in range(c1 + 1, n_colors):
                clauses.append([-var(i, c1), -var(i, c2)])
                
    # Constraint 3: Adjacent nodes cannot have same color
    for u, v in G.edges():
        for c in range(n_colors):
            # NAND: NOT (X_{u,c} AND X_{v,c}) <=> (NOT X_{u,c} OR NOT X_{v,c})
            clauses.append([-var(u, c), -var(v, c)])
            
    num_vars = num_nodes * n_colors
    return num_vars, clauses

def decode_coloring(solution, num_nodes, n_colors):
    coloring = {}
    for i in range(num_nodes):
        assigned = []
        for c in range(n_colors):
            # 1-based indexing
            idx = (i * n_colors + c)
            if solution[idx] == 1:
                assigned.append(c)
        coloring[i] = assigned
    return coloring

def main():
    # Use the "Evil" Graph Settings but smaller scale first because SAT encoding blows up
    # Standard Evil: N=150, p=0.06.
    N_NODES = 150
    EDGE_PROB = 0.06 # The "Evil" case
    N_COLORS = 4
    SEED = 666
    
    print(f"Generating Graph N={N_NODES}, p={EDGE_PROB}...")
    G = nx.erdos_renyi_graph(N_NODES, EDGE_PROB, seed=SEED)
    
    print("Encoding to SAT...")
    num_vars, clauses = encode_graph_coloring(G, N_COLORS)
    print(f"SAT Instance: {num_vars} variables, {len(clauses)} clauses")
    
    # Run Physics Solver
    print("\nRunning Physics-Inspired SAT Solver...")
    solution, best_energy = physics_sat.solve_sat(
        num_vars, 
        clauses, 
        steps=5000,          # Slow cooling
        learning_rate=0.05,  # Moderate steps
        beta_max=6.0,        # Standard freeze
        seed=123
    )
    
    print(f"\nFinal Discrete Unsat Clauses: {int(best_energy)}")
    
    # Verify Coloring Validity
    coloring = decode_coloring(solution, N_NODES, N_COLORS)
    
    # Check 1: Exactly one color per node?
    valid_assignment = True
    for i, colors in coloring.items():
        if len(colors) != 1:
            # print(f"Node {i} has invalid colors: {colors}")
            valid_assignment = False
            
    if not valid_assignment:
        print("❌ FAILED: Invalid assignment (nodes with 0 or >1 colors)")
    else:
        print("✅ Assignment Validity: All nodes have exactly 1 color.")
        
    # Check 2: Edge conflicts
    conflicts = 0
    state = {i: colors[0] for i, colors in coloring.items() if len(colors) == 1}
    for u, v in G.edges():
        if u in state and v in state and state[u] == state[v]:
            conflicts += 1
            
    print(f"Graph Coloring Conflicts: {conflicts}")
    
    if conflicts == 0 and valid_assignment:
        print("🚀 SUCCESS: Physics Solver cracked it!")
        sys.exit(0)
    else:
        print("failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
