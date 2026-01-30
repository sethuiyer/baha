import networkx as nx
import pybaha
import random
import sys
import time

# "MONSTER MODE" Parameters
N_NODES = 1300        # Large
EDGE_PROB = 0.314     # Dense (~265k edges)
N_COLORS = 4          # 4-Coloring
DOMAIN_SIZE_MIN = 4   
DOMAIN_SIZE_MAX = 4   
SEED = 999            

random.seed(SEED)

def create_problem():
    print(f"Creating 'MONSTER' Erdos-Renyi graph with N={N_NODES}, p={EDGE_PROB}...")
    G = nx.erdos_renyi_graph(N_NODES, EDGE_PROB, seed=SEED)
    
    # Pre-fetch edges to a list to avoid iterator overhead in inner loop
    edges = list(G.edges())
    
    # Generate domains
    domains = {}
    all_colors = list(range(N_COLORS))
    for i in range(N_NODES):
        k = random.randint(DOMAIN_SIZE_MIN, DOMAIN_SIZE_MAX)
        domains[i] = random.sample(all_colors, k)
        
    print(f"Graph generated: {G.number_of_nodes()} nodes, {len(edges)} edges")
    print(f"Constraints: Nodes have exactly {DOMAIN_SIZE_MAX} valid colors.")
    return G, edges, domains

def energy(state, edges):
    # Optimized Python energy loop
    conflicts = 0
    for u, v in edges:
        if state[u] == state[v]:
            conflicts += 1
    return float(conflicts)

def sampler(domains):
    state = []
    for i in range(N_NODES):
        state.append(random.choice(domains[i]))
    return state

def greedy_sampler(G, domains):
    print("Running Greedy Initialization (DSATUR-ish)...")
    state = [None] * N_NODES
    nodes_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
    
    for node, _ in nodes_by_degree:
        neighbor_colors = {state[nbr] for nbr in G.neighbors(node) if state[nbr] is not None}
        valid_freedom = [c for c in domains[node] if c not in neighbor_colors]
        
        if valid_freedom:
            state[node] = random.choice(valid_freedom)
        else:
            state[node] = random.choice(domains[node]) # Conflict
            
    return state

def neighbors(state, domains):
    nbrs = []
    # Sample 20 neighbors
    for _ in range(20):
        node = random.randint(0, N_NODES - 1)
        current_color = state[node]
        domain = domains[node]
        if len(domain) > 1:
            new_color = random.choice([c for c in domain if c != current_color])
            new_state = list(state)
            new_state[node] = new_color
            nbrs.append(new_state)
    return nbrs

def main():
    G, edges, domains = create_problem()
    
    e_wrapper = lambda s: energy(s, edges)
    s_wrapper = lambda: sampler(domains) # Back to Raw Random Init
    n_wrapper = lambda s: neighbors(s, domains)
    
    print("Initializing BAHA Optimizer...")
    opt = pybaha.Optimizer(e_wrapper, s_wrapper, n_wrapper)
    
    config = pybaha.Config()
    config.timeout_ms = 15000.0 # 15s timeout
    config.samples_per_beta = 5 # Reduce overhead of greedy sampling
    
    print("Optimization started (running for 15s, fast sampling)...")
    start_time = time.time()
    result = opt.optimize(config)
    end_time = time.time()
    
    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    print(f"Time Taken: {end_time - start_time:.4f} seconds")
    print(f"Final Energy (Conflicts): {result.best_energy}")
    
    try:
        print(f"Branch Jumps: {result.branch_jumps}")
        print(f"Fractures Detected: {result.fractures_detected}")
    except:
        pass

    if result.best_energy == 0.0:
        print("\n✅ SUCCESS: The Monster is Tamed.")
        sys.exit(0)
    else:
        print(f"\n❌ FAILURE: Monster won with {int(result.best_energy)} conflicts.")
        sys.exit(1)

if __name__ == "__main__":
    main()
