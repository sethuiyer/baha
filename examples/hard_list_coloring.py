import networkx as nx
import pybaha
import random
import sys
import time

# "Minimalist Mode" Parameters
N_NODES = 13          # Tiny
TARGET_EDGES = 8      # Explicit edge count request
EDGE_PROB = -1.0      # Unused, we will use gnm_random_graph
N_COLORS = 3          # 3-Coloring
DOMAIN_SIZE_MIN = 3   # Standard 3-Coloring
DOMAIN_SIZE_MAX = 3   # Standard 3-Coloring
SEED = 123            # Simple seed

random.seed(SEED)

def create_problem():
    print(f"Creating Minimalist graph with N={N_NODES}, M={TARGET_EDGES}...")
    G = nx.gnm_random_graph(N_NODES, TARGET_EDGES, seed=SEED)
    
    # Generate domains
    domains = {}
    all_colors = list(range(N_COLORS))
    for i in range(N_NODES):
        # Dynamic choice based on parameters
        k = random.randint(DOMAIN_SIZE_MIN, DOMAIN_SIZE_MAX)
        domains[i] = random.sample(all_colors, k)
        
    print(f"Graph generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Constraints: Nodes have exactly {DOMAIN_SIZE_MAX} valid colors.")
    return G, domains

def energy(state, G):
    conflicts = 0
    for u, v in G.edges():
        if state[u] == state[v]:
            conflicts += 1
    return float(conflicts)

def sampler(domains):
    # Randomly pick a valid color for each node from its domain
    state = []
    for i in range(N_NODES):
        state.append(random.choice(domains[i]))
    return state

def greedy_sampler(G, domains):
    # Greedy initialization: Pick color that minimizes conflicts with already-colored neighbors
    state = [None] * N_NODES
    # Sort nodes by degree (DSATUR-lite heuristic) to color hard constraints first
    nodes_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
    
    for node, _ in nodes_by_degree:
        # Check colored neighbors
        neighbor_colors = {state[nbr] for nbr in G.neighbors(node) if state[nbr] is not None}
        
        # Find valid color in domain that isn't in neighbor_colors
        valid_freedom = []
        for color in domains[node]:
            if color not in neighbor_colors:
                valid_freedom.append(color)
        
        if valid_freedom:
            state[node] = random.choice(valid_freedom)
        else:
            # Conflict unavoidable (local), pick random from domain to respect constraints
            state[node] = random.choice(domains[node])
            
    return state

def neighbors(state, domains):
    nbrs = []
    # Increase neighbor sampling for harder landscape
    for _ in range(20):
        node = random.randint(0, N_NODES - 1)
        current_color = state[node]
        domain = domains[node]
        
        # In binary domain, there is only 1 other choice
        if len(domain) > 1:
            new_color = random.choice([c for c in domain if c != current_color])
            new_state = list(state)
            new_state[node] = new_color
            nbrs.append(new_state)
            
    return nbrs

def main():
    G, domains = create_problem()
    
    e_wrapper = lambda s: energy(s, G)
    s_wrapper = lambda: greedy_sampler(G, domains)  # UPGRADE: Smart Initialization
    n_wrapper = lambda s: neighbors(s, domains)
    
    print("Initializing BAHA Optimizer (The Fracture Hunter)...")
    opt = pybaha.Optimizer(e_wrapper, s_wrapper, n_wrapper)
    
    config = pybaha.Config()
    # config.timeout_ms = 10000.0  # Give it 10s of breathing room
    
    print("Optimization started...")
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
        print("\n✅ SUCCESS: The Fracture Hunter survived Evil Mode.")
        sys.exit(0)
    else:
        print(f"\n❌ FAILURE: Stuck with {int(result.best_energy)} conflicts.")
        sys.exit(1)

if __name__ == "__main__":
    main()
