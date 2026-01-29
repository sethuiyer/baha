import networkx as nx
import pybaha
import random
import sys
import time

# Parameters
N_NODES = 50
EDGE_PROB = 0.15
N_COLORS = 10
DOMAIN_SIZE_MIN = 2
DOMAIN_SIZE_MAX = 5
SEED = 42

random.seed(SEED)

def create_problem():
    print(f"Creating Erdos-Renyi graph with N={N_NODES}, p={EDGE_PROB}...")
    G = nx.erdos_renyi_graph(N_NODES, EDGE_PROB, seed=SEED)
    
    # Generate domains
    domains = {}
    all_colors = list(range(N_COLORS))
    for i in range(N_NODES):
        k = random.randint(DOMAIN_SIZE_MIN, DOMAIN_SIZE_MAX)
        domains[i] = random.sample(all_colors, k)
        
    print(f"Graph generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, domains

def energy(state, G):
    # State is a list where state[i] is the color of node i
    # Energy = number of conflicting edges
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

def neighbors(state, domains):
    # Change one node's color to another valid color from its domain
    nbrs = []
    # Generate a subset of neighbors for speed (Stochastic Hill Climbing)
    for _ in range(10):
        node = random.randint(0, N_NODES - 1)
        current_color = state[node]
        domain = domains[node]
        
        if len(domain) > 1:
            new_color = current_color
            while new_color == current_color:
                new_color = random.choice(domain)
            
            new_state = list(state)
            new_state[node] = new_color
            nbrs.append(new_state)
            
    return nbrs

def main():
    G, domains = create_problem()
    
    # Bind arguments to functions using lambdas to pass static data
    # Note: pybaha expects specialized signatures, so we wrap them
    e_wrapper = lambda s: energy(s, G)
    s_wrapper = lambda: sampler(domains)
    n_wrapper = lambda s: neighbors(s, domains)
    
    print("Initializing BAHA Optimizer...")
    opt = pybaha.Optimizer(e_wrapper, s_wrapper, n_wrapper)
    
    config = pybaha.Config()
    # config.timeout_ms = 5000.0  # Optional timeout
    
    print("Optimization started...")
    start_time = time.time()
    # Using default config, runs until convergence or max steps
    result = opt.optimize(config)
    end_time = time.time()
    
    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    print(f"Time Taken: {end_time - start_time:.4f} seconds")
    print(f"Final Energy (Conflicts): {result.best_energy}")
    
    # Attempt to access stats if available (bindings depending)
    try:
        jumps = result.branch_jumps
        print(f"Branch Jumps: {jumps}")
    except:
        pass

    if result.best_energy == 0.0:
        print("\n✅ SUCCESS: Valid list coloring found!")
        sys.exit(0)
    else:
        print(f"\n❌ FAILURE: Could not eliminate {int(result.best_energy)} conflicts.")
        sys.exit(1)

if __name__ == "__main__":
    main()
