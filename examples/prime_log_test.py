import networkx as nx
import pybaha
import random
import sys
import time
import math

# "MONSTER MODE" Parameters (Same as Scale Test)
N_NODES = 1300        
EDGE_PROB = 0.314     
N_COLORS = 4          
DOMAIN_SIZE_MIN = 4   
DOMAIN_SIZE_MAX = 4   
SEED = 999            

random.seed(SEED)

def sieve_of_eratosthenes(n):
    primes = []
    is_prime = [True] * (n + 1)
    for p in range(2, n + 1):
        if is_prime[p]:
            primes.append(p)
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
    return primes

def create_problem():
    print(f"Creating 'MONSTER' Erdos-Renyi graph with N={N_NODES}, p={EDGE_PROB}...")
    G = nx.erdos_renyi_graph(N_NODES, EDGE_PROB, seed=SEED)
    
    edges = list(G.edges())
    
    # Generate domains
    domains = {}
    all_colors = list(range(N_COLORS))
    for i in range(N_NODES):
        k = random.randint(DOMAIN_SIZE_MIN, DOMAIN_SIZE_MAX)
        domains[i] = random.sample(all_colors, k)
    
    # --- PRIME LOG WEIGHTING LOGIC ---
    print("Assigning Primes based on Node Degree...")
    primes = sieve_of_eratosthenes(12000)[:N_NODES]
    if len(primes) < N_NODES:
        raise ValueError("Not enough primes generated")
        
    nodes_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
    
    weights = [0.0] * N_NODES
    for rank, (node, degree) in enumerate(nodes_by_degree):
        p = primes[rank]
        # KEY INNOVATION: 1.0 / log(p)
        # Gentle decay, unique mass, breaks symmetry without massive distortion
        weights[node] = 1.0 / math.log(p) 
        
    print(f"Top Node (Deg {nodes_by_degree[0][1]}): Prime {primes[0]}, Weight {weights[nodes_by_degree[0][0]]:.4f}")
    print(f"Bot Node (Deg {nodes_by_degree[-1][1]}): Prime {primes[-1]}, Weight {weights[nodes_by_degree[-1][0]]:.4f}")

    return G, edges, domains, weights

def energy_weighted(state, edges, weights):
    # Energy = Sum of weights of conflicting nodes
    total_penalty = 0.0
    for u, v in edges:
        if state[u] == state[v]:
            total_penalty += (weights[u] + weights[v])
    return total_penalty

def sampler(domains):
    state = []
    for i in range(N_NODES):
        state.append(random.choice(domains[i]))
    return state

def neighbors(state, domains):
    nbrs = []
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
    G, edges, domains, weights = create_problem()
    
    e_wrapper = lambda s: energy_weighted(s, edges, weights)
    s_wrapper = lambda: sampler(domains) # Random Init
    n_wrapper = lambda s: neighbors(s, domains)
    
    print("Initializing BAHA Optimizer (Log-Prime-Weighted)...")
    try:
        opt = pybaha.Optimizer(e_wrapper, s_wrapper, n_wrapper)
    except AttributeError:
        # Fallback if pybaha bindings don't support it directly, but they should
        print("Error initializing optimizer")
        sys.exit(1)
    
    config = pybaha.Config()
    config.timeout_ms = 60000.0 # 60s timeout
    config.samples_per_beta = 64 # Proper sampling
    
    print("Optimization started (running for 15s)...")
    start_time = time.time()
    result = opt.optimize(config)
    end_time = time.time()
    
    # Calculate raw conflicts for comparison
    raw_conflicts = 0
    for u, v in edges:
        if result.best_state[u] == result.best_state[v]:
            raw_conflicts += 1
            
    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    print(f"Time Taken: {end_time - start_time:.4f} seconds")
    print(f"Final Weighted Energy: {result.best_energy:.4f}")
    print(f"Final RAW Conflicts: {raw_conflicts}")
    
    try:
        print(f"Branch Jumps: {result.branch_jumps}")
        print(f"Fractures Detected: {result.fractures_detected}")
    except:
        pass

    if raw_conflicts == 0:
        print("\n✅ SUCCESS: Log-Prime Weighting cracked the Monster!")
        sys.exit(0)
    else:
        print(f"\n❌ FAILURE: {raw_conflicts} conflicts remaining.")
        sys.exit(1)

if __name__ == "__main__":
    main()
