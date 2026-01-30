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
    
    # --- PRIME WEIGHTING LOGIC ---
    print("Assigning Primes based on Node Degree...")
    # 1. Generate enough primes
    # Prime number theorem: pi(x) ~ x/ln(x). For 1300 primes, we need roughly x ~ 11000.
    primes = sieve_of_eratosthenes(12000)[:N_NODES]
    if len(primes) < N_NODES:
        raise ValueError("Not enough primes generated")
        
    # 2. Sort nodes by degree (Higher Degree = Higher Priority = Smaller Prime)
    nodes_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
    
    # 3. Assign primes and calculate weights
    # User Formula: log(1/p^2)
    # Interpretation: We want to PENALIZE conflicts on important nodes.
    # log(1/p^2) is negative and magnitude decreases with p.
    # To use as a penalty, we can take magnitude or inverse.
    # Given we want p=2 to be MOST important, we use 1/p as a robust proxy.
    # Or strictly following user intuition of "lower primes" = "weigh more":
    # Weight = 1.0 / prime  (Harmonic Weighting)
    
    weights = [0.0] * N_NODES
    for rank, (node, degree) in enumerate(nodes_by_degree):
        p = primes[rank]
        # Using 1000/p to keep numbers in a reasonable float range
        weights[node] = 1000.0 / p 
        
    print(f"Top Node (Deg {nodes_by_degree[0][1]}): Prime {primes[0]}, Weight {weights[nodes_by_degree[0][0]]:.4f}")
    print(f"Bot Node (Deg {nodes_by_degree[-1][1]}): Prime {primes[-1]}, Weight {weights[nodes_by_degree[-1][0]]:.4f}")

    return G, edges, domains, weights

def energy_weighted(state, edges, weights):
    # Energy = Sum of weights of conflicting nodes
    # If u and v conflict, we add (W[u] + W[v]) to energy.
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
    
    # Bind weights to energy function
    e_wrapper = lambda s: energy_weighted(s, edges, weights)
    s_wrapper = lambda: sampler(domains) # Random Init
    n_wrapper = lambda s: neighbors(s, domains)
    
    print("Initializing BAHA Optimizer (Prime-Weighted)...")
    opt = pybaha.Optimizer(e_wrapper, s_wrapper, n_wrapper)
    
    config = pybaha.Config()
    config.timeout_ms = 15000.0 # 15s timeout
    config.samples_per_beta = 5 # Fast sampling
    
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
        print("\n✅ SUCCESS: Prime Weighting cracked the Monster!")
        sys.exit(0)
    else:
        print(f"\n❌ FAILURE: {raw_conflicts} conflicts remaining.")
        sys.exit(1)

if __name__ == "__main__":
    main()
