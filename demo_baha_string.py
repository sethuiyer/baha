import sys
import os
import random
import string

# Ensure valid import
sys.path.append(os.getcwd())
import pybaha

TARGET = "BAHA is the future of optimization!"
ALPHABET = string.ascii_letters + string.punctuation + " "

def energy(state):
    # Sum of squared differences in character codes
    # This provides a smooth gradient towards the target character
    diff = 0
    state_str = "".join(state)
    for s_char, t_char in zip(state_str, TARGET):
        diff += (ord(s_char) - ord(t_char))**2
    return float(diff)

def sampler():
    # Random initial string of same length
    return [random.choice(ALPHABET) for _ in range(len(TARGET))]

def neighbors(state):
    nbrs = []
    # Generate 10 neighbors by mutating one character
    for _ in range(10):
        new_state = list(state)
        idx = random.randint(0, len(state) - 1)
        # Mutate slightly or randomly
        if random.random() < 0.5:
            # Small shift
            char_code = ord(new_state[idx]) + random.choice([-1, 1])
            new_state[idx] = chr(max(32, min(126, char_code)))
        else:
            # Random replacement
            new_state[idx] = random.choice(ALPHABET)
        nbrs.append(new_state)
    return nbrs

def main():
    print(f"🎯 Target: '{TARGET}'")
    print(f"📏 Length: {len(TARGET)}")
    
    opt = pybaha.Optimizer(energy, sampler, neighbors)
    
    config = pybaha.Config()
    config.beta_steps = 2000
    config.timeout_ms = 5000
    config.verbose = True
    
    print("\nStarting Optimization...")
    result = opt.optimize(config)
    
    final_str = "".join(result.best_state)
    print("\n" + "="*50)
    print(f"Result: '{final_str}'")
    print(f"Energy: {result.best_energy}")
    print(f"Jumps:  {result.branch_jumps}")
    print("="*50)
    
    if final_str == TARGET:
        print("✅ Success!")
    else:
        print("❌ Not quite there.")

if __name__ == "__main__":
    main()
