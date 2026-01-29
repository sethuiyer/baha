import sys
import os
sys.path.append(os.getcwd())
import pybaha
import random

# Target: 32-bit key
TARGET_KEY = 0xDEADBEEF

def energy(candidate):
    # Simulate "leakage" (Hamming distance)
    leakage_distance = bin(candidate ^ TARGET_KEY).count('1')
    return float(leakage_distance)

def sampler():
    return random.getrandbits(32)

def neighbors(k):
    # Flip single bits in the key
    return [k ^ (1 << i) for i in range(32)]

print("🔐 RUNNING CRYPTANALYSIS TUTORIAL 🔐")
opt = pybaha.Optimizer(energy, sampler, neighbors)
config = pybaha.Config()
config.verbose = True
config.beta_steps = 1000
result = opt.optimize(config)

if result.best_state == TARGET_KEY:
    print(f"\n🚨 ALERT: KEY RECOVERED: {hex(result.best_state)}")
else:
    print(f"\n❌ FAILED. Best found: {hex(result.best_state)}")

print(f"Energy: {result.best_energy}")
print(f"Fractures: {result.fractures_detected}, Jumps: {result.branch_jumps}")
