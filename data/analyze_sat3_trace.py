import pandas as pd
import numpy as np

def analyze_traces():
    try:
        sa = pd.read_csv('data/sa_trace.csv')
        baha = pd.read_csv('data/baha_trace.csv')
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    print("--- 3-SAT (N=500) TRACE ANALYSIS ---")
    
    # SA Results
    print(f"SA Initial Energy: {sa['energy'].iloc[0]}")
    print(f"SA Final Energy: {sa['energy'].iloc[-1]}")
    
    # BAHA Results
    print(f"\nBAHA Initial Energy: {baha['energy'].iloc[0]}")
    print(f"BAHA Final Energy: {baha['energy'].iloc[-1]}")
    
    # Analyze BAHA Jumps
    jumps = baha[baha['type'] == 'branch_jumped'] # Assuming I logged this or something similar? 
    # Wait, my logger wrote 'fracture_detected' or 'step'. Let's check 'type'.
    
    fractures = baha[baha['type'] == 'fracture_detected']
    print(f"\nTotal Fractures Detected: {len(fractures)}")
    
    # Find Rho peaks
    baha['rho_smooth'] = baha['rho'].rolling(window=10).mean()
    peaks = baha.sort_values('rho', ascending=False).head(5)
    print("\nTop 5 Rho Peaks (Fractures):")
    print(peaks[['beta', 'rho', 'energy']])
    
    # Check if rho spikes at beta ~ 1.0
    beta_near_1 = baha[(baha['beta'] > 0.8) & (baha['beta'] < 1.2)]
    if not beta_near_1.empty:
        print(f"\nAvg Rho near beta=1.0: {beta_near_1['rho'].mean():.3f}")
        print(f"Max Rho near beta=1.0: {beta_near_1['rho'].max():.3f}")
    
    # Analyze Energy Trajectory
    # Find points where energy drops significantly
    baha['energy_diff'] = baha['energy'].diff()
    big_drops = baha[baha['energy_diff'] < -10]
    if not big_drops.empty:
        print("\nSignificant Energy Drops (Branch Jumps / Basin Discovery):")
        print(big_drops[['step', 'beta', 'energy', 'energy_diff']])

if __name__ == "__main__":
    analyze_traces()
